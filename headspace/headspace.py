#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import random
import sys
import time
import urllib.parse
import matplotlib.pyplot as plt

# Set tensorflow log level (0 = all, 1 is > INFO, 2 > WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow

# Make logging go to STDOUT.
root = logging.getLogger()
root.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s'))
root.addHandler(ch)

log = logging.getLogger(__name__)

# Not used right now, but could be useful if we want to parse grpc protocol URIs.
def register_scheme(scheme):
    for method in filter(lambda s: s.startswith('uses_'), dir(urllib.parse)):
        getattr(urllib.parse, method).append(scheme)
register_scheme('grpc')


def start_server(job_name, ports):
    log.info("Starting local workers for job {} at ports {}".format(job_name, ports))
    cluster = tensorflow.train.ClusterSpec({job_name:
        ["localhost:{0}".format(port) for port in ports]})

    # Create a number of workers and add their target URIs to the server list.
    workers = []
    for i, port in enumerate(ports):
        server = tensorflow.train.Server(cluster, job_name=job_name, task_index=i)
        worker_address = server.target.decode("utf-8")
        workers.append(worker_address)

    # Test that workers are running and log each running server:
    for i, worker_address in enumerate(workers):
        message = tensorflow.constant(
            "Tensorflow server running for job {} with task index {} at {}"
                .format(job_name, i, worker_address))
        with tensorflow.Session(worker_address) as sess:
            result = sess.run(message)
            log.info(result.decode("utf-8"))

    return workers


def run(job_function, job_name, workers=3, *args, **kwargs):
    # TODO: Allow specifying device here iso in benchmark.
    log.info("Starting job {}".format(job_name))
    # Create workers if needed.
    if isinstance(workers, int):
        r = random.randint(10000, 20000)
        workers = start_server(job_name, ports=[r + i for i in range(workers)])
    elif workers is None or len(workers) == 0:
        workers = start_server(job_name, ports=[random.randint(10000, 20000)])
    log.info("Workers: {}".format(workers))

    # This function should set up the calculation we want to perform.
    output_var = job_function(job_name=job_name,
                              workers=workers,
                              *args, **kwargs)

    # Now, run the calculation.
    with tensorflow.Session(workers[0]) as sess:
        result = sess.run(output_var)
        log.debug("Result is {}".format(result))
        return result


def make_benchmark_job(job_name, workers, device_name, size, data_type):
    shape = (size, size)

    # TODO: Use multiple workers if available.
    # Determine the device we want to use.
    device = "/job:" + job_name + "/task:0" + "/" + device_name
    log.info("Calculating {} for {} on {}".format(job_name, data_type.name, device))
    with tensorflow.device(device):
        # Initialize two matrices with random float values and compute their product.
        r1 = tensorflow.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
        r2 = tensorflow.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
        dot_operation = tensorflow.matmul(r2, r1)
    return dot_operation

def benchmark(max_time=10,
              workers=1,
              data_types=(
                  #tensorflow.float16, # Float16 is very slow on the CPU.
                  tensorflow.float32,
                  tensorflow.float64),
              devices=("gpu:0",
                       "cpu:0")):
    job_name = "benchmark"

    # Start some workers.
    r = random.randint(10000, 20000)
    servers = start_server(job_name, ports=[r + i for i in range(workers)])

    # Track run times per device.
    device_times = {data_type: {device: [] for device in devices} for data_type in data_types}

    # Create a list of matrix sizes to test.
    matrix_sizes = range(50, 5000, 100)
    # Run once to avoid measuring startup time for first device.
    run(job_function=make_benchmark_job,
        job_name=job_name,
        workers=servers,
        device_name=devices[0],
        size=1,
        data_type=data_types[0])

    # Track the matrix sizes used during the test for plotting purposes.
    used_matrix_sizes = []
    # Track the maximum time for this round to ensure that the test is not too slow.
    max_device_time = 0
    # Run the benchmark test.
    for size in matrix_sizes:
        used_matrix_sizes.append(size)

        for data_type in device_times:
            for device_name in device_times[data_type]:
                start_time = time.time()
                # Run a single benchmark computation for the selected device.
                run(job_function=make_benchmark_job,
                    job_name=job_name,
                    workers=servers,
                    device_name=device_name,
                    size=size,
                    data_type=data_type)
                # Determine computation time.
                time_taken = time.time() - start_time

                # Update the maximum time for this round if appropriate.
                max_device_time = max(max_device_time, time_taken)
                log.info("Run times for {}: {}".format(device_name, time_taken))
                device_times[data_type][device_name].append(time_taken)

        # Our test is becoming to slow, so break it off.
        if max_device_time > max_time:
            break

    # Report the run times.
    log.info("Run times: {}".format(device_times))

    line_styles = ["o-", "D-", "s-", ">-", "--", "-"]
    colors = ['orange', 'g', 'r', 'b', 'y', 'm', 'c', 'k']
    # Plot the resulting curves.
    for i, data_type in enumerate(device_times):
        for j, device in enumerate(device_times[data_type]):
            times = device_times[data_type][device]
            plt.plot(matrix_sizes[:len(times)],
                     times,
                     line_styles[i % len(line_styles)],
                     color=colors[j % len(colors)],
                     label="{}/{}".format(device, data_type.name))
    plt.ylabel('Time')
    plt.xlabel('Matrix size')
    plt.legend()
    plt.xlim([min(used_matrix_sizes), max(used_matrix_sizes)])
    plt.ylim([0, max_device_time])
    plt.show()


if __name__ == "__main__":
    benchmark()