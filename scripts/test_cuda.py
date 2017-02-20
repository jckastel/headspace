## Needs to go first because it doesn't like to be used together with theano.
print('Testing tensorflow...')
import tensorflow
import matplotlib
import matplotlib.pyplot as plt
import time

# Creates a graph.
a = tensorflow.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tensorflow.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tensorflow.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))


def get_times(maximum_time):
    device_times = {
        "/gpu:0": [],
        "/cpu:0": []
    }
    matrix_sizes = range(500, 50000, 50)

    for size in matrix_sizes:
        for device_name in device_times.keys():
            print("####### Calculating on the " + device_name + " #######")

            shape = (size, size)
            data_type = tensorflow.float16
            with tensorflow.device(device_name):
                r1 = tensorflow.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
                r2 = tensorflow.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
                dot_operation = tensorflow.matmul(r2, r1)

            with tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True)) as session:
                start_time = time.time()
                result = session.run(dot_operation)
                time_taken = time.time() - start_time
                print(result)
                device_times[device_name].append(time_taken)

            print(device_times)

            if time_taken > maximum_time:
                return device_times, matrix_sizes


device_times, matrix_sizes = get_times(1.5)
gpu_times = device_times["/gpu:0"]
cpu_times = device_times["/cpu:0"]

plt.plot(matrix_sizes[:len(gpu_times)], gpu_times, 'o-')
plt.plot(matrix_sizes[:len(cpu_times)], cpu_times, 'o-')
plt.ylabel('Time')
plt.xlabel('Matrix size')
plt.show()

print('\n\n----------------------------------------------------\n\nTesting theano...')
import os

os.environ['THEANO_FLAGS'] = "floatX=float32,allow_gc=False,nvcc.fastmath=True,optimizer_including=cudnn"
del os.environ['THEANORC']
import theano.sandbox.cuda

# Note: this only works if you do NOT specify a device in your .theanorc or THEANO_FLAGS environment variables.
theano.sandbox.cuda.use("cpu")

import theano
import theano.sandbox
import theano.tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = theano.shared(numpy.asarray(rng.rand(vlen), theano.config.floatX))
f = theano.function([], theano.tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
cputime = t1 - t0
print("Looping %d times took %f seconds" % (iters, cputime))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, theano.tensor.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')

# Note: this only works if you do NOT specify a device in your .theanorc or THEANO_FLAGS environment variables.
theano.sandbox.cuda.use("gpu")

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = theano.shared(numpy.asarray(rng.rand(vlen), theano.config.floatX))
f = theano.function([], theano.tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
gputime = t1 - t0
print("Looping %d times took %f seconds, a speedup of %.2fx" % (iters, gputime, (cputime / gputime)))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, theano.tensor.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')

