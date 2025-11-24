#!/usr/bin/env cs_python

import argparse
import json
import numpy as np
from scipy.sparse import coo_matrix

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder  # pylint: disable=no-name-in-module

struct_dtype = np.dtype([
    ('i', '<u2'),     # u16
    ('j', '<u2'),     # u16
    ('x', '<f4'),     # f32
], align=False)       # align=False -> tightly packed; here fields already align to 4 bytes

# Sanity checks
assert struct_dtype.itemsize == 8


def random_triplet_sparse(n, k):
    if k > n:
        raise ValueError("k cannot exceed matrix dimension n")

    triplets = np.zeros(N_per_PE*NZ_per_N, dtype=struct_dtype)

    for i in range(n):
        arri = np.random.choice(n, k, replace=False)
        arrj = np.full(shape=k, fill_value=i, dtype=np.uint16)
        arrx = np.random.randn(k)

        for j in range(k):
            triplets['i'][i*k + j] = arri[j]
            triplets['j'][i*k + j] = arrj[j]
            triplets['x'][i*k + j] = arrx[j]

    sp_matrix = coo_matrix((triplets['x'], (triplets['i'], triplets['j'])), shape=(n, n))

    return (triplets, sp_matrix)


# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
    compile_data = json.load(json_file)

# Matrix dimensions
N_per_PE = int(compile_data['params']['N_per_PE'])
NZ_per_N = int(compile_data['params']['NZ_per_N'])

# 2) Create and fill the structured array
# host_structs = np.zeros(N_per_PE*NZ_per_N, dtype=struct_dtype)
# host_structs['i'] = np.arange(N_per_PE*NZ_per_N, dtype=np.uint16) % N_per_PE
# host_structs['j'] = (np.arange(N_per_PE*NZ_per_N, dtype=np.uint16) * 2 + 1) % N_per_PE
# host_structs['x'] = np.linspace(0.1, 1.0, N_per_PE*NZ_per_N, dtype=np.float32)
(host_structs, sp_matrix) = random_triplet_sparse(N_per_PE, NZ_per_N)
host_as_u32 = np.ascontiguousarray(host_structs).view(np.uint32).reshape(-1) # shape: (N * 4,)

# Verify sizes:
num_32bit_words_per_struct = struct_dtype.itemsize // 4  # 8 bytes / 4 = 2 words
total_32bit_words = N_per_PE * NZ_per_N * num_32bit_words_per_struct

assert host_as_u32.nbytes == N_per_PE * NZ_per_N * struct_dtype.itemsize
assert host_as_u32.size == total_32bit_words

x = np.full(shape=N_per_PE, fill_value=1.0, dtype=np.float32)
y = np.full(shape=N_per_PE, fill_value=2.5, dtype=np.float32)

y_expected = y + sp_matrix*x

# Calculate expected y
# y_expected = A.reshape(N_per_PE, N_per_PE)@x + y

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for A, x, b, y on device
NZ_symbol = runner.get_id('NZ')
A_symbol = runner.get_id('A')
x_symbol = runner.get_id('x')
y_symbol = runner.get_id('y')

# Load and run the program
runner.load()
runner.run()

NZ_send = np.array([N_per_PE*NZ_per_N], dtype=np.uint32)
print(NZ_send)

# Copy A, x, b to device
print("Transfering NZ")
runner.memcpy_h2d(NZ_symbol, NZ_send, 0, 0, 1, 1, 1, streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  nonblock=False)

print("Transfering A")
runner.memcpy_h2d(A_symbol, host_as_u32, 0, 0, 1, 1, total_32bit_words, streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  nonblock=False)

print("Transfering x")
runner.memcpy_h2d(x_symbol, x, 0, 0, 1, 1, N_per_PE, streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  nonblock=False)

print("Transfering y")
runner.memcpy_h2d(y_symbol, y, 0, 0, 1, 1, N_per_PE, streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  nonblock=False)

# Launch the init_and_compute function on device
runner.launch('compute', nonblock=False)

# Copy y back from device
y_result = np.zeros(N_per_PE, dtype=np.float32)
print("Retrieving y")
runner.memcpy_d2h(y_result, y_symbol, 0, 0, 1, 1, N_per_PE, streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  nonblock=False)

# Stop the program
runner.dump_core("corefile.cs1")
runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected, atol=0.01, rtol=0)

print(f"Result y = {y_result}")

print("SUCCESS!")
