#!/usr/bin/env cs_python

import argparse
import json
import numpy as np
from scipy.sparse import coo_matrix, isspmatrix_csr

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder  # pylint: disable=no-name-in-module

from test_assembler import test_assembly


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
max_nz_per_n = int(compile_data['params']['max_nz_per_n'])

N_kernel = int(compile_data['params']['N_kernel'])
M_kernel = int(compile_data['params']['M_kernel'])
num_kernels = N_kernel * M_kernel

N_matrix = N_per_PE * N_kernel
M_matrix = N_per_PE * M_kernel
nz_per_row = max_nz_per_n

assert N_matrix % N_kernel == 0
assert M_matrix % M_kernel == 0


# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for A, x, b, y on device
iterations_symbol = runner.get_id('iterations')
NZ_symbol = runner.get_id('NZ')
A_symbol = runner.get_id('A')
x_symbol = runner.get_id('x')
y_symbol = runner.get_id('y')

# Load and run the program
runner.load()
runner.run()

(triplet_stream, nz, x, y, y_expected) = test_assembly(N_kernel, M_kernel, N_matrix, M_matrix, nz_per_row)

print(f"Stream size: {triplet_stream.size}")
print(f"nz: {nz}")

iterations = np.full(shape=(M_kernel*N_kernel), fill_value=2, dtype=np.uint32)

runner.memcpy_h2d(iterations_symbol, iterations, 0, 0, M_kernel, N_kernel, 1,
                  streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  nonblock=False)

print("Transfering NZ")
runner.memcpy_h2d(NZ_symbol, nz, 0, 0, M_kernel, N_kernel, 1,
                  streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  nonblock=False)

transfered = 0
# print(nz)
for pe_y in range(N_kernel):
    for pe_x in range(M_kernel):
        nz_ = nz[pe_x + pe_y*M_kernel]

        print(f"Transfering {nz_*2} bytes of A to PE ({pe_x}, {pe_y})")
        if nz_ == 0:
            continue

        # print(triplet_stream[transfered:])
        runner.memcpy_h2d(A_symbol, triplet_stream[transfered:], pe_x, pe_y, 1, 1, nz_*2,
                          streaming=False,
                          order=MemcpyOrder.ROW_MAJOR,
                          data_type=MemcpyDataType.MEMCPY_32BIT,
                          nonblock=False)

        transfered += nz_*2

print("Transfering x")
runner.memcpy_h2d(x_symbol, np.tile(x, N_kernel), 0, 0, M_kernel, N_kernel, M_matrix // M_kernel,
                  streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  nonblock=False)

print("Transfering y")
runner.memcpy_h2d(y_symbol, y, 0, 0, 1, N_kernel, N_matrix // N_kernel,
                  streaming=False,
                  order=MemcpyOrder.ROW_MAJOR,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  nonblock=False)

# Launch the init_and_compute function on device
runner.launch('compute', nonblock=False)

# Restreive y from each PE
for pe_y in range(N_kernel):
    for pe_x in range(M_kernel):
        y_partial_result = np.zeros(N_matrix // N_kernel, dtype=np.float32)
        runner.memcpy_d2h(y_partial_result, x_symbol, pe_x, pe_y, 1, 1, N_matrix // N_kernel, streaming=False,
                          order=MemcpyOrder.ROW_MAJOR,
                          data_type=MemcpyDataType.MEMCPY_32BIT,
                          nonblock=False)
        print(f"y ({pe_x},{pe_y}) = {y_partial_result}")

# Copy y back from device
y_result = np.zeros(N_matrix, dtype=np.float32)
for pe_y in range(N_kernel):
    print(f"Retrieving y {pe_y}")
    y_partial_result = y_result[pe_y * (N_matrix // N_kernel):(pe_y+1) * (N_matrix // N_kernel)]

    runner.memcpy_d2h(y_partial_result, x_symbol, pe_y, pe_y, 1, 1, N_matrix // N_kernel, streaming=False,
                      order=MemcpyOrder.ROW_MAJOR,
                      data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False)

    print(y_partial_result)
    # y_result = np.append(y_result, y_partial_result)


print(y_result)

# runner.memcpy_d2h(y_result, y_symbol, 0, 0, 1, 1, N_matrix, streaming=False,
#                   order=MemcpyOrder.ROW_MAJOR,
#                   data_type=MemcpyDataType.MEMCPY_32BIT,
#                   nonblock=False)
# print(y_result)

runner.dump_core("corefile.cs1")
# Stop the program
runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected, atol=0.01, rtol=0)

print(f"Result y = {y_result}")

print("SUCCESS!")
