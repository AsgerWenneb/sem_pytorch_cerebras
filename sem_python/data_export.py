
import numpy as np
import struct

def export_etov(output_file, Etov_table):
    with open(output_file, "w") as output:
        for row in Etov_table:
            line = ",".join(str(x) for x in row)
            output.write(line + "\n")


def export_data_header(d_file, x , y):
    with open(d_file, "wb") as fd:
        n_nodes = len(x)
        # Write n_nodes as a 64-bit float
        fd.write(int.to_bytes(n_nodes, length=8, byteorder='little', signed=True))
        fd.write(x.tobytes())
        fd.write(y.tobytes())

def export_solution(d_file, u):
    with open(d_file, "ab") as fd:
        fd.write(u.tobytes())