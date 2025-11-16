
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
        fd.write(struct.pack("d", float(n_nodes)))

        # Write all X coordinates (64-bit floats)
        for j in range(n_nodes):
            fd.write(struct.pack("d", x[j]))

        # Write all Y coordinates (64-bit floats)
        for j in range(n_nodes):
            fd.write(struct.pack("d", y[j]))


def export_solution(d_file, u):
    with open(d_file, "ab") as fd:
        n_nodes = len(u)
        # Write all solution values (64-bit floats)
        for j in range(n_nodes):
            fd.write(struct.pack("d", u[j]))