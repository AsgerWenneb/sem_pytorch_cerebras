from os import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri, cm, colors


def read_etov(etov_filename):
    evals = pd.read_csv(etov_filename, header=None)
    trigs = evals[1].size
    etov = np.zeros((trigs, 3))

    for i in range(0, trigs):
        for j in range(3):
            etov[i, j] = evals[i:i+1][j]

    return etov


def read_xy_data(filename):
    with open(filename, 'rb') as resfile:
        nnodes = int.from_bytes(resfile.read(8), "little")
        ncornernodes = int.from_bytes(resfile.read(8), "little")

        line_size = nnodes*8

        # resfile.seek(8)
        xcoords = np.frombuffer(resfile.read(line_size), dtype=np.float64)
        ycoords = np.frombuffer(resfile.read(line_size), dtype=np.float64)

        u = np.frombuffer(resfile.read(line_size), dtype=np.float64)

    return (xcoords[0:ncornernodes], ycoords[0:ncornernodes], u[0:ncornernodes])


def main():
    etov = read_etov(sys.argv[1])
    (xcoords, ycoords, u) = read_xy_data(sys.argv[2])
    print("min(u) =", min(u))
    print("max(u) =", max(u))

    lim = max([abs(min(u)), abs(max(u))])
    norm = colors.Normalize(vmin=-lim, vmax=lim)
    plot_args = {'norm': norm, 'cmap': cm.seismic}

    fig = plt.figure()
    ax = fig.add_subplot(111)

    tgrid = tri.Triangulation(xcoords, ycoords, triangles=etov)
    ax.tricontourf(xcoords, ycoords, u, 100,
                   tgrid.triangles, **plot_args)

    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')

    plt.gca().set_aspect('equal')

    plt.show()


if __name__ == "__main__":
    main()
