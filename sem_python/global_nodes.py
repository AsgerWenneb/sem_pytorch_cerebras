import numpy as np

def gen_node_coordinates(elemsx, elemsy, min_x=-1, max_x=1, min_y=-1, max_y=1):
    n_nodes_x = elemsx + 1
    n_nodes_y = elemsy + 1
    x = np.zeros((n_nodes_x * n_nodes_y, 1))
    y = np.zeros((n_nodes_x * n_nodes_y, 1))

    for i in range(n_nodes_x):
        for j in range(n_nodes_y):
            idx = j + i*n_nodes_y
            x[idx] = min_x + (max_x - min_x) * i / elemsx
            y[idx] = min_y + (max_y - min_y) * j / elemsy
    return x, y


