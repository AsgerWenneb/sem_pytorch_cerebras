import numpy as np
from scipy.sparse import coo_matrix, isspmatrix_csr

triplet_dtype = np.dtype([
    ('i', '<u2'),     # u16
    ('j', '<u2'),     # u16
    ('x', '<f4'),     # f32
], align=False)       # align=False -> tightly packed; here fields already align to 4 bytes

# Sanity check
assert triplet_dtype.itemsize == 8


def random_triplet_sparse(n, m, k):
    if k > n:
        raise ValueError("k cannot exceed matrix dimension n")

    triplets = np.zeros(n*k, dtype=triplet_dtype)

    for i in range(n):
        arri = np.full(shape=k, fill_value=i, dtype=np.uint16)
        arrj = np.random.choice(m, k, replace=False)
        arrx = np.random.randn(k)

        for j in range(k):
            triplets['i'][i*k + j] = arri[j]
            triplets['j'][i*k + j] = arrj[j]
            triplets['x'][i*k + j] = arrx[j]

    sp_matrix = coo_matrix((triplets['x'], (triplets['i'], triplets['j'])), shape=(n, m))

    return (triplets, sp_matrix.tocsr())


def csr_to_triplet(csr):
    coo = csr.tocoo()

    triplets = np.zeros(len(coo.data), dtype=triplet_dtype)

    triplets['i'] = coo.row
    triplets['j'] = coo.col
    triplets['x'] = coo.data

    return triplets


def split_sparse_matrix(mat, num_row_blocks, num_col_blocks):
    """
    Split a sparse matrix into a grid of submatrices.

    Parameters:
    -----------
    mat : scipy.sparse matrix
        The input sparse matrix (any format, will be converted to CSR).
    num_row_blocks : int
        Number of submatrices along the rows.
    num_col_blocks : int
        Number of submatrices along the columns.

    Returns:
    --------
    submatrices : list of list of sparse matrices
        A 2D list where submatrices[i][j] is the (i,j)-th submatrix.
    """
    if not isspmatrix_csr(mat):
        mat = mat.tocsr()

    n_rows, n_cols = mat.shape
    row_splits = np.linspace(0, n_rows, num_row_blocks + 1, dtype=int)
    col_splits = np.linspace(0, n_cols, num_col_blocks + 1, dtype=int)

    submatrices = []
    for i in range(num_row_blocks):
        row_start, row_end = row_splits[i], row_splits[i+1]
        row_blocks = []
        for j in range(num_col_blocks):
            col_start, col_end = col_splits[j], col_splits[j+1]
            sub = mat[row_start:row_end, col_start:col_end]
            row_blocks.append(sub)
        submatrices.append(row_blocks)

    return submatrices


def test_assembly(N_kernel, M_kernel, N_matrix, M_matrix, nz_per_row):
    (_, sp_matrix) = random_triplet_sparse(N_matrix, M_matrix, nz_per_row)
    print("Full matrix")
    print(sp_matrix.toarray())
    # print(sp_matrix)
    submatrices = split_sparse_matrix(sp_matrix, N_kernel, M_kernel)

    # Print submatrices shapes
    # for i, row in enumerate(submatrices):
    #     for j, sub in enumerate(row):
    #         print(f"Submatrix ({i},{j}) shape: {sub.shape}")

    all_triplets = np.array([], dtype=triplet_dtype)
    nz = np.array([], dtype=np.uint32)
    nz_total = 0

    x = np.full(shape=M_matrix, fill_value=1.0, dtype=np.float32)
    y = np.full(shape=N_matrix, fill_value=2.5, dtype=np.float32)

    for i in range(N_kernel):
        for j in range(M_kernel):
            matrix = submatrices[i][j]
            # print(f"matrix ({i}, {j})")
            # print(matrix.toarray())
            # print()
            triplets = csr_to_triplet(matrix)
            # print(triplets)
            # print()
            expected_part_result = matrix*x[i*N_kernel:(i+1)*N_kernel]
            if j == 0:
                expected_part_result += y[i*N_kernel:(i+1)*N_kernel]

            print(f"expected y={expected_part_result} for ({j},{i})")
            nz_sub = len(triplets)
            nz_total += nz_sub
            nz = np.append(nz, np.array(nz_sub, dtype=np.uint32))
            all_triplets = np.append(all_triplets, triplets)

    y_expected = y + sp_matrix*x

    # Calculate expected y
    # y_expected = A.reshape(N_per_PE, N_per_PE)@x + y

    triplet_stream = np.ascontiguousarray(all_triplets).view(np.uint32).reshape(-1)

    assert triplet_stream.nbytes == nz_total * 8

    return triplet_stream, nz, x, y, y_expected


if __name__ == "__main__":
    (triplet_stream, nz, x, y, y_expected) = test_assembly(2, 1, 2, 4, 2)
    # print(triplet_stream)
    # print(triplet_stream.size)
    # print(nz)
    # print(x)
    # print(y)
    # print(y_expected)
