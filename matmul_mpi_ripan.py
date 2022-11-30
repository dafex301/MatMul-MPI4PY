from mpi4py import MPI
import numpy as np
import sys

# Set random seeder
np.random.seed(0)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Count = int(sys.argv[1])


def matrix_mul(mat1, mat2):
    m, n = mat1.shape[0], mat2.shape[1]
    C = np.zeros([m, n])
    mat2 = np.transpose(mat2)
    for i in range(m):
        for j in range(n):
            C[i, j] = np.dot(mat1[i, :], mat2[j, :])
    return C


# Create a random integer matrix A and B from 1-9 of size Count x Count
# Split the matrix into parts based on the number of processors
# and send the parts to each processor
t1 = MPI.Wtime()
if rank == 0:
    A = np.random.randint(10, size=(Count, Count))
    B = np.random.randint(10, size=(Count, Count))

    tmp = np.array_split(A, size, axis=0)
# Print the parts of A and B on each processor
else:
    tmp = None
    B = None
A = comm.scatter(tmp, root=0)
B = comm.bcast(B, root=0)


# print("A = ", A, "on processor ", rank)
# print("B = ", B, "on processor ", rank)

# Perform matrix multiplication on each processor
tmp_c = matrix_mul(A, B)
result = comm.gather(tmp_c, root=0)

# Gather the parts of C on each processor and print the result
if rank == 0:
    result = np.vstack(result)
    t2 = MPI.Wtime()
    print(result)
    print("Time = ", t2-t1)
