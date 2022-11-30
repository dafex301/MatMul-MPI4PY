# =============================================
#          How to run the program
# =============================================
# mpiexec -n 4 python matmul_mpi.py 1000

# number after n is the number of processes
# number after filename is the size of matrix

from mpi4py import MPI
import numpy as np
import sys

# Set random seeder
np.random.seed(0)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get count from argument
Count = int(sys.argv[1])

t1 = MPI.Wtime()
if rank == 0:
    # Generate random matrix with the size of Count
    A = np.random.randint(10, size=(Count, Count))
    B = np.random.randint(10, size=(Count, Count))

    # Split matrix A
    A_local = np.array_split(A, size, axis=0)

else:
    # Initialize local matrix A and B
    A_local = None
    B = None

# Scatter matrix A
A = comm.scatter(A_local, root=0)

# Broadcast matrix B
B = comm.bcast(B, root=0)

# Multiply matrix
C = np.dot(A, B)

# Gather matrix C
result = comm.gather(C, root=0)

if rank == 0:
    # Concatenate matrix C
    result = np.concatenate(result)

    t2 = MPI.Wtime()
    print("\n@================ Result ================@\n", result)
    print("\n@================= Time =================@\n", t2-t1, "\n")
