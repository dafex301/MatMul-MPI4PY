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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get count from argument
Count = int(sys.argv[1])

t1 = MPI.Wtime()
if rank == 0:
    A = np.random.randint(10, size=(Count, Count))
    B = np.random.randint(10, size=(Count, Count))

    A_local = np.array_split(A, size, axis=0)

else:
    A_local = None
    B = None

A = comm.scatter(A_local, root=0)
B = comm.bcast(B, root=0)

C = np.dot(A, B)

result = comm.gather(C, root=0)

if rank == 0:
    result = np.concatenate(result)
    t2 = MPI.Wtime()
    print("\n@================ Result ================@\n", result)
    print("\n@================= Time =================@\n", t2-t1, "\n")
