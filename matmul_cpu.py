import numpy as np
import time
import sys

# Set random seeder
np.random.seed(0)

# Get count from argument
Count = int(sys.argv[1])

A = np.random.randint(10, size=(Count, Count))
B = np.random.randint(10, size=(Count, Count))

# Multiply matrix
t1 = time.time()
C = np.matmul(A, B)
t2 = time.time()

print("\n@================ Result ================@\n", C)
print("\n@================= Time =================@\n", t2-t1, "\n")
