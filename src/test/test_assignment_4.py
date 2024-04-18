import sys
from pathlib import Path
import numpy as np

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from main.assignment_4 import jacobi, gauss_seidel, SOR, iterative_refinement

A = np.array([
    [10, -1, 2, 0],
    [-1, 11, -1, 3],
    [2, -1, 10, -1],
    [0, 3, -1, 8]
])

b = np.array([6, 25, -11, 15])

def test_functions():
    print("Testing Jacobi method:")
    print(jacobi(A, b))
    print("Testing Gauss-Seidel method:")
    print(gauss_seidel(A, b))
    print("Testing SOR method:")
    print(SOR(A, b, 1.2))
    print("Testing iterative refinement:")
    print(iterative_refinement(A, b))

if __name__ == "__main__":
    test_functions()