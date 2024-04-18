import numpy as np

def jacobi(A, b, tolerance=1e-3, max_iterations=500):
    x = np.zeros_like(b, dtype=np.double)

    for k in range(max_iterations):
        x_old = np.copy(x)
        
        for i in range(A.shape[0]):
            sum_except_diagonal = np.dot(A[i, :], x) - A[i, i] * x[i]
            x[i] = (b[i] - sum_except_diagonal) / A[i, i]
        
        if np.linalg.norm(x - x_old, np.inf) / np.linalg.norm(x, np.inf) < tolerance:
            break
        
    
    return x

def gauss_seidel(A, b, tolerance=1e-3, max_iterations=10000):
    x = np.zeros_like(b, dtype=np.double)
    n = len(b)

    for itr in range(max_iterations):
        x_old = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tolerance:
            return x

        x_old[:] = x

    return x


def SOR(A, b, omega, tolerance=1e-3, max_iterations=10000):
    n = len(b)
    x = np.zeros_like(b)
    for itr in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sum1 - sum2)
        if np.linalg.norm(x_new - x, np.inf) < tolerance:
            return x_new
        x = x_new
    return x

def iterative_refinement(A, b, tolerance=1e-3, max_iterations=10000):
    x = np.linalg.solve(A, b)
    for _ in range(max_iterations):
        r = b - np.dot(A, x)
        e = np.linalg.solve(A, r)
        x += e
        
        if np.linalg.norm(e, np.inf) < tolerance:
            break
    return x