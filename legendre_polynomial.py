import numpy as np
from scipy.special import factorial, binom

def legendre_polynomial(n, x):
    """Input arguments:
    n: polynomial order
    x: array of x values over which to calculate the Legendre polynomial
    """
    if np.remainder(n, 2)==0 and np.mod(n,1)==0 and abs(max(x)) <= 1:
        P = np.zeros(len(x))
        p = int(n / 2.0)
        for m in range(p, n+1, 1):
            P = P + (-1)**(n-m)*binom(n,m)*factorial(2*m)/factorial(2*m-n)*x**(2*m-n)
        
        P = P / (2**n*factorial(n))
    elif np.remainder(n, 2) != 0 and np.mod(n, 1) == 0 and abs(max(x)) <= 1:
        P = np.zeros(len(x))
        p = int((n+1)/2.0)
        for m in range(p, n+1, 1):
            P = P + (-1)**(n-m)*binom(n,m)*factorial(2*m)/factorial(2*m-n)*x**(2*m-n)
        
        P = P / (2**n*factorial(n))
    return P
