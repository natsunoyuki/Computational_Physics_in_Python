import numpy as np

def make_fs_coeff(f, x, r):
    coeffa = np.zeros(r) # coefficient of cosine terms
    coeffb = np.zeros(r) # coefficient of sine terms
    L = np.abs(x[-1] - x[0]) # 'Period' of Fourier Series
    
    for count in range(1, r+1, 1):
        g = f * np.cos(2 * np.pi * count * x / L)
        h = f * np.sin(2 * np.pi * count * x / L)
        coeffa[count-1] = 2.0 / L * np.trapz(g, x)
        coeffb[count-1] = 2.0 / L * np.trapz(h, x)    
    
    return coeffa, coeffb
    
def real_fourier_series(f, x, r = 5):
    coeffa, coeffb = make_fs_coeff(f, x, r)   
    L = np.abs(x[-1] - x[0]) # 'Period' of Fourier Series
    F = np.zeros(len(x))
    for count in range(1,r+1,1):
        F = F + coeffa[count-1] * np.cos(2 * np.pi * count * x / L) + \
                coeffb[count-1] * np.sin(2 * np.pi * count * x / L)
    F = F + np.trapz(f, x) / L
    return F

def make_cplx_fs_coeff(f, x, r):
    # complex fourier series coefficients
    coeffcr = np.zeros(2*r+1)
    coeffci = np.zeros(2*r+1)
    L = np.abs(x[-1] - x[0]) # 'Period' of Fourier Series
    index = 0
    for count in range(-r, r+1, 1):
        C = 1.0 / L * np.trapz(f * np.exp(-2*np.pi*1j*count*x/L), x)
        coeffcr[index] = np.real(C)
        coeffci[index] = np.imag(C)
        index = index + 1
    return coeffcr, coeffci

def cplx_fourier_series(f, x, r):
    coeffcr, coeffci = make_cplx_fs_coeff(f, x, r)
    G = np.zeros(len(x))
    L = np.abs(x[-1] - x[0]) # 'Period' of Fourier Series
    index = 0
    for count in range(-r,r+1,1):
        G = G+(coeffcr[index]+coeffci[index]*1j)*np.exp(2*np.pi*1j*count*x/L)
        index = index + 1
    return np.real(G)

