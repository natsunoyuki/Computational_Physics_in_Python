###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from numpy.random import randn
from scipy import signal

# This code demonstrates the Sompi algorithm used to decompose a signal into
# both real and imaginary frequency parts. In contrast to traditional Fourier
# analysis techniques the Sompi method is able to give a direct interpretation
# of decaying/amplifying waveforms.

# BASED ON KUMAZAWA et al. 1990, A theory of spectral analysis based on the 
# characteristic property of a linear dynamic system

def sompi(X, dt, m_order_min=4, m_order_max=40, d_m_order=1):
    """
    Inputs:
    X: time series signal to analyse
    dt: time step size
    m_order_min: minimum Sompi order
    m_order_max: maximum Sompi order
    d_m_order: Sompi order step size
    Outputs:
    F, G: Sompi real and imaginary frequency components
    """
    F = []
    G = []
    
    n_order = len(X) # total number of data points in the time series
    for m_order in range(m_order_min, m_order_max+d_m_order, d_m_order):
        P = np.zeros([m_order+1, m_order+1]) # P(k,l) matrix
        for k in range(0, m_order+1, 1):
            for l in range(0, m_order+1, 1):
                for t in range(m_order, n_order, 1):
                    P[k, l] = P[k, l] + X[t-k] * X[t-l]
                    
        P = P / (n_order - m_order) # TAKE THE AVERAGE!
        [val, vct] = linalg.eig(P)
        val = np.real(val) # both val and vct should be real! Drop +0j parts
        #vct = np.real(vct)
        aj = vct[:, np.argmin(val)] # smallest eigen value is the noise power
        Z = np.roots(aj) # obtain the m independent roots
        giw = np.log(Z) # Note that Z = np.exp(gamma + 1j * omega)
        g = np.real(giw) / (2 * np.pi) / dt
        f = np.imag(giw) / (2 * np.pi) / dt
        
        F.append(f)
        G.append(g)
    return F, G

def demo():
    """
    This demo function shows how Sompi analysis is performed using a test
    waveform
    """
    # Create a test wavefunction:
    dt = 1
    T = np.arange(0, 200, dt)
    EFF = 0.1
    GEE = -0.003
    X = np.sin(EFF*2*np.pi*T)*np.exp(GEE*2*np.pi*T)+randn(len(T))*0.03
    
    F, G = sompi(X, dt) # Corresponds to EFF and GEE
    # obtain the FFT as a double check
    [f, p] = signal.periodogram(X, 1, None, 2**12)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot([EFF], [GEE], 'ro', markersize = 10)
    for count in range(len(F)):
        plt.plot(F[count], G[count], 'kx')
    plt.legend(["Actual","Estimated"])
    plt.axis([0, 0.5, GEE * 2,0])
    plt.grid('on')
    plt.subplot(2, 1, 2)
    plt.plot(f, abs(p))
    plt.xlim([0, 0.5])
    plt.grid('on')
    plt.show()
