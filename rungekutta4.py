import numpy as np
import matplotlib.pyplot as plt

# This function demonstrates explicitly the Runge-Kutta 4 algorithm to perform
# numerical integration of differential equations.
# Some well known nonlinear systems are included as demonstrators.

def ode_steps(ydot_fun, y0, t, params, step_fun):
    # This function uses a for loop to loop over all time steps for numerical
    # integration, calling the rk4 function for the actual integration steps
    Y = np.zeros([len(t), len(y0)])  
    Y[0, :] = y0
    
    for i in range(len(t) - 1):
        Y[i + 1, :] = step_fun(t[i], t[i + 1], Y[i, :], ydot_fun, params)
        
    return Y

def rk4(t0, t1, y0, xdot_fun, params):
    # This function contains the actual Runge-Kutta 4th order integration steps
    dt = t1 - t0
    k1 = (dt) * xdot_fun(y0, t0, params)
    k2 = (dt) * xdot_fun(y0 + 0.5 * k1, t0 + (dt) * 0.5, params)
    k3 = (dt) * xdot_fun(y0 + 0.5 * k2, t0 + (dt) * 0.5, params)
    k4 = (dt) * xdot_fun(y0 + k3, t0 + (dt), params)
    y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return y1

def plot(Y, x_ax, y_ax):
    # Plot the numerical integration results
    plt.figure(figsize=(8, 8))
    plt.plot(Y[:, x_ax], Y[:, y_ax])
    
    if x_ax == 0:
        x_ax = 'x'
    elif x_ax == 1:
        x_ax = 'y'
    elif x_ax == 2:
        x_ax = 'z'
    if y_ax == 0:
        y_ax = 'x'
    elif y_ax == 1:
        y_ax = 'y'
    elif y_ax == 2:
        y_ax = 'z'   
        
    plt.xlabel(x_ax)
    plt.ylabel(y_ax)
    plt.grid('on')
    
    plt.show()    

def lorentz_plot_demo(x_ax = 0, y_ax = 2):
    # Lorenz, E. N. (1963) Journal of the Atmospheric Sciences. 20 (2): 130–141
    x0 = np.array([1, 2, 3])
    t = np.arange(0, 100, 0.001)
    s = 10.0
    B = 8.0 / 3.0
    p = 28.0
    params = np.array([s, B, p])
    
    def xdot_fun(x, t, params):
        xdot = np.zeros(len(x))
        xdot[0] = params[0] * (x[1] - x[0])  # x
        xdot[1] = x[0] * (params[2] - x[2]) - x[1]  # y
        xdot[2] = x[0] * x[1] - params[1] * x[2]  # z
        return xdot
    
    Y = ode_steps(xdot_fun, x0, t, params, rk4)
    Y = Y[int(len(Y)*0.2):]
    
    plot(Y, x_ax, y_ax)

def rossler_plot_demo(x_ax = 0, y_ax = 1):
    # Rossler, O. E. (1976) Physics Letters, 57A (5): 397–398
    x0 = np.array([1, 2, 3])
    t = np.arange(0, 1000, 0.01)
    a = 0.1
    b = 0.1
    c = 13
    params = np.array([a, b, c])
    
    def xdot_fun(x, t, params):
        a = params[0]
        b = params[1]
        c = params[2]
        xdot = np.zeros(len(x))
        xdot[0] = -x[1] - x[2]  #x
        xdot[1] = x[0] + a * x[1]  #y
        xdot[2] = b + x[2] * (x[0] - c)  #z
        return xdot
    
    Y = ode_steps(xdot_fun, x0, t, params, rk4)
    Y = Y[int(len(Y)*0.75):]
    
    plot(Y, x_ax, y_ax)
    
def julian_plot_demo(x_ax = 1, y_ax = 2):
    # Julian, B. R. (1994) Volcanic tremor: Nonlinear excitation by fluid flow, 
    # J. geophys. Res., 99(B6), 11859–11877
    x0 = np.array([1, 2, 3])
    t = np.arange(0, 100, 0.001)
    
    k = 600 * 10**6
    M = (3 * 10**5) * 0
    rho = 2500
    eta = 50
    p2 = 0.1 * 10**6
    p1 = 18 * 10**6
    h0 = 1
    L = 10
    A = (10**7) * 1   
    
    params = np.array([k, M, rho, eta, p2, p1, h0, L, A])
    
    def xdot_fun(x, t, params):
        k = params[0]
        M = params[1]
        rho = params[2]
        eta = params[3]
        p2 = params[4]
        p1 = params[5]
        h0 = params[6]
        L = params[7]
        A = params[8]

        xdot = np.zeros(len(x))
        effectm = M + rho * L**3 / 12 / x[1]
        damping = A + L**3 / 12 / x[1] * (12 * eta / x[1]**2 - rho / 2 * x[2] / x[1])
        kcoeff = k * (x[1] - h0)
        Lcoeff = L * (p1 + p2) / 2 - L * rho * x[0]**2 / 2
        
        xdot[0] = (p1-p2)/(rho*L)-(12*eta*x[0])/(rho*x[1]**2)
        xdot[1] = x[2]
        xdot[2] = (Lcoeff-kcoeff-damping*x[2])/effectm
    
        return xdot
    
    Y = ode_steps(xdot_fun, x0, t, params, rk4)
    Y = Y[int(len(Y)*0.75):]
    
    plot(Y, x_ax, y_ax)
