import numpy as np
import matplotlib.pyplot as plt

# This function demonstrates explicitly the Runge-Kutta 4 algorithm to perform
# numerical integration of differential equations.

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

def lorentz_plot_demo(x_ax = 0, y_ax = 2):
    # Use the Lorentz attractor as a demonstration of the above rk4 functions
    x0 = np.array([1, 2, 3])
    t = np.arange(0, 50, 0.001)
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
    Y = Y[20000:]
    
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
    plt.show()
