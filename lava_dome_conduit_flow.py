import numpy as np
import matplotlib.pyplot as plt
import time

# This is a code to simulate the time forward modelling of a delay difference equation.

# This code solves the delay difference equations proposed by:
# M. Nakanishi, T. Koyaguchi, "A stability analysis of a conduit flow model for lave dome eruptions",
# Journal of Volcanology and Geothermal Research, 178, 46-57, 2008.
# http://www.eri.u-tokyo.ac.jp/people/ichihara/vp2008plan/NakanishiKoyaguchi2008.pdf

def lava_dome_conduit_flow(Qin = 0.8, mu = 10, y0 = np.array([1.5, 1.5]), t_star = 1, dt = None, t_end = None):
    """
    Inputs
    ------
    Qin: float
        DDE parameter
    mu: float
        DDE parameter
    y0: np.array
        initial conditions for [P, Q]
    t_star: float
        time parameter. Interesting values to use: 1, 0.2
    dt: np.float
        time step size. If set to None, dt will be calculated automatically using dt = t_star / 20000
    t_end: float
        end time. If set to None, t_end will be set automatically to 30

    Returns
    -------
    t, Y: np.array
        np.arrays of time steps and corresponding integrated variables [P, Q]
    """
    # The delay difference equations proposed by Nakanashi and Koyaguchi come in 2 different forms 
    # depending on the current system conditions. 

    # First set of DDE equations, when np.trapz(Q, T) <= 1 (equation (5) in the paper)
    # Equations (8), (10)~(12) in the paper
    def less_than_one(x, t, delayx):
        ydot = np.zeros(len(x))
        ydot[0] = Qin - x[1] # pressure P
        ydot[1] = x[1] / x[0] * (Qin - x[1] + (mu - 1) * (x[1] - delayx) * x[1]) # flux Q
        return ydot

    # Second set of DDE equations, when np.trapz(Q, T) > 1 (equation (5) in the paper)
    # Equations (8), (10)~(12) in the paper
    def more_than_one(x, t, delayx):
        ydot = np.zeros(len(x))
        ydot[0] = Qin - x[1] # pressure P
        ydot[1] = Qin - x[1] # flux Q
        return ydot
    
    # Runge-Kutta-4 algorithm for numerical integration of initial value problems
    def rk4(t0, t1, y0, ydot_fun, params):
        dt = t1 - t0
        k1 = (dt) * ydot_fun(y0, t0, params)
        k2 = (dt) * ydot_fun(y0 + 0.5 * k1, t0 + (dt) * 0.5, params)
        k3 = (dt) * ydot_fun(y0 + 0.5 * k2, t0 + (dt) * 0.5, params)
        k4 = (dt) * ydot_fun(y0 + k3, t0 + (dt), params)
        y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return y1
    
    starttime = time.time() #log start time

    if dt is None:
        dt = t_star / 20000 # set time step size
    if t_end is None:
        t_end = 30 # set the end time

    # set the number of delay difference histories to use
    delayindex = int(t_star / dt) 

    # length of time of the simulation
    t = np.arange(-t_star, t_end + dt, dt) 

    # set initial conditions for the simulation
    Y = np.zeros([len(t), len(y0)])
    Y[0, :] = y0 
    
    # Create delay difference history (history has length of delayindex)
    for i in range(delayindex):
        y0[0] = y0[0] - 1.0 / delayindex # pressure P history
        y0[1] = y0[1] - 1.0 / delayindex # flux Q history
        Y[i+1, :] = y0
    
    # Actual RK4 loop to solve the delay differential equation with the delay difference history
    for i in range(len(t) - 1 - delayindex):
        params = Y[i, 1]
        T = t[i:i + delayindex + 1]
        Q = Y[i:i + delayindex + 1, 1]
        QT = np.trapz(Q, T) # check the value of the integral of Q with respect to T
        if QT > 1:
            # if np.trapz(Q, T) > 1, the DDE equation to integrate is more_than_one
            Y[i + 1 + delayindex, :] = rk4(t[i+delayindex],t[i+1+delayindex],Y[i+delayindex,:], more_than_one, params)
        else:
            # if np.trapz(Q, T) <= 1, the DDE equation to integrate is less_than_one
            Y[i + 1 + delayindex, :] = rk4(t[i+delayindex],t[i+1+delayindex],Y[i+delayindex,:], less_than_one, params)

    print("Time elapsed: {:.2f}s".format(time.time()-starttime)) 

    # don't forget to remove the (artificial) history from the data!!!
    return t[delayindex:], Y[delayindex:] 
