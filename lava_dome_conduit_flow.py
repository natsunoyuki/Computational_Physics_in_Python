import numpy as np
import matplotlib.pyplot as plt

# This code solves the delay difference equations proposed by:
# M. Nakanishi, T. Koyaguchi, "A stability analysis of a conduit flow model for lave dome eruptions",
# Journal of Volcanology and Geothermal Research, 178, 46-57, 2008.
# http://www.eri.u-tokyo.ac.jp/people/ichihara/vp2008plan/NakanishiKoyaguchi2008.pdf

# The delay difference equations proposed by Nakanashi and Koyaguchi come in 2 different forms 
# depending on the current system conditions. 

def less_than_one(x, t, params, delayx):
    """
    First set of DDE equations, when np.trapz(Q, T) <= 1

    Inputs
    ------
    x: np.array
        variables to integrate. P and Q in this system.
    t: float
        time 
    params: np.array
        [Qin, mu] DDE parameters
    delayx: np.array
        DDE function values at some previous time step
    Returns
    -------
    ydot: np.array
        np.array of DDE values
    """
    Qin = params[0]
    mu = params[1]
    ydot = np.zeros(len(x))
    ydot[0] = Qin - x[1] # pressure P
    ydot[1] = x[1] / x[0] * (Qin - x[1] + (mu - 1) * (x[1] - delayx) * x[1]) # flux Q
    return ydot.copy()

def more_than_one(x, t, params, delayx):
    """
    Second set of DDE equations, when np.trapz(Q, T) > 1

    Inputs
    ------
    x: np.array
        variables to integrate. P and Q in this system.
    t: float
        time step
    params: np.array
        [Qin, mu] DDE parameters
    delayx: np.array
        DDE solution values at some previous time step
    Returns
    -------
    ydot: np.array
        np.array of DDE values
    """
    Qin = params[0]
    ydot = np.zeros(len(x))
    ydot[0] = Qin - x[1] # pressure P
    ydot[1] = Qin - x[1] # flux Q
    return ydot.copy()
    
def rk4(t0, t1, y0, ydot_fun, params, delayx):
    """
    Runge-Kutta-4 algorithm

    Inputs
    ------
    t0: float
        start of time step 
    t1: float
        end of time step
    y0: np.array
        DDE solution values at time step t0
    y_dot_fun: function
        function containing the DDE to integrate numerically
    params: np.array
        [Qin, mu] DDE parameters
    delayx: np.array
        DDE solution values at some previous time step
    Returns
    -------
    y1: np.array
        DDE solution values at time step t1
    """
    dt = t1 - t0
    k1 = dt * ydot_fun(y0, t0, params, delayx)
    k2 = dt * ydot_fun(y0 + 0.5 * k1, t0 + dt * 0.5, params, delayx)
    k3 = dt * ydot_fun(y0 + 0.5 * k2, t0 + dt * 0.5, params, delayx)
    k4 = dt * ydot_fun(y0 + k3, t0 + dt, params, delayx)
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

def lava_dome_conduit_flow(Qin = 0.8, mu = 10, y0 = np.array([1.5, 1.5]), 
                           t_star = 1, dt = None, t_end = None,
                           less_than_one = None, more_than_one = None, rk4 = None):
    """
    Inputs
    ------
    Qin: float
        DDE parameter for input flux
    mu: float
        DDE parameter for magma viscosity
    y0: np.array
        initial conditions for [P, Q]
    t_star: float
        time parameter.
    dt: np.float
        time step size. If set to None, dt will be calculated automatically using dt = t_star / 20000
    t_end: float
        end time. If set to None, t_end will be set automatically to 30
    less_than_one: function
        function containing the 1st set of DDEs to integrate when np.trapz(Q, T) <= 1
    more_than_one: function
        function containing the 2nd set of DDEs to integrate when np.trapz(Q, T) > 1
    rk4: function
        function containing the numerical integration algorithm

    Returns
    -------
    t, Y: np.array
        np.arrays of time steps and corresponding integrated variables [P, Q]
    """
    
    if (less_than_one is None) or (more_than_one is None) or (rk4 is None):
        return

    if dt is None:
        dt = t_star / 20000 # set time step size
    if t_end is None:
        t_end = 30

    delayindex = int(t_star / dt) # set the delay (history) index
    params = np.array([Qin, mu])

    # length of time of the simulation
    t = np.arange(-t_star, t_end + dt, dt) 
    Y = np.zeros([len(t), len(y0)])
    
    # set initial conditions for the simulation
    Y[0, :] = y0.copy() 
    
    # Create delay difference history (history has length of delayindex)
    for i in range(delayindex):
        y0[0] = y0[0] - 1.0 / delayindex # pressure P history
        y0[1] = y0[1] - 1.0 / delayindex # flux Q history
        Y[i+1, :] = y0.copy()
    
    # Actual RK4 loop to solve the delay differential equation
    for i in range(len(t) - 1 - delayindex):
        delayx = Y[i, 1].copy()
        T = t[i:i + delayindex + 1].copy()
        Q = Y[i:i + delayindex + 1, 1].copy()
        QT = np.trapz(Q, T) # check the value of the integral of Q with respect to T
        if QT > 1:
            # if np.trapz(Q, T) > 1, the DDE equation to integrate is more_than_one
            Y[i+1+delayindex, :] = rk4(t[i+delayindex], t[i+1+delayindex], Y[i+delayindex,:], more_than_one, params, delayx)
        else:
            # if np.trapz(Q, T) <= 1, the DDE equation to integrate is less_than_one
            Y[i+1+delayindex, :] = rk4(t[i+delayindex], t[i+1+delayindex], Y[i+delayindex,:], less_than_one, params, delayx)

    # don't forget to remove the (artificial) history from the data!!!
    return t[delayindex:].copy(), Y[delayindex:].copy()
