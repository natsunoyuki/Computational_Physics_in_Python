from __future__ import division
from scipy import *
import matplotlib.pyplot as plt
import time
#from mpl_toolkits.mplot3d import Axes3D

#This function shows how the Runge-Kutta 4 algorithm is used to solve ordinary differential
#equations. The algorithm is applied to damped and driven oscillators, Langevin equations
#as well as the Lorenz equations which describes chaotic motion in the atmosphere. This code
#also shows how the Euler method is unstable compared to the RK4 method.

def ode_steps(ydot_fun, y0, t, params, step_fun):

    Y = zeros([len(t), len(y0)])  # matrix to hold the final result over N+1 steps from t=0 to t=N
    Y[0, :] = y0  # first row is the state vector at time t=0 i.e. all the components like x,vx,y,vy,....
    # Hence, y0 will have the same number of columns as the number of states and the number of rows of y0
    # correspond to the number of time steps, including the time at t=0.
    
    for i in range(len(t) - 1):
        Y[i + 1, :] = step_fun(t[i], t[i + 1], Y[i, :], ydot_fun, params)  # compute the state vector at time t+dt
    return Y
    
# default derivative function with known solutions to test the solver:    
def ydot_fun(y, t, params):
    # General representation: for non linear systems this is best used:
    # for the default representation we use the same function but for different initial conditions:
    ydot = zeros(len(y))
    ydot[0] = 15 - 3 * y[0] 
    ydot[1] = 15 - 3 * y[1]
    # Matrix representation (for linear systems):
    # A=array([[-3,0],[2,-4]])
    # ydot=dot(A,y)+array([1,15]) # ydot = A*y for linear systems, where A is a matrix and ydot & y are vectors
    return ydot

# This function computes the Euler step: y[n+1]=y[n]+dt*F[y0,t0]
def forward_euler(t0, t1, y0, ydot_fun, params):
    dt = t1 - t0
    y1 = y0 + (dt) * ydot_fun(y0, t0, params) 
    return y1

# This function computes the Runge Kutta step: y[n+1]=y[n]+(k1+2*k2+2*k3+k4)/6.0
def rk4(t0, t1, y0, ydot_fun, params):
    dt = t1 - t0
    k1 = (dt) * ydot_fun(y0, t0, params)
    k2 = (dt) * ydot_fun(y0 + 0.5 * k1, t0 + (dt) * 0.5, params)
    k3 = (dt) * ydot_fun(y0 + 0.5 * k2, t0 + (dt) * 0.5, params)
    k4 = (dt) * ydot_fun(y0 + k3, t0 + (dt), params)
    y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return y1

def damped_oscillator_plot():
    x0 = array([1, 1])  # initial x and v values, col 1 = x, col 2 = v
    t = arange(0, 20 + 0.01, 0.01)  # this is the array of times to run the solvers
    # note that I use arange instead of linspace so I can control directly the step size.
    # If the step size value is unimportant, linspace may be used instead.
    # the following parameters work only for the under damped case
    w0 = 0.8
    g = 0.1
    params = array([w0, g])
    
    def xdot_fun(x, t, params):   
        # use matrix representation since this is a linear system:
        A = array([[0, 1], [-params[0] ** 2, -2 * params[1]]])
        xdot = dot(A, x)
        return xdot

    for i in range(2):  # 2 loops - under damped and over damped regimes:
        # the first loop handles the underdamped case:
        # obtain the numerical solutions
        xeuler = ode_steps(xdot_fun, x0, t, params, forward_euler)
        xrk4 = ode_steps(xdot_fun, x0, t, params, rk4)
        # Analytical solution
        A2 = -1.0 / (2 * sqrt(g ** 2 - w0 ** 2)) * (x0[1] + g * x0[0] - x0[0] * sqrt(g ** 2 - w0 ** 2))
        A1 = x0[0] - A2
        xt = real(exp(-g * t) * (A1 * exp(sqrt(g ** 2 - w0 ** 2) * t) + A2 * exp(-sqrt(g ** 2 - w0 ** 2) * t)))

        plt.subplot(2, 1, i + 1)
        plt.plot(t, xeuler[:, 0], 'yo')  # the first column contains the x-positions
        plt.plot(t, xrk4[:, 0], 'co')  # the first column contains the x-positions
        plt.plot(t, xt, 'r')
        plt.legend(['Euler', 'Runge Kutta 4', 'Analytical'])
        plt.xlabel('t')
        plt.ylabel('x')
        # now change the variables for the over damped case
        w0 = 0.8
        g = 1.0
        params = array([w0, g])
    # show all plots after the for loop has ended
    plt.show()
        
def driven_oscillator_plot():
    x0 = array([0, 0])  # initial conditions: x=0, v=0.
    # I use arange so I can define the stepsize. If this is unimportant, linspace may be 
    # used instead
    dt = 0.01
    t = arange(0, 100 + dt, dt)  # this is the array of times to run the solvers
    g = 0.1
    w0 = 0.8
    F0 = 2  # the case of F0=0 corresponds to the damped SHO problem
    w = 1
    params = array([w0, g, F0, w])
    
    def xdot_fun(x, t, params):
        w0 = params[0]
        g = params[1]
        F0 = params[2]
        w = params[3]
        xdot = zeros(len(x))
        xdot[0] = x[1]
        xdot[1] = -w0 ** 2 * x[0] - 2 * g * x[1] + F0 * cos(w * t)
        return xdot
    
    # obtain the numerical solution:
    xrk4 = ode_steps(xdot_fun, x0, t, params, rk4)
    Q = F0 / (w ** 2 - w0 ** 2 + 2 * 1j * w * g)
    # obtain the analytical solution
    # complimentary solution:
    # A2=-1.0/(2*sqrt(g**2-w0**2))*(x0[1]+g*x0[0]-x0[0]*sqrt(g**2-w0**2)-real(1j*w*Q)+real(-Q)*sqrt(g**2-w0**2))
    # A1=x0[0]-A2-real(-Q)
    A2 = 1.0 / (-2 * sqrt(g ** 2 - w0 ** 2)) * (x0[1] + g * (x0[0] - real(-Q)) - x0[0] * sqrt(g ** 2 - w0 ** 2) + sqrt(g ** 2 - w0 ** 2) * real(-Q) - real(1j * w * Q))
    A1 = x0[0] - A2 - real(-Q)
    xct = real(exp(-g * t) * (A1 * exp(sqrt(g ** 2 - w0 ** 2) * t) + A2 * exp(-sqrt(g ** 2 - w0 ** 2) * t)))
    # particular solution:
    # D=F0/sqrt((w0**2-w**2)**2+(2*w*g)**2)
    # delta=arctan(2*w*g/(w0**2-w**2))
    # xpt=D*cos(w*t-delta)
    xpt = real(-F0 / (w ** 2 + 2 * 1j * w * g - w0 ** 2) * exp(-1j * w * t))
    # The total solution is the sum of the complimentary and particular solution
    xt = xct + xpt
    # plot everything:
    plt.plot(t, xrk4[:, 0], 'co')
    plt.plot(t, xt, 'r')
    plt.legend(['Runge Kutta 4', 'Analytic'])
    plt.xlabel('t')
    plt.ylabel('x')
    plt.axis('tight')
    plt.show()

def driven_oscillator_error_plot():
    x0 = array([0, 0])
    g = 0.1
    w0 = 0.8
    F0 = 2  # the case of F0=0 corresponds to the damped SHO problem
    w = 1
    params = array([w0, g, F0, w])
    Q = F0 / (w ** 2 - w0 ** 2 + 2 * 1j * w * g)
    
    def xdot_fun(x, t, params):
        w0 = params[0]
        g = params[1]
        F0 = params[2]
        w = params[3]
        xdot = zeros(len(x))
        xdot[0] = x[1]
        xdot[1] = -w0 ** 2 * x[0] - 2 * g * x[1] + F0 * cos(w * t)
        # use Matrix representation because this is still a linear system
        # A=array([[0,1],[-w0**2,-2*g]])
        # C=array([0,F0*cos(w*t)])
        # xdot=dot(A,x)+C
        return xdot

    dT = arange(0.05, 1 + 0.01, 0.01)  # range of values of dt to use from 0.05 to 1 in 0.01 steps
    errorrk4 = zeros(len(dT))
    erroreuler = zeros(len(dT))
    index = 0
    
    for dt in dT:
        t = arange(0, 100 + dt, dt)  # this is the array of times to run the solvers from 0 to 100
        xeuler = ode_steps(xdot_fun, x0, t, params, forward_euler)
        xrk4 = ode_steps(xdot_fun, x0, t, params, rk4)
        A2 = 1.0 / (-2 * sqrt(g ** 2 - w0 ** 2)) * (x0[1] + g * (x0[0] - real(-Q)) - x0[0] * sqrt(g ** 2 - w0 ** 2) + sqrt(g ** 2 - w0 ** 2) * real(-Q) - real(1j * w * Q))
        A1 = x0[0] - A2 - real(-Q)
        xct = real(exp(-g * t) * (A1 * exp(sqrt(g ** 2 - w0 ** 2) * t) + A2 * exp(-sqrt(g ** 2 - w0 ** 2) * t)))
        xpt = real(-F0 / (w ** 2 + 2 * 1j * w * g - w0 ** 2) * exp(-1j * w * t))
        xt = xct + xpt
        # compute errors at time t=100 i.e. right at the end of the time vector the last time value
        errorrk4[index] = abs((xrk4[-1, 0] - xt[-1]) / xt[-1])
        erroreuler[index] = abs((xeuler[-1, 0] - xt[-1]) / xt[-1])
        index = index + 1

    plt.semilogy(dT, erroreuler, 'o')
    plt.semilogy(dT, errorrk4, '*')
    plt.xlabel('dt')
    plt.ylabel('Numerical Error')
    plt.legend(['Euler', 'RK'])
    plt.axis('tight')
    plt.show()

# rk4 solver for the langevin equations. We need to ensure that the same value of F is used
# in all 4 values of k, the standard code will generate a new value of F each time k is called
# for a total of 4 times in the rk4 solver.
def rk4lang(t0, t1, y0, ydot_fun, param):
    dt = t1 - t0
    r = random.random()
    if r <= 0.5:
        F = param[1]
    else:
        F = -param[1]
    params = array([param[0], F])
    k1 = (dt) * ydot_fun(y0, t0, params)
    k2 = (dt) * ydot_fun(y0 + 0.5 * k1, t0 + (dt) * 0.5, params)
    k3 = (dt) * ydot_fun(y0 + 0.5 * k2, t0 + (dt) * 0.5, params)
    k4 = (dt) * ydot_fun(y0 + k3, t0 + (dt), params)
    y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0   
    return y1
    
def langevin_plot():
    start = time.time()
    x0 = array([0, 0])  # initial conditions for x (col 1) and v (col 2)
    TMAX = 100  # default value should be 100
    t = linspace(0, TMAX, 10001)  # we want to control the vector length rather than step size
    dt = t[1] - t[0]  # step size
    g = 0.05
    f = 0.1  # f = dt F0**2 for dt=0
    F0 = sqrt(f / dt)
    params = array([g, F0])
    t0 = int32(20 / dt)  # index of t0=20
    print dt, t0, t[t0]
    N = 500  # number of samples to run i.e. solve the RK4 solver this amount of times
     
    def xdot_fun(x, t, params):
        g = params[0]
        F = params[1]
        xdot = zeros(len(x))
        xdot[0] = x[1]
        xdot[1] = -2 * g * x[1] + F
        return xdot
    
    tt = t[t0 + 1:]  # truncated time from t0=20+dt to 100 only
    tt = tt - tt[0]  # reset the truncated time vector so that now its from 0 to 80-dt
    MSD = zeros(len(tt))  # mean squared displacement array
    VCF = zeros(len(tt))  # velocity correlation function array
    # count=0 #counter for the rolling averages
    
    # loop over all values of N
    for i in range(N):
        xrk4 = ode_steps(xdot_fun, x0, t, params, rk4lang)  # col 0 is x, col 1 is v
        x = xrk4[:, 0]  # displacements
        v = xrk4[:, 1]  # velocities
        xt = x[t0 + 1:]  # truncated displacements
        vt = v[t0 + 1:]  # truncated velocities
        MSD = MSD + (xt - x[t0]) ** 2
        VCF = VCF + (vt * v[t0])
        """
        MSD=(MSD*count+(xt-x[t0])**2)/(count+1.0)
        VCF=(VCF*count+(vt*v[t0]))/(count+1.0)   
        count=count+1
        """
    MSD = MSD / N
    VCF = VCF / N  
    # Analytic results
    msd = f / (4 * g ** 2) * (tt - (1 - exp(-2 * g * tt)) / (2 * g))
    vcf = exp(-2 * g * tt) * f / (4 * g)
    end = time.time() - start
    print "Time taken:", end
    plt.subplot(1, 2, 1)
    plt.plot(tt, MSD, 'r+')
    plt.plot(tt, msd)
    plt.title('MSD')
    plt.subplot(1, 2, 2)
    plt.plot(tt, VCF, 'r+')
    plt.plot(tt, vcf)
    plt.title('VCF')
    plt.show()   

def lorenz_plot():
    x0 = array([1, 0, 0])
    t = linspace(0, 50, 10000)
    s = 10.0
    B = 8.0 / 3.0
    p = 28.0
    params = array([s, B, p])
    
    def xdot_fun(x, t, params):
        xdot = zeros(len(x))
        xdot[0] = params[0] * (x[1] - x[0])  # x
        xdot[1] = x[0] * (params[2] - x[2]) - x[1]  # y
        xdot[2] = x[0] * x[1] - params[1] * x[2]  # z
        return xdot
    
    Y = ode_steps(xdot_fun, x0, t, params, rk4)
    
    plt.figure().add_subplot(111, projection='3d')  # this line enables 3D plotting
    plt.plot(Y[:, 0], Y[:, 1], Y[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
     
def lorenz_deviation_plot():
    x0 = array([1, 0, 0])
    x1 = array([1 + 10 ** -6, 0, 0])
    dt = 0.01
    t = arange(0, 50 + dt, dt)
    s = 10.0
    B = 8.0 / 3.0
    p = 28.0
    params = array([s, B, p])
    
    def xdot_fun(x, t, params):
        xdot = zeros(len(x))
        xdot[0] = params[0] * (x[1] - x[0])  # x
        xdot[1] = x[0] * (params[2] - x[2]) - x[1]  # y
        xdot[2] = x[0] * x[1] - params[1] * x[2]  # z
        return xdot
     
    Y0 = ode_steps(xdot_fun, x0, t, params, rk4)
    Y1 = ode_steps(xdot_fun, x1, t, params, rk4)
    
    plt.subplot(3, 1, 1)
    plt.semilogy(t, abs(Y0[:, 0] - Y1[:, 0]))
    plt.xlabel('t')
    plt.ylabel('log x dev')
    plt.subplot(3, 1, 2)
    plt.semilogy(t, abs(Y0[:, 1] - Y1[:, 1]))
    plt.xlabel('t')
    plt.ylabel('log y dev')
    plt.subplot(3, 1, 3)
    plt.semilogy(t, abs(Y0[:, 2] - Y1[:, 2]))
    plt.xlabel('t')
    plt.ylabel('log z dev')
    plt.show()

def lorenz_error_plot():
    x0 = array([1, 0, 0])
    dt1 = 5 * 10 ** -3
    t1 = arange(0, 50 + dt1, dt1)
    dt2 = 5 * 10 ** -4
    t2 = arange(0, 50 + dt2, dt2)
    dt3 = 5 * 10 ** -5
    t3 = arange(0, 50 + dt3, dt3)
    s = 10.0
    B = 8.0 / 3.0
    p = 28.0
    params = array([s, B, p])
    
    def xdot_fun(x, t, params):
        xdot = zeros(len(x))
        xdot[0] = params[0] * (x[1] - x[0])  # x
        xdot[1] = x[0] * (params[2] - x[2]) - x[1]  # y
        xdot[2] = x[0] * x[1] - params[1] * x[2]  # z
        return xdot    
    
    xrk41 = ode_steps(xdot_fun, x0, t1, params, rk4)
    xrk42 = ode_steps(xdot_fun, x0, t2, params, rk4)
    xrk43 = ode_steps(xdot_fun, x0, t3, params, rk4)
    
    x1 = xrk41[:, 0]
    x2 = zeros(len(x1))
    x3 = zeros(len(x1))
    xrk42 = xrk42[:, 0]
    xrk43 = xrk43[:, 0]
    i = 0
    count = 0
    while i <= len(xrk42):
        x2[count] = xrk42[i]
        count = count + 1
        i = i + 10
    i = 0
    count = 0
    while i <= len(xrk43):
        x3[count] = xrk43[i]
        count = count + 1
        i = i + 100
        
    # print len(t1),len(x1),len(x2),len(x3)
    plt.subplot(2, 1, 1)
    plt.semilogy(t1, abs(x1 - x3), 'b.')
    plt.xlabel('time')
    plt.ylabel('abs(x1-x3)')
    plt.subplot(2, 1, 2)
    plt.semilogy(t1, abs(x2 - x3), 'g.')
    plt.xlabel('time')
    plt.ylabel('abs(x2-x3)')
    plt.show()
 
#########################################################################################
# Additional code for ECLIPSE IDE users:

# Y1=ode_steps(ydot_fun,array([0,1]),arange(0,0.5+0.1,0.1),0,forward_euler)
# print Y1
"""
Comments:
"""

#Y2=ode_steps(ydot_fun,array([0,1]),arange(0,0.5+0.1,0.1),0,rk4)
#print Y2
"""
Comments:
"""

#damped_oscillator_plot()
"""
Comments:
"""

#driven_oscillator_plot()
"""
Comments:
"""

#driven_oscillator_error_plot()
"""
Comments:
"""

langevin_plot()
"""
Comments:
"""
# lorenz_plot()
"""
Comments:
"""

# lorenz_deviation_plot()
"""
Comments:
"""

# lorenz_error_plot()
"""
Comments:
"""
