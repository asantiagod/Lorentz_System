# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 22:23:29 2018

@author: Santiago
"""

from integrator import rk4, euler
import numpy as np
import matplotlib.pyplot as plt

def Ec(data,t0): #las distancias están en Mm = 10^6m
    """    
    data: arreglo con los datos [V_x, x, V_y, y]
    t0: por compatibilidad.
    """
    G = 989.7#10^9m / (M_s day^2)
    r = np.sqrt(data[1]**2 + data[3]**2)
    k = -G/(r**3)
                    #dv_x     , dx     , dv_y     , dy 
    return np.array([k*data[1], data[0], k*data[3], data[2]])
    
y0 = np.array([0., 1.5*(10**2), 2.573, 0]) #[V_x, x, V_y, y]
sol = rk4.solve(Ec, y0, 0., 365.*5,365.*5,None)
t = sol[0]
sol = sol[1]

plt.plot(sol[:,1],sol[:,3])
plt.show()

y0 = np.array([0., 1.5*(10**2), 2.573, 0]) #[V_x, x, V_y, y]
sol = euler.solve(Ec, y0, 0., 365.*5,365.*5,None)
t = sol[0]
sol = sol[1]

plt.plot(sol[:,1],sol[:,3])
plt.show()