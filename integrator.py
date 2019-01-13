# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 22:22:14 2018

@author: Santiago
"""
from numpy import linspace, array

class rk4:   
    #def solve(self,f,y0,x0,xf,n,param = None):
    def solve(f,y0,x0,xf,n,param = None):
       """
       @brief     Soliciona un sistema de EDO por el método de Runge-Kutta 4.
                  Se basa el la función scipy.integrate.odeint para tomar los 
                  parámetros de entrada.
       @param f   f(y0,x0,...) Función que calcula la derivada de y en x0.
       @param y0  Valores iniciales (puede ser un vector).
       @param x0  Valor inicial de la variable independiente.
       @param xf  Valor fínal de la variable independiente.
       @param n   Número de puntos en los que se va a calcular la solución
       @param param   (Opcional) Lista con los parámetros adicionales que la 
                      función f necesita.       
       """
       #TODO: validar el tipo y el número de n
       x = linspace(x0,xf,n)       
       h = x[1] - x[0]
       res = [y0]
       yi = y0
       
       # ¿Hay parámetros adicionales?
       if param is None:
           fun  = f
       else:
           fun = lambda a,b : f(a,b,param)
        
       
       for i in range(len(x)):
           #TODO: Recordar que los parámetros son adicionales
           k1 = fun(yi,x[i])
           k2 = fun(yi + 0.5*h*k1,x[i]+ 0.5*h)
           k3 = fun(yi + 0.5*h*k2,x[i]+ 0.5*h)
           k4 = fun(yi + h, x[i] + h)
           yi = yi +(h/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
           res.append(yi)
       return [x,array(res)]
       
    def denseOutput():
       """
       TODO: hay hacerlo :P
       Permite enconctrar la solución en un punto arbitrario x
       """
       print("Dense output")