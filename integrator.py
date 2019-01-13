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
        
       
       for i in range(len(x)-1):
           #TODO: Recordar que los parámetros son adicionales
           k1 = fun(yi,x[i+1])
           k2 = fun(yi + 0.5*h*k1,x[i+1]+ 0.5*h)
           k3 = fun(yi + 0.5*h*k2,x[i+1]+ 0.5*h)
           k4 = fun(yi + h*k3, x[i+1] + h) #FIXME: parece haber un error
           yi = yi +(h/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
           res.append(yi)
       return [x,array(res)]
       
    def denseOutput(f, x, solveOutput, param = None):
       """
       @brief    Calcula el valor de la soculion para un x arbitrário
       @ref      "Computational methods for physicist"
       
       @param f   f(y0,x0,...) Función que calcula la derivada de y en x0.
       @param x   Punto de la variable independiente donde se quiere 
                  encontrar la solución.
       @param solveOutput   Arreglo retornado por el método "solve"       
       @param param   (Opcional) Lista con los parámetros adicionales que la 
                      función f necesita.
       """
       # ¿Hay parámetros adicionales?
       if param is None:
           fun  = f
       else:
           fun = lambda a,b : f(a,b,param)
           
       xArr = solveOutput[0]
       yArr = solveOutput[1]

       for i in range(len(xArr)):
           if x <= xArr[i] :
               break;
       if(i == len(xArr) and (x != xArr[i])):
           raise ValueError('El valor de la variable x está fuera de rango')
       
       if(x == xArr[i]):
           return yArr[i]
       
       h = xArr[i+1] - xArr[i]
       theta = (x - xArr[i])/h    
       k1 = fun(yArr[i],xArr[i])
       k2 = fun(yArr[i] + 0.5*h*k1,xArr[i]+ 0.5*h)
       k3 = fun(yArr[i] + 0.5*h*k2,xArr[i]+ 0.5*h)
       k4 = fun(yArr[i] + h*k3, xArr[i] + h)
       
       b1 = theta - (3*theta**2)/2. + (2*theta**3)/3.
       b2 = theta**2 - (2*theta**3)/3.
       b3 = b2
       b4 = -(theta**2)/2. + (2*theta**3)/3.
       
       return yArr[i] + h*(b1*k1 + b2*k2 + b3*k3 + b4*k4)