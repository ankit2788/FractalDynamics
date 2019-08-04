#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:58:45 2019

@author: ankitgupta
"""

            
            
        
import numpy as np
from IterativeFunctions import Koch, Sierpinski, IterativeFunction
from Geometry import Point, Line, Triangle
from matplotlib.lines import Line2D    
import matplotlib.pyplot as plt 


pointA = Point(0,0)
pointB = Point(1,0)
line = Line(pointA, pointB)

pointC = Point(0.5,np.sqrt(3)/2)
triangle = Triangle(pointA, pointB, pointC)


Koch = IterativeFunction("Koch", line, "PLOT")
Koch.Draw(order = 2)
Koch.Draw(order = 3)
Koch.Draw(order = 4)
Koch.ax.

Sierpinski = IterativeFunction("Sierpinski", triangle, "PLOT")
Sierpinski.Draw(order = 7)



