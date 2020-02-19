#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:26:54 2019

@author: ankitgupta
"""
from Fractals import IteratedFunctions, Transform, FractalImage, LinearTransform
import matplotlib.pyplot as plt
import math
import numpy as np
from IteratedFunctionSystems.IFS import IteratedFunctions, Transform, LinearTransform
from IteratedFunctionSystems.Drawing import FractalImage
from Fractals import IteratedFunctions, Transform, FractalImage, LinearTransform


#Barnsley Fern
Fern = IteratedFunctions()
Linear1 = LinearTransform(0.01, False, [0, 0, 0, 0.16], [0, 0])
Linear2 = LinearTransform(0.85, False, [0.85, 0.04, -0.04, 0.85], [0, 1.6])
Linear3 = LinearTransform(0.07, False, [0.2, -0.26, 0.23, 0.22], [0, 1.6])
Linear4 = LinearTransform(0.07, False, [-0.15, 0.28, 0.26, 0.24], [0, 0.44])

Fern.add_transform(Linear1)
Fern.add_transform(Linear2)
Fern.add_transform(Linear3)
Fern.add_transform(Linear4)

x_fern, y_fern = FractalImage(Fern)

#Dentrite 
Dentrite = IteratedFunctions()
Linear2 = LinearTransform(0.167, False, [0.342, -0.701, 0.071, 0.341], [0, 0])
Linear1 = LinearTransform(0.167, False, [0.038, -0.346, 0.346, 0.038], [0.341, 0.071])
Linear3 = LinearTransform(0.166, False, [0.341, -0.071, 0.071, 0.341], [.379, .418])
Linear4 = LinearTransform(0.167, False, [-0.234, 0.258, -0.258, -0.234], [0.72, 0.489])
Linear5 = LinearTransform(0.167, False, [0.173, 0.302, -0.302, 0.173], [0.486, 0.231])
Linear6 = LinearTransform(0.166, False, [0.341, -0.071, 0.071, 0.341], [0.659, -0.071])

Dentrite.add_transform(Linear1)
Dentrite.add_transform(Linear2)
Dentrite.add_transform(Linear3)
Dentrite.add_transform(Linear4)
Dentrite.add_transform(Linear5)
Dentrite.add_transform(Linear6)

x_den, y_den = FractalImage(Dentrite)


#Sierpinski
Sierpinski = IteratedFunctions()
Linear1 = LinearTransform(1/3, False, [0.5, 0, 0, 0.5], [0, 0])
Linear2 = LinearTransform(1/3, False, [0.5, 0, 0, 0.5], [0.5, 0.5])
Linear3 = LinearTransform(1/3, False, [0.5, 0, 0, 0.5], [1, 0])

Sierpinski.add_transform(Linear1)
Sierpinski.add_transform(Linear2)
Sierpinski.add_transform(Linear3)

x_sier, y_sier = FractalImage(Sierpinski)

#Koch Curve
cos60 = math.cos(math.radians(60))
sin60 = math.sin(math.radians(60))

Koch = IteratedFunctions()

Linear1 = LinearTransform(1/4, False, [1/3, 0, 0, 1/3], [0, 0])
Linear2 = LinearTransform(1/4, False, [1/3 * cos60, -1/3*sin60, 1/3*sin60, 1/3 * cos60], [1/3, 0])
Linear3 = LinearTransform(1/4, False, [1/3 * cos60, 1/3*sin60, -1/3*sin60, 1/3* cos60], [0.5, 1/3*sin60])
Linear4 = LinearTransform(1/4, False, [1/3, 0, 0, 1/3], [2/3, 0])

Koch.add_transform(Linear1)
Koch.add_transform(Linear2)
Koch.add_transform(Linear3)
Koch.add_transform(Linear4)

x_koch, y_koch = FractalImage(Koch)


#random IFS
random1 = IteratedFunctions()
Linear1 = LinearTransform(0.1, True)
Linear2 = LinearTransform(0.3, True)
Linear3 = LinearTransform(0.05, True)
Linear4 = LinearTransform(0.02, True)
Linear5 = LinearTransform(0.23, True)
Linear6 = LinearTransform(0.2, True)
Linear7 = LinearTransform(0.1, True)

random1.add_transform(Linear1)
random1.add_transform(Linear2)
random1.add_transform(Linear3)
random1.add_transform(Linear4)
random1.add_transform(Linear5)
random1.add_transform(Linear6)
random1.add_transform(Linear7)

x_ran1, y_ran1 = FractalImage(random1)

#random1
random2 = IteratedFunctions()
Linear1 = LinearTransform(0.01, False, [0, 0, 0, 0.16], [0, 0])
Linear2 = LinearTransform(0.62, False, [1.2, 0.44, -1.04, 0.3], [0, 1.6])
Linear3 = LinearTransform(0.23, False, [-0.2, 0.44, .04, 0.3], [0, 1.6])
Linear4 = LinearTransform(0.07, False, [0.2, -0.26, 0.23, 0.22], [0, 1.6])
Linear5 = LinearTransform(0.07, False, [-0.15, 0.28, 0.26, 0.24], [0, 0.44])

random2.add_transform(Linear1)
random2.add_transform(Linear2)
random2.add_transform(Linear3)
random2.add_transform(Linear4)
random2.add_transform(Linear5)

x_ran2, y_ran2 = FractalImage(random2)

#random2
random3 = IteratedFunctions()
Linear1 = LinearTransform(0.01, False, [0, 0, 0, 0.16], [0, 0])
Linear2 = LinearTransform(0.85, False, [1.2, 0.44, -1.04, 0.3], [0, 1.6])
Linear3 = LinearTransform(0.07, False, [0.2, -0.26, 0.23, 0.22], [0, 1.6])
Linear4 = LinearTransform(0.07, False, [-0.15, 0.28, 0.26, 0.24], [0, 0.44])

random3.add_transform(Linear1)
random3.add_transform(Linear2)
random3.add_transform(Linear3)
random3.add_transform(Linear4)

x_ran3, y_ran3 = FractalImage(random3)

fig, ax = plt.subplots(1,2)    
ax[0].scatter(np.array(x_ran2), np.array(y_ran2), c = "black", s = 0.15, alpha = 0.6)
ax[0].
ax[1].scatter(np.array(x_ran3), np.array(y_ran3), c = "black", s = 0.15, alpha = 0.4)

plt.scatter(np.array(x_sier), np.array(y_sier), c = "black", s = 0.15, alpha = 0.4)

plt.title("Koch Curve")
plt.axis("off")


x1, y1 = x, y


maxi = 0
for color in allcolors:
    max2 = max(color)         
    if max2 > maxi:
        maxi= max2
        