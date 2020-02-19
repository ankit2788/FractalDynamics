#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:26:54 2019

@author: ankitgupta
"""
from IteratedFunctionSystems.IFS import ClassicIteratedFunctions
from IteratedFunctionSystems.Drawing import FractalImage
from IteratedFunctionSystems.Transformations.Transforms import LinearTransform
from Utils import Config

import matplotlib.pyplot as plt
import math
import numpy as np

# ------------------------------------------------------------
#Barnsley Fern
Fern        = ClassicIteratedFunctions(name="Barnsley Fern")
Fern.applyConfiguration(configPath= "/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/IFS/Fern.ini")
Fern.draw()


# ------------------------------------------------------------
#Dentrite 
Dentrite    = ClassicIteratedFunctions(name="Dentrite")
Dentrite.applyConfiguration(configPath= "/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/IFS/Dentrite.ini")
Dentrite.draw()


# ------------------------------------------------------------
#Sierpinski
#1. Equilateral Triangle
Sierpinski  = ClassicIteratedFunctions(name="Sierpinski Eq. Triangle")
Sierpinski.applyConfiguration(configPath= "/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/IFS/Sierpinski1.ini")
Sierpinski.draw()

#2. Right Angled Triangle
Sierpinski  = ClassicIteratedFunctions(name="Sierpinski Right Triangle")
Sierpinski.applyConfiguration(configPath= "/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/IFS/Sierpinski2.ini")
Sierpinski.draw()


# ------------------------------------------------------------
#Koch Curve
Koch    = ClassicIteratedFunctions(name="Koch")
Koch.applyConfiguration(configPath= "/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/IFS/Koch.ini")
Koch.draw()

# ------------------------------------------------------------
#random IFS1
random1     = ClassicIteratedFunctions(name="Random1")
random1.applyConfiguration(configPath= "/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/IFS/Random1.ini")
random1.draw()

#random IFS2
random2     = ClassicIteratedFunctions(name="Random2")
random2.applyConfiguration(configPath= "/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/IFS/Random2.ini")
random2.draw()

#random IFS3
random3     = ClassicIteratedFunctions(name="Random3")
random3.applyConfiguration(configPath= "/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/IFS/Random3.ini")
random3.draw()

fig, ax = plt.subplots(1,2)    
ax[0].scatter(np.array(Fern._Xaxis), np.array(Fern._Yaxis), c = "black", s = 0.15, alpha = 0.4)
ax[0].set_axis_off()
ax[0].set_title("Leaf")

ax[1].scatter(np.array(random3._Xaxis), np.array(random3._Yaxis), c = "black", s = 0.15, alpha = 0.4)
ax[1].set_axis_off()
ax[1].set_title("Completely Random")

plt.suptitle("Guess the Transformations")



fig, ax = plt.subplots(2,2)    
ax[0,0].scatter(np.array(Koch._Xaxis), np.array(Koch._Yaxis), c = "black", s = 0.15, alpha = 0.4)
ax[0,0].set_axis_off()

ax[0,1].scatter(np.array(Sierpinski._Xaxis), np.array(Sierpinski._Yaxis), c = "black", s = 0.15, alpha = 0.4)
ax[0,1].set_axis_off()

ax[1,0].scatter(np.array(random3._Xaxis), np.array(random3._Yaxis), c = "black", s = 0.15, alpha = 0.4)
ax[1,0].set_axis_off()

ax[1,1].scatter(np.array(Fern._Xaxis), np.array(Fern._Yaxis), c = "black", s = 0.15, alpha = 0.4)
ax[1,1].set_axis_off()

