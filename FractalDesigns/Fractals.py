#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:37:47 2019

@author: ankitgupta

For details on the algorithm, please check:
https://flam3.com/flame_draves.pdf
"""

import random
import numpy as np
from bisect import bisect_left
from numpy.linalg import det    #to compute the determinant of Linear Transformation matrix
import matplotlib.pyplot as plt
import AllTransforms as Flames




class IteratedFunctions:
    
    def __init__(self):
        """
        The object IFS can have multiple Linear Transformations with respective weights.
        X_n+1 = A* X_n + B. 
        where A--> Linear transformation
        B --> Linear translation
        
        """
        self.Transforms = []
        self.Totalweight = 0
        self.CumProbWeight = [0]
        
    def add_transform(self, Transform):
        # a transform object is added
        
        if (Transform.Probability > 0):
            if (Transform.Probability + self.Totalweight <= 1):
                
                self.Transforms.append(Transform)
                self.Totalweight += Transform.Probability
                self.CumProbWeight.append(self.Totalweight)
                
    def choose_transform(self, index = None):
        """
        out of available transforms, choose a random transform to be applied. 
        """
        if index is None:
            return self.Transforms[random.choice(range(len(self.Transforms)))]
        else:
            return self.Transforms[index]
        
    def get_index(self, probability):
        """
        returns the index of the Transform to be applied given the prob
        """
        index = bisect_left(self.CumProbWeight, probability) - 1
        return index
        
        
                
                
                

class Transform:
    def __init__(self):
        #defines the rgb coor
        
        self.r = random.random()
        self.g = random.random()
        self.b = random.random()
        
    
    def transform_colour(self, r, g, b):
        r = (255* (self.r + r) / 2)%255
        g = (255* (self.g + g) / 2)%255
        b = (255* (self.b + b) / 2)%255
        return r, g, b
        
        
    
    
class LinearTransform(Transform):
    def __init__(self, Probability, AllRandom = True, LinearMatrix = None, TranslationMatrix = None):
        
        super(LinearTransform, self).__init__()
        if AllRandom:
            #creating the Linear Transformation matrix as per random
            #[[a, b], [c,d]]
            #[e,f]   This is for Linear translation
            a = random.uniform(-1, 1)
            b = random.uniform(-1, 1)
            c = random.uniform(-1, 1)
            d = random.uniform(-1, 1)
            
            e = random.uniform(-1, 1)
            f = random.uniform(-1, 1)
            
            self.LinearMatrix   = np.array([[a, b], [c, d]])
            self.Translation    = np.array([e, f])
            self.Probability    = Probability
            
        else:
            if LinearMatrix is not None:
                #this is in a list format
                a = LinearMatrix[0]
                b = LinearMatrix[1]
                c = LinearMatrix[2]
                d = LinearMatrix[3]
                
                self.LinearMatrix = np.array([[a, b], [c, d]])
                
                
                
            else:
                exit

            if TranslationMatrix is not None:
                e = TranslationMatrix[0]
                f = TranslationMatrix[1]
            else:
                e, f = 0, 0
                
            self.Translation    = np.array([e, f])
            self.Probability    = Probability
            
                
                
    def applyTransform(self, Point):
        return np.matmul(self.LinearMatrix, Point) + self.Translation
    

class GeneralizedIFS():
    def __init__(self):
        self.Transforms     = []
        self.Totalweight    = 0
        self.CumProbWeight  = [0]

    def add_transform(self, Transform):
        #Transform --> a GeneralizedTransform object
        if Transform.Probability > 0:
            if (Transform.Probability + self.Totalweight <= 1):
                
                self.Transforms.append(Transform)
                self.Totalweight += Transform.Probability
                self.CumProbWeight.append(self.Totalweight)
        

    def choose_transform(self, index = None):
        """
        out of available transforms, choose a random transform to be applied. 
        """
        if index is None:
            return self.Transforms[random.choice(range(len(self.Transforms)))]
        else:
            return self.Transforms[index]
        
    def get_index(self, probability):
        """
        returns the index of the Transform to be applied given the prob
        """
        index = bisect_left(self.CumProbWeight, probability) - 1
        return index


    

class Variation:
    def __init__(self, type = Flames.none, weight = 1, externalparams = [0,0,0,0]):
        self.type   = type
        self.weight = weight
        self.params = externalparams


class GeneralizedTransform:
    def __init__(self, ProbImplementation = None, Probability = 0.0, Color = 1,  Random = False, 
                IFSparams = [1,0,0,1,0,0], PostTransformParams = [0,0,0,0]):
        """
        Parameters list consist of the below:
        ProbImplementation: Probability can be either based on the contraction factor determined by IFS params
                            Applicable values: [CONTRACTION, NONE]
        Probability:        If ProbImplementation is None, input > 0 is required
        Random:             Choice of IFS params turns to be random completely. 
        IFSparams:
            a,b,c,d --> Linear Transformation
            e,f --> Linear Translation
            Default parameters set as for Linear Transformation
        """

        if Random is True:
            a = random.uniform(-1, 1)
            b = random.uniform(-1, 1)
            c = random.uniform(-1, 1)
            d = random.uniform(-1, 1)
            e = random.uniform(-1, 1)
            f = random.uniform(-1, 1)
            
            self.IFSparams      = [a,b,c,d,e,f]
        else:
            self.IFSparams      = IFSparams

        if ProbImplementation is None:
            self.Probability    = Probability
        else:
            #get the contraction factor
            self.Probability    = self.__getContractionFactor()
        
        if (self.Probability > 1):
            print("Probability weight cant be greater than 1. Exiting!")
            exit(1)

        self.Color          = Color
        self.Variations     = []


        #setting the post transform parameters
        self.PostTransform  = PostTransformParams

    def __getContractionFactor(self):
        LinearMatrix = np.array([[self.IFSparams[0], self.IFSparams[1]], [self.IFSparams[2], self.IFSparams[3]]])
        return abs(det(LinearMatrix))


    def addVariation(self, variation = Variation()):
        self.Variations.append(variation)


    def apply_transform(self, point):
        totalweight = 0
        for variation in self.Variations:
            weight          = variation.weight
            totalweight     += weight
        
        for variation in self.Variations:
            #applying the transformation based on the weight of all variations
            weight          = variation.weight
            func            = variation.type            
            externalparams  = variation.params
            
            point           = (weight/totalweight) * func(point, self.IFSparams, externalparams)

        return point







    
def FractalImage(IFS, InitialPoint = np.array([0,0]), Iterations = 1000000):
    """
    IFS --> an Iterated Function system object containing the functions to be applied
    """
    
    point = InitialPoint
    allpoints = [point]

    #r, g, b = 0, 0, 0
    #allcolors = [[r, g, b]]
    
    for _ in range(Iterations):
        rand = random.uniform(0, 1)
        
        index = IFS.get_index(rand)
        t = IFS.choose_transform(index)
        
        #apply transform
        point = t.applyTransform(point)
        allpoints.append(point)
    
    #    r, g, b = t.transform_colour(r, g, b)
    #    allcolors.append([r, g, b])
    
    x =[]
    y = []
    for item in allpoints:
        x.append(item[0])
        y.append(item[1])

    return x, y
    
    

from math import radians, cos, atan2, sin
from AllTransforms import getRadius
def spiral(point, dependentParams = None, ExternalParams = None ):    
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)      #atan2 returns the value in radians

    return (1/radius) * np.array([cos(theta) + sin(radians(radius)), sin(theta) - cos(radians(radius))])

    
spiral = Variation(spiral)
spiral = Variation(Flames.spiral)
heart = Variation(Flames.heart)
swirl = Variation(Flames.swirl)
linear = Variation(Flames.linear)


    
transform1 = GeneralizedTransform(Probability = 1)
transform1.addVariation(spiral)
transform1.addVariation(heart)
transform1.addVariation(swirl)

#Sierpinski
Linear1 = GeneralizedTransform(1/3, IFSparams = [0.5, 0, 0, 0.5, 0, 0])
Linear1.addVariation(linear)

Linear2 = GeneralizedTransform(1/3, IFSparams = [0.5, 0, 0, 0.5, 0.5, 0.5])
Linear2.addVariation(linear)

Linear3 = GeneralizedTransform(1/3, IFSparams = [0.5, 0, 0, 0.5, 1, 0])
Linear3.addVariation(linear)

Sierpinski = GeneralizedIFS()
Sierpinski.add_transform(Linear1)
Sierpinski.add_transform(Linear2)
Sierpinski.add_transform(Linear3)

#random1

Linear1 = GeneralizedTransform(Probability = .01, Random=False, IFSparams = [0, 0, 0, 0.16, 0, 0])
Linear1.addVariation(linear)

Linear2 = GeneralizedTransform(Probability = 0.62, Random=False, IFSparams = [1.2, 0.44, -1.04, 0.3, 1.6, -1.6])
Linear2.addVariation(linear)

Linear3 = GeneralizedTransform(Probability = 0.23, Random=False, IFSparams = [-0.2, 0.44, .04, 0.3,0, 1.6])
Linear3.addVariation(linear)

Linear4 = GeneralizedTransform(Probability = 0.01, Random=False, IFSparams = [0.2, -0.26, 0.23, 0.22,0, 1.6])
Linear4.addVariation(linear)

Linear5 = GeneralizedTransform(Probability = 0.13, Random=False, IFSparams = [-0.15, 0.28, 0.26, 0.24,0, 0.44])
Linear5.addVariation(swirl)

Random1 = GeneralizedIFS()
Random1.add_transform(Linear1)
Random1.add_transform(Linear2)
Random1.add_transform(Linear3)
Random1.add_transform(Linear4)
Random1.add_transform(Linear5)






point = np.array([0.1,0])
x = [point[0]]
y = [point[1]]

for i in range(1000000):
    
        rand = random.uniform(0, 1)
        
        index = Random1.get_index(rand)
        t = Random1.choose_transform(index)
        
        #apply transform
        point = t.apply_transform(point)
        #allpoints.append(point)
        x.append(point[0])
        y.append(point[1])
 
    #    point = transform1.apply_transform(point)
#    x.append(point[0])
#    y.append(point[1])
    
plt.scatter(x,y, color = "black", s = 0.2)


def swirl(point, dependentParams = None, ExternalParams = None ):

    x = point[0]
    y = point[1]
    radius = getRadius(point)
    return np.array([x*sin(radians(radius**2)) - y*cos(radians(radius**2)), x*cos(radians(radius**2)) + y*sin(radians(radius**2))])

def sinusodial(point, dependentParams = None, ExternalParams = None ):
    x = point[0]
    y = point[1]
    
    return np.array([sin(radians(x)), sin(radians(y))])


x = []
y = []
X = np.arange(-20,20,0.5)

for hor in X:
    for yer in np.arange(-20,20,0.5):
        x.append(hor)
        y.append(yer)
        
        




x_changed = []
y_changed = []
myvariation = Variation(Flames.polar)

weight          = myvariation.weight
func            = myvariation.type            
externalparams  = myvariation.params

for i in range(len(x)):
    point = np.array([x[i], y[i]] )
    for iter in range(1000):
        
        point           = func(point)
        
    x_changed.append(point[0])
    y_changed.append(point[1])

plt.figure()
plt.scatter(x_changed,y_changed, color = "black", s = 0.2)



        
