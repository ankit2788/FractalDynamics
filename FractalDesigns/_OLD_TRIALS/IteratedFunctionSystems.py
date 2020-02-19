#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:44:34 2019

@author: ankitgupta
"""

import random
from PIL import Image
 
 
class BarnsleyFern(object):
    def __init__(self, img_width, img_height, paint_color=(0, 150, 0),
                 bg_color=(255, 255, 255)):
        self.img_width, self.img_height = img_width, img_height
        self.paint_color = paint_color
        self.x, self.y = 0, 0
        self.age = 0
 
        self.fern = Image.new('RGB', (img_width, img_height), bg_color)
        self.pix = self.fern.load()
        self.pix[self.scale(0, 0)] = paint_color
 
    def scale(self, x, y):
        h = (x + 2.182)*(self.img_width - 1)/4.8378
        k = (9.9983 - y)*(self.img_height - 1)/9.9983
        return h, k
 
    def transform(self, x, y):
        rand = random.uniform(0, 100)
        if rand < 1:
            return 0, 0.16*y
        elif 1 <= rand < 86:
            return 0.85*x + 0.04*y, -0.04*x + 0.85*y + 1.6
        elif 86 <= rand < 93:
            return 0.2*x - 0.26*y, 0.23*x + 0.22*y + 1.6
        else:
            return -0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44
 
    def iterate(self, iterations):
        for _ in range(iterations):
            self.x, self.y = self.transform(self.x, self.y)
            self.pix[self.scale(self.x, self.y)] = self.paint_color
        self.age += iterations
 
fern = BarnsleyFern(500, 500)
fern.iterate(1000000)
fern.fern.show()


import numpy as np
from bisect import bisect_left

class IteratedFunctions:
    
    def __init__(self):
        """
        The object IFS can have multiple Linear Transformations with respective weights.
        X_n+1 = A* X_n + B. 
        where A--> Linear transformation
        B --> Linear translation
        
        """
        self.Transforms = []
        self.weights = []
        self.Translations = []
        self.Totalweight = 0
        self.CumProbWeight = [0]
        
    def add_transform(self, transform, weight = 0, translation = None):
        
        if (weight > 0):
            if (weight + self.Totalweight <= 1):
                
                self.Transforms.append(transform)
                self.Translations.append(translation)
                self.weights.append(weight)
                self.Totalweight += weight
                self.CumProbWeight.append(self.Totalweight)
                
    def choose_transform(self):
        """
        out of available transforms, choose a random transform to be applied. 
        """
        
        return self.Transforms(random.choice(range(len(self.Transforms))))
        
                
                
                

class Transform:
    def __init__(self):
        #defines the rgb coor
        self.r = random.random()
        self.g = random.random()
        self.b = random.random()
    
    def transform_colour(self, r, g, b):
        r = (self.r + r) / 2
        g = (self.g + g) / 2
        b = (self.b + b) / 2
        return r, g, b
    
    


class Barnsley() :
    def __init__(self):
        #super(Barnsley, self).__init__()
        
        self.IFS = IteratedFunctions()
        self.__CreateIFS__()
        self.allpoints = []
        
        
    def __CreateIFS__(self):
        #add 4 different transformations with respective probability
        self.IFS.add_transform(np.array([[0, 0], [0, 0.16]]), 
                               weight = 0.01, 
                               translation= np.array([0,0]))
        self.IFS.add_transform(np.array([[0.85, 0.04], [-0.04, 0.85]]), 
                               weight = 0.85,
                               translation= np.array([0,1.6]))
        self.IFS.add_transform(np.array([[0.2, -0.26], [0.23, 0.22]]), 
                               weight = 0.07, 
                               translation= np.array([0,1.6]))
        self.IFS.add_transform(np.array([[-0.15, 0.28], [0.26, 0.24]]), 
                               weight = 0.07, 
                               translation= np.array([0,0.44]))
        
        
        
    def transform(self, point):
        
        #generate a random number implyig the probability of selection. 
        #based on the probability, select the iterative function
        
        prob = random.uniform(0,1)
        #get the index for  the iterative function
        index = bisect_left(self.IFS.CumProbWeight, prob) - 1
        
        #print(prob,index)
        
        transformation = np.matmul(self.IFS.Transforms[index], point) + self.IFS.Translations[index]
        return transformation
    

    def iterate(self, iterations = 100):
        
        point = np.array([0,0])     #starting from origin
        
        for _ in range(iterations):
            point = self.transform(point)
            self.allpoints.append(point)
            
            
            
        
fern = Barnsley()
fern.iterate(100000)        

a = fern.allpoints
        
x =[]
y = []
for item in a:
    x.append(item[0])
    y.append(item[1])
    
import matplotlib.pyplot as plt
plt.scatter(x, y)
        
        
                
        
        
        