from abc import ABC, abstractclassmethod
import random
import numpy as np
from numpy.linalg import det    #to compute the determinant of Linear Transformation matrix
from IteratedFunctionSystems.Transformations import AllVariations as Flames

class Transform(ABC):
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





class Variation:
    def __init__(self, type = Flames.linear, weight = 1, externalparams = [0,0,0,0]):
        """
        Variation to be applied after the Linear Transformation.
        Type : a Variation function to be applied. By default: No variation
        weight factor: Default weight given to this variation: 1
         
        """
        self.type   = type
        self.weight = weight
        self.params = externalparams





        
