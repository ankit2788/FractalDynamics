"""
Contains various Geometry objects such as Point, Line, Triangle
Also, different functions that operate on these Geometry Objects
"""

import numpy as np
import math
from scipy import misc
import matplotlib.pyplot as plt



class Point:
    
    def __init__(self, x = None, y = None):
        self.x = x
        self.y = y
        
    def __repr__(self):
        return ("({0},{1})".format(self.x,self.y))
        
    def getCoords(self):
        return (self.x, self.y)
    
    def setCoords(self, x, y):
        self.x = x
        self.y = y
        
        
class Line:
    
    def __init__(self, PointA, PointB):
        #points are Point objects
        self.pointA = PointA
        self.pointB = PointB
        
        
    def __repr__(self):
        return ("Line joining points:{0} and {1}".format(self.pointA, self.pointB))
        

        

    def getLength(self):
        
        length = np.square(self.pointA.x - self.pointB.x) + np.square(self.pointA.y - self.pointB.y)
        return np.sqrt(length)
        
            
    def getPortion(self, portionSize, directionfrom = "left"):

        
        dividedPoint = Point()
        if directionfrom.upper() == "LEFT":
            x = self.pointA.x * (1- portionSize) + self.pointB.x * portionSize 
            y = self.pointA.y * (1- portionSize) + self.pointB.y * portionSize 
            
        else:
            x = self.pointB.x * (1- portionSize) + self.pointA.x * portionSize 
            y = self.pointB.y * (1- portionSize) + self.pointA.y * portionSize 
            
        dividedPoint.setCoords(x,y)
            
        return dividedPoint
            
             
    def getMid(self):

        return self.getPortion(0.5, "left")    

    def Draw(self, ax):
        if ax is None:
            fig, ax = plt.subplots(1,1)

        ax.plot([self.pointA.x, self.pointB.x ], [self.pointA.y, self.pointB.y ], color = "black")
        return ax
        
    

class Triangle:
    def __init__(self, pointA, pointB, pointC):
        self.A = pointA
        self.B = pointB
        self.C = pointC

    def __repr__(self):
        return ("Triangle joining:{0}, {1}, {2}".format(self.A, self.B, self.C))

    def getMidpoints(self):
        #returns the mid of all 3 sides
        mid1 = Line(self.A, self.B).getMid()
        mid2 = Line(self.B, self.C).getMid()
        mid3 = Line(self.C, self.A).getMid()

        return (mid1, mid2, mid3)
    
    
    def Draw(self, ax):
        if ax is None:
            fig, ax = plt.subplots(1,1)
            
        ax.plot([self.A.x, self.B.x ], [self.A.y, self.B.y ], color = "black")
        ax.plot([self.B.x, self.C.x ], [self.B.y, self.C.y ], color = "black")
        ax.plot([self.C.x, self.A.x ], [self.C.y, self.A.y ], color = "black")
        
        return ax
        

    

def RotationMatrix(theta):
    '''
    returns the counterclockwise rotation matrix for linear transformation
    theta --> in radians
    '''
    
    mat =  np.array([[math.cos(theta), - math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]] )
    
    return mat

def DrawTriangle(line, triangletype = "EQ", theta1 = 60, theta2 = 60):
    '''
    Given 2 points (or a line) and angle between them, construct the triangle
    '''

    #get the vector from pointB to pointA
    myVector = np.array([line.pointB.x - line.pointA.x, line.pointB.y - line.pointA.y] )

    if triangletype == "EQ":
        
            #rotate this vector by 60 degrees keeping the length same
        rotationMatrix = RotationMatrix(math.radians(60))
        newvector = np.matmul(rotationMatrix, myVector)        
        
    else:
        theta3 = 180 - theta1 - theta2
        linelength = line.getLength()
        length_newvector = linelength * math.sin(math.radians(theta2))/ math.sin(math.radians(theta3))
            
        rotationMatrix = RotationMatrix(math.radians(theta3))
            
        newvector = length_newvector * np.matmul(rotationMatrix, myVector)        
            
    point = Point(newvector[0] + line.pointA.x, newvector[1] + line.pointA.y)
    return point
        
    
        

    