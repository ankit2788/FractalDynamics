from math import cos, sin, sqrt, tan, pi, atan2, hypot
import numpy as np


"""
All different functions as mentioned in the paper:
https://flam3.com/flame_draves.pdf

Input: point --> a 2-D point  (a numpy array)
"""

def getRadius(point):
    x = point[0]
    y = point[1]

    if ((x == 0) and (y == 0)):
        x = 1e-1
        y = 1e-1

    radius = hypot(x,y)
    return radius



def none(point):
    return np.array(point)

def linear(point, dependentParams = None, ExternalParams = None ):
    a = dependentParams[0]
    b = dependentParams[1]
    c = dependentParams[2]
    d = dependentParams[3]

    e = dependentParams[4]
    f = dependentParams[5]

    LinearMatrix    = np.array([[a,b ], [c, d]])
    Translation     = np.array([e,f])
    return np.matmul(LinearMatrix, point) + Translation

def sinusodial(point, dependentParams = None, ExternalParams = None ):
    transformation = []
    for dim in point:
        transformation.append(sin(dim))
    
    return np.array(transformation)


def spherical(point, dependentParams = None, ExternalParams = None ):

    radius = getRadius(point)

    return (1/ radius) * np.array(point)

def swirl(point, dependentParams = None, ExternalParams = None ):

    x = point[0]
    y = point[1]
    radius = getRadius(point)
    return np.array([x*sin(radius**2) - y*cos(radius**2), x*cos(radius**2) + y*sin(radius**2)])

def horseshoe(point, dependentParams = None, ExternalParams = None ):

    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    
    return (1/radius)*np.array([(x-y)*(x+y), 2*x*y])

def polar(point, dependentParams = None, ExternalParams = None ):
    x = point[0]
    y = point[1]
    
    radius = getRadius(point)    
    theta = atan2(x,y)

    return np.array([theta/pi, radius- 1])

def handkerchief(point, dependentParams = None, ExternalParams = None ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)        
    theta = atan2(x,y)

    return radius * np.array([sin(theta + radius), cos(theta - radius)])


def heart(point, dependentParams = None, ExternalParams = None ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    return radius * np.array([sin(theta * radius),  -cos(theta * radius)])


def disc(point, dependentParams = None, ExternalParams = None ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    return (theta/pi) * np.array([sin(pi*radius), cos(pi* radius)])

def spiral(point, dependentParams = None, ExternalParams = None ):    
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    return (1/radius) * np.array([cos(theta) + sin(radius), sin(theta) - cos(radius)])

def hyperbolic(point, dependentParams = None, ExternalParams = None ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    return np.array([sin(theta)/radius, cos(theta)* radius])

def diamond(point, dependentParams = None, ExternalParams = None ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    return np.array([sin(theta) * cos(radius), cos(theta) * sin(radius)])

def ex(point, dependentParams = None, ExternalParams = None ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    p0 = sin(theta + radius)
    p1 = cos(theta - radius)

    return radius * np.array([p0**3 + p1**3, p0**3 - p1**3])
















