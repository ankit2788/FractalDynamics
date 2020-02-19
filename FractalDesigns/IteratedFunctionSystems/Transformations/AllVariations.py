from math import cos, sin, sqrt, tan, pi, atan2, hypot, cosh, sinh
import numpy as np
import random


"""
All different functions as mentioned in the paper:
https://flam3.com/flame_draves.pdf

Input: point --> a 2-D point  (a numpy array)
"""


NO_TRANSFORMATION = np.array([[1, 0], [0, 1]])
NO_TRANSLATION = np.array([0,0])



def chooseRandom(type = "OMEGA"):
    prob = random.uniform(0,1)

    if type.upper() == "OMEGA":
        if prob < 0.5:
            return 0
        else:
            return pi

    if type.upper() == "TAU":
        if prob < 0.5:
            return -1
        else:
            return 1

    if type.upper() == "CHI":
        return prob

def getRadius(point):
    x = point[0]
    y = point[1]

    radius = hypot(x,y)
    return radius



def none(point):
    return np.array(point)

#variation 0
def linear(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    """
    dependentParams --> tuple of (LinearTransformation Matrix, Translation Matrix)
                        Default params: (No Transformation, No Translation)
    """

    LinearMatrix    = dependentParams[0]
    Translation     = dependentParams[1]
    return np.matmul(LinearMatrix, point) + Translation

#variation 1
def sinusodial(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    transformation = []
    for dim in point:
        transformation.append(sin(dim))
    
    return np.array(transformation)

#variation 2
def spherical(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):

    x = point[0]
    y = point[1]
    radius = getRadius(point)

    r = 1/(radius**2)


    return r * np.array([x,y])

#variation 3
def swirl(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):

    x = point[0]
    y = point[1]
    radius = getRadius(point)
    rad_2 = radius**2

    return np.array([x*sin(rad_2) - y*cos(rad_2), x*cos(rad_2) + y*sin(rad_2)])

#variation 4
def horseshoe(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):

    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    
    return (1/radius)*np.array([(x-y)*(x+y), 2*x*y])

#variation 5
def polar(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):
    x = point[0]
    y = point[1]
    
    radius = getRadius(point)    
    theta = atan2(x,y)

    return np.array([theta/pi, radius- 1])

#variation 6
def handkerchief(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)        
    theta = atan2(x,y)

    return radius * np.array([sin(theta + radius), cos(theta - radius)])


#variation 7
def heart(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    return radius * np.array([sin(theta * radius),  -cos(theta * radius)])

#variation 8
def disc(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    return (theta/pi) * np.array([sin(pi*radius), cos(pi* radius)])

#variation 9
def spiral(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):    
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    return (1/radius) * np.array([cos(theta) + sin(radius), sin(theta) - cos(radius)])

#variation 10
def hyperbolic(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    return np.array([sin(theta)/radius, cos(theta)* radius])

#variation 11
def diamond(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    return np.array([sin(theta) * cos(radius), cos(theta) * sin(radius)])

#variation 12
def ex(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    p0 = sin(theta + radius)
    p1 = cos(theta - radius)

    return radius * np.array([p0**3 + p1**3, p0**3 - p1**3])


#variation 13
def julia(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):

    x = point[0]
    y = point[1]
    theta = atan2(x,y)
    omega = chooseRandom(type="OMEGA")
    radius = getRadius(point)    

    return np.sqrt(radius) * np.array([cos(theta/2 + omega), sin(theta/2 + omega)])


#variation 14
def bent(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):

    x = point[0]
    y = point[1]

    if x >= 0 and y >= 0:
        return point

    elif x < 0 and y >= 0:
        return np.array([2*x, y])

    elif x >= 0 and y < 0:
        return np.array([x, y/2])
    
    elif x < 0 and y < 0:
        return np.array([2*x, y/2])



#variation 15
def waves(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):
    x = point[0]
    y = point[1]

    b = dependentParams[0][0][1]
    e = dependentParams[0][1][1]
    c = dependentParams[1][0]
    f = dependentParams[1][1]

    return np.array([x + b*sin(y/c**2), y + e*sin(x/f**2)])



#variation 16
def fisheye(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):

    x = point[0]
    y = point[1]
    radius = getRadius(point)    

    return 2/(radius + 1) * np.array([y, x])

#variation 17
def popcorn(point, dependentParams= (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0) ):
    x = point[0]
    y = point[1]

    c = dependentParams[1][0]
    f = dependentParams[1][1]

    return np.array([x + c*sin(tan(3*y)), y + f*sin(tan(3*x))])


#variation 18
def exponential(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]

    return np.exp(x-1) * np.array([cos(pi*y), sin(pi*y)])


#variation 19
def power(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)


    return pow(radius, sin(theta)) * np.array([cos(theta), sin(theta)])

#variation 20
def cosine(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]

    return np.array([cos(pi*x) * cosh(y), -sin(pi*x) * sinh(y)])


#variation 21
def rings(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    c_2 = (dependentParams[1][0])**2
    prefix = (radius + c_2)%(2*c_2) - c_2 + radius*(1-c_2)

    return prefix * np.array([cos(theta), sin(theta)])


#variation 22
def fan(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    c_2 = (dependentParams[1][0])**2
    f   = dependentParams[1][1]
    t   = pi*c_2

    if (theta + f)%t > t/2:
        return radius * np.array([cos(theta - t/2), sin(theta - t/2)])
    else:
        return radius * np.array([cos(theta + t/2), sin(theta + t/2)])


#variation 23
def blob(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):

    p1 = externalparams[0]          #represents high
    p2 = externalparams[1]          #represents high
    p3 = externalparams[2]          #represents waves
    x = point[0]
    y = point[1]
    radius = getRadius(point)    
    theta = atan2(x,y)

    prefix = radius * (p2 + .5*(p1 - p2)*(sin(p3*theta) + 1))

    return prefix * np.array([cos(theta), sin(theta)])



#variation 24
def pdj(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):

    p1 = externalparams[0]          
    p2 = externalparams[1]          
    p3 = externalparams[2]          
    p4 = externalparams[3]          
    x = point[0]
    y = point[1]

    return np.array([sin(p1*y) - cos(p2*x), sin(p3*x) - cos(p4*y)])

#variation 25
def fan2(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):

    p1 = pi * externalparams[0]**2          
    p2 = externalparams[1]          
    x = point[0]
    y = point[1]

    radius = getRadius(point)    
    theta = atan2(x,y)

    t = theta + p2 - p1 * int(2*theta*p2/p1)

    if t > p1/2:
        return radius * np.array([sin(theta -0.5*p1), cos(theta -0.5*p1)])

    else:
        return radius * np.array([sin(theta +0.5*p1), cos(theta +0.5*p1)])


#variation 26
def rings2(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):

    p = externalparams[0]**2          
    x = point[0]
    y = point[1]

    radius = getRadius(point)    
    theta = atan2(x,y)

    prefix = radius - 2*p*int((radius + p)/(2*p)) + radius * (1-p)

    return prefix * np.array([cos(theta), sin(theta)])



#variation 27
def eyefish(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    

    return 2/(radius + 1) * np.array([x, y])


#variation 28
def bubble(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]
    radius = getRadius(point)    

    return 4/(radius**2 + 4) * np.array([x, y])


#variation 29
def cylinder(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]
    
    return np.array([sin(x), y])


#variation 30
def perspective(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):

    p1 = externalparams[0]      #angle
    p2 = externalparams[1]      #distance    
    x = point[0]
    y = point[1]

    prefix = p2 / (p2 - y*sin(p1))
    return prefix * np.array([x, y*cos(p1)])


#variation 31
def noise(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]

    chi1 = chooseRandom(type="CHI")
    chi2 = chooseRandom(type="CHI")

    return chi1* np.array([x*cos(2*pi*chi2), y*sin(2*pi*chi2)])


#variation 32
def juliaN(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]
    radius = getRadius(point)   
    chi = chooseRandom(type="CHI")
    phi = atan2(y,x)

    p1 = externalparams[0]      #power
    p2 = externalparams[1]      #distance    
    p3 = int(np.abs(p1)*chi)
 
    t = (phi + 2*pi*p3)/p1

    prefix = radius**(p2/p1)

    return prefix * np.array([cos(t), sin(t)])


#variation 33
def juliaScope(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]
    radius = getRadius(point)   
    chi = chooseRandom(type="CHI")
    tau = chooseRandom(type="TAU")
    phi = atan2(y,x)

    p1 = externalparams[0]      #power
    p2 = externalparams[1]      #distance    
    p3 = int(np.abs(p1)*chi)
 
    t = (tau*phi + 2*pi*p3)/p1

    prefix = radius**(p2/p1)

    return prefix * np.array([cos(t), sin(t)])



#variation 34
def blur(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]

    chi1 = chooseRandom(type="CHI")
    chi2 = chooseRandom(type="CHI")

    return chi1* np.array([cos(2*pi*chi2), sin(2*pi*chi2)])

#variation 35
def gaussian(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]

    chi = []
    for _ in np.arange(5):
        chi.append(chooseRandom(type="CHI"))

    return (np.sum(chi[:4]) - 2)* np.array([cos(2*pi*chi[4]), sin(2*pi*chi[4])])


#variation 36
def radialblur(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]

    radius = getRadius(point)   

    p1      = externalparams[0] * pi * 0.5      #angle
    coeff   = externalparams[1]                 # variation coefficient
    phi = atan2(y,x)

    chi = []
    for _ in np.arange(5):
        chi.append(chooseRandom(type="CHI"))

    t1 = coeff * (np.sum(chi) - 2)        #TODO: check again
    t2 = phi + t1 * sin(p1)
    t3 = t1 * cos(p1) - 1


    return 1/coeff* np.array([radius * cos(t2) + t3*x, radius*sin(t2) + t3*y])



#variation 37
def pie(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]

    chi1 = chooseRandom(type="CHI")
    chi2 = chooseRandom(type="CHI")
    chi3 = chooseRandom(type="CHI")

    p1      = externalparams[0]       #slices
    p2      = externalparams[1]       #rotation
    p3      = externalparams[2]       #thickness

    t1 = int(chi1 * p1 + 0.5)
    t2 = p2 + 2*pi*(t1 + chi2*p3)/ p1
    
    return chi3 * np.array([cos(t2), sin(t2)])


#variation 42
def tangent(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    x = point[0]
    y = point[1]

    return np.array([sin(x)/ cos(y), tan(y)])

#variation 43
def square(point, dependentParams = (NO_TRANSFORMATION, NO_TRANSLATION), externalparams = (0,0,0,0)):
    chi1 = chooseRandom(type="CHI")
    chi2 = chooseRandom(type="CHI")

    return np.array([chi1 - 0.5, chi2 - 0.5])








#Mapping for Configuration
def Mapping(typeName):
    if typeName.upper() == "LINEAR":
        return linear
    elif typeName.upper() == "SINUSODIAL":
        return sinusodial
    elif typeName.upper() == "SPHERICAL":
        return spherical
    elif typeName.upper() == "SWIRL":
        return swirl
    elif typeName.upper() == "HORSESHOE":
        return horseshoe
    elif typeName.upper() == "POLAR":
        return polar
    elif typeName.upper() == "HANDKERCHIEF":
        return handkerchief
    elif typeName.upper() == "HEART":
        return heart
    elif typeName.upper() == "DISC":
        return disc
    elif typeName.upper() == "SPIRAL":
        return spiral
    elif typeName.upper() == "HYPERBOLIC":
        return hypot
    elif typeName.upper() == "DIAMOND":
        return diamond
    elif typeName.upper() == "EX":
        return ex
    elif typeName.upper() == "JULIA":
        return julia
    elif typeName.upper() == "BENT":
        return bent
    elif typeName.upper() == "WAVES":
        return waves
    elif typeName.upper() == "FISHEYE":
        return fisheye
    elif typeName.upper() == "POPCORN":
        return popcorn
    elif typeName.upper() == "EXPONENTIAL":
        return exponential
    elif typeName.upper() == "POWER":
        return power
    elif typeName.upper() == "COSINE":
        return cosine
    elif typeName.upper() == "RINGS":
        return rings
    elif typeName.upper() == "FAN":
        return fan
    elif typeName.upper() == "BLOB":
        return blob
    elif typeName.upper() == "PDJ":
        return pdj
    elif typeName.upper() == "FAN2":
        return fan2
    elif typeName.upper() == "RINGS2":
        return rings2
    elif typeName.upper() == "EYEFISH":
        return eyefish
    elif typeName.upper() == "BUBBLE":
        return bubble
    elif typeName.upper() == "CYLINDER":
        return cylinder
    elif typeName.upper() == "PERSPECTIVE":
        return perspective
    elif typeName.upper() == "NOISE":
        return noise
    elif typeName.upper() == "JULIAN":
        return juliaN
    elif typeName.upper() == "JULIASCOPE":
        return juliaScope
    elif typeName.upper() == "BLUR":
        return blur
    elif typeName.upper() == "GAUSSIAN":
        return gaussian
    elif typeName.upper() == "RADIALBLUR":
        return radialblur
    elif typeName.upper() == "PIE":
        return pie
    elif typeName.upper() == "TANGENT":
        return tangent
    elif typeName.upper() == "SQUARE":
        return square

    else:
        return linear















