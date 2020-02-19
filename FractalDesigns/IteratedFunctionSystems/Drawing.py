import random
import numpy as np

def FractalImage(IFS, InitialPoint = np.array([0,0]), Iterations = 1000000):

    """
    IFS --> an Iterated Function system object containing the functions to be applied
    returns: the final fractal image with x,y  coordinates
    """
    
    point = InitialPoint
    allpoints = [point]

    #r, g, b = 0, 0, 0
    #allcolors = [[r, g, b]]
    
    for _ in range(Iterations):
        rand = random.uniform(0, 1)
        
        index = IFS.get_index(rand)
        #t = IFS.choose_transform(index)
        t = IFS.choose_transform()
        
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
    