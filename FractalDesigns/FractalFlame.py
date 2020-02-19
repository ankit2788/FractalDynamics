from IteratedFunctionSystems.IFS import GeneralizedIFS, FractalFlame
from IteratedFunctionSystems.Transformations import  Variation
from IteratedFunctionSystems.Transformations.Transforms import GeneralizedTransform
import IteratedFunctionSystems.Transformations.AllVariations as Flames
from IteratedFunctionSystems.Drawing import FractalImage
import matplotlib.pyplot as plt
import numpy as np
import random




def generateNonLinear(X, Y, functions = ["linear","linear"]):
    
    xnew = []
    ynew = []
    for _x, _y in zip(np.ravel(X),np.ravel(Y)):
    
        point = (_x, _y)
        #choose a random function
        fn = random.choices(functions, weights=np.ones(len(functions)))[0]
        _x, _y = Flames.Mapping(fn)(point)
        xnew.append(_x)
        ynew.append(_y)
    
    
    fig,ax = plt.subplots(1,2)
    #fig.suptitle("Non Linear Transformations: {0}".format(function.upper()))

    ax[0].scatter(np.ravel(X), np.ravel(Y), s=0.5, alpha = 0.5, color = "black")
    ax[0].set_title("Original 2D")
    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[0].get_yticklabels(), visible=False)
    ax[0].tick_params(axis='both', which='both', length=0)

    ax[1].scatter(xnew, ynew, s=0.5, alpha = 0.5, color = "green")
    ax[1].set_title("After Non Linear Transformation")
    plt.setp(ax[1].get_xticklabels(), visible=False)
    plt.setp(ax[1].get_yticklabels(), visible=False)
    ax[1].tick_params(axis='both', which='both', length=0)


    


x = np.arange(-20,20,.5)
y = np.arange(-20,20,.5)
X,Y = np.meshgrid(x,y)


generateNonLinear(X,Y,functions=["disc"])
generateNonLinear(X,Y,functions=["handkerchief", "disc", "polar"])





Flame = FractalFlame(name="Test")
Flame.applyConfiguration(configPath="/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/Flames/Test1.ini")
Flame.draw()


Flame = FractalFlame(name="Random", NbSamples=100, NbIters=1000, xres=550, yres=310)
Flame.applyConfiguration(configPath="/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/Flames/Test3.ini")
Flame.draw()
Flame.save(path="/Users/ankitgupta/Documents/git/anks/github_page/data/pics/2020/01/Flame1_small.png")

Flame = FractalFlame(name="Random", NbSamples=100, NbIters=1000, xres=550, yres=310)
Flame.applyConfiguration(configPath="/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/Flames/Test2.ini")
Flame.draw()
Flame.save(path="/Users/ankitgupta/Documents/git/anks/github_page/data/pics/2020/01/Flame2_small.png")


Flame = FractalFlame(name="Random", NbSamples=100, NbIters=1000)
Flame.applyConfiguration(configPath="/Users/ankitgupta/Documents/git/anks/FractalDynamics/FractalDesigns/Configs/Flames/Test4.ini")
Flame.draw()
