
import random
import numpy as np
from bisect import bisect_left
import matplotlib.pyplot as plt
from Utils import getStringtoList_Float, getStringtoBool, Config, convertListItemstoFloat
from IteratedFunctionSystems.Transformations.Transforms import LinearTransform, GeneralizedTransform
from IteratedFunctionSystems.Transformations import Variation
from IteratedFunctionSystems.Transformations.AllVariations import Mapping
from PIL import Image

ERR_THRESHOLD = 10e-16

class ClassicIteratedFunctions:
    
    def __init__(self, name):
        """
        The object IFS can have multiple Linear Transformations with respective weights.
        X_n+1 = A* X_n + B. 
        where A--> Linear transformation
        B --> Linear translation
        
        """
        self.Transforms = []
        self.Totalweight = 0
        self.CumProbWeight = [0]
        self.Name = name

        self._Xaxis =[]
        self._Yaxis =[]

    
    def add_transform(self, Transform):
        # a transform object is added
        
        if (Transform.Probability > 0):
            if (Transform.Probability + self.Totalweight - 1 ) <= ERR_THRESHOLD:
                
                self.Transforms.append(Transform)
                self.Totalweight += Transform.Probability
                self.CumProbWeight.append(self.Totalweight)
                
            else:
                print("Cumulative Prob exceeding 1. Unable to add this Transformation")
                
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


    def applyConfiguration(self, configPath):
        """
        config --> a configfile path
        parses through the config file and adds the transform to the IFS  
        """

        config = Config(configPath =configPath)
        for transform in config.parser.sections:

            #get key value in this transform
            _transformType  = config.getSectionValue(transform, "TYPE").upper()
            _probability    = float(config.getSectionValue(transform, "PROB"))
            _allRandom      = getStringtoBool(config.getSectionValue(transform, "ALL_RANDOM").upper())
            _LinearMatrix   = convertListItemstoFloat(config.getSectionValue(transform, "LINEAR_TRANSFORM"))
            _translationMatrix = convertListItemstoFloat(config.getSectionValue(transform, "TRANSLATION"))

            if _allRandom is False:
                if _transformType == "LINEAR":

                    #this is a linear transform
                    _mytransform = LinearTransform(Probability = _probability, AllRandom=_allRandom, 
                                                    LinearMatrix=_LinearMatrix, TranslationMatrix=_translationMatrix)

            else:
                _mytransform = LinearTransform(Probability = _probability, AllRandom=_allRandom)


            #adding the transform to my IFS
            self.add_transform(_mytransform)


    def draw(self, InitialPoint = np.array([0,0]), Iterations = 1000000):
        """
        returns: the final fractal image with x,y  coordinates
        """
    
        point       = InitialPoint
    
        #r, g, b = 0, 0, 0
        #allcolors = [[r, g, b]]

        self._Xaxis =[]
        self._Yaxis = []

        
        for _ in range(Iterations):
            rand = random.uniform(0, 1)
            
            index = self.get_index(rand)
            t = self.choose_transform(index)
            #t = self.choose_transform()
            
            #apply transform
            point = t.applyTransform(point)

            self._Xaxis.append(point[0])
            self._Yaxis.append(point[1])

        
        #    r, g, b = t.transform_colour(r, g, b)
        #    allcolors.append([r, g, b])
        

        #plotting
        plt.figure()
        plt.scatter(np.array(self._Xaxis), np.array(self._Yaxis), c = "black", s = 0.15, alpha = 0.4, label = self.Name)
        plt.legend()

        
        
    

class GeneralizedIFS():
    """
    In the generalized Iterated Function system, there need not be a Linear transformation. 
    This is a generalised transformation
    The object IFS can have multiple Transformations  with respective weights.
    Eg: X_n+1 = V(A* X_n + B)
    here, V defines the Non linear transformation applied 
    
    """

    def __init__(self, name):
        self.Transforms     = []
        self.Totalweight    = 0
        self.CumProbWeight  = [0]
        self.Name           = name 
        self.randGen        = random
        self.rand           = lambda hi,lo: lo + ((hi - lo) * self.randGen.random())

    def add_transform(self, Transform):
        #Transform --> a GeneralizedTransform object
        
        if (Transform.Probability > 0):
            if (Transform.Probability + self.Totalweight - 1 ) <= ERR_THRESHOLD:
                
                self.Transforms.append(Transform)
                self.Totalweight += Transform.Probability
                self.CumProbWeight.append(self.Totalweight)
                
            else:
                print("Cumulative Prob exceeding 1. Unable to add this Transformation")


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



    def draw(self, InitialPoint = np.array([0,0]), Iterations = 1000000):
        """
        returns: the final fractal image with x,y  coordinates
        """
    
        self._Xaxis =[]
        self._Yaxis = []

        #r, g, b = 0, 0, 0
        #allcolors = [[r, g, b]]

        for _ in range(Iterations):
            rand = random.uniform(0, 1)
            
            index = self.get_index(rand)
            t = self.choose_transform(index)
            
            cur_x = self.rand(-10, 10)
            cur_y = self.rand(-10, 10)
            point       = [cur_x, cur_y]

            #apply transform
            point = t.applyTransform(point)

            self._Xaxis.append(point[0])
            self._Yaxis.append(point[1])
        
        #    r, g, b = t.transform_colour(r, g, b)
        #    allcolors.append([r, g, b])

        #plotting
        plt.figure()
        plt.scatter(np.array(self._Xaxis), np.array(self._Yaxis), c = "black", s = 0.15, alpha = 0.4, label = self.Name)
        plt.legend()




class FractalFlame():
    """
    In the generalized Iterated Function system, there need not be a Linear transformation. 
    This is a generalised transformation
    The object IFS can have multiple Transformations  with respective weights.
    Eg: X_n+1 = V(A* X_n + B)
    here, V defines the Non linear transformation applied 
    
    """

    def __init__(self, name, xres = 1920, yres = 1080, sup = 1,
                xmin = -1, xmax = 1, ymin = -1, ymax = 1,
                NbSamples = 1000, NbIters = 100):

        self.Transforms     = []
        self.Totalweight    = 0
        self.CumProbWeight  = [0]
        self.Name           = name 
        self.randGen        = random
        self.rand           = lambda hi,lo: lo + ((hi - lo) * self.randGen.random())

        #setting limits for image
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin

        #image resolution
        self.xres   = xres
        self.yres   = yres
        self.sup    = sup
        self.pixels = np.zeros((xres*sup, yres*sup,4),dtype=np.uint8)      #contains image histogram as well 
        self.image = np.zeros((xres*sup, yres*sup,3),dtype=np.uint8)       

        self.FinalTransform = None

        self.NbSamples  = NbSamples
        self.NbIters    = NbIters

        self.colors     = np.zeros((256,3))



    def applyConfiguration(self, configPath):
        """
        config --> a configfile path
        parses through the config file and adds the transform to the IFS  
        """

        config = Config(configPath =configPath)
        for transform in config.parser.sections:

            _probability    = float(config.getSectionValue(transform, "PROB"))
            _color          = convertListItemstoFloat(config.getSectionValue(transform, "COLOR"))

            _LinearMatrix   = convertListItemstoFloat(config.getSectionValue(transform, "LINEAR_TRANSFORM"))
            _translationMatrix = convertListItemstoFloat(config.getSectionValue(transform, "TRANSLATION"))

            #Check for Post Transform
            _PostLinear     = convertListItemstoFloat(config.getSectionValue(transform, "POST_LINEAR"))
            _Posttranslation = convertListItemstoFloat(config.getSectionValue(transform, "POST_TRANSLATION"))

            #get all variations
            _variations = []
            for variation in config.parser[transform]["VARIATIONS"].keys():

                _type =  Mapping(config.parser[transform]["VARIATIONS"][variation]["TYPE"])

                _weight = 1
                if "WEIGHT" in config.parser[transform]["VARIATIONS"][variation].keys():
                    _weight =  float(config.parser[transform]["VARIATIONS"][variation]["WEIGHT"])

                _externalParams = [0,0,0,0]
                if "EXTERNAL_PARAMS" in config.parser[transform]["VARIATIONS"][variation].keys():
                    _externalParams =  convertListItemstoFloat(config.parser[transform]["VARIATIONS"][variation]["EXTERNAL_PARAMS"])

                
                _thisVar = Variation(type = _type, weight=_weight, externalparams=_externalParams)

                _variations.append(_thisVar)

            
            # Creating a Generalised Transform object
            _mytransform = GeneralizedTransform(Probability=_probability, Color=_color, Random=False, 
                                                LinearTransform=_LinearMatrix, LinearTranslation=_translationMatrix,
                                                Post_Linear=_PostLinear, Post_Translation=_Posttranslation, 
                                                Variations=_variations)



            #adding the transform to my IFS

            if transform.upper() != "FINALTRANSFORM":
                self.add_transform(_mytransform)

            else:
                self.FinalTransform = _mytransform



    def add_transform(self, Transform):
        #Transform --> a GeneralizedTransform object
        
        if (Transform.Probability > 0):
            self.Transforms.append(Transform)
                


    def choose_transform(self, index = None):
        """
        out of available transforms, choose a random transform to be applied. 
        """

        prob    = [_t.Probability for _t in self.Transforms]
        index   = self.randGen.choices(range(len(self.Transforms)), weights=prob)[0]

        return self.Transforms[index]

        

    def draw(self):
        """
        returns: the final fractal image with x,y  coordinates
        """
    

        #r, g, b = 0, 0, 0
        #allcolors = [[r, g, b]]

        for _ in range(self.NbSamples):

            #for each sample, choose a random point from a bi-unit square
            cur_x = self.rand(self.xmin, self.xmax)
            cur_y = self.rand(self.ymin, self.ymax)
            point = [cur_x, cur_y]
            color = [self.randGen.randint(0,255), self.randGen.randint(0,255), self.randGen.randint(0,255)]

            for _iter in range(self.NbIters):
                _tform  = self.choose_transform()

                #apply transform 
                point   = _tform.applyTransform(point)
                color   = list(map(int, np.add(color, _tform.Color)/2))         #getting the integer part for rgb color format

                point   = self.FinalTransform.applyTransform(point)
                color   = list(map(int, np.add(color, self.FinalTransform.Color)/2))

                cur_x, cur_y = point[0], point[1]
                
                


                if _iter >= 20:
                    #drawing the image only after 20 steps

                    #procure the pixel to be drawn
                    if self.xmin < cur_x < self.xmax and self.ymin < cur_y < self.ymax:
                        x = int( (cur_x - self.xmin)/(self.xmax - self.xmin) * self.xres * self.sup )
                        y = int( (cur_y - self.ymin)/(self.ymax - self.ymin) * self.yres * self.sup )   

                        if 0 < x < self.xres * self.sup and 0 < y < self.yres * self.sup:
                            _pixel = self.pixels[x][y]

                            _pixel[0] = color[0]
                            _pixel[1] = color[1]
                            _pixel[2] = color[2]

                            _pixel[3] += 1 # Hitcounter

                            self.image[x][y][0] = _pixel[0]
                            self.image[x][y][1] = _pixel[1]
                            self.image[x][y][2] = _pixel[2]

                            #Blend colors
                            """
                            r, g, b = self.colors[function.color]
                            point[0] = int(r / 2.0 + point[0] / 2.0)
                            point[1] = int(g / 2.0 + point[1] / 2.0)
                            point[2] = int(b / 2.0 + point[2] / 2.0)

                            point[3] += 1 # Hitcounter
                            """
                            
        #    r, g, b = t.transform_colour(r, g, b)
        #    allcolors.append([r, g, b])


        Image.fromarray(self.image).show()

        
        #plotting
        """
        plt.figure()
        plt.scatter(np.array(self._Xaxis), np.array(self._Yaxis), c = "black", s = 0.15, alpha = 0.4, label = self.Name)
        plt.legend()
        """


    def save(self, path):
        Image.fromarray(self.image).save(path)

        
        