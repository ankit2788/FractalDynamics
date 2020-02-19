from IteratedFunctionSystems.Transformations import Transform, Variation
from IteratedFunctionSystems.Transformations import AllVariations as Flames
import random
import numpy as np
from numpy.linalg import det


class LinearTransform(Transform):
    def __init__(self, Probability, AllRandom = True, LinearMatrix = None, TranslationMatrix = None):
        
        super().__init__()
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
    




class GeneralizedTransform:
    def __init__(self, ProbImplementation = None, Probability = 0.0, Color = 1,  Random = False, 
                LinearTransform = [1,0,0,1], LinearTranslation = [0,0], 
                Post_Linear = [1,0,0,1], Post_Translation = [0,0],
                Variations = []):
        """
        Parameters list consist of the below:
        ProbImplementation: Probability can be either based on the contraction factor determined by IFS params
                            Applicable values: [CONTRACTION, NONE]
        Probability:        If ProbImplementation is None, input > 0 is required
        Random:             Choice of IFS params turns to be random completely. 
        Params:
            Linear Transform and Linear Translation
            Post Linear Transform and Post Linear Translation
        """

        if Random is True:
            a = random.uniform(-1, 1)
            b = random.uniform(-1, 1)
            c = random.uniform(-1, 1)
            d = random.uniform(-1, 1)
            e = random.uniform(-1, 1)
            f = random.uniform(-1, 1)
            
            self.LinearMatrix   = np.array([[a, b], [c, d]])
            self.Translation    = np.array([e, f])
            
        else:
            self.LinearMatrix   = np.array([[LinearTransform[0], LinearTransform[1]],
                                            [LinearTransform[2], LinearTransform[3]]])
            self.Translation    = np.array(LinearTranslation)

        if ProbImplementation is None:
            self.Probability    = Probability
        else:
            #get the contraction factor
            self.Probability    = self.__getContractionFactor()

        self.Color          = Color
        self.Variations     = Variations




        #setting the post transform parameters
        self.Post_LinearTransform   = np.array([[Post_Linear[0], Post_Linear[1]],
                                                [Post_Linear[2], Post_Linear[3]]])
        self.Post_Translation       = np.array(Post_Translation)
        

    def getIFSCoeff(self):
        return self.LinearMatrix, self.Translation

    def setIFSCoeff(self, LineatMatrix, Translation):
        self.LinearMatrix = LineatMatrix
        self.Translation = Translation



    def __getContractionFactor(self):
        return abs(det(self.LinearMatrix))


    def addVariation(self, variation = Variation()):
        self.Variations.append(variation)


    def applyTransform(self, point):

        # apply all variations
        point = self.__applyVariations(point)
        return point


    def __applyVariations(self, point):

        totalweight = 0
        for variation in self.Variations:
            weight          = variation.weight
            totalweight     += weight
        
        for variation in self.Variations:
            #applying the transformation based on the weight of all variations
            weight          = variation.weight
            func            = variation.type            
            externalparams  = variation.params
            
            #apply Linear Transformation and Translation first
            point           = Flames.linear(point, dependentParams=(self.LinearMatrix, self.Translation))

            #apply the variation now
            point           = func(point, dependentParams = (self.LinearMatrix, self.Translation), externalparams = externalparams)

            #apply the weights now
            point           = (weight/totalweight) * point     

        #apply the post transformation  
        point = Flames.linear(point, dependentParams=(self.Post_LinearTransform,self.Post_Translation))

        return point
