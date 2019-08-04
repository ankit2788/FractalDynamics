from Geometry import *

COLORS = ["red", "blue", "green", "cyan", "magenta", "yello"]

class IterativeFunction:
    def __init__(self, FractalType, Initiator, OutputType = "PLOT"):
        """
        Fractal Type --> string (example: Koch, Sierpinski)
        Initiator --> a geometry object (Line, triangle)
        OutputType --> (DATA, PLOT)
        """
        
        self.fractal = FractalType
        self.Initiator = Initiator
        if OutputType == "PLOT":
            self.fig, self.ax = plt.subplots(1,1)
#        elif OutputType == "DATA":
#            self.ax = None
#            self.data = [[], []]    #list of x and y coordinates      
        
    
    def Draw(self, order = 3):
        """
        Starts with Initiator and recursively draws using the generator
        """
        
        if (self.fractal.upper() == "KOCH"):
            Koch(self.Initiator, order, self.ax)
            
        elif (self.fractal.upper() == "SIERPINSKI"):
            Sierpinski(self.Initiator, order, self.ax)
        
        



def Koch(line, order, ax):
    """
    Creates Koch curve of nth order
    Functions is used recursively
    Inputs:
        
    1. Initiator --> line object
    2. order --> order of the Koch curve
    """
        
    if (order == 1):
        #print(line)    #printing the line

        ax = line.Draw(ax)
        return line
    else:
        
        #divide line in 3 equal portions
        #line = self.MyStack.get()
        point1 = line.pointA
        point2 = line.getPortion(1/3)
        point3 = line.getPortion(2/3)
        point4 = line.pointB
        
        #create triangle from middle portion
        point5 = DrawTriangle(Line(point2, point3), triangletype="EQ")
        #point5 = DrawTriangle(Line(point2, point3), triangletype="Other", theta1=70, theta2=60)
        
        #repeat for 4 segments
        Koch(Line(point1, point2), order-1, ax)
        Koch(Line(point2, point5), order-1, ax)
        Koch(Line(point5, point3), order-1, ax)
        Koch(Line(point3, point4), order-1, ax)



def Sierpinski(triangle, order, ax):
    
     
    if (order == 1):
            
        ax = triangle.Draw(ax)

        return triangle
    else:
        
        #divide line in 3 equal portions
        #line = self.MyStack.get()
        mids = triangle.getMidpoints()
        triangle1 = Triangle(triangle.A, mids[0], mids[2])
        triangle2 = Triangle(mids[0], triangle.B, mids[1])
        triangle3 = Triangle(mids[2], mids[1], triangle.C)

        #repeat for 3 triangles
        Sierpinski(triangle1, order-1, ax)
        Sierpinski(triangle2, order-1, ax)
        Sierpinski(triangle3, order-1, ax)
        
        