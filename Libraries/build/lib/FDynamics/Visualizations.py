from abc import ABC, abstractmethod
from FDynamics.Utils import GetReadableTime
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, LinearAxis, Range1d


class Animation(ABC):
    '''
    ABC class for all animation objects
    '''
    
    @abstractmethod
    def initFigure(self, Plotlimits = None , fig = None, axis = None):
        '''
        This is a generic method and shoudl be used by all animation logics while initializing the figure.
        Sets up the figure for animation.
        Returns fig and axis objects.

        If Plotlimits are not provided --> fig, axis are returned as None
        '''

        pass
        
    
    @abstractmethod
    def initPlotElements(self, axis = None):
        '''
        initializes the plot elements which are supposed to be animated
        '''
        pass
    
    @abstractmethod
    def initAnimation(self):
        '''
        Initialization to plot the background of each frame
        '''
        pass

    @abstractmethod
    def SequentialAnimate(self, frame):
        '''
        Animation function and is called sequentially
        '''
        pass

    @abstractmethod
    def RunAnimation(self, Plotlimits = None, fig = None, axis = None, interval = 10):
        pass




class Stills(ABC):
    """
    ABC for all still plots
    """
    
    @abstractmethod
    def Plot(self):
        '''
        plots the timeseries
        '''
        pass


class CurveAnimation(Animation):

    @abstractmethod
    def __init__(self, X_AxisDetails, CurveDetails, CurveLabel):
        '''
        Initialize the Animation object with below attributes:
        Inputs: 
            X_AxisDetails --> Dataframe object to be plotted on X-axis. 
            CurveDetails --> tuple of Curve at every instant 
            
        '''
        self.CurveDetails       = CurveDetails
        self.CurveLabel         = CurveLabel
        self.X_AxisDetails      = X_AxisDetails
        
        #Nb of IV curves to be plotted along side (Bid, Ask, Fit etc)
        self.NbCurves = len(CurveDetails)
        self.Timers = list(X_AxisDetails.index)


    def initFigure(self, Plotlimits = None , fig = None, axis = None):
        '''
        If plot limits are not provided, this method sets the limits, 
        and then calls the base class method of initailizing the figure
        '''

        #set plot limits
        if Plotlimits is None:
            X_lim, Y_lim    = None, None
        else:
            X_lim, Y_lim    = Plotlimits[0], Plotlimits[1]
        
        #set up the figure and axis and plot elements we need to animate
        if ((fig is None) & (axis is None) & (Y_lim is not None)):            
            fig     = plt.figure()
            if X_lim is not None:
                axis    = plt.axes(xlim = X_lim, ylim = Y_lim)
            else:
                axis    = plt.axes(ylim = Y_lim)
        else:
            fig     = None
            axis    = None  

        return fig, axis


    def initPlotElements(self, axis = None):
        '''
        initializes the plot elements which are supposed to be animated
        '''

        if axis is None:
            pass
        else:
            self.lines = []
            for i in np.arange(self.NbCurves):
                objline,    = axis.plot([], [], ls = "dashed",lw = 0.5, marker = "x", markersize = 5, markerfacecolor = "red", label = self.CurveLabel[i])
                self.lines.append(objline)
            
            #self.TimeText      = axis.text(0.02, 0.5, "", transform = axis.transAxes)
            self.TimeText      = axis.text(0.45, 0.9, "", transform = axis.transAxes)
            



    def initAnimation(self):
        '''
        Initialization to plot the background of each frame
        '''

        for line in self.lines:
            line.set_data([],[])
        
        return tuple(self.lines) 
        

    def SequentialAnimate(self, frame, timer_in_epoch = False, time_dateobject = False):
        '''
        Animation function and is called sequentially
        '''
        timer = self.Timers[frame]
        if timer_in_epoch:
            self.TimeText.set_text("{0}".format(GetReadableTime(timer)))
        else:
            if time_dateobject:
                self.TimeText.set_text("{0}".format(timer.strftime("%Y, %b %d")))
            else:
                self.TimeText.set_text("{0}".format(timer))

        x = self.X_AxisDetails.loc[timer]      #Strikes to be plotted on X axis

        for Nbline, line in enumerate(self.lines):
            y = self.CurveDetails[Nbline].loc[timer]
            line.set_data(x, y)
        
        return tuple(self.lines) +  (self.TimeText,)


    def RunAnimation(self, Plotlimits = None, fig = None, axis = None, interval = 10, timer_in_epoch = True, time_dateobject = False):

        pass


class BokehVisuals():
    """
    create plots with interactive charts (to be saved on html pages)
    """

    def __init__(self, data):
        self.data = data            #a dataframe object

    
    def createPlot(self, outputpath, imagetitle, XAxisDetails, PrimaryAxisDetails, SecondaryAxis = True, SecondaryAxisDetails = None):
        """
        creates interactive plot and saves them
        Input:
        PrimaryAxisDetails/ SecondaryAxisDetails/ XAxisDetails: dictionary item with keys as below
            a. data column name (COL_NAME)
            b. data name to be present on chart (LABEL)
            c. item to be present while hovering (HOVER)

        """

        y_axis_max = [self.data[PrimaryAxisDetails["COL_NAME"]].max() * 1.1]        #setting upper limit as 110% of max available value
        y_axis_min = [self.data[PrimaryAxisDetails["COL_NAME"]].min() * 1.1]        #setting upper limit as 110% of max available value
        #y_axis_min = [0]
        if SecondaryAxis:
            y_axis_max.append(self.data[SecondaryAxisDetails["COL_NAME"]].max() * 1.1)
            y_axis_min.append(0)

        hoverfields = [(PrimaryAxisDetails["LABEL"], PrimaryAxisDetails["HOVER"])]
        if SecondaryAxis:
            hoverfields.append((SecondaryAxisDetails["LABEL"], SecondaryAxisDetails["HOVER"]))

        hover = HoverTool(tooltips= hoverfields)

        #creating bokeh figure object

        p = figure(title = imagetitle, plot_height=400, plot_width=800, tools=[hover, "pan,reset,wheel_zoom"])
        p.title.align = 'center'

        p.xaxis.axis_label = XAxisDetails["LABEL"]
        p.yaxis.axis_label = PrimaryAxisDetails["LABEL"]

        p.y_range = Range1d(start=y_axis_min[0], end=y_axis_max[0])
        p.line(XAxisDetails["COL_NAME"], PrimaryAxisDetails["COL_NAME"], source=ColumnDataSource(self.data), color = "black")


        #creating secondary axis
        if SecondaryAxis:

            p.extra_y_ranges = {SecondaryAxisDetails["LABEL"]: Range1d(start=y_axis_min[1], end=y_axis_max[1])}

            # Adding the second axis to the plot.  
            p.add_layout(LinearAxis(y_range_name=SecondaryAxisDetails["LABEL"], axis_label=SecondaryAxisDetails["LABEL"]), 'right')

            # Using the aditional y range named "foo" and "right" y axis here. 
            p.line(XAxisDetails["COL_NAME"], SecondaryAxisDetails["COL_NAME"], source=ColumnDataSource(self.data),  y_range_name=SecondaryAxisDetails["LABEL"], color = "green")



        output_file(outputpath)

        return p

        

