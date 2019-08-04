#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:49:53 2019

@author: ankitgupta
"""


import pandas as pd
import numpy as np
import os
from functools import reduce
from datetime import datetime
from FDynamics.Visualizations import CurveAnimation
from matplotlib import animation
import matplotlib.pyplot as plt


class BondYieldCurve(CurveAnimation):
    
    def __init__(self, Maturity, YieldDetails, YieldLabels):
        '''
        Initialize the Animation object with below attributes:
        Inputs: 
            YieldLabels 
            YieldDetails --> tuple of Yield Curves at every instant 
            YieldLabels --> name of the YieldCurves to be plotted
            
        '''

        super().__init__(Maturity, YieldDetails, YieldLabels)


    def initFigure(self, Plotlimits = None , fig = None, axis = None):
        '''
        If plot limits are not provided, this method sets the limits, 
        and then calls the base class method of initailizing the figure
        '''

        #set plot limits
        if Plotlimits is None:
            Plotlimits      = self._getLimits()

        #calling base class method
        fig, axis = super().initFigure(Plotlimits, fig, axis)
        axis.set_xlabel("Maturity (in years)")
        axis.set_ylabel("Bond Yield (%)")
        return fig, axis


    def initPlotElements(self, axis = None):
    
        pass


    def initAnimation(self):
        pass

    def SequentialAnimate(self, frame, timer_in_epoch):
        pass

    def RunAnimation(self, Plotlimits = None, fig = None, axis = None, interval = 10):

        timer_in_epoch = False
        time_dateobject = True

        #Create the figure for animation
        figure, axes = self.initFigure(Plotlimits, fig, axis)
        

        #Initialize the Plot element which is supposed to be animated
        super().initPlotElements(axes)
        axes.legend()

        anim = animation.FuncAnimation(figure, 
                                    super().SequentialAnimate, init_func=super().initAnimation, 
                                    frames=len(self.Timers), fargs = (timer_in_epoch, time_dateobject,), interval=interval, 
                                    blit=True, repeat = False)



        return anim

        
    def _getLimits(self):
        '''
        Function to set the limits required for plotting purposes
        '''
        
        
        #get the plot limits        
        temp_Mat = np.ravel(self.X_AxisDetails)
        temp_Yield = np.ravel(self.CurveDetails[0])

        LowYield = np.nanmin(temp_Yield)
        HighYield = np.nanmax(temp_Yield)

        MatLimit            = (0, np.nanmax(temp_Mat)+5)
        YieldLimit           = (LowYield - 1, HighYield + 1)

        return MatLimit, YieldLimit


path = "/Users/ankitgupta/Data/IndiaBondData/"
file = "IndiaBondYields.csv"


def convert(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

#-----------Arrange Data----------


files = []

for filename in os.listdir(path):
    
    if "Bond Yield" in filename:
        #maturity = filename.split("-")[0][6:] + "Y"
        maturity = filename.split("-")[0][6:]
        month_year = filename.split("-")[1][:5]
        if month_year.upper() == "MONTH":
            maturity = str(float(maturity)/12)
        
        yearyield = pd.read_csv(path + filename)
        yearyield[maturity] = yearyield["Price"]
        yearyield["DATE"] = yearyield["Date"].apply(lambda x:datetime.strptime(x, '%b %d, %Y').date() )
        yearyield = yearyield[["DATE", maturity]]
        
        yearyield = pd.DataFrame(yearyield)
    
    
        files.append(yearyield)




df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['DATE'],
                                            how='outer'), files).fillna(np.nan)
df_merged.set_index("DATE", drop=True, inplace=True)
df_merged.sort_index(axis=0, inplace = True)

#sort columns
temp_col = []
cols = list(df_merged.columns)
temp_col = [convert(year) for year in cols]
temp_col.sort()

new_cls = [str(year) for year in temp_col]
#new_cls = temp_col

IndiaBondYields = df_merged[new_cls]
IndiaBondYields.to_csv(path + file)

maturity = [temp_col] * len(list(IndiaBondYields.index))
Maturity = pd.DataFrame(maturity, index = IndiaBondYields.index)



#animation
a = BondYieldCurve(Maturity, (IndiaBondYields, ), ("India Yield Curve",))
anim = a.RunAnimation(interval=70)
anim.save(path + 'YieldCurveDance.gif', dpi=80, writer='imagemagick')


#2002 - 2004
columns = ['0.25',
 '0.5',
 '1',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '10',
 '15']

mat_columns = [0,1,2,4,5,6,7,8,9,11,16]

b = IndiaBondYields[206:365][columns]
m = Maturity[206:365][mat_columns]

a = BondYieldCurve(m, (b, ), ("India Yield Curve",))
anim = a.RunAnimation(interval=70)

dates =[datetime(2002, 1, 20), datetime(2002, 7, 28), datetime(2002, 12, 29),
 datetime(2003, 6, 29), datetime(2003, 12, 28), datetime(2004, 6, 27),
 datetime(2004, 12, 26)]



import matplotlib.pyplot as plt
#fig, ax = plt.subplots()

plt.figure()
for item in dates:
    date = item.date()
    plt.plot(list(m.loc[date]), list(b.loc[date]),  ls = "dashed",lw = 0.5, marker = "x", markersize = 5, markerfacecolor = "red", label = item.strftime("%b %Y"))

plt.annotate('Rates declining throughout!', xy=(2, 4.2), xytext=(4, 8),
            arrowprops=dict(facecolor='black', shrink=0.03, width=1),
            horizontalalignment='left', verticalalignment='top',
            )


plt.xlabel("Maturity (in years)")
plt.ylabel("Yield (%)")
plt.title("Bond Curve. Year 2002 - 2004")
plt.legend()
plt.show()



#2004 - mid 2008
columns = ['0.25',
 '0.5',
 '1',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '10',
 '15']

mat_columns = [0,1,2,4,5,6,7,8,9,11,16]

b = IndiaBondYields[366:570][columns]
m = Maturity[366:570][mat_columns]

a = BondYieldCurve(m, (b, ), ("India Yield Curve",))
anim = a.RunAnimation(interval=70)

dates =[datetime(2005, 1, 30), datetime(2005, 7, 10), datetime(2005, 12, 25),
 datetime(2006, 6, 25), datetime(2006, 12, 24), datetime(2007, 6, 24),
 datetime(2007, 12, 30),datetime(2008, 6, 29)]



import matplotlib.pyplot as plt
#fig, ax = plt.subplots()

plt.figure()
for item in dates:
    date = item.date()
    plt.plot(list(m.loc[date]), list(b.loc[date]),  ls = "dashed",lw = 0.5, marker = "x", markersize = 5, markerfacecolor = "red", label = item.strftime("%b %Y"))

plt.annotate('Rates shooting up!', xy=(0.5, 9.2), xytext=(4, 5.5),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2"),
            )



plt.xlabel("Maturity (in years)")
plt.ylabel("Yield (%)")
plt.title("Bond Curve. Year 2005 - mid 2008")
plt.legend()
plt.show()


#Financial Crisis - 2010

columns = ['0.25',
 '0.5',
 '1',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '10',
 '15', '19']

mat_columns = [0,1,2,4,5,6,7,8,9,11,16,17]

b = IndiaBondYields[530:675][columns]
m = Maturity[530:675][mat_columns]

a = BondYieldCurve(m, (b, ), ("India Yield Curve",))
anim = a.RunAnimation(interval=70)



#---------
plt.figure()
plt.plot(b.index, b["0.5"], label = "6 months", lw = 0.8, ls = "-")
plt.plot(b.index, b["1"], label = "1 year", lw = 0.8, ls = "-")
plt.plot(b.index, b["5"], label = "5 years", lw = 0.8, ls = "-")
plt.plot(b.index, b["10"], label = "10 years", lw = 0.8, ls = "-")
plt.xlabel("Date")
plt.ylabel("Yield(in %)") 

date = datetime(2008,7,7)
plt.annotate('1. Short and long rates merging,\n implying a flat yield curve!', xy=(date.date(), 8.7) ,
            xytext=(date.date(), 5),arrowprops=dict(facecolor='black', shrink=0.03, width=0.5),
            horizontalalignment='center', verticalalignment='top',)

date1 = datetime(2008,12,28)
date2 = datetime(2008,9,14)

plt.annotate('2. Crisis looming up! \nMoney becomes cheap', xy=(date1.date(), 6), xytext=(date2.date(), 9.2),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2"),
            )

date1 = datetime(2009,7,26)
date2 = datetime(2010,11,28)

plt.annotate("",
            xy=(date2.date(), 7.6), xycoords='data',
            xytext=(date1.date(), 5.5), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
date3 = datetime(2009,8,30)
plt.text(date3.date(), 6.6, "3. Recovery in the sighting. Yield Curve returing to normal.", 
            ha="left", va="center", rotation=22,
            size=10,)

plt.title("Financial Crisis and Recovery")
plt.legend()




#2011 - 2013 --> case of Curve inversion in due time

columns = ['0.25',
 '0.5',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '10',
 '15', '19','24']

mat_columns = [0,1,2,3,4,5,6,7,8,9,11,16,17,18]

b = IndiaBondYields[650:830][columns]
m = Maturity[650:830][mat_columns]

a = BondYieldCurve(m, (b, ), ("India Yield Curve",))
anim = a.RunAnimation(interval=70)


plt.figure()
plt.plot(b.index, b["0.5"], label = "6 months", lw = 0.8, ls = "--")
plt.plot(b.index, b["1"], label = "1 year", lw = 0.8, ls = "--")
plt.plot(b.index, b["5"], label = "5 years", lw = 0.8, ls = "--")
plt.plot(b.index, b["10"], label = "10 years", lw = 0.8, ls = "--")
plt.plot(b.index, b["15"], label = "15 years", lw = 0.8, ls = "--")
plt.xlabel("Date")
plt.ylabel("Yield(in %)") 
plt.legend()

date1 = datetime(2010,9,26)
date2 = datetime(2011,10,23)

plt.annotate("",
            xy=(date2.date(), 8.2), xycoords='data',
            xytext=(date1.date(), 5.6), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
date3 = datetime(2011,3,6)
plt.text(date3.date(), 6.8, "1. Flattening of the curve", 
            ha="left", va="center", rotation=42,
            size=10,)


date1 = datetime(2013,7,21)
date2 = datetime(2013,2,24)

plt.annotate('2. Now, Curve Inversion!', xy=(date1.date(), 9.6), xytext=(date2.date(), 10.4),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2"),
            )



plt.title("Curious Case of Curve Inversion")

dates =[datetime(2010,9,26), datetime(2011,10,23), datetime(2013,2,24),
 datetime(2013,7,21)]
plt.figure()
for item in dates:
    date = item.date()
    plt.plot(list(m.loc[date]), list(b.loc[date]),  ls = "dashed",lw = 0.5, marker = "x", markersize = 5, markerfacecolor = "red", label = item.strftime("%b %Y"))

plt.legend()
plt.xlabel("Maturity (in years)")
plt.ylabel("Yield (%)")

plt.title("Curious Case of Curve Inversion")


#2014 - 2016 --> Returning back to normal (Election time)
columns = ['0.25',
 '0.5',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '10',
 '15', '19','24']

mat_columns = [0,1,2,3,4,5,6,7,8,9,11,16,17,18]

b = IndiaBondYields[830:990][columns]
m = Maturity[830:990][mat_columns]

a = BondYieldCurve(m, (b, ), ("India Yield Curve",))
anim = a.RunAnimation(interval=70)


plt.figure()
plt.plot(b.index, b["0.5"], label = "6 months", lw = 0.8, ls = "--")
plt.plot(b.index, b["1"], label = "1 year", lw = 0.8, ls = "--")
plt.plot(b.index, b["5"], label = "5 years", lw = 0.8, ls = "--")
plt.plot(b.index, b["10"], label = "10 years", lw = 0.8, ls = "--")
plt.plot(b.index, b["15"], label = "15 years", lw = 0.8, ls = "--")
plt.xlabel("Date")
plt.ylabel("Yield(in %)") 
plt.legend()

date1 = datetime(2014,3,9)
date2 = datetime(2016,7,3)

plt.annotate("",
            xy=(date2.date(), 6.1), xycoords='data',
            xytext=(date1.date(), 8), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
date3 = datetime(2014,8,17)
plt.text(date3.date(), 7.25, "1. A Downward shift in the curve", 
            ha="left", va="center", rotation=-23,
            size=10,)

date3 = datetime(2015,8,16)
plt.text(date3.date(), 6.7, "2. Curve returning to normal", 
            ha="left", va="center", rotation=-23,
            size=10,)


plt.title("Returning to normalcy!")




#2017 - Till now--> Returning back to normal (Election time)
columns = ['0.25',
 '0.5',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '10',
 '15', '19','24']

mat_columns = [0,1,2,3,4,5,6,7,8,9,11,16,17,18]

b = IndiaBondYields[990:][columns]
m = Maturity[990:][mat_columns]

a = BondYieldCurve(m, (b, ), ("India Yield Curve",))
anim = a.RunAnimation(interval=70)


plt.figure()
plt.plot(b.index, b["0.5"], label = "6 months", lw = 0.8, ls = "--")
plt.plot(b.index, b["1"], label = "1 year", lw = 0.8, ls = "--")
plt.plot(b.index, b["5"], label = "5 years", lw = 0.8, ls = "--")
plt.plot(b.index, b["10"], label = "10 years", lw = 0.8, ls = "--")
plt.plot(b.index, b["15"], label = "15 years", lw = 0.8, ls = "--")
plt.xlabel("Date")
plt.ylabel("Yield(in %)") 
plt.legend()

date1 = datetime(2018,6,24)
date2 = datetime(2017,9,17)

plt.annotate("",
            xy=(date1.date(), 7.6), xycoords='data',
            xytext=(date2.date(), 6.4), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
date3 = datetime(2018,1,7)
plt.text(date3.date(), 7.05, "1. Period of rising interest rates", 
            ha="left", va="center", rotation=38,
            size=10,)


date1 = datetime(2019,2,17)
date2 = datetime(2018,10,14)

plt.annotate("",
            xy=(date1.date(), 6.4), xycoords='data',
            xytext=(date2.date(), 7.1), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
date3 = datetime(2018,11,18)
plt.text(date3.date(), 6.72, "2. Steepning of the curve\n (Short term rates declining)", 
            ha="center", va="center", rotation=-45,
            size=10,)


plt.title("Now a days!")
