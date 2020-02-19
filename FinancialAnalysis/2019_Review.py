#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:38:43 2019

@author: ankitgupta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from FDynamics.Visualizations import BokehVisuals
import seaborn as sns
import scipy.stats as stats
import numpy as np

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, LinearAxis, Range1d



path = "/Users/ankitgupta/Downloads/"
nifty = "^NSEI.csv"

data = pd.read_csv(path + nifty)


data.fillna(method = "ffill", inplace = True)

data["ret"] = data["Close"].pct_change()
data["std"] = data["ret"].rolling(30).std()
data["AnnualizedVol_month"] = data["std"] * np.sqrt(252)

data["VolofVolreturn"] = data["std"].rolling(252).std() * np.sqrt(252)

data["AnnualizedVol_year"] = data["ret"].rolling(252).std() * np.sqrt(252)
data["newDate"] = data["Date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

a = BokehVisuals(data)

b = a.createPlot(outputpath=path + "newplot.html", imagetitle = "returns", XAxisDetails = {"LABEL": "Date", "HOVER": "Date", "COL_NAME": "newDate"}, 
             PrimaryAxisDetails = {"LABEL": "ret", "HOVER": "@ret", "COL_NAME": "ret"}, SecondaryAxis = False)

show(b)

fig, ax = plt.subplots(2,1)
ax[0].plot(data["newDate"], data["Close"], label = "Daily Price", color = "black")
ax[0].set_ylabel("Daily Price levels")
ax[0].legend()

ax[1].plot(data["newDate"], data["AnnualizedVol_month"], label = "Annualized Vol of Returns (30 days rolling)", color = "green")
ax[1].plot(data["newDate"], data["AnnualizedVol_year"], label = "Annualized Vol of Returns (1 year rolling)", color = "black")
ax[1].set_ylabel("Vol of Returns")
ax[1].legend()

plt.suptitle("Nifty 50")

fig, ax = plt.subplots()
ax.plot(data["newDate"], data["AnnualizedVol_month"], label = "30 days Annualized Volatility", color = "green")
ax.set_ylabel("30 days Annualized Volatility")
ax.legend()

ax1 = ax.twinx()
ax1.plot(data["newDate"], data["VolofVolreturn"], label = "Volatlity of 30 days Volatility", color = "black")
ax1.set_ylabel("VolofVol returns")
ax1.legend()


#split the returns post the elections date
#1. Since 2018 (Start of Election campaigning)
section1 = data.query('newDate > datetime(2018,1,1)')

a = section1.ret
stdreturn = a.std()
meanreturn = a.mean()

sigma_2_pos = meanreturn + 2*stdreturn
sigma_2_neg = meanreturn - 2*stdreturn

sns.distplot(a, bins = 100, label = "2018 Onwards")
plt.ylabel("Frequency of Returns")
plt.xlabel("Daily Returns")
plt.suptitle("Nifty 50 returns (2018 Onwards)")

plt.vlines(sigma_2_neg, 0, 5, colors = "red", label = "2 sigma event")
plt.vlines(sigma_2_pos, 0, 5, colors = "red")
plt.legend()


#2 Since Elections counting
section2 = data.query('newDate > datetime(2019,4,1)')

print("Total Days passed: {0}".format(len(section2)) )
print("Days with abs(return) > 1%: {0}".format(len(section2.query('abs(ret) > 0.01'))) )
print("Days with abs(return) > 2%: {0}".format(len(section2.query('abs(ret) > 0.02'))) )
print("Average Annualsized Volatility of returns in this period: {0}%".format(np.round(section2["ret"].std() * np.sqrt(252) * 100,2)))

meanreturn = section2["ret"].mean()
stdreturn = section2["ret"].std()

sigma_2_pos = meanreturn + 2*stdreturn
sigma_2_neg = meanreturn - 2*stdreturn


plt.figure()
for Nbiter, row in section2.iterrows():
    
    if row["ret"] > 0:
        plt.vlines(row["newDate"], ymin = 0, ymax = row["ret"], color = "green")
    else:
        plt.vlines(row["newDate"], ymax = 0, ymin = row["ret"], color = "red")

plt.ylabel("Daily Percentage Return")
#plt.hlines(y = 0.01, xmin=section2["newDate"].iloc[0], xmax = section2["newDate"].iloc[-1], color = "black", label = "1% return")
#plt.hlines(y = -0.01, xmin=section2["newDate"].iloc[0], xmax = section2["newDate"].iloc[-1], color = "black")
plt.suptitle("Nifty 50 (Post Elections 2019)")


#mark specific days
date = datetime(2019,5,20)  #Budget Day (exit Polls)
date1 = datetime(2019,5,1)

plt.annotate("Exit Polls\n20 May 2019", xy=(mdates.date2num(date), 0.02),
            xytext=(mdates.date2num(date1), 0.04),arrowprops=dict(facecolor='black', shrink=0.03, width=0.5),
            horizontalalignment='center', verticalalignment='top',)


date = datetime(2019,7,5)  #Budget Day
date1 = datetime(2019,5,28)

plt.annotate("Interim Budget\n5 July 2019", xy=(mdates.date2num(date), -0.01),
            xytext=(mdates.date2num(date1), -0.020),arrowprops=dict(facecolor='black', shrink=0.03, width=0.5),
            horizontalalignment='center', verticalalignment='top',)



date = datetime(2019,8,5)  #Kashmir referendum
date1 = datetime(2019,7,15)

plt.annotate("Kashmir dispute (Article 370 and 35-A)\n5 August 2019", xy=(mdates.date2num(date), -0.01),
            xytext=(mdates.date2num(date1), -0.020),arrowprops=dict(facecolor='black', shrink=0.03, width=0.5),
            horizontalalignment='center', verticalalignment='top',)



date = datetime(2019,8,5)  #Kashmir referendum
date1 = datetime(2019,7,15)

plt.annotate("Kashmir dispute (Article 370 and 35-A)\n5 August 2019", xy=(mdates.date2num(date), -0.01),
            xytext=(mdates.date2num(date1), -0.020),arrowprops=dict(facecolor='black', shrink=0.03, width=0.5),
            horizontalalignment='center', verticalalignment='top',)


date = datetime(2019,8,7)  #RBI cuts rate by 35 bps
date1 = datetime(2019,7,15)

plt.annotate("RBI cuts rate\n7 August 2019", xy=(mdates.date2num(date), -0.004),
            xytext=(mdates.date2num(date1), 0.020),arrowprops=dict(facecolor='black', shrink=0.03, width=0.5),
            horizontalalignment='center', verticalalignment='top',)

date = datetime(2019,8,23)  #FinMin Withdraws surcharge
date1 = datetime(2019,8,1)

plt.annotate("Economic measures revision\n23 August 2019", xy=(mdates.date2num(date), 0.005),
            xytext=(mdates.date2num(date1), 0.030),arrowprops=dict(facecolor='black', shrink=0.03, width=0.5),
            horizontalalignment='center', verticalalignment='top',)


date = datetime(2019,8,30)  #Banks MegaMerger
date1 = datetime(2019,8,15)

plt.annotate("PSBs Mega Merger\n30 August 2019", xy=(mdates.date2num(date), 0.005),
            xytext=(mdates.date2num(date1), 0.040),arrowprops=dict(facecolor='black', shrink=0.03, width=0.5),
            horizontalalignment='center', verticalalignment='top',)


date = datetime(2019,9,16)  #Saudi Oil supply cut
date1 = datetime(2019,9,1)

plt.annotate("Saudi Aramco Attack\n16 September 2019", xy=(mdates.date2num(date), -0.005),
            xytext=(mdates.date2num(date1), -0.0150),arrowprops=dict(facecolor='black', shrink=0.03, width=0.5),
            horizontalalignment='center', verticalalignment='top',)



date = datetime(2019,9,21)  #Saudi Oil supply cut
date1 = datetime(2019,9,10)

plt.annotate("Corporate Tax Cut Announcement\n21 September 2019", xy=(mdates.date2num(date), 0.025),
            xytext=(mdates.date2num(date1), 0.0550),arrowprops=dict(facecolor='black', shrink=0.03, width=0.5),
            horizontalalignment='center', verticalalignment='top',)


z_score_sept21 = (0.0531911 - section2["ret"].mean())/section2["ret"].std()

