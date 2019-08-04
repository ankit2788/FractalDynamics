#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:05:20 2019

@author: ankitgupta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

from numpy import cumsum, log, polyfit, sqrt, std, subtract


##COMM = 2bps  

# ------------------------------------------------------

#data cleansing
#data analysis --> see ACF (returns, sign of returns)
#identify what to predict --> sign of returns at what period?
#feature extraction
    #1. AR(1),... AR(5)
    #2. RSI indicator
    #3. z-score of returns (take sample size equivalent to 1 minute)
    
#create a ML based model to predict the next sign--> Classification algo
#perform cross validation
#grid search on the parameters
    
#Trading strategy based on predicted values




#data = pd.read_csv("final_train.csv")
data = pd.read_csv("final_test.csv")

data = data[data.bid >0]
data.reset_index(inplace = True)

data = data.iloc[1:]

print(data.bid.isnull().sum())
print(data.bid.isna().sum())


plt.plot(data.bid)

data["abs_returns"] = data.bid.diff()
data["pct_return"] = data.bid.pct_change()

data.abs_returns.fillna(0)
data.pct_return.fillna(0)
data["sign_returns"] = np.sign(data.abs_returns)


acf_returns = acf(data.pct_return[1:],nlags = 50)
acf_signret = acf(data.sign_returns[1:],nlags = 50)

plt.figure()
plt.bar(np.arange(0,51), acf_returns)
plt.plot([.2]*len(np.arange(0,51)), color = "red")
plt.title("ACF for returns")

plt.figure()
plt.bar(np.arange(0,51), acf_signret)
plt.plot([.2]*len(np.arange(0,51)), color = "red")
plt.title("ACF for sign of returns")

pacf_signret = pacf(data.sign_returns[1:],nlags = 50)

plt.figure()
plt.bar(np.arange(0,51), pacf_signret)
plt.plot([.2]*len(np.arange(0,51)), color = "red")
plt.title("PACF for sign of returns")

#to predict signs
data["sign_returns_1"] = data["sign_returns"].shift(1)
data["sign_returns_2"] = data["sign_returns"].shift(2)
data["sign_returns_3"] = data["sign_returns"].shift(3)

n_samples = 3
#n_samples_2 = 5

data["rollingmean"] = data.pct_return.rolling(n_samples).mean()
data["rollingstd"] = data.pct_return.rolling(n_samples).std()
data["rolling_zscore"] = (data.pct_return - data.rollingmean) / data.rollingstd
data["rollingmaxminrange"] = data.bid.rolling(n_samples).max() - data.bid.rolling(n_samples).min()
data["rollingupticks"] = data.abs_returns.gt(0).rolling(n_samples).sum()/ n_samples
data["rollingcum_ret"] = data.abs_returns.rolling(n_samples).sum()
#data["rollingcum_ret_sample2"] = data.abs_returns.rolling(n_samples_2).sum()

data["rollingcum_ret_shift1"] = data.rollingcum_ret.shift(1)
data["rollingcum_ret_shift2"] = data.rollingcum_ret.shift(2)

plt.hist(data.rollingcum_ret[3:], bins = 1000)


def applysign(data):
    range1 = -0.01     #threshold defined to capture negative returns
    range2 = 0.01      #threshold defined to capture positive returns
    if data <= range1:
        return -1
    elif data >= range2:
        return 1
    else:
        return 0
    
    
data["sign_rollingcum_ret_new"] = data.rollingcum_ret.apply(applysign)
#data["sign_rollingcum_ret"] = np.sign(data.rollingcum_ret)

data["rollingmaxminrange_ret"] = data.abs_returns.rolling(n_samples).max() - data.abs_returns.rolling(n_samples).min()

#sample data every n ticks
sampled_data = data.iloc[n_samples::n_samples]
#sampled_data = data.iloc[n_samples::n_samples].reset_index(inplace = False, drop = True)

#acf_signret_cumsum = acf(sampled_data.sign_rollingcum_ret,nlags = 20)
acf_signret_cumsum = acf(sampled_data.sign_rollingcum_ret_new,nlags = 20)


plt.figure()
plt.bar(np.arange(0,21), acf_signret_cumsum)
plt.plot([.2]*len(np.arange(0,21)), color = "red")
plt.title("ACF for sign of cum returns")

#sampled_data["sampled_signshift1"] = sampled_data.sign_rollingcum_ret.shift(1)
#sampled_data["sampled_signshift2"] = sampled_data.sign_rollingcum_ret.shift(2)
#sampled_data["sampled_upticksshift1"] = sampled_data.rollingupticks.shift(1)
#sampled_data["sampled_upticksshift2"] = sampled_data.rollingupticks.shift(2)

sampled_data["sampled_signshift1"] = sampled_data.sign_rollingcum_ret_new.shift(1)
sampled_data["sampled_signshift2"] = sampled_data.sign_rollingcum_ret_new.shift(2)
sampled_data["sampled_upticksshift1"] = sampled_data.rollingupticks.shift(1)
sampled_data["sampled_upticksshift2"] = sampled_data.rollingupticks.shift(2)


features_1 = ["rollingmaxminrange", "rolling_zscore",  "rollingmaxminrange_ret", "sampled_signshift1", "sampled_signshift2", "sampled_upticksshift1", 'sampled_upticksshift2', 'rollingcum_ret_shift1', 'rollingcum_ret_shift2']

newdata = sampled_data[features_1]

#required_data = features_1 + ["sign_rollingcum_ret"]
required_data = features_1 + ["sign_rollingcum_ret_new"]

df_required_data = sampled_data[required_data].iloc[n_samples-1:]


#get correlations of each features in dataset
import seaborn as sns
corrmat = df_required_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df_required_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")



#indicates only 4 major features to be incorporated
#normalizing all features 
#1. Cant use StandardScalar --> features are not normally distributed
#2. Try MinMaxScalar

from sklearn import preprocessing
#preprocessed = preprocessing.StandardScaler(copy = False)
preprocessed = preprocessing.MinMaxScaler()


selected_features = ["sampled_signshift1", "sampled_signshift2",  'rollingcum_ret_shift1', 'rollingcum_ret_shift2']
selected_features = ["sampled_upticksshift1", 'sampled_upticksshift2',  'rollingcum_ret_shift1', 'rollingcum_ret_shift2']
X_scaled = preprocessed.fit_transform(df_required_data[selected_features])

#y = df_required_data["sign_rollingcum_ret"]
y = df_required_data["sign_rollingcum_ret_new"]



#simple classification model
y_check = y.reset_index(drop = "index")


#trying few classificcation algos


#perform cross validation
from sklearn.model_selection import KFold
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier


classifiermodel = SVC(C = 1.0)
knnclassifier = KNeighborsClassifier(n_neighbors = 5)


scores_svm = []
scores_knn = []

cross_validation = KFold(n_splits=5, random_state=42, shuffle=False)

for train_index, test_index in cross_validation.split(X_scaled):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X_scaled[train_index], X_scaled[test_index], y_check[train_index], y_check[test_index]
    classifiermodel.fit(X_train, y_train)
    knnclassifier.fit(X_train, y_train)
    scores_svm.append(classifiermodel.score(X_test, y_test))
    scores_knn.append(knnclassifier.score(X_test, y_test))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_check, test_size=0.33, random_state=42)


k_scores = []    
for item in range(3,50):
    model = KNeighborsClassifier(n_neighbors = item)
    model.fit(X_train, y_train)
    k_scores.append(model.score(X_test, y_test))

plt.plot(range(3,50), k_scores)
#k --> 20
    

from sklearn.model_selection import GridSearchCV

def svc_param_selection(X, y, nfolds):
    Cs = [0.1, 1, 10]
    gammas = [0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    #grid_search.best_params_
    #return grid_search.best_params_
    return grid_search
    

svm_grid_search = svc_param_selection(X_scaled, y, nfolds=5)
#print("C: {0}, gamma: {1}".format(best_params["C"], best_params["gamma"]))





# Creating trading strategy
# get predicted values of y
SVCmodel = SVC(kernel='rbf', C = 1)       #chosen best params
KNNmodel = KNeighborsClassifier(n_neighbors=30)
from sklearn.metrics import confusion_matrix, classification_report


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

SVCmodel.fit(X_train, y_train)
KNNmodel.fit(X_train, y_train)

y_pred_svm = SVCmodel.predict(X_test)
y_pred_knn = KNNmodel.predict(X_test)

report_knn = classification_report(y_test, y_pred_knn)
report_svm = classification_report(y_test, y_pred_svm)


#print(report_knn, report_svm)
print(report_knn)


#------Now getting the predicted values of y and getting the trading strategy

y_pred_all_svm = SVCmodel.predict(X_scaled)
y_pred_all_knn = KNNmodel.predict(X_scaled)


class PerformanceMetrics:
    def __init__(self):
        
        self.CumPnL = 0
        self.PnLSummary = []
        self.PnLSummary_post = []
        self.TradesCounter = 0
        self.RoundTrades = 0
        self.ProfitableRoundTrades = 0
        
        
        

class PositionManagement:
    
    def __init__(self, position = 0):
        self.position = position
        self.TradingValue = 0
        self.TradeCost = 0
        
        self.TradeComm = 0.00002       #2bps
        
        self.LastBuyCost = None
        
        self.Performance = PerformanceMetrics()
        
        
        
        
    def addTradeComm(self, OrderQty, OrderPrice):
        self.TradeCost += OrderQty * OrderPrice * self.TradeComm
        
    def ExecuteOrder(self, OrderDetails, MarketPrice):
        #market order is assumed (not sending limit orders here
        #assumes we get the full fill always
        #Order Details --> [OrderQty, OrderDirection]
        
        self.Performance.TradesCounter += 1
        
        if OrderDetails[1] == "B":
            self.position += OrderDetails[0]
            self.TradingValue -= OrderDetails[0] * MarketPrice
            self.LastBuyCost = MarketPrice
            
        elif OrderDetails[1] == "S":
            self.position -= OrderDetails[0]
            self.TradingValue += OrderDetails[0] * MarketPrice
            
            self.Performance.RoundTrades += 1
            
            if MarketPrice > self.LastBuyCost:
                 self.Performance.ProfitableRoundTrades += 1
            
        self.addTradeComm(OrderDetails[0], MarketPrice)
        
        
        
    
        
        
class TickManagement:
    
    def __init__(self, n_sampled = 3 ):
        self.tickCounter = 0
        self.n_sampled = n_sampled      #information sampled every n ticks
        self.PositionHandle = PositionManagement()
        self.LastExecutedOrder = None
        self.LiveOrder = None
        

        
        
        
    def processTick(self, price, pred_returndirection):
        #price --> represents the current price
        
        #our model interprets the direction of cumulative return in the nth sampled tick
        # using current tick price, classify the direction accordingly.
        self.tickCounter += 1
        
        #creating orders every nth tick
        if (np.mod(self.tickCounter, self.n_sampled) == 0):
            
            #
            if (self.PositionHandle.position == 0 and ~np.isnan(pred_returndirection)):
                
                #pred_returndirection = self.prediction[self.tickCounter// self.n_sampled]            

                if pred_returndirection == 1:
                #predicted returns is positive --> BUY order for now, and to be sold at next time
                
                    self.LiveOrder = [1, "B"]
                    
            elif (self.PositionHandle.position > 0 and self.LastExecutedOrder is not None):
                if self.LastExecutedOrder[1] == "B":
                    
                    #Square off the position immediately at market
                    self.LiveOrder = [1, "S"]
                    
                    
        #Executing Orders
        TradePnL = 0
        
        #Executing BUY order at n+1th tick at market
        if (np.mod(self.tickCounter, self.n_sampled) == 1 and self.LiveOrder is not None):
            
            if self.LiveOrder[1] == "B":
                
                
                #execute order
                self.PositionHandle.ExecuteOrder(self.LiveOrder, price)
                self.LastExecutedOrder = self.LiveOrder
                self.LiveOrder =  None

        
        #Executing SELL order at nth tick at market
        if (np.mod(self.tickCounter, self.n_sampled) == 0 and self.LiveOrder is not None):
            if self.LiveOrder[1] == "S":
                self.PositionHandle.ExecuteOrder(self.LiveOrder, price)
                self.LastExecutedOrder = self.LiveOrder
                self.LiveOrder =  None

                #position squaring off, hence Pnl generation                
                TradePnL = price - self.PositionHandle.LastBuyCost
            

        #PnL summary
        self.PositionHandle.Performance.CumPnL += TradePnL
        self.PositionHandle.Performance.PnLSummary.append(self.PositionHandle.Performance.CumPnL)
        self.PositionHandle.Performance.PnLSummary_post.append(self.PositionHandle.Performance.CumPnL - self.PositionHandle.TradeCost)

                    

temp_knn = pd.DataFrame(y_pred_all_knn, index = y.index)
#temp_svm = pd.DataFrame(y_pred_all_svm, index = y.index)
temp_knn.columns =["predicted"]
#temp_svm.columns =["predicted"]

temp_knn["shift"] = temp_knn.predicted.shift(-1)
#temp_svm["shift"] = temp_svm.predicted.shift(-1)
   
#df_required_data["predicted_nextcum_sign_svm"] = temp_svm["shift"]
df_required_data["predicted_nextcum_sign_knn"] = temp_knn["shift"]

#data["predicted_nextcum_sign_svm"] = df_required_data["predicted_nextcum_sign_svm"]
data["predicted_nextcum_sign_knn"] = df_required_data["predicted_nextcum_sign_knn"]


#tick_handle_svm = TickManagement()       
tick_handle_knn = TickManagement()       

for index, row in data[1:].iterrows():
    
    price = row["bid"]
    prediction_knn = row["predicted_nextcum_sign_knn"]
    #prediction_svm = row["predicted_nextcum_sign_svm"]
    tick_handle_knn.processTick(price, prediction_knn)
    #tick_handle_svm.processTick(price, prediction_svm)    
    
    

pnl_knn = tick_handle_knn.PositionHandle.Performance.PnLSummary
pnl_postcost = tick_handle_knn.PositionHandle.Performance.PnLSummary_post
#pnl_svm = tick_handle_svm.PositionHandle.Performance.PnLSummary

#print("=====Stats (SVM) =====")
#print("Cum PnL(pre cost): {0}".format(tick_handle_svm.PositionHandle.Performance.CumPnL))
#print("Total RoundTrades: {0}".format(tick_handle_svm.PositionHandle.Performance.RoundTrades))
#print("Profitable RoundTrades: {0}".format(tick_handle_svm.PositionHandle.Performance.ProfitableRoundTrades))
#print("Net Trading Cost : {0}".format(tick_handle_svm.PositionHandle.TradeCost))
#
#
#plt.plot(pnl_svm)
            

print("=====Stats (KNN) =====")
print("Cum PnL(pre cost): {0}".format(tick_handle_knn.PositionHandle.Performance.CumPnL))
print("Total RoundTrades: {0}".format(tick_handle_knn.PositionHandle.Performance.RoundTrades))
print("Profitable RoundTrades: {0}".format(tick_handle_knn.PositionHandle.Performance.ProfitableRoundTrades))
print("Net Trading Cost : {0}".format(tick_handle_knn.PositionHandle.TradeCost))
print("Net PnL : {0}".format(pnl_postcost[-1]))


plt.plot(pnl_knn)
plt.plot(pnl_postcost)
            
            
        
        
            
            
        


