import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from datetime import datetime
from functools import reduce
import numpy as np
import locale

path = "/Users/ankitgupta/Data/Currencies/"

files = []

for filename in os.listdir(path):
    
    if "Historical Data" in filename:
        #maturity = filename.split("-")[0][6:] + "Y"
        currency = filename.split(" ")[0]
        
        df_currency = pd.read_csv(path + filename)
        df_currency[currency] = df_currency["Price"]
        df_currency["DATE"] = df_currency["Date"].apply(lambda x:datetime.strptime(x, '%b %d, %Y').date() )
        df_currency = df_currency[["DATE", currency]]
        
        df_currency = pd.DataFrame(df_currency)
    
    
        files.append(df_currency)




df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['DATE'],
                                            how='outer'), files).fillna(np.nan)
df_merged.set_index("DATE", drop=True, inplace=True)
df_merged.sort_index(axis=0, inplace = True)

df_merged.fillna(method = "ffill", inplace=True)

cols_ccy = ['USD_MXN', 'USD_MYR', 'USD_KRW', 'USD_TRY',
       'US_Dollar_Index_Futures', 'USD_NOK', 'USD_PHP', 'USD_NZD', 'USD_JPY',
       'USD_ZAR', 'USD_INR', 'GBP_USD',
       'USD_PKR', 'USD_AUD', 'USD_CAD', 'EUR_USD', 
       'USD_CNY', 'USD_BRL', 'USD_VND']

df_ccy = df_merged[cols_ccy]
df_ccy["USD_KRW"] = df_ccy["USD_KRW"].apply(lambda x: locale.atof(x))
df_ccy["USD_VND"] = df_ccy["USD_VND"].apply(lambda x: locale.atof(x))


df_rets = pd.DataFrame(index = df_ccy.index, columns = cols_ccy)
df_cumrets = pd.DataFrame(index = df_ccy.index, columns = cols_ccy)

for col in cols_ccy:
    df_rets[col] = df_ccy[col].pct_change()
    df_cumrets[col] = df_rets[col].cumsum()
    
df_cumrets.iloc[0] = 0
df_cumrets.fillna(method = "ffill", inplace = True)

#rebasing to 100
temp = np.tile(100, df_cumrets.shape)
rebased_array = np.add(temp, np.multiply(temp, np.array(df_cumrets)))

df_rebased = pd.DataFrame(rebased_array, index = df_ccy.index, columns = df_ccy.columns)


plt.figure()
for col in cols_ccy:
    
    plt.plot(df_rebased[col], label = col)
    
plt.legend()
    

#check the correlation of return of different currencies
import seaborn as sns
corrmat = df_rets.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df_rets[top_corr_features].corr(),annot=True,cmap="RdYlGn")


#split the data between 2 subsections..
#a. prior to financial crisis 
#b. since 2010

df_prior = 

