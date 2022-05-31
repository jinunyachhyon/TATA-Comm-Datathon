import pandas as pd
import matplotlib.pyplot as plt
#importing csv file
df=pd.read_csv('/Users/nyach/Downloads/Train Set - Train Set.csv')
#print('Shape of data',df.shape)
#print(df.head())

#renaming index col
df_month= df.rename(columns = {'Region_Name': 'Month'}, inplace = False)
#print(df_month)

#remove index
df_month=df_month.set_index('Month')
#print(df_month)

#transpose data
df_transpose=df_month.transpose()
#print(df_transpose)

region = input('Enter the Region that you want to predict the Sales for? Format:{Region \'Number\' }')

#Returns a new dataframe with Region 1 column
#insert region name to predict for that region
dfnew_month = df_transpose.reindex(columns = [region])
dfnew_month[region].plot()
plt.show()
print(dfnew_month)
dfnew_month=dfnew_month.dropna()

#ad test for p-value
from statsmodels.tsa.stattools import adfuller
def ad_test(dataset):
     dftest = adfuller(dataset, autolag = 'AIC')
     print("1. ADF : ",dftest[0])
     print("2. P-Value : ", dftest[1])
     print("3. Num Of Lags : ", dftest[2])
     print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
     print("5. Critical Values :")
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)
ad_test(dfnew_month[region])

#for AIC
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
stepwise_fit = auto_arima(dfnew_month[region], trace=True,
suppress_warnings=True)
#print(stepwise_fit.summary())
print(stepwise_fit)

from statsmodels.tsa.arima.model import ARIMA
string = str(stepwise_fit)
p = string[7]
d = string[9]
q = string[11]
model = ARIMA(dfnew_month[region],order =(int(p),int(d),int(q)) )
results = model.fit()
#model=ARIMA(train[region],order=(2,1,2))
#model=model.fit()
#print(results.summary())

dfnew_month['forecast']=results.predict(start=60,end=71,dynamic=True)
dfnew_month[[region,'forecast']].plot()
plt.show()


for i in range(73,85):
    dict = {'Month': 'Month '+ str(i)}
    dfnew_month = dfnew_month.append(dict,ignore_index= True)
print(dfnew_month)

dfnew_month['forecast'] = results.predict(start = 72, end = 85, dynamic= True)
dfnew_month[[region, 'forecast']].plot()
plt.show()

from pandas import  DataFrame
residuals = DataFrame(results.resid)
residuals.plot()
residuals.plot(kind = 'kde')
print(residuals.describe())
plt.show()

