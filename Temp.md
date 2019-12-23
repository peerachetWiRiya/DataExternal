import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import sklearn

from datetime import datetime, timedelta
from sklearn import preprocessing;
from sklearn import model_selection;
from sklearn import linear_model;
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out);#creating new column called label with the last few rows are nan
    X = np.array(df[[forecast_col]]); #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True); #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=test_size) #cross validation 
    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, ran-dom_state=chosen_random_state)


    response = [X_train,X_test , Y_train, Y_test , X_lately];
    return response;
   df = pd.read_csv('SEC_B/DCC_4.csv')
df.tail()
df = pd.read_csv('SEC_B/DCC_4.csv')
df.tail()
from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_true, y_predict)
dfreg.tail()
forecast_col = 'Close'#choosing which column to forecast
# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
#forecast_out = 5 #how far to forecast 
test_size = 0.2; #the size of my test set

X_train, X_test, Y_train, Y_test , X_lately =prepare_data(dfreg,forecast_col,forecast_out,test_size); #calling the method were the cross validation and data preperation is in

# Linear regression
learner_linear = linear_model.LinearRegression(); #initializing linear regression model
learner_linear.fit(X_train,Y_train); #training the linear regression model

score_linear=learner_linear.score(X_test,Y_test);#testing the linear regression model
forecast= learner_linear.predict(X_lately); #set that will contain the forecasted data
dfreg.tail()
# Plot Linear Prediction
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
last_date = df.iloc[-1].Date

last_unix = datetime.strptime(last_date, '%m/%d/%Y')

#print(date.today())
#print(date.today()+datetime.timedelta(days=1))
#print(last_unix+timedelta(days=1))
next_unix = last_unix + timedelta(days=1)

dfreg['Forecast'] = np.nan
for i in forecast:
    next_date = next_unix
    next_unix += timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
%config InlineBackend.figure_format = 'retina'
plt.legend(loc=1)
plt.title(r'DCC_ALL',fontsize=12)
plt.title(r'mean_squared_error =  15083.297190',fontsize=13,x=0.65,y=0.02,loc='right')
plt.xlabel('CountDay')
plt.ylabel('Price')
fig = plt.gcf()
fig.set_size_inches(12, 7)
plt.show()
plt.show()
# Plot Linear Prediction
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
last_date = df.iloc[-1].Date

last_unix = datetime.strptime(last_date, '%m/%d/%Y')

#print(date.today())
#print(date.today()+datetime.timedelta(days=1))
#print(last_unix+timedelta(days=1))
next_unix = last_unix + timedelta(days=1)

dfreg['Forecast'] = np.nan
for i in forecast:
    next_date = next_unix
    next_unix += timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
%config InlineBackend.figure_format = 'retina'
plt.legend(loc=1)
plt.title(r'DCC_ALL',fontsize=12)
plt.title(r'mean_squared_error =  15083.297190',fontsize=13,x=0.65,y=0.02,loc='right')
plt.xlabel('CountDay')
plt.ylabel('Price')
fig = plt.gcf()
fig.set_size_inches(12, 7)
plt.show()
plt.show()
