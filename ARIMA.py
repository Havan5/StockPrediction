import itertools
import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
from pandas_datareader import data as web
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from matplotlib.widgets import Slider
from mpl_interactions import ioff, panhandler, zoom_factory
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras. layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
stock_symbol = 'NVDA'
start_date = datetime(2010, 1, 1)  # or any other start date you prefer

yf.pdr_override()

stock_data = web.DataReader(stock_symbol, start=start_date, end=datetime.now())

data=stock_data.filter(['Close'])
dataset=data.values

def test_stationarity(dataFrame, var):
    dataFrame['rollMean']  = dataFrame[var].rolling(window=60).mean()
    dataFrame['rollStd']  = dataFrame[var].rolling(window=60).std()
    
    adfTest = adfuller(dataFrame[var],autolag='AIC')
    stats = pd.Series(adfTest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
    print(stats)
    
    for key, values in adfTest[4].items():
        print('criticality',key,":",values)

    sns.lineplot(data=dataFrame,x=dataFrame.index,y=var)
    sns.lineplot(data=dataFrame,x=dataFrame.index,y='rollMean')
    sns.lineplot(data=dataFrame,x=dataFrame.index,y='rollStd')
    # plt.show()

# test_stationarity(data.dropna(),'Close')

air_df = data[['Close']]
air_df['shift'] = air_df['Close'].shift()
air_df['shiftDiff'] = air_df['Close'] - air_df['shift']
print("--------shift-------")
# test_stationarity(air_df.dropna(),'shiftDiff')
print()

log_df = data[['Close']]
log_df['log'] = np.log(log_df['Close'])
print("--------log-------")
# test_stationarity(log_df,'log')
print()

sqrt_df = data[['Close']]
sqrt_df['sqrt'] = np.sqrt(sqrt_df['Close'])
sqrt_df.head()
print("--------sqrt-------")
# test_stationarity(sqrt_df,'sqrt')
print()

cbrt_df = data[['Close']]
cbrt_df['cbrt'] = np.cbrt(cbrt_df['Close'])
cbrt_df.head()
print("--------cbrt-------")
# test_stationarity(cbrt_df,'cbrt')
print()

log_df2 = log_df[['Close','log']]
log_df2['log_sqrt'] = np.sqrt(log_df['log'])
log_df2['logShiftDiff'] = log_df2['log_sqrt'] - log_df2['log_sqrt'].shift()
log_df2.head()
print("--------log_sqrt-------")
# test_stationarity(log_df2.dropna(),'logShiftDiff')
print()

airP = data[['Close']].copy(deep=True)
airP['firstDiff'] = data['Close'].diff()
airP['Diff60'] = data['Close'].diff(60)

# plot_pacf(airP['firstDiff'].dropna(),lags=20)
# plot_acf(airP['firstDiff'].dropna(),lags=20)
# plt.show()
train_size = int(len(data) * 0.85)
train, test = data[0:train_size], data[train_size:]

p = range(0, 8)
d = range(0, 8)
q = range(0, 2)
pdq_comb = list(itertools.product(p,d,q))
rmse=[]
order1=[]

def evaluate_arima_model():
    for pdq in pdq_comb:
        model = ARIMA(data['Close'], order=pdq).fit()
        # prediction = model.forecast(steps=len(test))
        prediction = model.predict(start=len(train), end=len(train)+len(test)-1)
        error = np.sqrt(mean_squared_error(test['Close'], prediction))
        order1.append(pdq)
        rmse.append(error)

warnings.filterwarnings("ignore")
# evaluate_arima_model()
# r =pd.DataFrame(index=order1, data=rmse, columns=['rmse'])
# r.to_csv("PDQ.csv")

# model = auto_arima(data["Close"],trace=True, error_action='ignore', start_p=1,start_q=1,max_p=10,max_q=10,m=12, suppress_warnings=True,stepwise=False,seasonal=True)
# print(model.summary())

model = SARIMAX(data['Close'],order=(5,2,1), seasonal_order=(5,2,1,12)).fit()
prediction = model.predict(start=len(train), end=len(data)-1)
test['arimaPred'] = prediction.values
error = np.sqrt(mean_squared_error(test['Close'], prediction))

last_date = data.index.max()
future_days = pd.DataFrame(pd.date_range(start=last_date + timedelta(days=1), periods=61, freq='D'),columns=['Dates'])
future_days.set_index('Dates', inplace=True)
prediction = model.predict(start=len(data), end=len(data)+60)
prediction.index = future_days.index
new_index = test.index.append(future_days.index)
test = test.reindex(new_index)
test['futurePred'] = np.nan
test.loc[future_days.index, 'futurePred'] = prediction.values

print(error)
plt.figure(figsize=(14,7))
plt.plot(train['Close'].dropna(), label='Training Data')
plt.plot(test['Close'].dropna(), label='Actual Data', color='orange')
plt.plot(test['arimaPred'].dropna(), label='Forecasted Data', color='green')
plt.plot(test['futurePred'].dropna(), label='Future Forecasted Data', color='red')
plt.title('ARIMA Model Evaluation')
plt.xlabel('Date')
plt.ylabel('Number of Births')
plt.legend()
plt.show()