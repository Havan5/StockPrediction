import itertools
import time
import warnings
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from matplotlib import pyplot as plt
import numpy as np
from pandas_datareader import data as web
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from data import *
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
plt.style.use('ggplot')
st.set_option('deprecation.showPyplotGlobalUse', False)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima

def update_layout(fig, title):
    fig.update_layout(
            autosize=True,
            height=600,
            title= title,
            xaxis_title="Days",
            yaxis_title="Close Price USD ($)",
            template='plotly_white'
    )
    fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=3,
                            label="3m",
                            step="month",
                            stepmode="backward"),              
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="YTD",
                            step="year",
                            stepmode="todate"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )

def plot_graph(figsize, label, extra_data, full_data, values, extra_dataset = None, extra_dataset2 = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Red', label=label)
    plt.plot(full_data.Close, 'b', label='Full Date')
    if extra_data != 0:
        plt.plot(extra_dataset, label='MA_for_100_days')
        plt.plot(extra_dataset2, label='MA_for_250_days')
    plt.legend()
    return fig

def print_helper(reddit_df):

    sorted_df = reddit_df.sort_values(by=['ticker'], ascending=True)

    top_picks = sorted_df['ticker'].value_counts().head(3) 
    symbols = top_picks.index.tolist()

    times = []
    top = []
    for value, count in top_picks.items():
        times.append(count)
        top.append(f"{value}: {count}")

    squarify.plot(sizes=times, label=top, alpha=.7 )
    plt.axis('off')
    plt.title("Most mentioned picks")
    st.pyplot(plt.show())
    return symbols

def visualization(vaders_df,symbols):
    vaders_df = vaders_df[vaders_df['ticker'].isin(symbols)]
    
    mean_compound_by_ticker_vader = vaders_df.groupby('ticker')[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']].mean().reset_index()
    mean_compound_by_ticker_roberta = vaders_df.groupby('ticker')[['roberta_neg', 'roberta_neu', 'roberta_pos']].mean().reset_index()

    # Sentiment analysis
    colors = ['red', 'springgreen', 'forestgreen', 'coral']
    mean_compound_by_ticker_vader.plot(x='ticker', kind = 'bar', color=colors, title="Vader Sentiment analysis of top 3 picks")
    plt.ylabel('scores')
    st.pyplot(plt.show())

    colors = ['red', 'springgreen', 'forestgreen']
    mean_compound_by_ticker_roberta.plot(x='ticker', kind = 'bar', color=colors, title="Roberta Sentiment analysis of top 3 picks")
    plt.ylabel('scores')
    st.pyplot(plt.show())

    weekly_df = vaders_df
    weekly_df['date'] = pd.to_datetime(weekly_df.date).dt.to_period('M')
    mean_compound_by_ticker_vader = weekly_df.groupby(['ticker', 'date'])[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']].mean().unstack()
    mean_compound_by_ticker_roberta = weekly_df.groupby('ticker')[['roberta_neg', 'roberta_neu', 'roberta_pos']].mean().unstack()
    
    mean_df = mean_compound_by_ticker_vader.xs('vader_compound', axis="columns").transpose()
    mean_df.plot(kind='bar', title="Monthly Vader Compound Score of top 3 picks")
    plt.ylabel('Compound Score')
    st.pyplot(plt.show())

    sns.pairplot(data=vaders_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='ticker',
            palette='tab10')
    st.pyplot(plt.show())

st.title("Stock Price Prediction")
yf.pdr_override()

model = load_model("Latest_stock_price_model.keras")
history=np.load('my_history.npy',allow_pickle='TRUE').item()

reddit_df = pd.read_csv('new_both_df.csv')
st.subheader("Social Media Sentiment")
st.write(reddit_df)
symbols = print_helper(reddit_df)
visualization(reddit_df, symbols)

stock = st.text_input("Enter the Stock Ticker", "NVDA")

end = datetime(2024, 4, 26)
start = datetime(2010, 1, 1)  # or any other start date you prefer

stock_data = web.DataReader(stock, start=start, end=end)

st.subheader("Stock Data " + stock)
st.write(stock_data)

st.subheader('Original Close Price and MA for 250 days')
stock_data['MA_for_250_days'] = stock_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), 'MA_for_250_days', 0, stock_data, stock_data['MA_for_250_days']))

st.subheader('Original Close Price and MA for 200 days')
stock_data['MA_for_200_days'] = stock_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), 'MA_for_200_days', 0, stock_data, stock_data['MA_for_200_days']))

st.subheader('Original Close Price and MA for 100 days')
stock_data['MA_for_100_days'] = stock_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), 'MA_for_100_days', 0, stock_data, stock_data['MA_for_100_days']))

st.subheader('Original Close Price and MA for 60 days')
stock_data['MA_for_60_days'] = stock_data.Close.rolling(60).mean()
st.pyplot(plot_graph((15,6), 'MA_for_60_days', 0, stock_data, stock_data['MA_for_60_days']))

st.subheader('Original Close Price and MA for 60, 100 and 250 days')
st.pyplot(plot_graph((15,6), 'MA_for_60_days', 1, stock_data, stock_data['MA_for_60_days'], stock_data['MA_for_100_days'], stock_data['MA_for_250_days']))

data=stock_data.filter(['Close'])
dataset=data.values
lookback = 60

training_data_len = int(len(dataset)* 0.85)

scaler = MinMaxScaler(feature_range=(-1,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len,:]
test_data = scaled_data[training_data_len-lookback: , : ]

x_train=[]
y_train=[]

for i in range(lookback, len(train_data)):
    x_train.append(train_data[i-lookback:i,0])
    y_train.append(train_data[i,0])
    if i<=lookback:
        # print(x_train)
        # print(y_train)
        print()

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(lookback, len(test_data)):
    x_test.append(test_data[i-lookback:i,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

# history = model.fit(x_train, y_train, batch_size=16,epochs=32, validation_split = 0.2)
fig = plt.figure(figsize=(15,6))
plt.plot(history['loss'], label='train')
plt.plot(history['val_loss'], label='validation')
plt.title('Train vs. Validation Loss')
plt.legend()
st.pyplot(fig)

test_predictions=model.predict(x_test)
train_predictions=model.predict(x_train)
test_predictions=scaler.inverse_transform(test_predictions)
train_predictions=scaler.inverse_transform(train_predictions)

train=data[:training_data_len]
valid=data[training_data_len:]
valid['predictions'] = test_predictions

fig = go.Figure()
fig.add_trace(go.Scatter(mode='lines', x=train.index, y=train['Close'], line_color='blue', name='training'))
fig.add_trace(go.Scatter(mode='lines', x=valid.index, y=valid['Close'], line_color='red', name='testing'))
fig.add_trace(go.Scatter(mode='lines', x=valid.index, y=valid['predictions'], line_color='green', name='predicted'))
fig.update_layout = update_layout(fig, stock + " Stock Price Prediction Model(LSTM Model)")
st.plotly_chart(fig)

lookback_future = 300
future_days_to_predict = 5
future_predictions = []
past_days = test_data[-lookback_future:]

for _ in range(future_days_to_predict):
    x_input = np.reshape(past_days, (1, lookback_future, 1))
    next_day_prediction = model.predict(x_input)
    future_predictions.append(next_day_prediction)
    past_days = np.roll(past_days, -1)
    past_days[-1] = next_day_prediction

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

last_date = data.index.max()
future_days = pd.date_range(start=last_date + timedelta(days=1), periods=future_days_to_predict, freq='D')
predicted_prices_df = pd.DataFrame({'Date': future_days, 'Predictions': future_predictions.flatten()})
predicted_prices_df = predicted_prices_df.set_index('Date')
combined_data = pd.concat([data, predicted_prices_df], axis=1)

fig = go.Figure()
fig.add_trace(go.Scatter(mode='lines', x=combined_data.index, y=combined_data['Close'], line_color='blue', name='Actual Stock Price'))
fig.add_trace(go.Scatter(mode='lines', x=combined_data.index, y=combined_data['Predictions'], line_color='red', name='Predicted Stock Price'))
fig.update_layout = update_layout(fig, stock + " Stock Price Prediction For " + str(future_days_to_predict) + " Days Ahead")
st.plotly_chart(fig)

def test_stationarity(dataFrame, var):
    dataFrame['rollMean']  = dataFrame[var].rolling(window=60).mean()
    dataFrame['rollStd']  = dataFrame[var].rolling(window=60).std()
    
    adfTest = adfuller(dataFrame[var],autolag='AIC')
    stats = pd.Series(adfTest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
    st.write(stats)
    
    for key, values in adfTest[4].items():
        st.write('criticality',key,":",values)

    sns.lineplot(data=dataFrame,x=dataFrame.index,y=var)
    sns.lineplot(data=dataFrame,x=dataFrame.index,y='rollMean')
    sns.lineplot(data=dataFrame,x=dataFrame.index,y='rollStd')
    st.pyplot(plt.show())

st.subheader("--------Stationary Check-------")

st.subheader("--------Actual Data-------")
test_stationarity(data.dropna(),'Close')

air_df = data[['Close']]
air_df['shift'] = air_df['Close'].shift()
air_df['shiftDiff'] = air_df['Close'] - air_df['shift']
st.subheader("--------shift-------")
test_stationarity(air_df.dropna(),'shiftDiff')

log_df = data[['Close']]
log_df['log'] = np.log(log_df['Close'])
st.subheader("--------log-------")
test_stationarity(log_df,'log')

sqrt_df = data[['Close']]
sqrt_df['sqrt'] = np.sqrt(sqrt_df['Close'])
sqrt_df.head()
st.subheader("--------sqrt-------")
test_stationarity(sqrt_df,'sqrt')

cbrt_df = data[['Close']]
cbrt_df['cbrt'] = np.cbrt(cbrt_df['Close'])
cbrt_df.head()
st.subheader("--------cbrt-------")
test_stationarity(cbrt_df,'cbrt')

log_df2 = log_df[['Close','log']]
log_df2['log_sqrt'] = np.sqrt(log_df['log'])
log_df2['logShiftDiff'] = log_df2['log_sqrt'] - log_df2['log_sqrt'].shift()
log_df2.head()
st.subheader("--------log_sqrt-------")
test_stationarity(log_df2.dropna(),'logShiftDiff')

airP = data[['Close']].copy(deep=True)
airP['firstDiff'] = data['Close'].diff()
airP['Diff60'] = data['Close'].diff(60)

st.pyplot(plot_pacf(airP['firstDiff'].dropna(),lags=20))
st.pyplot(plot_acf(airP['firstDiff'].dropna(),lags=20))

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
        arima_model = ARIMA(data['Close'], order=pdq).fit()
        prediction = arima_model.predict(start=len(train), end=len(train)+len(test)-1)
        error = np.sqrt(mean_squared_error(test['Close'], prediction))
        order1.append(pdq)
        rmse.append(error)

warnings.filterwarnings("ignore")
evaluate_arima_model()
r = pd.DataFrame(index=order1, data=rmse, columns=['rmse'])
st.subheader("Best P,Q,D values")
st.write(r)

auto_model = auto_arima(data["Close"],trace=True, error_action='ignore', start_p=1,start_q=1,max_p=10,max_q=10,m=12, suppress_warnings=True,stepwise=False,seasonal=True)
st.write(auto_model.summary())

while True:
    P = st.text_input("Enter P", "7", key="25")
    D = st.text_input("Enter D", "1",  key="35")
    Q = st.text_input("Enter Q", "1",  key="46")
    time.sleep(20)

    if P is not None and D is not None and Q is not None:

        st.write(P,D,Q)
        sarima_model = SARIMAX(data['Close'],order=(int(P),int(D),int(Q)), seasonal_order=(int(P),int(D),int(Q),12)).fit()
        prediction = sarima_model.predict(start=len(train), end=len(data)-1)
        test['arimaPred'] = prediction.values
        error = np.sqrt(mean_squared_error(test['Close'], prediction))

        last_date = data.index.max()
        future_days = pd.DataFrame(pd.date_range(start=last_date + timedelta(days=1), periods=61, freq='D'),columns=['Dates'])
        future_days.set_index('Dates', inplace=True)
        prediction = sarima_model.predict(start=len(data), end=len(data)+60)
        # prediction.index = future_days.index
        new_index = test.index.append(future_days.index)
        test = test.reindex(new_index)
        test['futurePred'] = np.nan
        test.loc[future_days.index, 'futurePred'] = prediction.values

        st.write(error)

        fig = go.Figure()
        fig.add_trace(go.Scatter(mode='lines', x=train.index, y=train['Close'].dropna(), line_color='blue', name='training'))
        fig.add_trace(go.Scatter(mode='lines', x=test.index, y=test['Close'].dropna(), line_color='red', name='testing'))
        fig.add_trace(go.Scatter(mode='lines', x=test.index, y=test['arimaPred'].dropna(), line_color='green', name='predicted'))
        fig.add_trace(go.Scatter(mode='lines', x=future_days.index, y=test['futurePred'].dropna(), line_color='orange', name='Future predicted'))
        fig.update_layout = update_layout(fig, stock + " Stock Price Prediction Model(SARIMAX Model)")
        st.plotly_chart(fig)