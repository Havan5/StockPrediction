import math
from matplotlib import pyplot as plt
import numpy as np
from pandas_datareader import data as web
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from matplotlib.widgets import Slider
from mpl_interactions import ioff, panhandler, zoom_factory
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Define the stock symbol and start date
stock_symbol = 'NVDA'
start_date = datetime(2010, 1, 1)  # or any other start date you prefer

yf.pdr_override()

# Fetch stock data using get_data_yahoo
stock_data = web.DataReader(stock_symbol, start=start_date, end=datetime.now())

data=stock_data.filter(['Close'])
dataset=data.values
lookback = 60

scaler = MinMaxScaler(feature_range=(-1,1))
scaled_data = scaler.fit_transform(dataset)

training_data_len = int(len(dataset)* 0.8)
train_data = scaled_data[0:training_data_len,:]
test_data=scaled_data[training_data_len-lookback: , : ]

x_train=[]
y_train=[]
for i in range(lookback, len(train_data)):
    x_train.append(train_data[i-lookback:i,0])
    y_train.append(train_data[i,0])
    if i<=lookback:
        print(x_train)
        print(y_train)
        print()
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(lookback, len(test_data)):
    x_test.append(test_data[i-lookback:i,0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

# model = load_model("Latest_stock_price_model.keras")
# history=np.load('my_history.npy',allow_pickle='TRUE').item()
model=Sequential()
model.add(LSTM(256, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

from keras.callbacks import Callback
RMSE = []
EPOCHS = []
class RMSECallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 2 == 0:
            y_pred = self.model.predict(x_test)
            y_pred=scaler.inverse_transform(y_pred)
            rmse_test = math.sqrt(mean_squared_error(y_test, y_pred))
            RMSE.append(rmse_test)
            EPOCHS.append(epoch + 1)

model.compile(optimizer='adam', loss='mean_squared_error')
rmse_callback = RMSECallback()
history = model.fit(x_train, y_train, batch_size=16,epochs=20, validation_split = 0.2, callbacks=[rmse_callback])
model.summary()
# np.save('my_history.npy',history.history)
# model.save("Latest_stock_price_model.keras")

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

plt.bar(EPOCHS, RMSE,)
plt.title("RMSE Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.show()

test_predictions=model.predict(x_test)
train_predictions=model.predict(x_train)

test_predictions=scaler.inverse_transform(test_predictions)
train_predictions=scaler.inverse_transform(train_predictions)

rmse = np.sqrt(np.mean(((test_predictions - y_test) ** 2)))
print(rmse)
rmse = np.sqrt(np.mean(((train_predictions - y_train) ** 2)))
print(rmse)

# Calculate ROC curve and AUC
# y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_binary = np.where(test_predictions > np.median(test_predictions), 1, 0)
y_test_binary = np.where(y_test > np.median(y_test), 1, 0)

fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

train=data[:training_data_len]
valid=data[training_data_len:]
valid['test_predictions'] = test_predictions

#Visualize the data
fig = go.Figure()
fig.add_trace(go.Scatter(mode='lines', x=train.index, y=train['Close'], line_color='blue', name='training'))
fig.add_trace(go.Scatter(mode='lines', x=valid.index, y=valid['Close'], line_color='red', name='testing'))
fig.add_trace(go.Scatter(mode='lines', x=valid.index, y=valid['test_predictions'], line_color='green', name='test_predictions'))
fig.update_layout(
    autosize=True,
    height=600,
    title="Apple's Stock Price Prediction Model(Decision Tree Regressor Model)",
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
fig.show()
