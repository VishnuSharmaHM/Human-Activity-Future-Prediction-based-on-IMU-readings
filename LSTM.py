import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataframe = read_csv('1.csv', usecols=[1])
plt.plot(dataframe)
dataset = dataframe.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print(dataset.shape)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

seq_size = length =  10
batch_size = 1
train_generator = TimeseriesGenerator(train,train,length=length,batch_size=batch_size)
validation_generator = TimeseriesGenerator(test, test, length=length ,batch_size=batch_size)
num_features = 1

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(length, num_features)))
model.add(LSTM(50, activation='relu'))
#model.add(Dense(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit_generator(generator=train_generator, verbose=2, epochs=100, validation_data=validation_generator)

trainPredict = model.predict(train_generator)
testPredict = model.predict(validation_generator)

trainPredict = scaler.inverse_transform(trainPredict)
trainY_inverse = scaler.inverse_transform(train)
testPredict = scaler.inverse_transform(testPredict)
testY_inverse = scaler.inverse_transform(test)

trainScore = math.sqrt(mean_squared_error(trainY_inverse[length:], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY_inverse[length:], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[length:len(trainPredict)+length, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train)+(length)-1:len(dataset)-1, :] = testPredict

plt.plot(scaler.inverse_transform(dataset))
plt.title("LSTM Future Prediction")
plt.xlabel("Time in ms")
plt.ylabel("Accelerometer X-axis measurement")
plt.plot(scaler.inverse_transform(dataset),label='Actual Data')
plt.plot(trainPredictPlot, label="LSTM Train Data Prediction")
plt.plot(testPredictPlot,label='LST Future Prediction')
plt.legend()
plt.show()

file1 = open("Final.csv", "w")
count=0
Out=testPredictPlot
for i in Out:
    count+=1
    file1.write(str(i))
    file1.write("\n")
file1.close()
print("Number of value in .csv file -",count  )
