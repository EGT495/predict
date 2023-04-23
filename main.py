from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib as plt

dataset = np.cos(np.arange(1000)*(20*np.pi/1000))

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)

look_back=1
#分割训练集和测试集
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[:train_size], dataset[train_size:]

#产生时间序列x和y
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 将输入x转换成[samples, time steps, features]的形式
trainX = np.reshape(trainX, (trainX.shape[0], 1,trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0], 1, testX.shape[1]))

#建模
model = Sequential()
model.add(LSTM(32,input_shape=(look_back,1)))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(trainX,trainY,batch_size=32,epochs=10)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print('trainPredict = ',trainPredict)
print('testPredict = ',testPredict)
