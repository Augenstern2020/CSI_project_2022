from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import random
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import plot_model
import graphviz


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 读取一个路径下所有的.dat文件，默认路径设为..\\DATA\\xxxx
os.chdir(os.path.realpath(__file__) + "\\..")
train_path_list = os.listdir("DATA\\pre3")
os.chdir("DATA\\pre3")


data0 = pd.read_csv(r'20210725Apre3.csv')
data1 = pd.read_csv(r'20210725Ppre3.csv')
# data = pd.read_csv(r'20210720pre3.csv')

print(data0.shape)

list = range(1, 75)
randomlist = random.sample(list, 15)
#randomlist = [35, 8, 3, 44, 18, 55, 60, 68, 54, 49, 74, 45, 17, 27, 57]

print(randomlist)

data0_train = data0[data0['th'].isin(randomlist)]
data0_test = data0[~data0['th'].isin(randomlist)]

data1_train = data1[data1['th'].isin(randomlist)]
data1_test = data1[~data1['th'].isin(randomlist)]



Y_train = data0_train.iloc[:, data0.columns == "bin_type"]
Y_test = data0_test.iloc[:, data0.columns == "bin_type"]


# 0/1 _ link

# link 1
X_train0_1 = data0_train.iloc[:, 5:215]
X_train1_1 = data1_train.iloc[:, 5:215]
X_train_1 = pd.concat([X_train0_1, X_train1_1], axis=1)
for i in range(21):
    X_train_1[str(i)] = 0

X_test0_1 = data0_test.iloc[:, 5:215]
X_test1_1 = data1_test.iloc[:, 5:215]
X_test_1 = pd.concat([X_test0_1, X_test1_1], axis=1)
for i in range(21):
    X_test_1[str(i)] = 0

standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train_1)

X_train_1 = standard_scaler.transform(X_train_1)
X_test_1 = standard_scaler.transform(X_test_1)


X_train_1 = X_train_1.reshape((X_train_1.shape[0], 21, 21, 1)).astype('float32')
X_test_1 = X_test_1.reshape((X_test_1.shape[0], 21, 21, 1)).astype('float32')



# link 2
X_train0_2 = data0_train.iloc[:, 215:425]
X_train1_2 = data1_train.iloc[:, 215:425]
X_train_2 = pd.concat([X_train0_2, X_train1_2], axis=1)
for i in range(21):
    X_train_2[str(i)] = 0

X_test0_2 = data0_test.iloc[:, 215:425]
X_test1_2 = data1_test.iloc[:, 215:425]
X_test_2 = pd.concat([X_test0_2, X_test1_2], axis=1)
for i in range(21):
    X_test_2[str(i)] = 0

standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train_2)

X_train_2 = standard_scaler.transform(X_train_2)
X_test_2 = standard_scaler.transform(X_test_2)

X_train_2 = X_train_2.reshape((X_train_2.shape[0], 21, 21, 1)).astype('float32')
X_test_2 = X_test_2.reshape((X_test_2.shape[0], 21, 21, 1)).astype('float32')



# link 3
X_train0_3 = data0_train.iloc[:, 425:635]
X_train1_3 = data1_train.iloc[:, 425:635]
X_train_3 = pd.concat([X_train0_3, X_train1_3], axis=1)
for i in range(21):
    X_train_3[str(i)] = 0

X_test0_3 = data0_test.iloc[:, 425:635]
X_test1_3 = data1_test.iloc[:, 425:635]
X_test_3 = pd.concat([X_test0_3, X_test1_3], axis=1)
for i in range(21):
    X_test_3[str(i)] = 0

standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train_3)

X_train_3 = standard_scaler.transform(X_train_3)
X_test_3 = standard_scaler.transform(X_test_3)

X_train_3 = X_train_3.reshape((X_train_3.shape[0], 21, 21, 1)).astype('float32')
X_test_3 = X_test_3.reshape((X_test_3.shape[0], 21, 21, 1)).astype('float32')




# link 4
X_train0_4 = data0_train.iloc[:, 635:845]
X_train1_4 = data1_train.iloc[:, 635:845]
X_train_4 = pd.concat([X_train0_4, X_train1_4], axis=1)
for i in range(21):
    X_train_4[str(i)] = 0

X_test0_4 = data0_test.iloc[:, 635:845]
X_test1_4 = data1_test.iloc[:, 635:845]
X_test_4 = pd.concat([X_test0_4, X_test1_4], axis=1)
for i in range(21):
    X_test_4[str(i)] = 0

standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train_4)

X_train_4 = standard_scaler.transform(X_train_4)
X_test_4 = standard_scaler.transform(X_test_4)


X_train_4 = X_train_4.reshape((X_train_4.shape[0], 21, 21, 1)).astype('float32')
X_test_4 = X_test_4.reshape((X_test_4.shape[0], 21, 21, 1)).astype('float32')



# link 5
X_train0_5 = data0_train.iloc[:, 845:1055]
X_train1_5 = data1_train.iloc[:, 845:1055]
X_train_5 = pd.concat([X_train0_5, X_train1_5], axis=1)
for i in range(21):
    X_train_5[str(i)] = 0

X_test0_5 = data0_test.iloc[:, 845:1055]
X_test1_5 = data1_test.iloc[:, 845:1055]
X_test_5 = pd.concat([X_test0_5, X_test1_5], axis=1)
for i in range(21):
    X_test_5[str(i)] = 0

standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train_5)

X_train_5 = standard_scaler.transform(X_train_5)
X_test_5 = standard_scaler.transform(X_test_5)

X_train_5 = X_train_5.reshape((X_train_5.shape[0], 21, 21, 1)).astype('float32')
X_test_5 = X_test_5.reshape((X_test_5.shape[0], 21, 21, 1)).astype('float32')



# link 6
X_train0_6 = data0_train.iloc[:, 1055:1265]
X_train1_6 = data1_train.iloc[:, 1055:1265]
X_train_6 = pd.concat([X_train0_6, X_train1_6], axis=1)
for i in range(21):
    X_train_6[str(i)] = 0

X_test0_6 = data0_test.iloc[:, 1055:1265]
X_test1_6 = data1_test.iloc[:, 1055:1265]
X_test_6 = pd.concat([X_test0_6, X_test1_6], axis=1)
for i in range(21):
    X_test_6[str(i)] = 0

standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train_6)

X_train_6 = standard_scaler.transform(X_train_6)
X_test_6 = standard_scaler.transform(X_test_6)

X_train_6 = X_train_6.reshape((X_train_6.shape[0], 21, 21, 1)).astype('float32')
X_test_6 = X_test_6.reshape((X_test_6.shape[0], 21, 21, 1)).astype('float32')
""""""




""""""
X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6), axis=3)
X_test = np.concatenate((X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6), axis=3)


print(X_train.shape)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

print(Y_test.shape)

num_classes = Y_test.shape[1]

model = Sequential()

# 一层卷积层 包含64个卷积核 大小5*5
model.add(Conv2D(64, (3, 3), input_shape=(21, 21, 6), activation='relu', data_format="channels_last", padding="same"))
model.add(Conv2D(64, (3, 3), input_shape=(21, 21, 6), activation='relu', data_format="channels_last", padding="same"))

# 一个最大池化层 池化大小为2*2
model.add(MaxPooling2D(pool_size=(2, 2)))

# 一个卷积层包含128个卷积核 3*3
model.add(Conv2D(128, (3, 3), activation='relu', data_format="channels_last", padding="same"))
model.add(Conv2D(128, (3, 3), activation='relu', data_format="channels_last", padding="same"))

# 一个池化层
model.add(MaxPooling2D(pool_size=(2, 2)))

# 遗忘层
model.add(Dropout(0.20))

# 压平层
model.add(Flatten())

# 全连接
model.add(Dense(128, activation='relu'))

# 分类
model.add(Dense(num_classes, activation='softmax'))

# plot_model(model, to_file='model_plot_2.png', show_shapes=True, show_layer_names=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=256, verbose=2)




# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

Ytrue = Y_test
Ypred = model.predict_classes(X_test)
C2 = confusion_matrix(Ytrue.argmax(axis=-1), Ypred)
print(C2)  # 打印出来看看


model.save('weights.model')
model.save_weights("model.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)