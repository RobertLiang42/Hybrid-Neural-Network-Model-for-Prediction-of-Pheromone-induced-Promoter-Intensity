import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models

df = pd.read_csv('Table03.csv')
X = df['Promoter sequence'].values
Y = df['class'].values



for i in range(len(X)):
        X[i] = X[i] + '00000'
        while len(X[i]) < 2925+5:
            X[i] = X[i] + '0'


videos = np.zeros((1,2930,6,4,1))

for i in range(len(X)):
    array = np.zeros((5, 4, 1))
    frames = np.zeros((1, 6, 4, 1))

    for j in X[i]:

        if j == "a":
                array = np.append(array, [[[1], [0], [0], [0]]], axis=0)
                # print(np.shape(array)) #(6,4,1) 当前面有（5，4，1）打底的时候，array加第一个碱基刚好构成一帧（6，4，1）
        if j == "t":
                array = np.append(array, [[[0], [1], [0], [0]]], axis=0)
        if j == "c":
                array = np.append(array, [[[0], [0], [1], [0]]], axis=0)
        if j == "g":
                array = np.append(array, [[[0], [0], [0], [1]]], axis=0)
        if j == "0":
                array = np.append(array, [[[0], [0], [0], [0]]], axis=0)

        frame = array[np.newaxis, :] #将array升维（1，6，4，1）

        frames = np.append(frames, frame, axis=0)

        #去除头列，以添加新的尾列
        # array = array.reshape((6,4,1))
        array = np.delete(array, 0, axis=0)
        # print(np.shape(array))
    frames = np.delete(frames, 0, axis=0) #除掉空帧
    # print(np.shape(frames))
    video = frames[np.newaxis, :] #(1,2930,6,4,1)
    videos = np.append(videos, video, axis=0)
videos = np.delete(videos, 0, axis=0)



X = videos








from sklearn.model_selection import StratifiedKFold
k = 5
KFolds = StratifiedKFold(n_splits=k)
fold_counter = 0
result = []
proba = []

for train, test in KFolds.split(X, Y):
    fold_counter += 1
    print(f"fold #{fold_counter}")

    X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]



    model = models.Sequential([
        layers.ConvLSTM2D(filters=5, kernel_size=(6,4), activation='relu', padding='same', input_shape=(2930, 6, 4, 1), return_sequences = True),
        # layers.MaxPool2D(pool_size=(2, 2)),

        layers.BatchNormalization(),

        layers.Conv3D(filters=1, kernel_size=(1,1,1), activation="sigmoid", padding="same"),

        layers.Flatten(),
        # # layers.Dense(16, activation='relu'), #对比测试，提升不明显
        layers.Dense(1, activation='sigmoid')  # softmax一般默认10，只有两类，所以2
    ])

    model.summary()


    # compile model
    model.compile(optimizer='adadelta', #没有不行，不然不跑
                loss='binary_crossentropy',
                  metrics=['accuracy'])




    model.fit(X_train, Y_train, epochs=5, batch_size=5)



    evaluate = model.evaluate(X_test, Y_test)

    # history.loss_plot('epoch') #作acc-loss图

    Y_pred = model.predict(X_test) #predict_proba 被删除，现在就predict好用
    Y_score = [np.argmax(element) for element in Y_pred]


    Y_score = pd.DataFrame(Y_score, columns=['Pred'])
    Y_test = pd.DataFrame(Y_test, columns=['Real'])
    Y_proba = pd.DataFrame(Y_pred)

    Fold_result = pd.concat([Y_test, Y_score], axis=1)
    Fold_proba = pd.concat([Y_proba], axis=1)

    result.append(Fold_result)
    proba.append(Fold_proba)

all_result = pd.concat(result, axis=0)
all_proba = pd.concat(proba, axis=0)


