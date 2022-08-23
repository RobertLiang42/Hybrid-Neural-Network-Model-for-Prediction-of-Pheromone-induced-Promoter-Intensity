import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models

# 读取数据
df = pd.read_csv('Table03.csv')
X = df['Promoter sequence'].values
Y = df['class'].values



for i in range(len(X)):
        X[i] = X[i] + '00000'
        while len(X[i]) < 2925+5:
            X[i] = X[i] + '0'

arrays = np.zeros((1, 2935, 4, 1))
for i in range(len(X)):
    array = np.zeros((5, 4, 1))
    for j in X[i]:
        if j == "a":
                array = np.append(array, [[[1], [0], [0], [0]]], axis=0)
        if j == "t":
                array = np.append(array, [[[0], [1], [0], [0]]], axis=0)
        if j == "c":
                array = np.append(array, [[[0], [0], [1], [0]]], axis=0)
        if j == "g":
                array = np.append(array, [[[0], [0], [0], [1]]], axis=0)
        if j == "0":
                array = np.append(array, [[[0], [0], [0], [0]]], axis=0)
    array = array[np.newaxis, :]
    arrays = np.append(arrays, array, axis=0)
arrays = np.delete(arrays, 0, axis=0)
X = arrays


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
        layers.Conv2D(filters=8, kernel_size=(6, 4), activation='relu', padding='same', input_shape=(2935, 4, 1)),
        layers.MaxPool2D(pool_size=(2, 2)),

        layers.Flatten(),
        # layers.Dense(16, activation='relu'), #对比测试，提升不明显
        layers.Dense(2, activation='softmax')  # softmax一般默认10，只有两类，所以2
    ])

    model.summary()


    # compile model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


    model.fit(X_train, Y_train, epochs=5)



    evaluate = model.evaluate(X_test, Y_test)


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






import scikitplot as skplt
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")



Y_test_all = all_result['Real'].values
Y_pred_all= all_result['Pred'].values
Y_probs_all = all_proba.values




# confusion matrix#################################################################
fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
skplt.metrics.plot_confusion_matrix(Y_test_all, Y_pred_all,
                                    title="Confusion Matrix",
                                    cmap="Oranges",
                                    ax=ax1)
ax2 = fig.add_subplot(122)
skplt.metrics.plot_confusion_matrix(Y_test_all, Y_pred_all,
                                    normalize=True,
                                    title="Confusion Matrix",
                                    cmap="Purples",
                                    ax=ax2);


skplt.metrics.plot_roc_curve(Y_test_all, Y_probs_all,
                       title="ROC Curve", figsize=(12,6));

skplt.metrics.plot_precision_recall_curve(Y_test_all, Y_probs_all,
                       title="Precision-Recall Curve", figsize=(12,6))

skplt.metrics.plot_ks_statistic(Y_test_all, Y_probs_all, figsize=(10,6));

skplt.metrics.plot_lift_curve(Y_test_all, Y_probs_all, figsize=(10,6));0