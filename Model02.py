import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding




df = pd.read_csv('Table03.csv')
X = df['Promoter sequence'].values
Y = df['class'].values

import Igem_Sub_Mute as mt
X = mt.pre(X)

sentences = []
for i in range(len(X)):
    keys = list(X[i].keys())  # 提取所有的键,为一个列表

    sentence = []

    while len(keys) != 0:  # 当keys不是空集
        min = keys[0]  # 从第一个开始，比较找最小值 每次重置
        for j in range(len(keys)):
            if keys[j] < min:
                min = keys[j]

        # 将prebox从序列里提取出来，按顺序作为词汇装进sentence
        sentence.append(X[i][min])
        min_index = keys.index(min)
        del keys[min_index]

    # sentences 表示共213个句子
    sentences.append(sentence)

WVmodel = gensim.models.Word2Vec(
        window=10,
        min_count=1,
        workers=8,
    )

WVmodel.build_vocab(sentences, progress_per=5)

WVmodel.train(sentences, total_examples=WVmodel.corpus_count, epochs=10)

WVmodel.save("./word2vec-prebox-sequences.model")

w2v = gensim.models.Word2Vec.load("word2vec-prebox-sequences.model")
vocab = w2v.wv.vocab

all_words = ['tgaaac', 'gtttca', 'agaaac', 'cgaaac', 'ggaaac', 'gtttct', 'gtttcg', 'gtttcc', 'taaaac', 'ttaaac',
                 'tcaaac', 'gtttta', 'gtttaa', 'gtttga', 'tgtaac', 'tgcaac', 'tggaac', 'gttaca', 'gttgca', 'gttcca',
                 'tgatac', 'tgacac', 'tgagac', 'gtatca', 'gtgtca', 'gtctca', 'tgaatc', 'tgaacc', 'tgaagc', 'gattca',
                 'ggttca', 'gcttca', 'tgaaaa', 'tgaaat', 'tgaaag', 'ttttca', 'atttca', 'ctttca', ]

docs = sentences  # 统一变量名

t = Tokenizer()
t.fit_on_texts(docs)

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(docs)
max_length = 43

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


# 根据Word2Vec，得到权重矩阵
def get_weight_matrix():
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, w2v.vector_size))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for i in range(len(all_words)):
        weight_matrix[i + 1] = w2v[all_words[i]]
    return weight_matrix

embedding_vectors = get_weight_matrix()
emb_layer = Embedding(vocab_size, output_dim=w2v.vector_size, weights=[embedding_vectors], input_length=43, trainable=False)


model = keras.Sequential()
model.add(emb_layer)


model.add(
    layers.Bidirectional(
            layers.GRU(16, return_sequences=True, activation='tanh')
        )
)
model.add(
    layers.Bidirectional(
        layers.GRU(16, activation='tanh')
    )
)

# model.add(
#     layers.Bidirectional(
#             layers.LSTM(8, return_sequences=True, activation='tanh')
#         )
# )
# model.add(
#     layers.Bidirectional(
#         layers.LSTM(8, activation='tanh')
#     )
# )


model.add(layers.Dense(2))

print(model.summary())

model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

from sklearn.model_selection import StratifiedKFold

KFolds = StratifiedKFold(n_splits=5)  # 5 fold
fold_counter = 0
result = []
proba = []

for train, test in KFolds.split(padded_docs, Y):
    fold_counter += 1
    print(f"fold #{fold_counter}")

    from Igem_Sub_LossHistory import LossHistory

    history = LossHistory()

    X_train, X_test, Y_train, Y_test = padded_docs[train], padded_docs[test], Y[train], Y[test]

    # from keras import backend as K
    # X_train = K.cast_to_floatx(X_train)
    # Y_train = K.cast_to_floatx(Y_train)

    model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=2, callbacks=[history])

    # history.loss_plot('epoch') #作acc-loss图

    evaluate = model.evaluate(X_test, Y_test)

    Y_pred = model.predict(X_test)

    Y_score = [np.argmax(element) for element in Y_pred]

    # from sklearn.metrics import confusion_matrix, classification_report
    # import warnings
    #
    # warnings.simplefilter('ignore')
    # print("Classification Report: \n", classification_report(Y_test, Y_score))

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

import sys
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

# skplt.metrics.plot_cumulative_gain(Y_test_all,Y_probs_all, figsize=(10,6));

skplt.metrics.plot_lift_curve(Y_test_all, Y_probs_all, figsize=(10,6));0



#     Y_score = pd.DataFrame(Y_score, columns=['Pred'])
#     Y_test = pd.DataFrame(Y_test, columns=['Real'])
#     Fold_result = pd.concat([Y_test, Y_score], axis=1)
#     result.append(Fold_result)
#
#
# BiGRU_result = pd.concat(result, axis=0)
# BiGRU_result.to_csv('BiGRU_result.csv')

# import Igem_Sub_ROC as roc
# roc.ROC('BiGRU_result.csv', 'Bidirctional GRU')