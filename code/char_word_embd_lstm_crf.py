# import packages
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv2D, Flatten, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPool1D, MaxPool1D
from sklearn.metrics import classification_report
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy, crf_accuracy, crf_marginal_accuracy
from keras.initializers import Constant
from keras.initializers import RandomUniform
from keras.utils import plot_model
import fasttext
import string

# import configurations from separated python file
from config import WORD_MAX_LEN, CHAR_MAX_LEN, VECTOR_DIM, TAGS, UNK, PAD

print('WORD_MAX_LEN: {}'.format(WORD_MAX_LEN))
# read pre-processed dataset
FILE_NAME = "normalized_dataset.txt"
mode = 'fasttext'
if_embd = True
cwd = os.getcwd()


def flatten_2d_list(main_list):
    return [item for sublist in main_list for item in sublist]


# the list 'data' contains sublists where each sublist is a lower-cased sentence.
# the list 'labels' contain sublists where each sublist of labels corresponding to each token in sentence
data = []
labels = []
with open(os.path.join(cwd, FILE_NAME), 'r', encoding='utf-8') as file:
    sent = []
    sent_tag = []
    for line in file:
        try:
            if line.strip() != "":
                tok_tag_tuple = line.strip().split('\t')
                # if there is any unexpected tag we ignore it

                if "DSC" in tok_tag_tuple[1] or "APL" in tok_tag_tuple[1] or "RXN" in tok_tag_tuple[1]:
                    tok_tag_tuple[1] = "O"
                if tok_tag_tuple[1] not in TAGS:
                    continue
                token, tag = tok_tag_tuple[0], tok_tag_tuple[1]
                sent += [token]
                sent_tag += [tag]
            else:
                data += [sent]
                labels += [sent_tag]
                sent = []
                sent_tag = []
        except:
            continue

data = [[' '.join(i)] for i in data]

data = [[''.join([i if not i.isdigit() else '0' for i in s])] for s in data]
data = [sent[0].split(' ') for sent in data]

# get the total number and the list of unique lower-cased tokens
tokens = list(dict.fromkeys(flatten_2d_list(data)))
n_tokens = len(tokens)
# get the number of labels in the dataset
n_tags = len(TAGS)

# set index to each token
token2idx = {w: i + 2 for i, w in enumerate(tokens)}
token2idx[UNK] = 1
token2idx[PAD] = 0
idx2token = {i: w for w, i in token2idx.items()}

tag2idx = {t: i + 1 for i, t in enumerate(TAGS)}
tag2idx[PAD] = 0
idx2tag = {i: w for w, i in tag2idx.items()}

X_words = [[token2idx[w] for w in s] for s in data]

X_words = pad_sequences(maxlen=WORD_MAX_LEN, sequences=X_words, value=token2idx["PAD"],
                        padding="post", truncating="post")

chars = set([w_i for w in tokens for w_i in w])
n_chars = len(chars)  # there are 117 unique chars

char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx[UNK] = 1
char2idx[PAD] = 0

X_chars = []
for sentence in data:
    sent_seq = []
    for i in range(WORD_MAX_LEN):
        word_seq = []
        for j in range(CHAR_MAX_LEN):
            try:
                word_seq.append(char2idx.get(sentence[i][j]))
            except:
                word_seq.append(char2idx.get(PAD))
        sent_seq.append(word_seq)
    X_chars.append(np.array(sent_seq))

y = [[tag2idx[tag] for tag in label] for label in labels]
y = pad_sequences(maxlen=WORD_MAX_LEN, sequences=y, value=tag2idx[PAD], padding="post", truncating="post")

# import fasttext embd
fasttext_model = fasttext.load_model("./embds/fasttext_uncased_freq10_30_ws15_epoch15.bin")


def load_pretrained_embeddings_fasttext(words, model, token2idx):
    embedding_matrix = np.zeros((len(words) + 2, 300))
    for word in words:
        embedding_matrix[token2idx[word]] = model.get_word_vector(word)
    embedding_matrix[token2idx[UNK]] = np.random.randn(300)
    embedding_matrix[token2idx[PAD]] = np.random.randn(300)
    return embedding_matrix


def load_pretrained_embeddings_w2v(words, file_name, token2idx):
    embedding_matrix = np.zeros((len(words) + 2, 200))
    with open(file_name, 'r', encoding='utf-8') as f:
        c = 0
        for line in f:
            if c == 0:
                continue
            word, vec = line.split(' ', maxsplit=1)
            vec = np.fromstring(vec, 'f', sep=' ')
            if word in words:
                embedding_matrix[token2idx[word]] = vec
            c += 1
    embedding_matrix[token2idx[UNK]] = np.random.randn(200)
    embedding_matrix[token2idx[PAD]] = np.random.randn(200)
    return embedding_matrix


if mode == 'fasttext':
    embedding_matrix = load_pretrained_embeddings_fasttext(tokens, fasttext_model, token2idx)
if mode == 'w2v':
    embedding_matrix = load_pretrained_embeddings_w2v(tokens, 'patent_w2v.txt', token2idx)

X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_words, y, test_size=0.2, random_state=2021)
X_char_tr, X_char_te, _, _ = train_test_split(X_chars, y, test_size=0.2, random_state=2021)

# input and word embedding
word_in = Input(shape=(WORD_MAX_LEN,))
if mode == 'fasttext':
    emb_word = Embedding(input_dim=n_tokens + 2, output_dim=VECTOR_DIM,
                         weights=[embedding_matrix], trainable=True)(word_in)
if mode == 'w2v':
    emb_word = Embedding(input_dim=n_tokens + 2, output_dim=200,
                         weights=[embedding_matrix], trainable=True)(word_in)
# input and char embedding
char_in = Input(shape=(WORD_MAX_LEN, CHAR_MAX_LEN,))

emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=30,
                                     input_length=CHAR_MAX_LEN,
                                     embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)))(char_in)
char_enc = TimeDistributed(Conv1D(kernel_size=2, filters=10, padding='same', activation='tanh', strides=1))(emb_char)
maxpool_char = TimeDistributed(MaxPool1D(CHAR_MAX_LEN))(char_enc)
char = TimeDistributed(Flatten())(maxpool_char)
'''
# char LSTM to get word encoding by characters
char_enc = TimeDistributed(LSTM(units=20, return_sequences=False))(emb_char)
'''
if if_embd == True:
    x = concatenate([emb_word, char])
    # x = SpatialDropout1D(0.1)(x)
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True))(x)
    # dense_layer = TimeDistributed(Dense(16,activation='tanh'))(main_lstm)
    crf = CRF(n_tags + 1, sparse_target=True)
    out = crf(main_lstm)
    # out = TimeDistributed(Dense(n_tags + 1, activation="sigmoid"))(main_lstm)
    model = Model([word_in, char_in], out)
    # compile the model
    model.compile(optimizer="nadam", loss=crf_loss, metrics=[crf_viterbi_accuracy])
    # model.summary()
    # plot_model(model,to_file='0411.png',show_shapes=False, show_layer_names=True,rankdir='TB')
    history = model.fit([X_word_tr,
                         np.array(X_char_tr).reshape((len(X_char_tr), WORD_MAX_LEN, CHAR_MAX_LEN))],
                        np.array(y_tr).reshape(len(y_tr), WORD_MAX_LEN, 1),
                        batch_size=32, epochs=15, verbose=1)
else:
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True))(emb_word)
    crf = CRF(n_tags + 1, sparse_target=True)
    out = crf(main_lstm)
    model = Model([word_in], out)
    # compile the model
    model.compile(optimizer="nadam", loss=crf_loss, metrics=[crf_viterbi_accuracy])
    history = model.fit(X_word_tr,
                        np.array(y_tr).reshape(len(y_tr), WORD_MAX_LEN, 1),
                        batch_size=32, epochs=15, verbose=1)
if if_embd == True:
    # get the performance report
    test_pred = model.predict([X_word_te,
                               np.array(X_char_te).reshape((len(X_char_te), WORD_MAX_LEN, CHAR_MAX_LEN))], verbose=1)
else:
    test_pred = model.predict([X_word_te], verbose=1)
test_pred_class = np.argmax(test_pred, axis=-1)
pred_class = [idx2tag[int(i)] for i in flatten_2d_list(test_pred_class)]
true_class = [idx2tag[int(i)] for i in flatten_2d_list(y_te)]

pred_class2 = []
for i in pred_class:
    if 'I-' in i:
        i = i.replace('I-', '')
    if 'B-' in i:
        i = i.replace('B-', '')
    pred_class2 += [i]

true_class2 = []
for j in true_class:
    if 'I-' in j:
        j = j.replace('I-', '')
    if 'B-' in j:
        j = j.replace('B-', '')
    true_class2 += [j]

print(classification_report(true_class2, pred_class2))

import pandas as pd

y_actu = pd.Series(true_class2, name='Actual')
y_pred = pd.Series(pred_class2, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)

import matplotlib.pyplot as plt


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)  # imshow
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)


plot_confusion_matrix(df_confusion)

te_str = ""
for j in range(len(X_word_te)):
    te_sent = X_word_te[j]
    for i in range(len(te_sent)):
        if y_te[j][i] != 0 and y_te[j][i] != 8:
            te_str += '[' + idx2token[te_sent[i]] + ']' + ' '
        if test_pred_class[j][i] != 0 and test_pred_class[j][i] != 8:
            te_str += '{' + idx2token[te_sent[i]] + '}' + ' '
        if idx2token[te_sent[i]]!='PAD':
            te_str += idx2token[te_sent[i]]+ ' '
    te_str += '\n'

with open('eval.txt','w',encoding='utf-8') as f:
    f.write(te_str)