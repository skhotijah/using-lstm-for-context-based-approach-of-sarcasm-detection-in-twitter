#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/reverse-balance-8700/rb 8700.csv',delimiter=',',encoding='latin-1')
#df['Tweet_Reverse'] = df.loc[:,'Tweet'].apply(lambda x: x[::-1])
df = df[['Tweet','Kelas']]
df = df[pd.notnull(df['Tweet'])]
df.rename(columns = {'Tweet':'Tweet'}, inplace = True)
df.head()


# In[ ]:


df.shape


# In[ ]:


df.index = range(8700)
df['Tweet'].apply(lambda x: len(x.split(' '))).sum()


# In[ ]:


cnt_pro = df['Kelas'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Kelas', fontsize=12)
plt.xticks(rotation=90)
plt.show();


# In[ ]:


def print_complaint(index):
    example = df[df.index == index][['Tweet', 'Kelas']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Tweet:', example[1])
print_complaint(12)


# In[ ]:


print_complaint(0)


# Text Preprocessing Below we define a function to convert text to lower-case and strip punctuation/symbols from words and so on.

# In[ ]:


from bs4 import BeautifulSoup
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
df['Tweet'] = df['Tweet'].apply(cleanText)


# In[ ]:


df['Tweet'] = df['Tweet'].apply(cleanText)
train, test = train_test_split(df, test_size=0.000001 , random_state=42)
import nltk
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            #if len(word) < 0:
            if len(word) <= 0:
                continue
            tokens.append(word.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Tweet']), tags=[r.Kelas]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Tweet']), tags=[r.Kelas]), axis=1)

# The maximum number of words to be used. (most frequent)
max_fatures = 500000

# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 50

#tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Tweet'].values)
X = tokenizer.texts_to_sequences(df['Tweet'].values)
X = pad_sequences(X)
print('Found %s unique tokens.' % len(X))


# In[ ]:


X = tokenizer.texts_to_sequences(df['Tweet'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


# In[ ]:


print(X[0])


# In[ ]:


print(X[1])


# In[ ]:


print(X[2])


# In[ ]:


train_tagged


# In[ ]:


#train_tagged.values[2173]
train_tagged.values


# In[ ]:


kata=train_tagged.values
print(kata[2170:2190])


# In[ ]:


#train_tagged[2173]
train_tagged


# In[ ]:


test_tagged.values


# ## Model DM = 1
# 
# This work use DM=1 (it preserve word order)
# 
# 

# In[ ]:


d2v_model = Doc2Vec(dm=1, dm_mean=1, size=20, window=8, min_count=1, workers=1, alpha=0.065, min_alpha=0.065)
d2v_model.build_vocab([x for x in tqdm(train_tagged.values)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    d2v_model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)\n    d2v_model.alpha -= 0.002\n    d2v_model.min_alpha = d2v_model.alpha')


# In[ ]:


print(d2v_model)


# In[ ]:


len(d2v_model.wv.vocab)


# ##  save the vectors in a new matrix

# In[ ]:


#print(word_model) 
# save the vectors in a new matrix
#embedding_matrix = np.zeros((len(model.wv.vocab)+ 1, 100))
#embedding_matrix = np.ones((len(model.wv.vocab)+ 1, 100))
#nonzero ndarray
embedding_matrix = np.zeros((len(d2v_model.wv.vocab)+ 1, 20))

for i, vec in enumerate(d2v_model.docvecs.vectors_docs):
    while i in vec <= 1000:
    #print(i)
    #print(model.docvecs)
          embedding_matrix[i]=vec
    #print(vec)
    #print(vec[i])


# ## Measuring distance between two vectors (related to cosine similarity)

# In[ ]:


d2v_model.wv.most_similar(positive=['optimis','sabar'], topn=50)


# In[ ]:


d2v_model.wv.doesnt_match(['selalu', 'optimis','dan','sabar','pak','jokowi'])


# In[ ]:


d2v_model.wv.doesnt_match(['selamat', 'ulang', 'tahun', 'bagi', 'negara', 'khusus', 'islam', ',', 'indonesia', '.', 'semoga', 'ke', 'depannya', 'bisa', 'semakin', 'menyudutkan', 'rakyat', 'non-muslim'])


# In[ ]:


d2v_model.wv.most_similar(positive=['jokowi'], topn=35)


# ## Plot Similarity word in Doc2vec

# In[ ]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in d2v_model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=500, random_state=42)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[ ]:


tsne_plot(d2v_model)


# ## KeyedVectors : Store and query word vectors
# 
# You can perform various syntactic/semantic NLP word tasks with the trained vectors. Some of them are already built-in

# In[ ]:


d2v_model.wv.vocab


# # Model LSTM
# 
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding


# init layer
model = Sequential()

# emmbed word vectors
model.add(Embedding(len(d2v_model.wv.vocab)+1,20,input_length=X.shape[1],weights=[embedding_matrix],trainable=True))

# learn the correlations
def split_input(sequence):
     return sequence[:-1], tf.reshape(sequence[1:], (-1,1))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(2,activation="softmax"))

# output model skeleton
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['acc'])


# In[ ]:


Y = pd.get_dummies(df['Kelas']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


batch_size = 32
model.fit(X_train, Y_train, epochs =20, batch_size=batch_size, verbose = 2)


# ### validation

# In[ ]:


validation_size = 860

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)
#score,acc = model.evaluate(X_test, Y_test)

print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


# ###  Evaluate the model (train split 90:10)
# The test from 90:10 split data should give as only the quality of the model.

# In[ ]:


# evaluate the model
_, train_acc = model.evaluate(X_train, Y_train, verbose=2)
_, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print('Train: %.3f, Test: %.4f' % (train_acc, test_acc))


# In[ ]:


# predict probabilities for test set (validation)
yhat_probs = model.predict(X_test, verbose=0)
#print(yhat_probs)
# predict crisp classes for test set(validation)
yhat_classes = model.predict_classes(X_test, verbose=0)
#print(yhat_classes)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
#yhat_classes = yhat_classes[:, 1]


# In[ ]:


import numpy as np
rounded_labels=np.argmax(Y_test, axis=1)
rounded_labels


# ### The confusion matrix

# In[ ]:


# The confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

lstm_val = confusion_matrix(rounded_labels, yhat_classes)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(lstm_val, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('LSTM Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()


# # Real Test 
# 
# We compared a more realistic test accuracy of the model using new and different data set aside from the data to build the model.

# In[ ]:


new_test = ["#4niesbebalsokhebat tanggul. jebol perbaiki tuk anda menunggu patungan harus sampai ya? hebat @aniesbaswedan anda warga "] #reverse tweet 
seq = tokenizer.texts_to_sequences(new_test)

padded = pad_sequences(seq, maxlen=X.shape[1], dtype='int32', value=0)

pred = model.predict(padded)

# 0:Non. 1: Sarcasm
labels = ['0','1']

#print vektor asli
print(seq)

#print yang di pad
print(padded)

#print prediction
print(pred, labels[np.argmax(pred)])


# Test 120 Tweets (in which 56 are sarcastic and the other are not). you can find it on test folder : https://github.com/skhotijah/using-lstm-for-context-based-approach-of-sarcasm-detection-in-twitter/tree/main/test

# In[ ]:


model.predict_classes(
    pad_sequences(
         tokenizer.texts_to_sequences(
             [ ". dicintainya pantas yang siapa tau lebih akan terluka pernah yang cewek tetapi . seseorang mencintai bagaimana tau pasti pintar yang cewek ",
                "hehe flashmopnya dikit ikutan juga tadi gabung untuk juga serta ikut yang warga banyak rame kegiatanya #rismaselamanya pendukung simpatisan gerakan ada pagi tadi bungkul taman cfd ",
                 "#4niesbebalsokhebat alhamdulilah syukur @wagimandeep ",
                "#4niesbebalsokhebat .. siapa klaim yang siapa kerja yang ... ya hebat memang @aniesbaswedan bapak ",
                "#4niesbebalsokhebat ..??????????????? hebat..aduuuh merasa susudah doang ngomong bisa hanya ya kerjakan di bisa tidak tapi ceritakan anies bisa semua lewat ilmunya lain orang .... hebat memang anies @73hermantjoe "]),
         maxlen=MAX_SEQUENCE_LENGTH))


# In[ ]:


model.save('mycoolmodel.h5')

