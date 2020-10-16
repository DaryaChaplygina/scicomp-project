import numpy as np
import pandas as pd
import pysam

from tensorflow.keras import models, layers, losses, preprocessing as kprocessing
from tensorflow.keras import backend as K
import gensim

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

import matplotlib.pyplot as plt

ref = pysam.FastaFile('male.hg19.fa')

def fetch_seq(start, end, region):
    seq = ref.fetch(start=start - 1,
                    end=end,
                    region=region)
    return seq.lower()

def split_to_kmers(x, k=5):
    return [x[i:min(i+k, len(x))] for i in range(0, len(x), k)]

def upsample(data, label, n_times=6, rs=0):
    minor_class = data[data.label == label]
    minor_class = minor_class.sample(minor_class.shape[0] * n_times, 
                                     replace=True, random_state=rs)
    new_data = pd.concat((data, minor_class)).sample(frac=1, 
                                                     random_state=rs) # concat and shuffle the data
    return new_data

class dna2vec:
    def __init__(self, s=20, min_count=2, window=2):
        self.size = s
        self.w2v = gensim.models.Word2Vec(size=s, 
                                            window=window, 
                                            min_count=min_count, 
                                            workers=7)
    
    def fit(self, corpus):
        self.w2v.build_vocab(corpus)
        self.w2v.train(corpus,
                       total_examples=self.w2v.corpus_count,
                       epochs=30,
                       report_delay=1)
        self.w2v.init_sims(replace=True)
        
def transform_to_avg_vec(corpus, model, name):
    s = model.size
    vectors = np.array([
            np.mean([model.w2v.wv[w] for w in sent if w in model.w2v.wv] or [np.zeros(s)], 
                    axis=0)
            for sent in corpus
        ])
    dataset = dict(zip([f'{name}_{i}' for i in range(s)], vectors.T))
    return pd.DataFrame(dataset)

class dna_tokenizer:
    def __init__(self, sent_size=500):
        self.size = sent_size
        self.tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', 
                                                    oov_token="NaN")
        
    def fit(self, corpus):
        self.tokenizer.fit_on_texts(list(map(lambda x: ' '.join(x),
                                             corpus)))
        dic_vocabulary = self.tokenizer.word_index
        
    def transform(self, corpus):
        lst_text2seq = self.tokenizer.texts_to_sequences(
            list(map(lambda x: ' '.join(x), corpus))
        )
        res = kprocessing.sequence.pad_sequences(lst_text2seq, 
                    maxlen=self.size, padding="post", truncating="post")
        return res
    
    def vocabulary(self):
        return self.tokenizer.word_index
    
def get_embeddings_matrix(w2v, tokenizer):
    embeddings = np.zeros((len(tokenizer.vocabulary())+1, w2v.size))
    for word, idx in tokenizer.vocabulary().items():
        try:
            embeddings[idx] =  w2v.w2v.wv[word]
        except:
            pass
    return embeddings

def filter_seq_length(df, l=2500):
    return df[(df.promoter_seq.apply(len) <= l) &
             (df.enhancer_seq.apply(len) <= l)]

def get_model(emb_prom, emb_enh, sent_size=500, activation='relu', n_filters=4, kernel_size=100):
    
    # input
    proms = layers.Input(shape=(sent_size,), name='promoters')
    enhs = layers.Input(shape=(sent_size,), name='enhancers')
    
    # embedding&convolution for promoters
    x_prom = layers.Embedding(input_dim=emb_prom.shape[0],  
                              output_dim=emb_prom.shape[1], 
                              weights=[emb_prom],
                              input_length=sent_size, trainable=False)(proms)
    x_prom = layers.Conv1D(filters=n_filters, kernel_size=kernel_size, activation=activation)(x_prom)
    x_prom = layers.Conv1D(filters=n_filters, kernel_size=kernel_size, activation=activation)(x_prom)
    x_prom = layers.Flatten()(x_prom)
    prom_model = models.Model(proms, x_prom)

    # embedding&convolution for enhancers
    x_enh = layers.Embedding(input_dim=emb_enh.shape[0],  
                         output_dim=emb_enh.shape[1], 
                         weights=[emb_enh],
                         input_length=sent_size, trainable=False)(enhs)
    x_enh = layers.Conv1D(filters=n_filters, kernel_size=kernel_size, activation=activation)(x_enh)
    x_enh = layers.Conv1D(filters=n_filters, kernel_size=kernel_size, activation=activation)(x_enh)
    x_enh = layers.Flatten()(x_enh)
    enh_model = models.Model(enhs, x_enh)

    # convolution for enhancers and promoters together
    combined = layers.Concatenate()([prom_model.output, enh_model.output])
    x = layers.Reshape((2, n_filters * (sent_size - kernel_size*2 + 2), 1))(combined)
    x = layers.Conv2D(filters=n_filters, kernel_size=(2, kernel_size), activation=activation)(x)
    x = layers.Conv2D(filters=n_filters, kernel_size=(1, kernel_size), activation=activation)(x)

    # dense layers to get the final result
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation=activation)(x)
    y_out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model([prom_model.input, enh_model.input], y_out)
    model.compile(loss=losses.BinaryCrossentropy(),
                  optimizer='adam', metrics=['accuracy'])

    return model

def fit_nn(model, x, y, valid=None, epochs=5):
    f1_scores = []
    acc = []
    for i in range(epochs):
        history = m.fit(x=x,
                        y=y,
                        validation_data=valid,
                        epochs=1, batch_size=8)
        
        train_pred = m.predict(x)
        
        if valid is not None:
            acc.append((history.history['acc'], history.history['val_acc']))
            valid_pred = m.predict(valid[0])
            f1_scores.append((f1_score(y, train_pred > 0.5), 
                             f1_score(valid[1], valid_pred > 0.5)))
        else:
            acc.append(history.history['acc'])
            f1_scores.append(f1_score(y, train_pred > 0.5))             
    return model, f1_scores, acc