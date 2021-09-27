# import necessary modules
from nltk.tokenize import sent_tokenize
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
import gensim
from gensim.models import Word2Vec
import pandas as pd
import os
import re
import logging
from datetime import datetime
import fasttext

def main():
    # create log file to record running process
    logging.basicConfig(filename='run.log', filemode='w', level=logging.DEBUG)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("############### Script Started ###############")
    logging.info("Started at {}".format(dt_string))
    # set the directory of this py file as current working directory
    cwd = os.path.dirname(os.path.realpath(__file__))
    print("Working directory: {}".format(str(cwd)))
    # get corpus file collected from elsevier
    df_corpus = pd.read_csv(os.path.join(cwd, 'complete_corpus_elsevier.csv'))
    # text content in the corpus
    abstracts = list(df_corpus['abstract'])
    # process text content
    string_processing(cwd, abstracts, cased=True)


def string_processing(cwd, abstracts, cased=False):
    str_abstracts = ' '.join(abstracts)
    str_abstracts = re.sub(r"\©(.*?)\.", "", str_abstracts)

    str_abstracts = ''.join([i if not i.isdigit() else '0' for i in str_abstracts])
    if not cased:
        str_abstracts = str_abstracts.lower()
        with open(os.path.join(cwd, 'uncased_corpus_elsevier.txt'), 'w', encoding='utf-8') as file:
            file.write(str_abstracts)
    else:
        with open(os.path.join(cwd, 'cased_corpus_elsevier.txt'), 'w', encoding='utf-8') as file:
            file.write(str_abstracts)
    return str_abstracts


def data_tokenization(str_abstracts):
    # use the Chemwordtokenizer to perform tokenization process
    cwt = ChemWordTokenizer()
    data = []
    # parse paragraphs to sentences first
    sentenized_abstracts = sent_tokenize(str_abstracts)
    # then tokenize each sentence
    for sentence in sentenized_abstracts:
        tokenized_sent = []
        for tok in cwt.tokenize(sentence):
            tokenized_sent.append(tok)
        data.append(tokenized_sent)
    return data


def w2v_model_generation(data, ver='sg'):
    # if skipgram is selected:
    if ver == 'sg':
        # create Skipgram model
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("start generating skipgram model at {}".format(dt_string))
        w2v_sg = gensim.models.Word2Vec(data, min_count=10, vector_size=300, sg=1, window=15, workers=4, epochs=15)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("finish generating skipgram model at {}".format(dt_string))
        w2v_sg.save('skipgram_w2v_ws15.model')
        w2v_sg.save('skipgram_w2v_binary_ws15.bin')
        # save keyedvector - skipgram
        word_vectors_sg = w2v_sg.wv
        word_vectors_sg.save('sg_vectors_ws15.kv')
    if ver == 'cbow':
        # create CBOW model
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("start generating cbow model at {}".format(dt_string))
        w2v_cbow = gensim.models.Word2Vec(data, min_count=10, vector_size=300, window=15, workers=4, epochs=15)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("end generating skipgram model at {}".format(dt_string))
        w2v_cbow.save('w2v_cbow_ws15.model')
        w2v_cbow.save('w2v_cbow_binary_ws15.bin')
        # save keyedvector - cbow
        word_vectors_cbow = w2v_cbow.wv
        word_vectors_cbow.save('cbow_vectors_ws15.kv')

def fasttext_embd_generation(cased=False):
    if cased:
        fasttext_model = fasttext.train_unsupervised('tokenized_cased_corpus.txt', 'skipgram',minn=2,maxn=30,dim=300)
        fasttext_model.save_model('fasttext_cased.bin')
    else:
        fasttext_model = fasttext.train_unsupervised('tokenized_uncased_corpus.txt', 'skipgram',minn=2,maxn=30,dim=300,ws=15,minCount=10,epoch=15,thread=4)
        fasttext_model.save_model('fasttext_uncased_freq10_30_ws15_epoch15.bin')

if __name__ == "__main__":
    main()
