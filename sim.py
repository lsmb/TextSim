#!/bin/python3

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize

#import nltk

from sentence_transformers import SentenceTransformer
#nltk.download('punkt')

filebeechly = open('beechly.txt',mode='r')
beechly = filebeechly.read()

people = []
peoplenames = []

import glob, os
os.chdir("./")
for file in glob.glob("*.txt"):
    if "beechly" not in file:
        print(file)
        temp = open(file, mode='r').read()
        people.append(temp)
        peoplenames.append(file.replace('.txt',''))

base_document = beechly
documents = people

def addnames(a, b):
    return (b, a)

def bert_similarity():
        model = SentenceTransformer('bert-base-nli-mean-tokens')

        sentences = sent_tokenize(base_document)
        base_embeddings_sentences = model.encode(sentences)
        base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)

        vectors = []
        for i, document in enumerate(documents):

                sentences = sent_tokenize(document)
                embeddings_sentences = model.encode(sentences)
                embeddings = np.mean(np.array(embeddings_sentences), axis=0)

                vectors.append(embeddings)

                print("making vector at index:", i)

        scores = cosine_similarity([base_embeddings], vectors).flatten()

        highest_score = 0
        highest_score_index = 0
        for i, score in enumerate(scores):
                if highest_score < score:
                        highest_score = score
                        highest_score_index = i

        most_similar_document = documents[highest_score_index]
        print("Most similar person by BERT:", peoplenames[highest_score_index], ': ',
              highest_score)
        x = map(addnames, scores, peoplenames)
        print(list(x))

bert_similarity()
