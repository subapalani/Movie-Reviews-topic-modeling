import sys, string, re
import csv
import numpy as np
from numpy import zeros
from scipy import linalg,array,dot,mat,transpose
from math import log
from numpy import asarray, sum
from collections import Counter
import math

with open('C:/Users/Ramkumar.Ramkumar-Lappy/Desktop/Topic Modeling Using Python/data.csv', 'rb') as csvfile:
    
     data = csv.reader(csvfile)
     textlist1 = list(data)
     
textlist=[]

for t in textlist1:
    processedtext=''.join(t).lower()
    processedtext = re.sub("[^a-z A-Z]","",processedtext)
    textlist.append(processedtext)
    
with open('C:/Users/Ramkumar.Ramkumar-Lappy/Desktop/Topic Modeling Using Python/notes.txt','r') as f:
    stopwords = [x.strip('\n') for x in f.readlines()]

def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]
textlist = removeStopwords(textlist,stopwords)

for t in textlist:
    c = Counter()
    for word in t.split():
        c[word] +=1

def dictionary(corpus):
    dict = set()
    for doc in corpus:
        dict.update([word for word in doc.split()])
    return dict
    
def term_frequency(term, document):
  return frequency(term, document)

def frequency(term, document):
  return document.split().count(term)

dictoftext = dictionary(textlist)

dtm =[]

for doc in textlist:
    tf_vector = [term_frequency(word, doc) for word in dictoftext]
    dtm.append(tf_vector)
    print(tf_vector)

def no_doc(term, textlist):
    document_count = 0
    for document in textlist:
        if frequency(term, document) > 0:
            document_count +=1
            
    return document_count
    
  

def idf(term, textlist):
    number_docs = len(textlist)
    df = no_doc(term, textlist)
    return np.log(number_docs / 1+df)

idf_vector = [idf(word,textlist) for word in dictoftext]



def idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat

idf_matrix1 =  idf_matrix(idf_vector)


dtm_tfidf = []


for tf_vector in dtm:
    dtm_tfidf.append(np.dot(tf_vector,idf_matrix1))

def normalization(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]
    
normalizeddtm_tfidf = []
for t in dtm_tfidf:
    normalizeddtm_tfidf.append(normalization(t))
print(normalizeddtm_tfidf)

u, s, vt = linalg.svd(normalizeddtm_tfidf)
k = 15
u = u[:, :k]
sigma = linalg.diagsvd(s[:k],k,k)
vt = vt[:k, :]
vtk = vt[:k] 
uk = transpose(transpose(u)[:k])
finalmatrix=dot(uk,sigma), transpose(dot(sigma,vtk))

for term in textlist :
    print term
for i in range(u.shape[0]):
    print " ".join([str(v) for v in u[i].tolist()])
print " ".join([str(v) for v in s.tolist()])
for i in range(vt.shape[0]):
    print " ".join([str(v) for v in vt[i].tolist()])
    
print(finalmatrix)
print(normalizeddtm_tfidf)
print(dictoftext)