# set the params 
# set params['values_objfile']="./values" the file which has pickled class Values
# if want to run on new data :
    # set params['data_file']='test.csv'
    # set params['do_preprocessing']=1
    # set params['values_objfile']="./values" the file which has pickled class Values
    # specify the pmatrix_file to store the pmatrix obj
# else:
    # set params['do_preprocessing']=0
    # set params['pmatrix_file'] to already pickled numpy array

# returns the p_matrix array and values_obj object to help in decoding

import pandas as pd
import pickle
import numpy as np
import difflib
import ast
import re
from loguru import logger
logger.add("logs.log")
params={}
import nltk

# When running set these
params['data_file']='./Flipkart/train10.csv'
params['values_objfile']="./values"
params['do_preprocessing']=0
# this is to store the pmatrix_file 
params['pmatrix_file']="./pmat.npy"

from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  
nltk.download('punkt')
data=pd.read_csv(params['data_file'])
n_items=len(data['attributes'])
flat_list=[]
train_data_values=set()

class Values:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  
        self.word_embeddings=None
    def addWord(self, word):
        word=re.sub(r'[^\x00-\x7F]+',' ',word)
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
values_obj=Values("Values")
picklefile = open(params['values_objfile'], 'rb')
#unpickle the dataframe
values_obj = pickle.load(picklefile)


if params['do_preprocessing']:

    for i in range(n_items):
        list_of_values=data['attributes'][i]
        d=ast.literal_eval(list_of_values)
        del[d['vertical']]
    #     Here need to make change if we don't want the category [the vertical] attribute as part of the classifier
        temp_list = [item for sublist in d.values() for item in sublist]
        temp_set=set(temp_list)
        train_data_values=train_data_values.union(temp_set)
        flat_list.append(temp_list)

    all_attrs_list=[item for sublist in flat_list for item in sublist]

    distinct_vals=set()
    no_matches_set=set()
    replacements={}

    all_allowed_attrs_set=set(values_obj.word2index.keys())
    for val in all_attrs_list:
        if val not in all_allowed_attrs_set:        
            match=difflib.get_close_matches(val,all_allowed_attrs_set,n=1)
            if len(match)!=0:
                replacements[val]=match[0]
            if(len(match)==0) and val not in distinct_vals:
                wordList = re.sub("[^\w]", " ",  val).split()
                wordList=[w for w in wordList if w not in stop_words]
                flag=0
                for w in wordList:
                    match=difflib.get_close_matches(w,all_allowed_attrs_set,n=1)
                    if (len(match)!=0):
                        distinct_vals.add(w)
                        flag=1
                        break
                if flag==0:
                    no_matches_set.add(val)
                else:
                    replacements[val]=match[0]
            distinct_vals.add(val)

    logger.info("Values with a suitable edit distance match in domain: ",len(distinct_vals))
    logger.info("Values with no matches in domain: ",len(no_matches_set))
    for i in range(len(flat_list)):
        flat_list[i]=[replacements[word] if word not in all_allowed_attrs_set else word for word in flat_list[i] if word in all_allowed_attrs_set or word in replacements.keys()]
    logger.info("Preprocessing the p matrix")
    p_matrix=np.zeros((len(all_allowed_attrs_set),len(all_allowed_attrs_set)),dtype=np.float32)
    # Normalized p matrix - formed by simply dividing the value of each value in row by diagonal
    for ind,word in values_obj.index2word.items(): 
    #     outer loop over words
        for values in flat_list:
    #         inner loop over the training data labels
            if word in values:
                for val in values:
                    p_matrix[ind][values_obj.word2index[val]]+=1   
        if p_matrix[ind][ind]!=0:
            p_matrix[ind,:]=p_matrix[ind,:]/p_matrix[ind][ind]
    np.fill_diagonal(p_matrix,1)
    p_matrix=p_matrix-np.eye(len(all_allowed_attrs_set))
    np.save(params['pmatrix_file'],p_matrix)
    
else:
    logger.info("Loading the pmatrix file",params['pmatrix_file'])
    p_matrix=np.load(params['pmatrix_file'])