import numpy as np
import glob

import torch.utils.data as data
import pandas as pd
import os
from PIL import Image
import ast
import pickle
import re 
import cv2


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


class Flipkart2021(data.Dataset):

	def __init__(self, root, train_csvfile, vertical_attributes_npy, allowed_values_npy, embedding_file,flag):
	    self.root = root
	    self.embedding_file = embedding_file
	    picklefile = open(embedding_file, 'rb')
	    values_obj = pickle.load(picklefile)
	    self.inp = values_obj.word_embeddings
	    self.df = pd.read_csv(train_csvfile)
	    self.is_training=flag
	    self.category_index_dict, self.category_attribute_map = self.get_category_index_dict(
	        vertical_attributes_npy)
	    self.allowed_values = np.load(
	        allowed_values_npy, allow_pickle=True).item()
	    self.allowed_values_set = self.get_allowed_values_set()
	    self.allowed_values_index_dict = self.get_index_dict()
	    self.n_items = len(self.df)
        
	def get_category_index_dict(self, vertical_attributes_npy):
		vertical_attributes = np.load(vertical_attributes_npy, allow_pickle=True).item()
		category_list = [k for v, k in enumerate(vertical_attributes.keys())]
		category_list.sort()

		category_index_dict = {k:v for v, k in enumerate(category_list)}
		return category_index_dict, vertical_attributes

	def get_allowed_values_set(self):
		allowed_values_set = set()

		for k,v in self.allowed_values.items():
			if k=='vertical':
				continue
			v = [re.sub(r'[^\x00-\x7F]+', ' ', word) for word in v]
			allowed_values_set = allowed_values_set.union(v)

		return allowed_values_set

	def get_index_dict(self):
		allowed_values_list = list(self.allowed_values_set)
		allowed_values_list.sort()
		return {k:v for v, k in enumerate(allowed_values_list)}

	def crop_or_pad1(self, image: np.array):
		h_new = 512*3
		w_new = 512

		h,w,c = image.shape
		if h == 499 or h == 500:
			image = image.transpose(1,0,2)

		h,w,c = image.shape

		diff_w = w_new - w
		if diff_w % 2 == 0:
			image = np.pad(image, ((0, 0), (diff_w//2, diff_w//2), (0,0)), mode='constant', constant_values=255)
		else:
			image = np.pad(image, ((0, 0), (diff_w//2, diff_w//2 + 1), (0,0)), mode='constant', constant_values=255)

		if h > h_new:
			diff_h = h - h_new
			if diff_h % 2 == 0:
				image = image[diff_h//2:h - diff_h//2]
			else:
				image = image[diff_h//2:h - diff_h//2 - 1]
		else:
			diff_h = h_new - h
			if diff_h % 2 == 0:
				image = np.pad(image, ((diff_h//2, diff_h//2), (0,0), (0,0)), mode='constant', constant_values=255)
			else:
				image = np.pad(image, ((diff_h//2, diff_h//2 + 1), (0,0), (0,0)), mode='constant', constant_values=255)

		return image

	def normalise(self, image: np.array):
		image = image.astype(np.float32)/255 #*2 -1
		return image

	def __len__(self):
	    return len(self.df.index)

	def __getitem__(self, index):
	    item = self.df.iloc[index]
	    try:
	        return self.get(item)
	    except:
	        i=1
	        item=self.df.iloc[index]
	        while True:
	            i+=1
	            item=self.df.iloc[(index+i) % self.n_items]
	            try:
	                 return self.get(item)
	            except:
	                 continue
	    return 

	def get_attribute_value_label(self, attributes):
		s = set()
		for k, v in attributes.items():
			if k=='vertical':
				continue
			s = s.union(v)

		common_values = self.allowed_values_set.intersection(s)

		labels = [0]*len(self.allowed_values_set)

		for val in common_values:
			labels[self.allowed_values_index_dict[val]] = 1

		return np.array(labels)

	def get_category_label(self, category):
		labels = [0]*len(self.category_index_dict)
		labels[self.category_index_dict[category]] = 1
		return np.array(labels)

	def get(self, item):
	    category = item['category']
	    category_label = self.get_category_label(category)

	    attributes = ast.literal_eval(item['attributes'])
	    attribute_value_label = self.get_attribute_value_label(attributes)

	    img_path = os.path.join(self.root, item['filename'])
	    img = np.asarray(Image.open(img_path).convert('RGB'))
	    img = self.transform(img)
	    if self.is_training:
	        return (img,item['filename'],self.inp), attribute_value_label
	    else:
	        return (img,item['attributes'],self.inp), attribute_value_label,category_label
	def transform(self, image: np.array):
		image = self.crop_or_pad1(image)
		image = self.normalise(image)
		image = cv2.resize(image, (224, 224))
		image = image.transpose(2,0,1)
# 		print(image.shape)    
		return image
class Flipkart2021_eval(data.Dataset):

	def __init__(self, root, train_csvfile, vertical_attributes_npy, allowed_values_npy, embedding_file,flag):
	    self.root = root
	    self.embedding_file = embedding_file
	    picklefile = open(embedding_file, 'rb')
	    values_obj = pickle.load(picklefile)
	    self.inp = values_obj.word_embeddings
	    self.df = pd.read_csv(train_csvfile)
	    self.is_training=flag
	    self.category_index_dict, self.category_attribute_map = self.get_category_index_dict(
	        vertical_attributes_npy)
	    self.allowed_values = np.load(
	        allowed_values_npy, allow_pickle=True).item()
	    self.allowed_values_set = self.get_allowed_values_set()
	    self.allowed_values_index_dict = self.get_index_dict()
	    self.n_items = len(self.df)
        
	def get_category_index_dict(self, vertical_attributes_npy):
		vertical_attributes = np.load(vertical_attributes_npy, allow_pickle=True).item()
		category_list = [k for v, k in enumerate(vertical_attributes.keys())]
		category_list.sort()

		category_index_dict = {k:v for v, k in enumerate(category_list)}
		return category_index_dict, vertical_attributes

	def get_allowed_values_set(self):
		allowed_values_set = set()

		for k,v in self.allowed_values.items():
			if k=='vertical':
				continue
			v = [re.sub(r'[^\x00-\x7F]+', ' ', word) for word in v]
			allowed_values_set = allowed_values_set.union(v)

		return allowed_values_set

	def get_index_dict(self):
		allowed_values_list = list(self.allowed_values_set)
		allowed_values_list.sort()
		return {k:v for v, k in enumerate(allowed_values_list)}

	def crop_or_pad1(self, image: np.array):
		h_new = 512*3
		w_new = 512

		h,w,c = image.shape
		if h == 499 or h == 500:
			image = image.transpose(1,0,2)

		h,w,c = image.shape

		diff_w = w_new - w
		if diff_w % 2 == 0:
			image = np.pad(image, ((0, 0), (diff_w//2, diff_w//2), (0,0)), mode='constant', constant_values=255)
		else:
			image = np.pad(image, ((0, 0), (diff_w//2, diff_w//2 + 1), (0,0)), mode='constant', constant_values=255)

		if h > h_new:
			diff_h = h - h_new
			if diff_h % 2 == 0:
				image = image[diff_h//2:h - diff_h//2]
			else:
				image = image[diff_h//2:h - diff_h//2 - 1]
		else:
			diff_h = h_new - h
			if diff_h % 2 == 0:
				image = np.pad(image, ((diff_h//2, diff_h//2), (0,0), (0,0)), mode='constant', constant_values=255)
			else:
				image = np.pad(image, ((diff_h//2, diff_h//2 + 1), (0,0), (0,0)), mode='constant', constant_values=255)

		return image

	def normalise(self, image: np.array):
		image = image.astype(np.float32)/255 #*2 -1
		return image

	def __len__(self):
	    return len(self.df.index)

	def __getitem__(self, index):
	    item = self.df.iloc[index]
	    try:
	        return self.get(item)
	    except:
	        i=1
	        item=self.df.iloc[index]
	        while True:
	            i+=1
	            item=self.df.iloc[(index+i) % self.n_items]
	            try:
	                 return self.get(item)
	            except:
	                 continue
	    return 

	def get_attribute_value_label(self, attributes):
		s = set()
		for k, v in attributes.items():
			if k=='vertical':
				continue
			s = s.union(v)

		common_values = self.allowed_values_set.intersection(s)

		labels = [0]*len(self.allowed_values_set)

		for val in common_values:
			labels[self.allowed_values_index_dict[val]] = 1

		return np.array(labels)

	def get_category_label(self, category):
		labels = [0]*len(self.category_index_dict)
		labels[self.category_index_dict[category]] = 1
		return np.array(labels)

	def get(self, item):
# 	    category = item['category']
# 	    category_label = self.get_category_label(category)

# 	    attributes = ast.literal_eval(item['attributes'])
# 	    attribute_value_label = self.get_attribute_value_label(attributes)

	    img_path = os.path.join(self.root, item['filename'])
	    img = np.asarray(Image.open(img_path).convert('RGB'))
	    img = self.transform(img)
# 	    if self.is_training:
	    return (img,item['filename'],self.inp)
# 	    else:
# 	        return (img,item['attributes'],self.inp), attribute_value_label,category_label
	def transform(self, image: np.array):
		image = self.crop_or_pad1(image)
		image = self.normalise(image)
		image = cv2.resize(image, (224, 224))
		image = image.transpose(2,0,1)
# 		print(image.shape)    
		return image


if __name__ == '__main__':
	dataset = Flipkart2021('train10_images', 'Sample Data_Readme and other docs/train10.csv', 'Sample Data_Readme and other docs/vertical_attributes.npy', 'Sample Data_Readme and other docs/Attribute_allowedvalues.npy')
	dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True)
	for img, category_label, attribute_value_label in dataloader:
		print('image dim:',img.shape) 
		print('category dim:',category_label.shape)
		print('attribute_value_label:', attribute_value_label.shape)
		exit()
# allowed_values = np.load('Sample Data_Readme and other docs/Attribute_allowedvalues.npy', allow_pickle=True).item()
# allowed_values_set = set()

# for k,v in allowed_values.items():
# 	allowed_values_set = allowed_values_set.union(v)

# allowed_values_list = list(allowed_values_set)
# allowed_values_list.sort()
# allowed_values_index_dict = {k:v for v, k in enumerate(allowed_values_list)}

# # print(allowed_values_set)

# df = pd.read_csv('Sample Data_Readme and other docs/train10.csv')
# item = df.iloc[10]
# d = ast.literal_eval(item['attributes'])
# s = set()
# for k, v in d.items():
# 	s = s.union(v)

# common_values = allowed_values_set.intersection(s)
# print(common_values)

# labels = [0]*len(allowed_values_list)
# for val in common_values:
# 	labels[allowed_values_index_dict[val]] = 1

# print(len(labels))
# print(labels)
