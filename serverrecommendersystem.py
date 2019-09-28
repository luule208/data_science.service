# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 04:01:18 2019

@author: ASUS
"""

import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
class CF(object):
    """docstring for CF"""
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        #self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_users = len(np.unique(Y_data[:,0]))
        #self.n_items = int(np.max(self.Y_data[:, 1])) + 1
        self.n_items = len(np.unique(Y_data[:,1]))
    
    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
    
    def normalize_Y(self):
        users = self.Y_data[:, 0] # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        #self.mu = np.zeros((self.n_users,))
        self.mu = np.zeros((len(np.unique(Y_data[:,0])),))
        #print(self.mu)
        for n in range(self.n_users):
            #print(n)
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, 1] 
            # and the corresponding ratings 
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings) 
            #print(m)
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        ################################################
        # form the rating matrix as a sparse matrix. Sparsity is important 
        # for both memory and computing efficiency. For example, if #user = 1M, 
        # #item = 100k, then shape of the rating matrix would be (100k, 1M), 
        # you may not have enough memory to store this. Then, instead, we store 
        # nonzeros only, and, of course, their locations.
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        eps = 1e-6
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
    
        
    def refresh(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self.normalize_Y()
        self.similarity() 
        
    def fit(self):
        self.refresh()
        
    
    def __pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2: 
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others 
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:] 
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        if self.uuCF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)
            
    
    def recommend(self, u):
        """
        Determine all items should be recommended for user u.
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which 
        have not been rated by u yet. 
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0: 
                    recommended_items.append(i)
        
        return recommended_items 
    
    def recommend2(self, u):
        """
        Determine all items should be recommended for user u.
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which 
        have not been rated by u yet. 
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        recommended_items = []
    
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0: 
                    recommended_items.append(i)
        
        return recommended_items 

    def print_recommendation(self):
        """
        print all items which should be recommended for each user 
        """
        print( 'Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print( '    Recommend item(s):', recommended_items, 'for user', u)
            else: 
                print( '    Recommend item', u, 'for user(s) : ', recommended_items)
 
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
               
import os
import pickle
import re
import json
app = Flask(__name__)

from flask import Flask, request, jsonify
rs = None
try:
    with open('C:/Users/ASUS/JupyterNoteBook_workspace/Choto_Recommender_System_model/ChotoRecommenderSystemModel.pkl', 'rb') as model:
        rs = pickle.load(model)
except IOError:
    print("File not found!!")
    c

@app.route("/recommend/user/<id>", methods=["POST"])
def recommendforuser(id):
    list_item = rs.recommend(int(id))
    #list_item=[4, 11, 12, 21, 25, 29, 40, 47, 58, 89, 124, 127, 142, 159, 184, 192, 195, 198, 213, 252, 253, 260, 261, 268, 293, 339, 347, 353, 363, 367, 380, 384, 398, 400, 411, 435, 447, 448, 474, 503, 514, 517, 528, 529, 564, 572, 595, 627, 665, 677, 696, 714, 761, 780, 786, 801, 810, 848, 878, 922, 938, 1041, 1056, 1058, 1070, 1097, 1099, 1135, 1146, 1154, 1191, 1192, 1193, 1215, 1225, 1232, 1279, 1285, 1312, 1317, 1336, 1344, 1387, 1425, 1429, 1491, 1521, 1538, 1566, 1623, 1624, 1632, 1642, 1674, 1677, 1718, 1749, 1765, 1794, 1800, 1801, 1818, 1825, 1830, 1874, 1878, 1890, 1904, 1951, 1982, 1992, 2060, 2076, 2086, 2092, 2096, 2112, 2134, 2138, 2149, 2170, 2189, 2194, 2241, 2255, 2275, 2279, 2309, 2339, 2376, 2378, 2385, 2413, 2421, 2461, 2463, 2468, 2494, 2515, 2545, 2619, 2623, 2624, 2627, 2673, 2718, 2743, 2749, 2773, 2778, 2793, 2822, 2856, 2861, 2884, 2888, 2894, 2899, 2912, 2928, 2930, 2942, 2961, 2979, 2992, 3006, 3031, 3053, 3061, 3078, 3082, 3085, 3095, 3099, 3106, 3110, 3115, 3143, 3147, 3148, 3165, 3186, 3203, 3236, 3246, 3254, 3257, 3283, 3294, 3296, 3302, 3329, 3380, 3383, 3438, 3449, 3456, 3495, 3514, 3524, 3534, 3558, 3573, 3580, 3581, 3610, 3633, 3643, 3648, 3685, 3815, 3828, 3844, 3907, 3949, 3952, 3955, 3979, 4005, 4042, 4146, 4152, 4184, 4245, 4263, 4266, 4316, 4320, 4321, 4388, 4436, 4531, 4533, 4537, 4551, 4641, 4801, 4838, 4840, 4908, 4937, 4940, 4948, 4976, 4982, 5015, 5016, 5020, 5026, 5105, 5161, 5186, 5194, 5195, 5219, 5227, 5307, 5352, 5464, 5493, 5504, 5523, 5586, 5589, 5611, 5616, 5666, 5734, 5738, 5743, 5760, 5762, 5763, 5785, 5807, 5809, 5817, 5859, 5916, 5927, 5984, 5994, 6000, 6021, 6035, 6080, 6121, 6125, 6128, 6130, 6137, 6150, 6159, 6171, 6203, 6239, 6311, 6384, 6399, 6400, 6403, 6416, 6516, 6592, 6605, 6625, 6659, 6683, 6686, 6694, 6697, 6699, 6700, 6708, 6735, 6749, 6848, 6870, 6877, 6914, 6915, 6945, 6946, 6954, 6965, 6986, 7028, 7096, 7097, 7099, 7109, 7164, 7171, 7175, 7272, 7273, 7285, 7304, 7311, 7332, 7348, 7355, 7368, 7389, 7402, 7441, 7466, 7484, 7507, 7532, 7552, 7569, 7571, 7607, 7630, 7632, 7698, 7707, 7721, 7753, 7769, 7818, 7837, 7947, 7950, 7961, 7983, 7986, 8016, 8049, 8075, 8086, 8095, 8136, 8150, 8158, 8173, 8174, 8187, 8195, 8197, 8213, 8215, 8219, 8229, 8271, 8281, 8297, 8308, 8310, 8325, 8330, 8338, 8347, 8353, 8355, 8356, 8376, 8420, 8438, 8446, 8447, 8489, 8490, 8494, 8502, 8505, 8515, 8529, 8535, 8538, 8558, 8574, 8575, 8579, 8592, 8596, 8603, 8606, 8608, 8612, 8625, 8635, 8640, 8650, 8691, 8711, 8728, 8735, 8739, 8760, 8781, 8784, 8790, 8792, 8794, 8809, 8819, 8838, 8841, 8875, 8893, 8904, 8909, 8913, 8957, 8974, 8977, 9020, 9095, 9113, 9121, 9141, 9147, 9188, 9227, 9256, 9260, 9265, 9272, 9285, 9317, 9337, 9338, 9365, 9405, 9418, 9450, 9456, 9487, 9499, 9524, 9585, 9599, 9616, 9644, 9653, 9655, 9658, 9668, 9670, 9713, 9728, 9730, 9735, 9738, 9740, 9751, 9752, 9760, 9761, 9767, 9768, 9781, 9799, 9812, 9818, 9820, 9826, 9875, 9879, 9904, 9911, 9918, 9939, 9942, 9945, 9949, 9965, 9986, 9987, 9990, 9993, 10009, 10030, 10040, 10042, 10061, 10064, 10072, 10079, 10095, 10128, 10163, 10238, 10266, 10268, 10281, 10312, 10314, 10317, 10347, 10356, 10369, 10378, 10379, 10380, 10383, 10395, 10398, 10410, 10447, 10458, 10459, 10470, 10486, 10493, 10542]

    #return jsonify({'item': list_item})
    #return user_schema.jsonify(list_item)
    return jsonify({'item': list_item})
#recommendforuser()
app.run(debug=True)