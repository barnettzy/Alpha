#Predict Missing Value
#-*- coding:utf-8 -*-

import numpy as np
import PCC_similarity as sim
import logger
import model

__logger__ = logger.getLogger()

class Prediction:
    def __init__(self, model):
        self.__model__ = model
        self.__user_similarity__ = sim.User_similartiy(self.__model__)
        self.__item_similarity__ = sim.Item_similartiy(self.__model__)
    
    def user_based_value(self, userid, itemid, K):
        value = self.__model__.preference_value(userid, itemid)
        if not np.isnan(value):
            return value
        average = self.__user_similarity__.get_average(self.__model__.preference_values_from_user(userid))
        sims = self.__user_similarity__.get_similarities(userid)
        
        sim_up = 0.0
        sim_down = 0.0
        for i in range(K+1):
            if sims[i][1] == userid or sims[i][0] < 0.0:
                continue 
            uRvi = self.__model__.preference_value(sims[i][1], itemid)
            
            if np.isnan(uRvi):
                continue
                
            uRv  = self.__user_similarity__.get_average(self.__model__.preference_values_from_user(sims[i][1]))
            sim_up = sim_up + sims[i][0] * (uRvi - uRv)
            sim_down = sim_down + sims[i][0]
        if sim_down == 0.0:
            return average
        return average + (sim_up / sim_down)
    
    def item_based_value(self, itemid, userid, K):
        value = self.__model__.preference_value(userid, itemid)
        if not np.isnan(value):
            return value
        average = self.__item_similarity__.get_average(self.__model__.preferences_for_item(itemid))
        sims = self.__item_similarity__.get_similarities(itemid)
        
        sim_up = 0.0
        sim_down = 0.0
        for i in range(K+1):
            if sims[i][1] == userid or sims[i][0] < 0.0:
                continue
            iRvi = self.__model__.preference_value(userid, sims[i][1])
            
            if np.isnan(iRvi):
                continue
            print 'iRvi ', iRvi
            
            iRv  = self.__item_similarity__.get_average(self.__model__.preferences_for_item(sims[i][1]))
            sim_up = sim_up + sims[i][0] * (iRvi - iRv)
            sim_down = sim_down + sims[i][0]
        if sim_down == 0.0:
            return average
        return average + (sim_up / sim_down)
    
    def predict_value(self, userid, itemid, lam, K):
        u = self.user_based_value(userid, itemid, K)
        i = self.item_based_value(itemid, userid, K)
        
        return lam * u + (1-lam) * i
  
if __name__ == '__main__':
    movies = {'Marcel Caraciolo': {'Lady in the Water': 2.5, \
     'Snakes on a Plane': 3.5, \
     'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, \
     'The Night Listener': 3.0}, \
     'Paola Pow': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, \
     'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, \
     'You, Me and Dupree': 3.5}, \
    'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, \
     'Superman Returns': 3.5, 'The Night Listener': 4.0}, \
    'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, \
     'The Night Listener': 4.5, 'Superman Returns': 4.0, \
     'You, Me and Dupree': 2.5}, \
    'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
     'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0, \
     'You, Me and Dupree': 2.0}, \
    'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
     'The Night Listener': 3.0, 'Superman Returns': 5.0, \
     'You, Me and Dupree': 3.5}, \
    'Penny Frewman': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0, \
    'Superman Returns':4.0}, \
    'Maria Gabriela': {'KKK':4}}
    
    m = model.MatrixFromData(movies)
    pmv = Prediction(m)
    print pmv.predict_value('Sheldom', 'Just My Luck', 0.8, 3)
    