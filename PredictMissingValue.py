#Predict Missing Value
#-*- coding:utf-8 -*-

import numpy as np
import PCC_similarity as sim
import logger

__logger__ = logger.getLogger()

class Prediction:
    def __init__(self, model):
        self.__model__ = model
        self.__user_similarity__ = sim.User_similartiy(self.__model__)
        self.__item_similarity__ = sim.Item_similartiy(self.__model__)
    
    def user_based_value(self, userid, itemid, K):
        value = self.__user_similarity__.get_similarity(userid, itemid)
        if not np.isnan(value):
            return value
        
        