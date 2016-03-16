#-*- coding:utf-8 -*-

import numpy as np
import logger
import model

__loc_logger__ = logger.getLogger()

def find_common_elements(source, target):
    src = dict(source)
    tgt = dict(target)
    
    inter = np.intersect1d(src.keys(), tgt.keys())
    common_preferences = zip(*[(src[item], tgt[item]) for item in inter if not np.isnan(src[item]) and not np.isnan(tgt[item])])
    if common_preferences:
        return np.asarray(common_preferences[0]), np.asarray(common_preferences[1]), inter
    else:
        return np.asarray([]), np.asarray([]), inter

def PCC_similarity_Value(vect_A, vect_B, average_A, average_B, name=''):
        
        if vect_A.ndim != 1 or vect_B.ndim != 1:
            __loc_logger__.warning('__PCC_similarity_Value__ warning!!!')
            __loc_logger__.warning('%s preference vector A ndim : %d' % (name, vect_A.ndim))
            __loc_logger__.warning('%s preference vector B ndim : %d' % (name, vect_B.ndim))
            raise Exception
        
        if len(vect_A) != len(vect_B):
            __loc_logger__.warning('__PCC_similarity_Value__ warning!!!')
            __loc_logger__.warning('%s preference vector A length : %d' % (name, len(vect_A)))
            __loc_logger__.warning('%s preference vector B length : %d' % (name, len(vect_B)))
            raise Exception
                                        
        #average_A = np.average(user_A)
        #average_B = np.average(user_B)
        
        sum_up = 0.0
        sum_down_A = 0.0
        sum_down_B = 0.0
         
        for i in range(len(vect_A)):
            Rai = vect_A[i] - average_A
            Rbi = vect_B[i] - average_B
            
            sum_up = sum_up + (Rai * Rbi)
            sum_down_A = Rai**2 + sum_down_A
            sum_down_B = Rbi**2 + sum_down_B
            
        if sum_up==0 and sum_down_A==0 and sum_down_B==0:
            return 1
        
        if sum_down_A == 0.0 or sum_down_B == 0.0:
            __loc_logger__.warning('__PCC_similarity_Value__ warning!!!')
            __loc_logger__.warning('%s preference vector: %s' % (name, str(vect_A)))
            __loc_logger__.warning('%s preference vector: %s' % (name, str(vect_B)))
            raise Exception
        
        __loc_logger__.warning('average vect A : %f ' % (average_A))
        __loc_logger__.warning('average vect B : %f ' % (average_B))
        __loc_logger__.warning('%f / sqrt(%f) * sqrt(%f) ' % (sum_up, sum_down_A, sum_down_B))
        
        return float(sum_up) / (np.sqrt(sum_down_A) * np.sqrt(sum_down_B))

class User_similartiy:
    def __init__(self, model):
        self.__model__ = model            
    
    def get_average(self, vect):
        i=0
        _sum=0.0
        for val in vect.flatten():
            if not np.isnan(val):
                _sum = _sum + val
                i = i+1
        if i==0:
            return 0
        return float(_sum) / i
    
    def get_similarity(self, source_id, target_id):
        source_preferences = self.__model__.preferences_from_user(source_id)
        target_preferences = self.__model__.preferences_from_user(target_id)
        
        average_A = self.get_average(self.__model__.preference_values_from_user(source_id))
        average_B = self.get_average(self.__model__.preference_values_from_user(target_id))
        
        if self.__model__.has_preference_values():
            source_preferences, target_preferences, common_item_ids = find_common_elements(source_preferences, target_preferences)
       
        __loc_logger__.info(str(source_preferences))
        __loc_logger__.info(str(target_preferences))
        __loc_logger__.info(str(common_item_ids))
        
        if len(source_preferences) == 0:
            return 0
        return PCC_similarity_Value(source_preferences, target_preferences, average_A, average_B, "%s and %s" % (str(source_id), str(target_id)))
    
    def get_similarities(self, source_id):
        return sorted([(self.get_similarity(source_id, tid), tid) for tid, val in self.__model__], reverse=True)
        
    def __iter__(self):
        for tid, preferences in self.__model__:
            yield tid, self[tid]
 
            
class Item_similartiy:
    def __init__(self, model):
        self.__model__ = model
        
    def get_average(self, vect):
        aver = 0.0
        i = 0
        for key, value in vect:
            aver = aver + value
            i = i + 1
        if i==0:
            __loc_logger__.warning('vect is empty ["%s"]' % str(vect))  
            raise Exception
        return aver / i   
             
    def get_similarities(self, source_id):
       return sorted([(self.get_similarity(source_id, tid), tid) for tid in self.__model__.item_ids()], reverse=True)
        
    def get_similarity(self, source_id, target_id):
        source_preferences = self.__model__.preferences_for_item(source_id)
        target_preferences = self.__model__.preferences_for_item(target_id)
        
        average_A = self.get_average(source_preferences)
        average_B = self.get_average(target_preferences)  
        
        if self.__model__.has_preference_values():
            source_preferences, target_preferences, common_item_ids = find_common_elements(source_preferences, target_preferences)

        __loc_logger__.info(str(source_id) + str(source_preferences))
        __loc_logger__.info(str(target_id) + str(target_preferences))
        __loc_logger__.info(str(common_item_ids))
        
        if len(source_preferences)==0:
            return 0
        return PCC_similarity_Value(source_preferences, target_preferences, average_A, average_B, 'user sim')
               
#test
          
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
    simu = User_similartiy(m)
    simi = Item_similartiy(m)
    #print m.preference_values_from_user('Sheldom')
    #print m.preference_values_from_user('Leopoldo Pires')
    #print m.item_ids()
    print simu.get_similarities('Sheldom')
    print simi.get_similarities('Just My Luck')