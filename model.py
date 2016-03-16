#model
#-*- coding:utf-8 -*-

import numpy as np
import logger
import copy

__logger__ = logger.getLogger()

class MatrixFromData:
    def __init__(self, dataset):
        self.dataset = copy.deepcopy(dataset)#防止原始数据被改变，使用深拷贝
        self.__buildmodel__()
    
    def __getitem__(self, user_id):
        return self.preferences_from_user(user_id)

    def __iter__(self):
        for index, user in enumerate(self.user_ids()):
            yield user, self[user]
    
    def __buildmodel__(self):
        #self
        self._user_ids = np.asanyarray(self.dataset.keys())
        self._user_ids.sort()
        
        self._item_ids = []
        for items in self.dataset.itervalues():
            self._item_ids.extend(items.keys())
        self._item_ids = np.unique(np.array(self._item_ids))
        self._item_ids.sort()
        
        self.max_pref = -np.inf
        self.min_pref = np.inf
        
        #数据矩阵
        '''
            item  1 2 3   
        user
           1      x z y
           2
           3
        
        '''
        self.matrix = np.empty(shape=(self._user_ids.size, self._item_ids.size))
        __logger__.info('create matrix for %d users and %d items' %(self._user_ids.size, self._item_ids.size))
        
        for userno, userid in enumerate(self._user_ids):
            if(userno % 2 ==0):
                __logger__.debug("PROGRESS: at user_id #%i/%i" % (userno, self._user_ids.size))
            for itemno, itemid in enumerate(self._item_ids):
                r = self.dataset[userid].get(itemid, np.NAN)
                self.matrix[userno, itemno] = r
        
        if self.matrix.size:
            self.max_pref = np.nanmax(self.matrix)
            self.min_pref = np.nanmin(self.matrix)
    
    def user_ids(self):
        return self._user_ids
    
    def item_ids(self):
        return self._item_ids
        
    def preference_values_from_user(self, user_id):
         user_id_loc = np.where(self._user_ids == user_id)
         if not user_id_loc[0].size:
             #userid not found
             raise Exception
         preference = self.matrix[user_id_loc]
         return preference
    
    def preferences_from_user(self, user_id, ordered_by_id=True):
        preferences = self.preference_values_from_user(user_id)
        
        data = zip(self._item_ids, preferences.flatten())
        if ordered_by_id:
            return [(item_id, preference) for item_id, preference in data if not np.isnan(preference)]
        else:
            return sorted([(item_id, preference) for item_id, preference in data if not np.isnan(preference)], key=lambda item:-item[1])
        
    def has_preference_values(self):
        return True
    
    def maximum_preference_value(self):
        return self.max_pref
    
    def minimum_preference_value(self):
        return self.min_pref
        
    def user_count(self):
        return self._user_ids.size
 
    def item_count(self):
        return self._item_ids.size
        
    def items_from_user(self, user_id):
        preferences = self.preferences_from_user(user_id)
        return [key for key, value in preferences]
    
    def preferences_for_item(self, item_id, ordered_by_id=True):
        item_id_loc = np.where(self._item_ids == item_id)
        
        if not item_id_loc[0].size:
            __logger__.warning('preference_for_item item_id:%s is not found' % str(item_id))
            raise Exception
        preferences = self.matrix[:, item_id_loc]
        
        data = zip(self._user_ids, preferences.flatten())
        if ordered_by_id:
            return [(user_id, preference) for user_id, preference in data if not np.isnan(preference)]
        else:
            return sorted([(user_id, preference)  for user_id, preference in data if not np.isnan(preference)], key=lambda user: -user[1])
    
    def preference_value(self, user_id, item_id):
        
        item_id_loc = np.where(self._item_ids == item_id)
        user_id_loc = np.where(self._user_ids == user_id)
        
        if not item_id_loc[0].size:
            __logger__.warning('preference_value item_id:%s is not found' % str(item_id))
            raise Exception
        if not user_id_loc[0].size:
            __logger__.warning('preference_value user_id:%s is not found' % str(user_id))
            raise Exception
            
        return self.matrix[user_id_loc, item_id_loc].flatten()[0]
    
    #for filling the missing values
    def set_preference(self, user_id, item_id, value):
        user_id_loc = np.where(self._user_ids == user_id)
        if not user_id_loc[0].size:
            __logger__.warning('set_preference user_id:%s is not found' % str(user_id))
            raise Exception
        item_id_loc = np.where(self._item_ids == item_id)
        if item_id_loc[0].size and not np.isnan(self.dataset[user_id][item_id]):
            __logger__.warning('set_preference user_id:%s item_id:%s is not allowed edit' % (str(user_id), str(item_id)))
            raise Exception
        self.dataset[user_id][item_id] = value
        self.__buildmodel__()
  
    #sparsering the matrix  
    def remove_preference(self, user_id, item_id):
        user_id_loc = np.where(self._user_ids == user_id)
        item_id_loc = np.where(self._item_ids == item_id)
        if not item_id_loc[0].size:
            __logger__.warning('remove_preference item_id:%s is not found' % str(item_id))
            raise Exception
        if not user_id_loc[0].size:
            __logger__.warning('remove_preference user_id:%s is not found' % str(user_id))
            raise Exception
        
        del self.dataset[user_id][item_id]
        self.__buildmodel__()
    
    def __repr__(self):
        return "<MatrixPreferenceDataModel (%d by %d)>" % (self.matrix.shape[0], self.matrix.shape[1])
    
    def _repr_matrix(self, matrix):
        s = ""
        cellWidth = 11
        shape = matrix.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    s += "---".center(cellWidth)
                else:
                    exp = np.log(abs(v))
                    if abs(exp) <= 4:
                        if exp < 0:
                            s += ("%9.6f" % v).ljust(cellWidth)
                        else:
                            s += ("%9.*f" % (6, v)).ljust(cellWidth)
                    else:
                        s += ("%9.2e" % v).ljust(cellWidth)
            s += "\n"
        return s[:-1]
    
    def __unicode__(self):
        """
        Write out a representative picture of this matrix.

        The upper left corner of the matrix will be shown, with up to 20x5
        entries, and the rows and columns will be labeled with up to 8
        characters.
        """
        matrix_ = self._repr_matrix(self.matrix[:20, :5])
        lines = matrix_.split('\n')
        headers = [repr(self)[1:-1]]
        if self._item_ids.size:
            col_headers = [('%-8s' % unicode(item)[:8]) for item in self._item_ids[:5]]
            headers.append(' ' + ('   '.join(col_headers)))

        if self._user_ids.size:
            for (i, line) in enumerate(lines):
                lines[i] = ('%-8s' % unicode(self._user_ids[i])[:8]) + line
            for (i, line) in enumerate(headers):
                if i > 0:
                    headers[i] = ' ' * 8 + line
        lines = headers + lines
        if self.matrix.shape[1] > 5 and self.matrix.shape[0] > 0:
            lines[1] += ' ...'
        if self.matrix.shape[0] > 20:
            lines.append('...')

        return '\n'.join(line.rstrip() for line in lines)
                        
    def __str__(self):
        return unicode(self).encode('utf-8')
            
       
