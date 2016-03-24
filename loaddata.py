#dataset

import numpy as np
from os.path import dirname
from os.path import join

class Bunch(dict):
    """
    Container object for datasets: dictionary-like object
    that exposes its keys and attributes. """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def load_datasets(dataset, title=None, timestamp=False):
    base_dir = join(dirname(__file__), 'data/')
    if timestamp:
        data_m = np.loadtxt(base_dir + dataset, delimiter='\t', dtype=int)
        data_sets = {}
        for userid, itemid, rating, stime in data_m:
            data_sets.setdefault(userid, {})
            data_sets[userid][itemid] = (stime, int(rating))
    else:
        data_m = np.loadtxt(base_dir + dataset, delimiter='\t', usecols=(0,1,2), dtype=int)
        data_sets = {}
        for userid, itemid, rating in data_m:
            data_sets.setdefault(userid, {})
            data_sets[userid][itemid] = int(rating)

    data_titles = None
    if not title == None:
        data_titles = np.loadtxt(base_dir + title, delimiter='|', usecols=(0, 1), dtype=str)
        data_t = []
        for itemid, label in data_titles:
            data_t.append((int(itemid), label))
        data_titles = dict(data_t)

    return Bunch(data=data_sets, item_ids=data_titles, user_ids=None)
    
