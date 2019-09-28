import json
import datetime
import numpy as np
import os
class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)
def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)
def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as json_file:
        dic = json.load(json_file)
    return dic
def get_dict(path_data):
    id_all_data = []
    cams = ["cam1","cam2","cam3","cam4","cam5","cam6"]
    for cam in cams:
        for ids in os.listdir(os.path.join(path_data,cam)):
            id_all_data.append(int(ids))
    ids_old = sorted(set(id_all_data))
    old2new_id = {old:new for (new,old) in enumerate(ids_old)}
    return old2new_id
