#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from REND import REND
import numpy as np
from tqdm import tqdm
import time
import networkx as nx
import pandas as pd
import pickle as cp
import random


def GetSolution(STEPRATIO, MODEL_FILE):
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    dqn = REND()
    ## data_test
    data_test_path = '../data/real/'
    data_test_name = ['crime', "DBLP", 'digg', 'Email-Enron', 'Epinions', 'Facebook', 'Flickr', 'Gnutella31', 'HI-II-14',
                  'Youtube']
    data_test_costType = ['uniform','degree','random']
    model_file = '../models/%s'%MODEL_FILE
    ## save_dir
    save_dir = '../results/real'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_dir_degree = save_dir + '/Data_degree'
    save_dir_random = save_dir + '/Data_random'
    save_dir_uniform = save_dir + '/Data_uniform'
    cost_type2path = {}
    cost_type2path['degree'] = save_dir_degree
    cost_type2path['random'] = save_dir_random
    cost_type2path['uniform'] = save_dir_uniform
    for costType in data_test_costType:
        if not os.path.exists(cost_type2path[costType]):
            os.mkdir(cost_type2path[costType])
    ## begin computing...
    print('The best model is :%s' % (model_file))
    dqn.LoadModel(model_file)

    for costType in data_test_costType:
        df = pd.DataFrame(np.arange(1 * len(data_test_name)).reshape((1, len(data_test_name))), index=['time'],
                          columns=data_test_name)
        #################################### modify to choose which stepRatio to get the solution
        stepRatio = STEPRATIO
        save_path = cost_type2path[costType]
        for j in range(len(data_test_name)):
            print('Testing dataset %s' % data_test_name[j])
            data_test = data_test_path + data_test_name[j] + '.txt'
            data_test_weight = data_test_path + data_test_name[j] + f'_{costType}_weight.txt'
            solution, time = dqn.EvaluateRealData(model_file, data_test, data_test_weight, save_path, stepRatio)
            df.iloc[0, j] = time

        save_dir_local = save_path + '/StepRatio_%.4f' % stepRatio
            
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)

        df.to_csv(save_dir_local + '/solution_%s_time.csv' % costType, encoding='utf-8', index=False)
        print('model has been tested!')


def main():
    model_file = 'REND.ckpt'
    GetSolution(0.01, model_file)

if __name__=="__main__":
    main()

