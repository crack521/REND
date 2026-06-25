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


def EvaluateSolution(STEPRATIO, STRTEGYID):
    #######################################################################################################################
    ##................................................Evaluate Solution.....................................................
    dqn = REND()
    ## data_test
    data_test_path = '../data/real/'
    data_test_name = ["crime", "HI-II-14", 'digg', 'Email-Enron', 'Gnutella31', 'Facebook', 'DBLP', 'Youtube', "Flickr"]
    baselines = ["BPD","CI","CoreHD","GND","HDA","MinSum","FINDER","REND","boundarycut","GDM"]

    for baseline in baselines:
        if baseline =="boundarycut":
            data_test_costType = ['degree','random']
        else:
            data_test_costType = ['uniform','degree','random']

        ## save_dir
        save_dir_degree = '../results/real/{}/degree_cost/'.format(baseline)
        save_dir_random = '../results/real/{}/random_cost/'.format(baseline)
        save_dir_uniform = '../results/real/{}/uniform_cost/'.format(baseline)
        if not os.path.exists(save_dir_degree):
            os.makedirs(save_dir_degree)
        if not os.path.exists(save_dir_random):
            os.makedirs(save_dir_random)
        if not os.path.exists(save_dir_uniform):
            os.makedirs(save_dir_uniform)
        ## begin computing...

        for costType in data_test_costType:
            if baseline =="GDM":
                current_data_test_name = data_test_name[:-2]
            else:
                current_data_test_name = data_test_name
            df = pd.DataFrame(np.zeros((2, len(current_data_test_name))),
                              index=['solution', 'time'], columns=current_data_test_name)
            for i in range(len(current_data_test_name)):
                print('Evaluating dataset %s' % current_data_test_name[i])
                data_test = data_test_path + current_data_test_name[i] + '.txt'
                data_test_weight = data_test_path + current_data_test_name[i] + f'_{costType}_weight.txt'
                if costType == 'degree':
                    solution = save_dir_degree + current_data_test_name[i] + '.txt'
                elif costType == 'random':
                    solution = save_dir_random + current_data_test_name[i] + '.txt'
                elif costType == 'uniform':
                    solution = save_dir_uniform + current_data_test_name[i] + '.txt'
                t1 = time.time()
                # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
                ################################## modify to choose which strategy to evaluate
                strategyID = STRTEGYID
                score, MaxCCList, solution = dqn.EvaluateSol(data_test,data_test_weight, solution, strategyID, reInsertStep=20)
                t2 = time.time()
                df.iloc[0, i] = score
                df.iloc[1, i] = t2 - t1
                if costType == 'degree':
                    result_file = save_dir_degree + '/MaxCCList_Strategy_' + current_data_test_name[i] + '.txt'
                elif costType == 'random':
                    result_file = save_dir_random + '/MaxCCList_Strategy_' + current_data_test_name[i] + '.txt'
                elif costType == 'uniform':
                    result_file = save_dir_uniform + '/MaxCCList_Strategy_' + current_data_test_name[i] + '.txt'
                with open(result_file, 'w') as f_out:
                    for j in range(len(MaxCCList)):
                        f_out.write('%.8f\n' % MaxCCList[j])

                print('cost_type: %s,Data:%s, score:%.6f!' % (costType,current_data_test_name[i], score))

            if costType == 'degree':
                df.to_csv(save_dir_degree + '/solution_%s_score.csv' % (costType), encoding='utf-8', index=False)
            elif costType == 'random':
                df.to_csv(save_dir_random + '/solution_%s_score.csv' % (costType), encoding='utf-8', index=False)
            elif costType == 'uniform':
                df.to_csv(save_dir_uniform + '/solution_%s_score.csv' % (costType), encoding='utf-8', index=False)

def main():
    EvaluateSolution(0.01, 0)

if __name__=="__main__":
    main()
