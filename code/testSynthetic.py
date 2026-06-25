#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from REND import REND
from tqdm import tqdm
import json
def main():
    dqn = REND()
    cost_types = ['degree_cost', 'random_cost']
    result_detail = {}
    file_path = '../results/synthetic'

    if not os.path.exists('../results'):
        os.mkdir('../results')
    if not os.path.exists('../results/synthetic'):
        os.mkdir('../results/synthetic')

    for cost in cost_types:
        result_detail[cost] = {}
        data_test_path = '../data/synthetic/%s/'%cost
        data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
        model_file = '../models/REND.ckpt'
        with open('%s/%s_score.txt'%(file_path, cost), 'w') as fout:
            for i in tqdm(range(len(data_test_name))):
                data_test = data_test_path + data_test_name[i]
                score_mean, score_std, time_mean, time_std,score_list = dqn.Evaluate(data_test, model_file)
                score_list = [ele*100 for ele in score_list]
                result_detail[cost][data_test_name[i]] = {}
                result_detail[cost][data_test_name[i]]['score'] = score_list
                result_detail[cost][data_test_name[i]]['time'] = [time_mean, time_std]

                fout.write('%.2f±%.2f,' % (score_mean * 100, score_std * 100))
                print('%.2f±%.2f,' % (score_mean * 100, score_std * 100))
                fout.flush()
                print('data_test_%s has been tested!' % data_test_name[i])
    with open('%s/result_REND.json'%(file_path), 'w') as fout:
        json.dump(result_detail, fout)



if __name__=="__main__":
    main()
