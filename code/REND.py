#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 2026

REND: deep Reinforcement learning for cost-Effective Network Dismantling
"""

from __future__ import print_function, division
import os
os.environ["KMP_AFFINITY"] = "disabled"
import tensorflow as tf
import numpy as np
import networkx as nx
import random
import time
import pickle as cp
import sys
from tqdm import tqdm
import PrepareBatchGraph
import graph
import nstep_replay_mem
import nstep_replay_mem_prioritized
import mvc_env
import utils
import scipy.linalg as linalg
import pandas as pd

# Hyper Parameters are defined in config.py
from config import *



class REND:

    def __init__(self):
        # init some parameters
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type =  'barabasi_albert'#'barabasi_albert', 'small-world'
        self.training_type = 'degree'
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.utils = utils.py_Utils()

        ############----------------------------- variants of DQN(start) ------------------- ###################################
        self.IsHuberloss = False
        self.IsDoubleDQN = False
        self.IsPrioritizedSampling = False


        self.IsDuelingDQN = False
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1)
        self.IsDistributionalDQN = False
        self.IsNoisyNetDQN = False
        self.Rainbow = False
        ############----------------------------- variants of DQN(end) ------------------- ###################################

        #Simulator
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[]
        self.g_list=[]
        # self.covered=[]
        self.pred=[]
        if self.IsPrioritizedSampling:
            self.nStepReplayMem = nstep_replay_mem_prioritized.py_Memory(epsilon,alpha,beta,beta_increment_per_sampling,TD_err_upper,MEMORY_SIZE)
        else:
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)

        for i in range(num_env):
            self.env_list.append(mvc_env.py_MvcEnv(NUM_MAX))
            self.g_list.append(graph.py_Graph())

        self.test_env = mvc_env.py_MvcEnv(NUM_MAX)

        # [batch_size, node_cnt]
        self.action_select = tf.sparse_placeholder(tf.float32, name="action_select")
        # [node_cnt, batch_size]
        self.rep_global = tf.sparse_placeholder(tf.float32, name="rep_global")
        # [node_cnt, node_cnt]
        self.n2nsum_param = tf.sparse_placeholder(tf.float32, name="n2nsum_param")
        # [node_cnt, node_cnt]
        self.laplacian_param = tf.sparse_placeholder(tf.float32, name="laplacian_param")
        # [batch_size, node_cnt]
        self.subgsum_param = tf.sparse_placeholder(tf.float32, name="subgsum_param")
        # [batch_size,1]
        self.target = tf.placeholder(tf.float32, [BATCH_SIZE,1], name="target")
        # [batch_size, aux_dim]
        self.aux_input = tf.placeholder(tf.float32, name="aux_input")
        # [node_cnt, 2]
        self.node_input = tf.placeholder(tf.float32, name="node_input")

        #[batch_size, 1]
        if self.IsPrioritizedSampling:
            self.ISWeights = tf.placeholder(tf.float32, [BATCH_SIZE, 1], name='IS_weights')


        # init Q network
        self.loss,self.trainStep,self.q_pred, self.q_on_all,self.Q_param_list = self.BuildNet() #[loss,trainStep,q_pred, q_on_all, ...]
        #init Target Q Network
        self.lossT,self.trainStepT,self.q_predT, self.q_on_allT,self.Q_param_listT = self.BuildNet()
        #takesnapsnot
        self.copyTargetQNetworkOperation = [a.assign(b) for a,b in zip(self.Q_param_listT,self.Q_param_list)]


        self.UpdateTargetQNetwork = tf.group(*self.copyTargetQNetworkOperation)
        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=None)
        #self.session = tf.InteractiveSession()
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config = config)

        # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        self.session.run(tf.global_variables_initializer())

#################################################New code for REND#####################################
    def BuildNet(self):
        # [2, embed_dim]
        w_n2l = tf.Variable(tf.truncated_normal([2, self.embedding_size], stddev=initialization_stddev), tf.float32)
        # [embed_dim, embed_dim]
        p_node_conv = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
        if embeddingMethod == 1:    #'graphsage'
            # [embed_dim, embed_dim]
            p_node_conv2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
            # [2*embed_dim, embed_dim]
            p_node_conv3 = tf.Variable(tf.truncated_normal([2*self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)

        #[reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0:
            #[2*embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            #[reg_hidden1, reg_hidden2]
            # h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden1, self.reg_hidden2], stddev=initialization_stddev), tf.float32)
            #[reg_hidden+aux_dim, 1]
            h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32)
            #[reg_hidden2 + aux_dim, 1]
            last_w = h2_weight
        else:
            #[2*embed_dim, reg_hidden]
            h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            last_w = h1_weight

        ## [embed_dim, 1]
        cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32)

        y_nodes_size = tf.shape(self.subgsum_param)[0]
        # [batch_size, 2]
        y_node_input = tf.ones((y_nodes_size,2))

        #[node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        # no sparse
        input_message = tf.matmul(tf.cast(self.node_input,tf.float32), w_n2l)
        #[node_cnt, embed_dim]  # no sparse
        input_potential_layer = tf.nn.relu(input_message)

        # # no sparse
        # [batch_size, embed_dim]
        y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        #[batch_size, embed_dim]  # no sparse
        y_input_potential_layer = tf.nn.relu(y_input_message)

        #input_potential_layer = input_message
        lv = 0
        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim]
        y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)


        while lv < max_bp_iter:
            lv = lv + 1
            #[node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense
            n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param,tf.float32), cur_message_layer)
            #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
            node_linear = tf.matmul(n2npool, p_node_conv)

            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            y_n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
            #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
            y_node_linear = tf.matmul(y_n2npool, p_node_conv)

            if embeddingMethod == 0: # 'structure2vec'
                #[node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                merged_linear = tf.add(node_linear,input_message)
                #[node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(merged_linear)

                #[batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
                y_merged_linear = tf.add(y_node_linear, y_input_message)
                #[batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(y_merged_linear)
            else:   # 'graphsage'
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2)
                #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                cur_message_layer = tf.nn.relu(tf.matmul(merged_linear, p_node_conv3))

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                y_cur_message_layer_linear = tf.matmul(tf.cast(y_cur_message_layer, tf.float32), p_node_conv2)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                y_merged_linear = tf.concat([y_node_linear, y_cur_message_layer_linear], 1)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                y_cur_message_layer = tf.nn.relu(tf.matmul(y_merged_linear, p_node_conv3))

            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)

        self.node_embedding = cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
        y_potential = y_cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer)
        # # [batch_size, embed_dim, embed_dim]
        temp = tf.matmul(tf.expand_dims(action_embed, axis=2),tf.expand_dims(y_potential, axis=1))
        # [batch_size, embed_dim]
        Shape = tf.shape(action_embed)
        # [batch_size, embed_dim], first transform
        embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])),Shape)

        #[batch_size, 2 * embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0:
            #[batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = tf.matmul(embed_s_a, h1_weight)
            #[batch_size, reg_hidden]
            last_output = tf.nn.relu(hidden)
            #[batch_size, reg_hidden1] * [reg_hidden1, reg_hidden2] = [batch_size, reg_hidden2]


        last_output = tf.concat([last_output, self.aux_input], 1)
        q_pred = tf.matmul(last_output, last_w)

        ## first order reconstruction loss
        loss_recons = 2 * tf.trace(tf.matmul(tf.transpose(cur_message_layer), tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param,tf.float32), cur_message_layer)))
        edge_num = tf.sparse_reduce_sum(self.n2nsum_param)
        loss_recons = tf.divide(loss_recons, edge_num)

        if self.IsPrioritizedSampling:
            self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred), axis=1)    # for updating Sumtree
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred)
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred)

        loss = loss_rl + Alpha * loss_recons


        trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        rep_y = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential)
        # # [node_cnt, embed_dim, embed_dim]
        temp1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2),tf.expand_dims(rep_y, axis=1))
        # [node_cnt embed_dim]
        Shape1 = tf.shape(cur_message_layer)
        # [batch_size, embed_dim], first transform
        embed_s_a_all = tf.reshape(tf.matmul(temp1, tf.reshape(tf.tile(cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)

        #[node_cnt, 2 * embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            #Relu, [node_cnt, reg_hidden1]
            last_output = tf.nn.relu(hidden)


        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), self.aux_input)
        last_output = tf.concat([last_output,rep_aux],1)
        q_on_all = tf.matmul(last_output, last_w)

        return loss,trainStep,q_pred,q_on_all,tf.trainable_variables()

    def gen_graph(self, num_min, num_max):
        max_n = num_max
        min_n = num_min
        cur_n = np.random.randint(max_n - min_n + 1) + min_n
        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        elif self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4)
        ### random weight
        if self.training_type == 'random':
            weights = {}
            for node in g.nodes():
                weights[node] = random.uniform(0,1)
        # ### degree weight
        elif self.training_type == 'degree':
            degree = nx.degree(g)
            maxDegree = max(dict(degree).values())
            weights = {}
            for node in g.nodes():
                weights[node] = degree[node]/maxDegree
        elif self.training_type == 'degree_noise':
            degree = nx.degree(g)
            #maxDegree = max(dict(degree).values())
            mu = np.mean(list(dict(degree).values()))
            std = np.std(list(dict(degree).values()))
            weights = {}
            for node in g.nodes():
                episilon = np.random.normal(mu, std, 1)[0]
                weights[node] = 0.5*degree[node] + episilon
                if weights[node] < 0.0:
                    weights[node] = -weights[node]
            maxDegree = max(weights.values())
            for node in g.nodes():
                weights[node] = weights[node] / maxDegree

        nx.set_node_attributes(g, weights,'weight')
        return g

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        for i in tqdm(range(1000)):
            g = self.gen_graph(num_min, num_max)
            self.InsertGraph(g, is_test=False)

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()

    def InsertGraph(self,g,is_test):
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))


    #pass
    def PrepareValidData(self):
        print('\ngenerating validation graphs...')
        sys.stdout.flush()
        result_degree = 0.0
        result_betweeness = 0.0

        for i in tqdm(range(n_valid)):
            g = self.gen_graph(NUM_MIN, NUM_MAX)
            g_degree = g.copy()
            g_betweenness = g.copy()
            result_degree += self.HXA(g_degree,'HDA')
            result_betweeness += self.HXA(g_betweenness,'HBA')
            self.InsertGraph(g, is_test=True)
        print ('Validation of HDA: %.16f'%(result_degree / n_valid))
        print ('Validation of HBA: %.16f'%(result_betweeness / n_valid))



    def Run_simulator(self, num_seq, eps, TrainSet, n_step):
        num_env = len(self.env_list)
        n = 0
        while n < num_seq:
            for i in range(num_env):
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].isTerminal():
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].isTerminal():
                        n = n + 1
                        self.nStepReplayMem.Add(self.env_list[i], n_step)
                    g_sample= TrainSet.Sample()
                    self.env_list[i].s0(g_sample)
                    self.g_list[i] = self.env_list[i].graph
            if n >= num_seq:
                break

            Random = False
            if random.uniform(0,1) >= eps:
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list])
            else:
                Random = True

            for i in range(num_env):
                if (Random):
                    a_t = self.env_list[i].randomAction()
                else:
                    a_t = self.argMax(pred[i])
                self.env_list[i].step(a_t)

    def PlayGame(self,n_traj, eps):
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)


    def SetupTrain(self, idxes, g_list, covered, actions, target):
        self.m_y = target
        self.inputs['target'] = self.m_y
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['node_input'] = prepareBatchGraph.node_feat
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat


    def SetupPredAll(self, idxes, g_list, covered):
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered)
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['node_input'] = prepareBatchGraph.node_feat
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        return prepareBatchGraph.idx_map_list

    def Predict(self,g_list,covered,isSnapSnot):
        n_graphs = len(g_list)
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            if isSnapSnot:
                result = self.session.run([self.q_on_allT], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2nsum_param: self.inputs['n2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.node_input: self.inputs['node_input'],
                    self.aux_input: np.array(self.inputs['aux_input'])
                })
            else:
                result = self.session.run([self.q_on_all], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2nsum_param: self.inputs['n2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.node_input: self.inputs['node_input'],
                    self.aux_input: np.array(self.inputs['aux_input'])
                })
            raw_output = result[0]
            pos = 0
            pred = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                    else:
                        cur_pred[k] = raw_output[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                pred.append(cur_pred)
            assert (pos == len(raw_output))
        return pred

    def PredictWithCurrentQNet(self,g_list,covered):
        result = self.Predict(g_list,covered,False)
        return result

    def PredictWithSnapshot(self,g_list,covered):
        result = self.Predict(g_list,covered,True)
        return result
    #pass
    def TakeSnapShot(self):
       self.session.run(self.UpdateTargetQNetwork)

    def Fit(self):
        sample = self.nStepReplayMem.Sampling(BATCH_SIZE)
        ness = False
        for i in range(BATCH_SIZE):
            if (not sample.list_term[i]):
                ness = True
                break
        if ness:
            if self.IsDoubleDQN:
                double_list_pred = self.PredictWithCurrentQNet(sample.g_list, sample.list_s_primes)
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)]
            else:
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)

        list_target = np.zeros([BATCH_SIZE, 1])

        for i in range(BATCH_SIZE):
            q_rhs = 0
            if (not sample.list_term[i]):
                if self.IsDoubleDQN:
                    q_rhs=GAMMA * list_pred[i]
                else:
                    q_rhs=GAMMA * self.Max(list_pred[i])
            q_rhs += sample.list_rt[i]
            list_target[i] = q_rhs
        if self.IsPrioritizedSampling:
            return self.fit_with_prioritized(sample.b_idx,sample.ISWeights,sample.g_list, sample.list_st, sample.list_at,list_target)
        else:
            return self.fit(sample.g_list, sample.list_st, sample.list_at,list_target)

    def fit_with_prioritized(self,tree_idx,ISWeights,g_list,covered,actions,list_target):
        loss = 0.0
        n_graphs = len(g_list)
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)

            for j in range(i, i + bsize):
                batch_idxes[j-i] = j

            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            result = self.session.run([self.trainStep,self.TD_errors,self.loss],feed_dict={
                                        self.action_select : self.inputs['action_select'],
                                        self.rep_global : self.inputs['rep_global'],
                                        self.n2nsum_param : self.inputs['n2nsum_param'],
                                        self.laplacian_param : self.inputs['laplacian_param'],
                                        self.subgsum_param : self.inputs['subgsum_param'],
                                        self.node_input: self.inputs['node_input'],
                                        self.aux_input : np.array(self.inputs['aux_input']),
                                        self.ISWeights : np.mat(ISWeights).T,
                                        self.target : self.inputs['target']})
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        return loss / len(g_list)


    def fit(self,g_list,covered,actions,list_target):
        loss = 0.0
        n_graphs = len(g_list)
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)

            for j in range(i, i + bsize):
                batch_idxes[j-i] = j

            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            result = self.session.run([self.loss,self.trainStep],feed_dict={
                                        self.action_select : self.inputs['action_select'],
                                        self.rep_global : self.inputs['rep_global'],
                                        self.n2nsum_param : self.inputs['n2nsum_param'],
                                        self.laplacian_param : self.inputs['laplacian_param'],
                                        self.subgsum_param : self.inputs['subgsum_param'],
                                        self.node_input: self.inputs['node_input'],
                                        self.aux_input: np.array(self.inputs['aux_input']),
                                        self.target : self.inputs['target']})
            loss += result[0]*bsize
        return loss / len(g_list)


    def Train(self):
        self.PrepareValidData()
        self.gen_new_graphs(NUM_MIN, NUM_MAX)
        for i in range(10):
            self.PlayGame(100, 1)
        self.TakeSnapShot()
        eps_start = 1.0
        eps_end = 0.05
        eps_step = 10000.0
        loss = 0

        save_dir = '../models/Model_%s'%(self.g_type)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        VCFile = '%s/ModelVC_%d_%d.csv'%(save_dir, NUM_MIN, NUM_MAX)
        f_out = open(VCFile, 'w')
        for iter in range(MAX_ITERATION):
            start = time.clock()
            ###########-----------------------normal training data setup(start) -----------------##############################
            if iter and iter % 5000 == 0:
                self.gen_new_graphs(NUM_MIN, NUM_MAX)
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)

            if iter % 10 == 0:
                self.PlayGame(10, eps)
            if iter % 300 == 0:
                if(iter == 0):
                    N_start = start
                else:
                    N_start = N_end
                frac = 0.0

                test_start = time.time()
                for idx in range(n_valid):
                    frac += self.Test(idx)
                test_end = time.time()
                f_out.write('%.16f\n'%(frac/n_valid))   #write vc into the file
                f_out.flush()
                print('iter', iter, 'eps', eps, 'average size of vc: ', frac / n_valid)
                print ('testing 100 graphs time: %.8fs'%(test_end-test_start))
                N_end = time.clock()
                print ('300 iterations total time: %.8fs'%(N_end-N_start))
                sys.stdout.flush()
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                self.SaveModel(model_path)
            if iter % UPDATE_TIME == 0:
                self.TakeSnapShot()
            self.Fit()
        f_out.close()


    def findModel(self):
        VCFile = '../models/ModelVC_%d_%d.csv'%(NUM_MIN, NUM_MAX)
        vc_list = []
        for line in open(VCFile):
            vc_list.append(float(line))
        start_loc = 33
        min_vc = start_loc + np.argmin(vc_list[start_loc:])
        best_model_iter = 300 * min_vc
        best_model = '../models/nrange_%d_%d_iter_%d.ckpt' % (NUM_MIN, NUM_MAX, best_model_iter)
        return best_model

    def Evaluate(self, data_test, model_file=None):
        if model_file == None: 
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        n_test = 100
        result_list_score = []
        result_list_time = []
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g_path = '%s/'%data_test + 'g_%d'%i
            g = nx.read_gml(g_path, label='id')
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = self.GetSol(i)
            t2 = time.time()
            result_list_score.append(val)
            result_list_time.append(t2-t1)
        self.ClearTestGraphs()
        score_mean = np.mean(result_list_score)
        score_std = np.std(result_list_score)
        time_mean = np.mean(result_list_time)
        time_std = np.std(result_list_time)
        return  score_mean, score_std, time_mean, time_std,result_list_score


    def EvaluateBaseLine(self, data_test,method="HDA"):
        sys.stdout.flush()
        n_test = 100
        result_list_score = []
        result_list_time = []
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g_path = '%s/'%data_test + 'g_%d'%i
            g = nx.read_gml(g_path, label='id')
            t1 = time.time()
            val = self.HXA(g,method)
            t2 = time.time()
            result_list_score.append(val)
            result_list_time.append(t2-t1)
        score_mean = np.mean(result_list_score)
        score_std = np.std(result_list_score)
        time_mean = np.mean(result_list_time)
        time_std = np.std(result_list_time)
        return  score_mean, score_std, time_mean, time_std,result_list_score

    def EvaluateRealData(self, model_file, data_test,data_test_weight, save_dir, stepRatio=0.0025):  #测试真实数据
        solution_time = 0.0
        test_name = data_test.split('/')[-1]
        save_dir_local = save_dir+'/StepRatio_%.4f'%stepRatio
        if not os.path.exists(save_dir_local):#make dir
            os.mkdir(save_dir_local)
        result_file = '%s/%s' %(save_dir_local, test_name)
        weight_file = '%s/%s_sol_weight.txt' %(save_dir_local, test_name.replace(".txt",""))
        g = nx.Graph()
        with open(data_test) as f:
            for line in f:
                g.add_edge(int(line.split()[0]), int(line.split()[1]))

        weight = {}
        total_weight = 0
        with open(data_test_weight) as f:
            for i,line in enumerate(f):
                weight[i] = float(line)
                total_weight += weight[i]
        nx.set_node_attributes(g, weight,'weight')
        with open(weight_file,"w") as f_weight_out:
            with open(result_file, 'w') as f_out:
                print ('testing')
                sys.stdout.flush()
                print ('number of nodes:%d'%(nx.number_of_nodes(g)))
                print ('number of edges:%d'%(nx.number_of_edges(g)))
                if stepRatio > 0:
                    step = int(stepRatio*nx.number_of_nodes(g)) #step size
                else:
                    step = 1
                # step = 1
                self.InsertGraph(g, is_test=True)
                t1 = time.time()
                solution = self.GetSolution(0,step)
                t2 = time.time()
                solution_time = (t2 - t1)
                for i in range(len(solution)):
                    f_out.write('%d\n' % solution[i])
                for i in range(len(solution)):
                    f_weight_out.write('%f\n' % (weight[solution[i]]/total_weight))

        self.ClearTestGraphs()
        return solution, solution_time

    def find_nodes_not_on_boundary(self,graph: nx.Graph):
        core_num = nx.core_number(graph)
        nodes_on_boundary = set()
        for u, v in graph.edges():
            if (core_num[u] == 1 and core_num[v] == 2) or (core_num[u] == 2 and core_num[v] == 1):
                if core_num[u] == 2:
                    nodes_on_boundary.add(u)
                if core_num[v] == 2:
                    nodes_on_boundary.add(v)

        all_nodes = set(graph.nodes())
        nodes_not_on_boundary = all_nodes - nodes_on_boundary
        return nodes_not_on_boundary

    def EvaluateRealDataTreeBreak(self, model_file, data_test,data_test_weight, save_dir, stepRatio=0.0025):  #测试真实数据
        solution_time = 0.0
        test_name = data_test.split('/')[-1]
        save_dir_local = save_dir+'/StepRatio_%.4f'%stepRatio
        if not os.path.exists(save_dir_local):#make dir
            os.mkdir(save_dir_local)
        result_file = '%s/%s' %(save_dir_local, test_name)
        g = nx.Graph()
        with open(data_test) as f:
            for line in f:
                g.add_edge(int(line.split()[0]), int(line.split()[1]))

        weight = {}
        with open(data_test_weight) as f:
            for i,line in enumerate(f):
                weight[i] = float(line)
        nx.set_node_attributes(g, weight,'weight')

        with open(result_file, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            print ('number of nodes:%d'%(nx.number_of_nodes(g)))
            print ('number of edges:%d'%(nx.number_of_edges(g)))
            if stepRatio > 0:
                step = int(stepRatio*nx.number_of_nodes(g)) #step size
            else:
                step = 1
            # step = 1

            self.ClearTestGraphs()
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            solution = self.GetSolutionTreeBreak(gid=0, original_graph = g, bridge_rmv_ratio=1, step=step)
            t2 = time.time()
            solution_time = (t2 - t1)
            for i in range(len(solution)):
                f_out.write('%d\n' % solution[i])
        self.ClearTestGraphs()
        return solution, solution_time



    def relabel_graph(self, _origin_graph):
        relabeled_graph = _origin_graph.copy()
        node2id = {}
        id2node = {}
        for i, node in enumerate(relabeled_graph.nodes()):
            node2id[node] = i
            id2node[i] = node
        nx.relabel_nodes(relabeled_graph, node2id)
        return relabeled_graph,node2id,id2node

    def GetSolution(self, gid, step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        start = time.time()
        iter = 0
        while (not self.test_env.isTerminal()):
            print ('Iteration:%d'%iter)
            iter += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        return sol

    def GetSolutionTreeBreak(self, gid, original_graph, bridge_rmv_ratio=0.2, step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        iter = 0
        real_time_graph = original_graph.copy()
        bridge_node_num = 0
        selected_bridge_count = 0
        select_nodes = set()
        max_bridge_iter = 10
        bridge_iter = 0
        not_bridge_nodes=[]
        while (not self.test_env.isTerminal()):
            print('Iteration:%d' % iter)
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])

            current_step = step
            remaining_bridge = bridge_node_num - selected_bridge_count

            if remaining_bridge<=0 and bridge_iter < max_bridge_iter:
                not_bridge_nodes = self.find_nodes_not_on_boundary(real_time_graph)
                bridge_node_num = len(real_time_graph)- len(not_bridge_nodes)
                bridge_node_num = int(bridge_node_num * bridge_rmv_ratio)
                selected_bridge_count = 0
                bridge_iter +=1

            if bridge_node_num > 0 and remaining_bridge > 0:
                masked_pred = list_pred[0].copy()
                for node in not_bridge_nodes:
                        masked_pred[node] = -inf

                if remaining_bridge < step:
                    current_step = remaining_bridge
                    print(len(select_nodes))

                batchSol = np.argsort(-masked_pred)[:current_step]
                selected_bridge_count += current_step
            else:
                batchSol = np.argsort(-list_pred[0])[:current_step]

            for new_action in batchSol:
                if new_action not in select_nodes:
                    if not self.test_env.isTerminal():
                        self.test_env.stepWithoutReward(new_action)
                        sol.append(new_action)
                        select_nodes.add(new_action)
                        real_time_graph.remove_node(new_action)
                    else:
                        break

            iter += 1

        return sol


    def EvaluateSol(self, data_test,data_test_weight, sol_file, strategyID=0, reInsertStep=20):
        #evaluate the robust given the solution, strategyID:0,count;2:rank;3:multipy
        sys.stdout.flush()
        g = nx.Graph()
        with open(data_test) as f:
            for line in f:
                g.add_edge(int(line.split()[0]), int(line.split()[1]))
        weight = {}
        normal_weight = 0
        with open(data_test_weight) as f:
            for i,line in enumerate(f):
                weight[i]= float(line)
                normal_weight += float(line)
        nx.set_node_attributes(g, weight,'weight')
        g_inner = self.GenNetwork(g)
        nodes = list(range(nx.number_of_nodes(g)))
        sol = []
        for line in open(sol_file):
            sol.append(int(line))

        sol_left = list(set(nodes)^set(sol))
        if strategyID > 0:
            start = time.time()
            sol_reinsert = self.utils.reInsert(g_inner, sol, sol_left, strategyID, reInsertStep)
            end = time.time()
            print ('reInsert time:%.6f'%(end-start))
        else:
            sol_reinsert = sol
        solution = sol_reinsert + sol_left
        sol_weight = [weight[i] for i in solution]
        sum = 0
        sum_weight_list = []
        for i in range(len(solution)):
            sum += sol_weight[i]/normal_weight
            sum_weight_list.append(sum)
        Robustness = self.utils.getRobustness(g_inner, solution)
        sol_path_list = sol_file.split('/')
        sol_name = sol_path_list[-1].replace(".txt","")
        sol_path = "/".join(sol_path_list[:-1])
        weight_path = sol_path+"/"+sol_name+"_sum_node_weight.txt"
        MaxCCList = self.utils.MaxWccSzList
        with open(weight_path,"w") as f:
            for i in range(len(sum_weight_list)):
                f.write(str(sum_weight_list[i])+"\n")
        return Robustness, MaxCCList, solution


    def Test(self, gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cost = 0.0
        sol = []
        while (not self.test_env.isTerminal()):
            # cost += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            new_action = self.argMax(list_pred[0])
            self.test_env.stepWithoutReward(new_action)
            sol.append(new_action)
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness


    def GetSol(self, gid, step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cost = 0.0
        sol = []
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness, sol


    def SaveModel(self,model_path):
        self.saver.save(self.session, model_path)
        print('model has been saved success!')

    def LoadModel(self,model_path):
        self.saver.restore(self.session, model_path)
        print('restore model from file successfully')


    def GenNetwork(self, g):    #networkx2four
        nodes = g.nodes()
        edges = g.edges()
        weights = []
        for i in range(len(nodes)):
            weights.append(g.nodes[i]['weight'])
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
            W = np.array(weights)
        else:
            A = np.array([0])
            B = np.array([0])
            W = np.array([0])
        return graph.py_Graph(len(nodes), len(edges), A, B, W)

    def HXA(self, g, method):
        # 'HDA', 'HBA', 'HPRA', 'HCA'
        sol = []
        G = g.copy()
        while (nx.number_of_edges(G)>0):
            if method == 'HDA':
                dc = nx.degree_centrality(G)
            elif method == 'HBA':
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA':
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':
                dc = nx.pagerank(G)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values)
            node = keys[maxTag]
            sol.append(node)
            G.remove_node(node)
        solution = sol + list(set(g.nodes())^set(sol))
        solutions = [int(i) for i in solution]
        Robustness = self.utils.getRobustness(self.GenNetwork(g), solutions)
        return Robustness

    def argMax(self, scores):
        n = len(scores)
        pos = -1
        best = -10000000
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos


    def Max(self, scores):
        n = len(scores)
        pos = -1
        best = -10000000
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best

