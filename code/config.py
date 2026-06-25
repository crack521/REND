#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 2026

REND hyper parameters
"""

# Hyper Parameters:
GAMMA = 1  # decay rate of past observations
UPDATE_TIME = 1000
EMBEDDING_SIZE = 64
MAX_ITERATION = 1000000
LEARNING_RATE = 0.0001   #dai
MEMORY_SIZE = 500000
Alpha = 0.001 ## weight of reconstruction loss
########################### hyperparameters for priority(start)#########################################
epsilon = 0.0000001  # small amount to avoid zero priority
alpha = 0.6  # [0~1] convert the importance of TD error to priority
beta = 0.4  # importance-sampling, from initial value increasing to 1
beta_increment_per_sampling = 0.001
TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
N_STEP = 5
NUM_MIN = 15
NUM_MAX = 20
REG_HIDDEN = 32
M = 4  # how many edges selected each time for BA model
BATCH_SIZE = 64
initialization_stddev = 0.01  # 权重初始化的方差
n_valid = 200
aux_dim = 4
num_env = 1
inf = 2147483647/2
#########################  embedding method ##########################################################
max_bp_iter = 3
aggregatorID = 0 #0:sum; 1:mean; 2:GCN
embeddingMethod = 1   #0:structure2vec; 1:graphsage
