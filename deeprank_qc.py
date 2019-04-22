from __future__ import print_function
import sys
import os
import shutil
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from deeprank.dataset import DataLoader, PairGenerator, ListGenerator
from deeprank import utils

tag = 'model/deeprank_qc/'

if not os.path.isdir(tag):
    os.mkdir(tag)

shutil.copy2('./deeprank_qc.py', tag+'deeprank_qc.py')

seed = 4321
#1.set random seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

loader = DataLoader('./config/letor07_mp_fold1.model')

import json
letor_config = json.loads(open('./config/letor07_mp_fold1.model').read())
#device = torch.device("cuda")
#device = torch.device("cpu")
select_device = torch.device("cpu")
rank_device = torch.device("cuda")

Letor07Path = letor_config['data_dir']

letor_config['fill_word'] = loader._PAD_
letor_config['embedding'] = loader.embedding
letor_config['feat_size'] = loader.feat_size
letor_config['vocab_size'] = loader.embedding.shape[0]
letor_config['embed_dim'] = loader.embedding.shape[1]
letor_config['pad_value'] = loader._PAD_

pair_gen = PairGenerator(rel_file=Letor07Path + '/relation.train.fold%d.txt'%(letor_config['fold']), 
                         config=letor_config)

from deeprank import select_module
from deeprank import rank_module

letor_config['q_limit'] = 20
letor_config['d_limit'] = 2000

letor_config['max_match'] = 20
letor_config['win_size'] = 5
select_net = select_module.QueryCentricNet(config=letor_config, out_device=rank_device)
select_net = select_net.to(select_device)
select_net.train()

letor_config["dim_q"] = 1
letor_config["dim_d"] = 1
letor_config["dim_weight"] = 1
letor_config["c_reduce"] = [1, 1]
letor_config["k_reduce"] = [1, 50]
letor_config["s_reduce"] = 1
letor_config["p_reduce"] = [0, 0]

letor_config["c_en_conv_out"] = 4
letor_config["k_en_conv"] = 3
letor_config["s_en_conv"] = 1
letor_config["p_en_conv"] = 1

letor_config["en_pool_out"] = [1, 1]
letor_config["en_leaky"] = 0.2

letor_config["dim_gru_hidden"] = 3

letor_config['lr'] = 0.005
letor_config['finetune_embed'] = False

rank_net = rank_module.DeepRankNet(config=letor_config)
rank_net = rank_net.to(rank_device)
rank_net.embedding.weight.data.copy_(torch.from_numpy(loader.embedding))
rank_net.qw_embedding.weight.data.copy_(torch.from_numpy(loader.idf_embedding))
rank_net.train()
rank_optimizer = optim.Adam(rank_net.parameters(), lr=letor_config['lr'])

log_file = open(tag + 'log.txt', 'w')

def print_log(*msg):
    print(*msg)
    print(*msg, file=log_file)
    sys.stdout.flush()

def to_device(*variables, device):
    return (torch.from_numpy(variable).to(device) for variable in variables)

def to_device_raw(*variables, device):
    return (variable.to(device) for variable in variables)

def show_text(x):
    print_log(' '.join([loader.word_dict[w.item()] for w in x]))

def evaluate_test(select_net_e, rank_net_e):
    list_gen = ListGenerator(rel_file=Letor07Path+'/relation.test.fold%d.txt'%(letor_config['fold']),
                             config=letor_config)
    map_v = 0.0
    map_c = 0.0
    
    with torch.no_grad():
        for X1, X1_len, X1_id, X2, X2_len, X2_id, Y, F in \
            list_gen.get_batch(data1=loader.query_data, data2=loader.doc_data):
            #print_log(X1.shape, X2.shape, Y.shape)
            X1, X1_len, X2, X2_len, Y, F = to_device(X1, X1_len, X2, X2_len, Y, F, device=select_device)
            X1, X2, X1_len, X2_len, X2_pos = select_net_e(X1, X2, X1_len, X2_len, X1_id, X2_id)
            X2, X2_len = utils.data_adaptor(X2, X2_len, select_net, rank_net, letor_config)
            #print_log(X1.shape, X2.shape, Y.shape)
            pred = rank_net_e(X1, X2, X1_len, X2_len, X2_pos)
            map_o = utils.eval_MAP(pred.tolist(), Y.tolist())
            #print_log(pred.shape, Y.shape)
            map_v += map_o
            map_c += 1.0
        map_v /= map_c
    
    print_log('[Test]', map_v)
    return map_v

import time

rank_loss_list = []
select_loss_list = []
ret_map_list = []

it = 1000

start_t = time.time()
for i in range(10001):
    print_log('[Iter]', i)
    # One Step Forward
    X1, X1_len, X1_id, X2, X2_len, X2_id, Y, F = \
        pair_gen.get_batch(data1=loader.query_data, data2=loader.doc_data)
    X1, X1_len, X2, X2_len, Y, F = \
        to_device(X1, X1_len, X2, X2_len, Y, F, device=select_device)
    X1, X2, X1_len, X2_len, X2_pos = select_net(X1, X2, X1_len, X2_len, X1_id, X2_id)
    X2, X2_len = utils.data_adaptor(X2, X2_len, select_net, rank_net, letor_config)
    output = rank_net(X1, X2, X1_len, X2_len, X2_pos)
    
    # Update Rank Net
    rank_loss = rank_net.pair_loss(output, Y)
    print_log('[Rank Loss]', rank_loss.item())
    rank_loss_list.append(rank_loss.item())
    if True or i // it % 2 == 0:
    #if i < 1000:
        rank_optimizer.zero_grad()
        rank_loss.backward()
        rank_optimizer.step()
    
    if i % 200 == 0:
        ret_map = evaluate_test(select_net, rank_net)
        ret_map_list.append(ret_map)
    
end_t = time.time()
print_log('Time Cost: %s s' % (end_t-start_t))

fout = open(tag + 'select_loss.log', 'w')
for x in select_loss_list:
    fout.write(str(x))
    fout.write('\n')
fout.close()

fout = open(tag + 'rank_loss.log', 'w')
for x in rank_loss_list:
    fout.write(str(x))
    fout.write('\n')
fout.close()

fout = open(tag + 'map.log', 'w')
for x in ret_map_list:
    fout.write(str(x))
    fout.write('\n')
fout.close()

torch.save(select_net, tag + "select.model")
torch.save(rank_net, tag + "rank.model")

select_net_e = torch.load(f=tag + 'select.model')
rank_net_e = torch.load(f=tag + 'rank.model')

evaluate_test(select_net_e, rank_net_e)
