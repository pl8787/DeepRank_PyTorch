#! encoding: utf-8
#! author: pangliang

import json
import numpy as np
import random
import torch
import torch.nn.functional as F

# Read Word Dict and Inverse Word Dict
def read_word_dict(filename):
    word_dict = {}
    iword_dict = {}
    for line in open(filename):
        line = line.strip().split()
        word_dict[int(line[1])] = line[0]
        iword_dict[line[0]] = int(line[1])
    print('[%s]\n\tWord dict size: %d' % (filename, len(word_dict)))
    return word_dict, iword_dict

# Read Embedding File
def read_embedding(filename):
    embed = {}
    for line in open(filename):
        line = line.strip().split()
        embed[int(line[0])] = list(map(float, line[1:]))
    print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)))
    return embed

# Read old version data
def read_data_old_version(filename):
    data = []
    for idx, line in enumerate(open(filename)):
        line = line.strip().split()
        len1 = int(line[1])
        len2 = int(line[2])
        data.append([list(map(int, line[3:3+len1])),
                     list(map(int, line[3+len1:]))])
        assert len2 == len(data[idx][1])
    print('[%s]\n\tInstance size: %d' % (filename, len(data)))
    return data

# Read Relation Data
def read_relation(filename):
    data = []
    for line in open(filename):
        line = line.strip().split()
        data.append( (int(line[0]), line[1], line[2]) )
    print('[%s]\n\tInstance size: %s' % (filename, len(data)))
    return data

# Read Data Dict
def read_data(filename):
    data = {}
    for line in open(filename):
        line = line.strip().split()
        data[line[0]] = list(map(int, line[2:]))
    print('[%s]\n\tData size: %s' % (filename, len(data)))
    return data

# Convert Embedding Dict 2 numpy array
def convert_embed_2_numpy(embed_dict, max_size=0, embed=None):
    feat_size = len(embed_dict[list(embed_dict.keys())[0]])
    if embed is None:
        embed = np.zeros( (feat_size, max_size), dtype = np.float32 )
    for k in embed_dict:
        embed[k] = np.array(embed_dict[k])
    print('Generate numpy embed:', embed.shape)
    return embed

# evaluate MAP
def eval_MAP(pred, gt):
    map_value = 0.0
    r = 0.0
    c = list(zip(pred, gt))
    random.shuffle(c)
    c = sorted(c, key = lambda x:x[0], reverse=True)
    for j, (p, g) in enumerate(c):
        if g != 0:
            r += 1
            map_value += r/(j+1.0)
    if r == 0:
        return 0.0
    else:
        return map_value/r

# select-rank adaptor
def data_adaptor(d_data, d_len, select_net, rank_net, config):
    if select_net.output_type != rank_net.input_type:
        if select_net.output_type == 'LL':
            if rank_net.input_type == 'S':
                d_data_new = torch.cat(
                    [ F.pad(x.view(1, -1), 
                          pad=(0, config['d_limit']),
                          value=config['pad_value']
                      )[:,:config['d_limit']] for x in d_data],
                    dim=0).to(d_data[0].device)

                d_len_new = torch.clamp(torch.tensor([x.shape[0] * x.shape[1] for x in d_data]), max=config['d_limit']).to(d_data[0].device)
                return d_data_new, d_len_new
            elif rank_net.input_type == 'L':
                return d_data, d_len
        elif select_net.output_type == 'L':
            if rank_net.input_type == 'S':
                d_data_new = [x.view(-1) for x in d_data]
                d_len_new = [x.shape[0] * x.shape[1] for x in d_data]
                return d_data_new, d_len_new
    return d_data, d_len
                
