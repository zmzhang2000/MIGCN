from utils import SyntacticDependencyParser, WordEmbedding
import numpy as np
import pickle
import json
from tqdm import tqdm
import os
import h5py
from collections import defaultdict

'''
info:
{
    'v_QOlSCBRmfWY':
    {
       'duration':
       82.73,
       
       'timestamps':
       [[0.83, 19.86], [17.37, 60.81], [56.26, 79.42]],
       
       'sentences':
       ['A young woman is seen standing in a room and leads into her dancing.', 
       ' The girl dances around the room while the camera captures her movements.', 
       ' She continues dancing around the room and ends by laying on the floor.'],
       
       'tokens':
       [['a', 'young', 'woman', 'is', 'seen', 'standing', 'in', 'a', 'room', 'and', 'leads', 'into', 'her', 'dancing'],
       ...,
       ...],
       
       'embeddings':
       [numpy array with shape [word_num, embedding_size],
       ...,
       ...],
       
       'adj_mats':
        [numpy array with shape [word_num, word_num],
       ...,
       ...],
    }
    ...
}

features:
fin = h5py.File(data_path, 'r')
fin['v_---9CpRcKoU'][:] = numpy array with shape [moment_num, feat_size]
'''


def process_activitynet_info(aim_path):
    data_path = "data/raw_data/activitynet/captions/"
    with open(os.path.join(data_path, 'train.json'), 'r') as f:
        train = json.load(f)
    with open(os.path.join(data_path, 'val_1.json'), 'r') as f:
        val1 = json.load(f)
    with open(os.path.join(data_path, 'val_2.json'), 'r') as f:
        val2 = json.load(f)

    for idx, data in enumerate((train, val1, val2)):
        for name in tqdm(data.keys(), desc="activitynet info" + str(idx)):
            sentences = data[name]['sentences']
            tokens = []
            embeddings = []
            adj_mats = []
            for s in sentences:
                token, adj_mat = parser.parse(s)
                embedding = np.zeros((len(token), 300))
                for i in range(len(token)):
                    embedding[i] = word2vec.get(token[i])
                tokens.append(token)
                embeddings.append(embedding)
                adj_mats.append(adj_mat)
            data[name]['tokens'] = tokens
            data[name]['embeddings'] = embeddings
            data[name]['adj_mats'] = adj_mats

    if len(val1) < len(val2):
        val1, val2 = val2, val1  # val1 >= val2
    for k in val1:
        if k in val2:
            for kk in val1[k]:
                if kk != 'duration':
                    val1[k][kk].extend(val2[k][kk])
    val = val1

    with open(os.path.join(aim_path, 'train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(aim_path, 'val.pkl'), 'wb') as f:
        pickle.dump(val, f)


def process_activitynet_c3d(aim_path):
    data_path = "data/raw_data/activitynet/sub_activitynet_v1-3.c3d.hdf5"
    fin = h5py.File(data_path, 'r')
    fout = h5py.File(os.path.join(aim_path, "activitynet.c3d.hdf5"), 'w')
    for k in tqdm(fin, desc="activitynet c3d"):
        fout[k] = fin[k]['c3d_features'][:]


def process_charades_info(aim_path):
    with open('data/raw_data/charades/ref_info/charades_movie_length_info.txt',
              'r') as f:
        lines = f.readlines()
        video_duration = dict()
        for line in tqdm(lines, desc='read video duration'):
            line = line.split()
            video_duration[line[0]] = float(line[1])

    train = defaultdict(dict)
    with open('data/raw_data/charades/charades_sta_train.txt', 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc='charades info train'):
        info, sent = line[:-1].split('##')
        video, start, end = info.split()
        start, end = float(start), float(end)

        token, adj_mat = parser.parse(sent)
        embedding = np.zeros((len(token), 300))
        for i in range(len(token)):
            embedding[i] = word2vec.get(token[i])

        train[video]['duration'] = video_duration[video]
        for k in ('timestamps', 'sentences', 'tokens', 'embeddings', 'adj_mats'):
            if k not in train[video]:
                train[video][k] = list()
        train[video]['timestamps'].append((start, end))
        train[video]['sentences'].append(sent)
        train[video]['tokens'].append(token)
        train[video]['embeddings'].append(embedding)
        train[video]['adj_mats'].append(adj_mat)
    with open(os.path.join(aim_path, 'train.pkl'), 'wb') as f:
        pickle.dump(dict(train), f)

    val = defaultdict(dict)
    with open('data/raw_data/charades/charades_sta_test.txt', 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc='charades info val'):
        info, sent = line[:-1].split('##')
        video, start, end = info.split()
        start, end = float(start), float(end)

        token, adj_mat = parser.parse(sent)
        embedding = np.zeros((len(token), 300))
        for i in range(len(token)):
            embedding[i] = word2vec.get(token[i])

        val[video]['duration'] = video_duration[video]
        for k in ('timestamps', 'sentences', 'tokens', 'embeddings', 'adj_mats'):
            if k not in val[video]:
                val[video][k] = list()
        val[video]['timestamps'].append((start, end))
        val[video]['sentences'].append(sent)
        val[video]['tokens'].append(token)
        val[video]['embeddings'].append(embedding)
        val[video]['adj_mats'].append(adj_mat)
    with open(os.path.join(aim_path, 'val.pkl'), 'wb') as f:
        pickle.dump(dict(val), f)


def process_charades_c3d_twostream(aim_path):
    data_path = "data/raw_data/charades"
    origin_feats_path = os.path.join(data_path, 'Charades_localfeature_npy')
    fc3d = h5py.File(os.path.join(aim_path, 'charades.c3d.hdf5'), 'w')
    fts = h5py.File(os.path.join(aim_path, 'charades.two-stream.hdf5'), 'w')

    video_info = defaultdict(list)
    npy_list = os.listdir(origin_feats_path)
    for npy in tqdm(npy_list, desc='charades c3d two-stream collecting'):
        video, start, end = npy[:-4].split('_')
        start, end = int(start[:-2]), int(end[:-2])
        video_info[video].append((start, end))
    for video in tqdm(video_info, desc='charades c3d two-stream sorting and stacking'):
        video_info[video].sort(key=lambda x: x[0])
        feats = []
        for start, end in video_info[video]:
            start, end = str(start) + '.0', str(end) + '.0'
            npy = video + '_' + start + '_' + end + '.npy'
            feats.append(np.load(os.path.join(origin_feats_path, npy)))
        feats = np.vstack(feats)
        fc3d[video] = feats[:, 0:4096]
        fts[video] = feats[:, 4096:]


parser = SyntacticDependencyParser()
word2vec = WordEmbedding()

activitynet_path = "data/activitynet"
charades_path = "data/charades"
if not os.path.exists(activitynet_path):
    os.makedirs(activitynet_path)
if not os.path.exists(charades_path):
    os.makedirs(charades_path)

process_activitynet_info(activitynet_path)
process_activitynet_c3d(activitynet_path)

process_charades_info(charades_path)
process_charades_c3d_twostream(charades_path)
