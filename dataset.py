import os
import numpy as np
import h5py
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import generate_candidates, calculate_IoU_batch


class RawData(Dataset):
    def __init__(self, feature_path, data_path):
        self.feature_path = feature_path

        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        features = h5py.File(self.feature_path, 'r')
        video_names = [x for x in self.data.keys() if x in self.data and x in features]

        # [(video1, sentence_num1), (video2, sentence_num1)...]
        self.data_idx = []
        for video in video_names:
            sentence_num = len(self.data[video]['sentences'])
            for i in range(sentence_num):
                self.data_idx.append((video, i))

    def __getitem__(self, index):
        video_name, sentence_idx = self.data_idx[index]
        item = self.data[video_name]
        duration = item['duration']
        timestamps = item['timestamps'][sentence_idx]
        sentences = item['sentences'][sentence_idx]
        tokens = item['tokens'][sentence_idx]
        embeddings = item['embeddings'][sentence_idx]
        word_mat = item['adj_mats'][sentence_idx]
        vfeats = h5py.File(self.feature_path, 'r')[video_name][:]

        frame_num = vfeats.shape[0]
        start_frame = int(1.0 * frame_num * timestamps[0] / duration)
        end_frame = int(1.0 * frame_num * timestamps[1] / duration)
        if end_frame >= frame_num:
            end_frame = frame_num - 1
        if start_frame > end_frame:
            start_frame = end_frame
        assert start_frame <= end_frame
        assert 0 <= start_frame < frame_num
        assert 0 <= end_frame < frame_num
        label = np.asarray([start_frame, end_frame]).astype(np.int32)

        return vfeats, embeddings, word_mat, label, duration, timestamps

    def __len__(self):
        return len(self.data_idx)


class VideoSentencePair(RawData):
    def __init__(self, feature_path, data_path, max_frames_num, max_words_num, window_widths, window_stride):
        super().__init__(feature_path, data_path)
        self.max_frames_num = max_frames_num
        self.max_words_num = max_words_num
        self.window_stride = window_stride
        self.candidates, _ = generate_candidates(max_frames_num, window_widths, window_stride)

    def __getitem__(self, index):
        vfeats, word_vecs, word_mat, label, duration, timestamps = super().__getitem__(index)
        vfeats, frame_mask, frame_mat, label = self.__video_sampling(vfeats, label)
        word_vecs, word_mask, word_mat = self.__word_padding(word_vecs, word_mat)
        scores = self.__cal_candidate_scores(label, frame_mask, duration)
        return vfeats, frame_mask, frame_mat, word_vecs, word_mask, word_mat, label, scores, duration, timestamps

    def __video_sampling(self, vfeats, label):
        # sample frame to a fixed num and recompute the label
        ori_video_len = vfeats.shape[0]
        frame_mask = np.ones([self.max_frames_num], np.uint8)
        index = np.linspace(start=0, stop=ori_video_len - 1, num=self.max_frames_num).astype(np.int32)
        new_video = []
        for i in range(len(index) - 1):
            start = index[i]
            end = index[i + 1]
            if start == end or start + 1 == end:
                new_video.append(vfeats[start])
            else:
                new_video.append(np.mean(vfeats[start: end], 0))
        new_video.append(vfeats[-1])
        vfeats = np.stack(new_video, 0)
        v_sqrt = np.sqrt((vfeats * vfeats).sum(-1))[:, np.newaxis]
        frame_mat = vfeats.dot(vfeats.T) / v_sqrt.dot(v_sqrt.T)
        frame_mat = np.zeros_like(frame_mat) + np.triu(frame_mat, 2) + np.tril(frame_mat, -2)
        frame_mat[frame_mat == 0] = 1
        frame_mat[frame_mat < 0.7] = 0
        label[0] = min(np.where(index >= label[0])[0])
        if label[1] == ori_video_len - 1:
            label[1] = self.max_frames_num - 1
        else:
            label[1] = max(np.where(index <= label[1])[0])
        if label[1] < label[0]:
            label[0] = label[1]
        return vfeats, frame_mask, frame_mat, label

    def __word_padding(self, word_vecs, word_mat):
        # padding words to a fixed num and adjust the word_mat
        ori_words_num = word_vecs.shape[0]
        if ori_words_num < self.max_words_num:
            word_mask = np.zeros([self.max_words_num], np.uint8)
            word_mask[range(ori_words_num)] = 1
            word_vecs = np.pad(word_vecs, ((0, self.max_words_num - ori_words_num), (0, 0)), mode='constant')
            word_mat = np.pad(word_mat,
                              ((0, self.max_words_num - word_mat.shape[0]),
                               (0, self.max_words_num - word_mat.shape[1])),
                              mode='constant')
        else:
            word_mask = np.ones([self.max_words_num], np.uint8)
            word_vecs = word_vecs[:self.max_words_num]
            word_mat = word_mat[:self.max_words_num, :self.max_words_num]
        word_mat = (word_mat != 0).astype(np.float)
        word_mat = word_mat + word_mat.T - np.ones(word_mat.shape[0])
        return word_vecs, word_mask, word_mat

    def __cal_candidate_scores(self, label, frame_mask, duration):
        seq_len = np.sum(frame_mask, -1)
        labels = np.repeat(np.expand_dims(label, 0), self.candidates.shape[0], 0)  # candidate_num x 2
        IoUs = calculate_IoU_batch(
            (self.candidates[:, 0] * duration / seq_len, (self.candidates[:, 1] + 1) * duration / seq_len),
            (labels[:, 0] * duration / seq_len, (labels[:, 1] + 1) * duration / seq_len)
        )
        max_IoU = np.max(IoUs)
        if max_IoU == 0.0:
            IoUs[label[0] // self.window_stride] = 1
            max_IoU = 1
        IoUs[IoUs < 0.3 * max_IoU] = 0.0
        IoUs = IoUs / max_IoU
        scores = IoUs.astype(np.float32)
        return scores
