import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_candidates


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.gru.flatten_parameters()

    def forward(self, x, seq_len, max_num_frames):
        sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
        _, original_idx = torch.sort(sorted_idx, dim=0, descending=False)
        if self.batch_first:
            sorted_x = x.index_select(0, sorted_idx)
        else:
            sorted_x = x.index_select(1, sorted_idx)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            sorted_x, sorted_seq_len.cpu().data.numpy(), batch_first=self.batch_first)

        out, state = self.gru(packed_x)

        unpacked_x, unpacked_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=self.batch_first)

        if self.batch_first:
            out = unpacked_x.index_select(0, original_idx)
            if out.shape[1] < max_num_frames:
                out = F.pad(out, [0, 0, 0, max_num_frames - out.shape[1]])
        else:
            out = unpacked_x.index_select(1, original_idx)
            if out.shape[0] < max_num_frames:
                out = F.pad(out, [0, 0, 0, 0, 0, max_num_frames - out.shape[0]])

        return out


class NodeInitializer(nn.Module):
    def __init__(self, node_num, input_dim, node_dim, dropout):
        super().__init__()
        self.node_num = node_num
        self.dropout = dropout
        self.rnn = DynamicGRU(input_dim, node_dim >> 1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(node_dim, node_dim)

    def forward(self, x, mask):
        length = mask.sum(dim=-1)
        x = self.rnn(x, length, self.node_num)
        x = F.leaky_relu(self.fc(x))
        x = F.dropout(x, self.dropout, self.training)
        return x


class GraphConvolution(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.node_dim = node_dim
        self.wvv = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.wss = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.wvs = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.wsv = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.wgatev = nn.Linear(self.node_dim << 1, self.node_dim)
        self.wgates = nn.Linear(self.node_dim << 1, self.node_dim)

    def forward(self, v, avv, s, ass):
        vs = torch.matmul(v, s.transpose(2, 1))
        avs = torch.softmax(vs, -1)
        asv = torch.softmax(vs.transpose(2, 1), -1)
        v = F.leaky_relu(self.wvv(torch.matmul(avv, v)))
        s = F.leaky_relu(self.wss(torch.matmul(ass, s)))

        hv = self.wsv(torch.matmul(avs, s))  # batch_size x T x node_dim
        zv = torch.sigmoid(self.wgatev(torch.cat([v, hv], dim=-1)))  # batch_size x T x node_dim
        v = zv * v + (1 - zv) * hv  # batch_size x T x node_dim
        v = F.leaky_relu(v)

        hs = self.wvs(torch.matmul(asv, v))
        zs = torch.sigmoid(self.wgates(torch.cat([s, hs], dim=-1)))
        s = zs * s + (1 - zs) * hs
        s = F.leaky_relu(s)
        return v, s


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.alpha = opt.alpha
        self.beta = opt.beta
        self.dropout = opt.dropout
        self.vnode_initializer = NodeInitializer(node_num=opt.max_frames_num,
                                                 input_dim=opt.frame_feature_dim,
                                                 node_dim=opt.node_dim, dropout=self.dropout)
        self.wnode_initializer = NodeInitializer(node_num=opt.max_words_num,
                                                 input_dim=opt.word_feature_dim,
                                                 node_dim=opt.node_dim, dropout=self.dropout)
        self.s_embed = nn.Linear(opt.max_words_num, 1)
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(opt.node_dim)
            for _ in range(opt.gcn_layers_num)
        ])
        self.candidates, self.window_widths = generate_candidates(opt.max_frames_num,
                                                                  opt.window_widths, opt.window_stride)
        self.candidates = torch.from_numpy(self.candidates).float().cuda()
        self.conv_cls = nn.ModuleList([
            nn.Conv1d(opt.node_dim << 1, 1, w * 2, padding=w // 2, stride=opt.window_stride)
            for w in self.window_widths
        ])
        self.conv_reg = nn.ModuleList([
            nn.Conv1d(opt.node_dim << 1, 2, w * 2, padding=w // 2, stride=opt.window_stride)
            for w in self.window_widths
        ])
        self.criterion_BCE = nn.BCEWithLogitsLoss()
        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_reg = nn.SmoothL1Loss()

    def forward(self, vfeats, frame_mask, frame_mat, wfeats, word_mask, word_mats, label, scores):
        v = self.vnode_initializer(vfeats, frame_mask)  # batch_size x T x node_dim
        s = self.wnode_initializer(wfeats, word_mask)  # batch_size x L x node_dim

        for g in self.gcn_layers:
            v, s = g(v, frame_mat, s, word_mats)
            v = F.dropout(v, self.dropout, self.training)
            s = F.dropout(s, self.dropout, self.training)

        s = F.leaky_relu(self.s_embed(s.permute(0, 2, 1)).permute(0, 2, 1))
        s = s.expand(s.shape[0], v.shape[1], s.shape[2])  # batch_size x window_num x node_dim
        v = torch.cat((v, s), 2)  # batch_size x window_num x 2node_dim
        predict_scores = torch.cat([
            self.conv_cls[i](v.permute(0, 2, 1)).permute(0, 2, 1)
            for i in range(len(self.conv_cls))
        ], dim=1).squeeze(2)
        offset = torch.cat([
            self.conv_reg[i](v.permute(0, 2, 1)).permute(0, 2, 1)
            for i in range(len(self.conv_reg))
        ], dim=1)

        if self.training:
            indices = scores.max(dim=1)[1]
        else:
            indices = predict_scores.max(dim=1)[1]
        predict_box = self.candidates[indices]  # batch_size x 2
        predict_reg = offset[range(offset.shape[0]), indices]  # batch_size x 2
        refined_box = predict_box + predict_reg  # batch_size x 2

        bce_loss = self.criterion_BCE(predict_scores, scores)

        indices_label = scores.max(-1)[-1]  # batch_size x 1
        ce_loss = self.criterion_CE(predict_scores, indices_label)

        reg_loss = self.criterion_reg(refined_box, label.float())

        loss = bce_loss + self.alpha * ce_loss + self.beta * reg_loss
        return refined_box, loss
