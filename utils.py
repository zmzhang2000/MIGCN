import numpy as np


def generate_candidates(max_frames_num, window_widths, window_stride=1):
    widths = np.array(window_widths)
    candidates = []
    for w in widths:
        for start in range(0, max_frames_num - w + 1, window_stride):
            candidates.append([start, start + w - 1])
    return np.array(candidates), tuple(int(w) for w in widths)


def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
