import numpy as np

def oversample_indices(y):
    num_classes = y.shape[1]
    y = np.argmax(y, axis = 1)
    idx = [np.where(y == i)[0] for i in range(num_classes)]
    n = np.histogram(y, bins = num_classes)[0]
    max_n = np.max(n)
    reps = [int(max_n/i) for i in n]
    idx_reps = [np.concatenate([idx[i]]*reps[i], axis = 0) for i in range(num_classes)]
    idx_reps = np.concatenate(idx_reps, axis = 0)
    return idx_reps
