import torch
import numpy as np
arr = np.asarray

def conservation2num(conservation_seq):
    return torch.from_numpy(arr(conservation_seq.split(',')).astype(float))

def seq2vec(seq, le, n_tokens):
    n = len(seq)
    seqvec = [x for x in seq]
    aa_ind = torch.arange(0, n, dtype=torch.long)
    inds = le.transform(seqvec)
    vec = torch.zeros((n, n_tokens))
    vec[aa_ind, inds] = 1
    return vec