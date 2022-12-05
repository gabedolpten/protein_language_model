import torch
import numpy as np
arr = np.asarray

def mutate_at_random(vec, conservation_score, masking_index, mutprob = torch.tensor(.05), **kwargs):
    """
    Mask indices uniformly at random
    """
    n = len(vec)
    mutvec = vec.detach().clone()

    # Add a uniform mutation probability 
    weightvec = torch.ones(n)*mutprob
    assert (weightvec.mean().isclose(torch.tensor(mutprob, dtype=torch.float)))
    to_mutate = torch.bernoulli(weightvec)
    to_mutate = to_mutate.bool()
    mutvec[to_mutate, :] = 0
    mutvec[to_mutate, masking_index] = 1
    return mutvec, weightvec

def mutate_with_conservation_score(vec, conservation_scores, masking_index, mutprob = .05, beta = 1, **kwargs):
    """
    Mask indices weighted by conservation score
    """
    n = len(vec)
    mutvec = vec.detach().clone()
    beta = torch.Tensor([beta])
    # Conservation scores range from 0 <-> 5, normalize to get to 0 <-> 1-mutprob 
    conservation_scores = (1-mutprob)*arr(conservation_scores)/5 
    
    # Add a uniform mutation probability 
    uniform_weightvec = torch.ones(n, dtype=torch.long)*mutprob # Uniform baseline weight
    # Add conservation probability and uniform mutation probability
    unnormalized_weightvec = ((uniform_weightvec + beta*conservation_scores)/(beta+mutprob)).float() # Increase weight by conservation scores

    # Renormalize sum of probabilities to have same overall mutprob
    weightvec = (unnormalized_weightvec/unnormalized_weightvec.mean())*mutprob 
    assert (weightvec.mean().isclose(torch.tensor([mutprob], dtype=torch.float)))
    assert torch.max(weightvec) < 1, print(torch.max(weightvec))

    # Indices to mask
    to_mutate = torch.bernoulli(weightvec)
    to_mutate = to_mutate.bool()
    mutvec[to_mutate, :] = 0
    mutvec[to_mutate, masking_index] = 1
    return mutvec, weightvec


import random
def get_crop(mutvec, seqlength):
    """
    Crop protein to be a length of seqlength
    """
    # Length of protein
    n = mutvec.shape[0]
    # If protein > seqlength get a random position
    if n > seqlength:
        lastend = n - seqlength
        start = random.randint(0, lastend)
        end = start + seqlength
        cropvec = mutvec[start:end, ]
    # Else select entire protein and pad the rest
    else:
        cropvec = mutvec
        start = 0; end = n
    return cropvec, n, start, end