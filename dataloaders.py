#from torchvision import transforms, utils
import torch
from torch.utils.data import Dataset, DataLoader
from aux_functions import seq2vec, conservation2num
from masking_functions import mutate_at_random, get_crop
class ProtSeqDataset(Dataset):
    def __init__(self, input_prot_dict,  n_tokens, le, conservation_key,
				 padding_index, masking_index, masking_function = mutate_at_random, 
                 seqlength = 64, n_crop_per_protein = 1, mut_prob=.05, device=None, beta=1):
        """
        Dataset consisting of masked crops of different proteins
        """
        # Total number of crops is #(proteins)*#(crops per protein)
        n_proteins = len(input_prot_dict)*n_crop_per_protein
        inputmat = torch.zeros((n_proteins, seqlength, n_tokens))
        outputmat = torch.zeros((n_proteins, seqlength, n_tokens))
        
        # Vector for masking the padded indices
        notPadded = torch.zeros(n_proteins)

        # Initialize with the padding index
        inputmat[:, :, padding_index] = 1 
        outputmat[:, :, padding_index] = 1
        counter = 0
        protnames = []
        for _, protname in enumerate(input_prot_dict):
            seq = input_prot_dict[protname]['seq']
            # Get one hot representation of AA sequence
            vec = seq2vec(seq, le, n_tokens)

            # Get the conservation scores
            conservation_scores = conservation2num(input_prot_dict[protname][conservation_key])
            for randomsample in range(n_crop_per_protein):
                cropvec, length, start, end = get_crop(vec, seqlength) # Get a random 64 AA crop
                crop_conservation_scores = conservation_scores[start:end] # Crop conservation scores
                mutvec, weightvec = masking_function(cropvec, crop_conservation_scores, masking_index, mutprob = mut_prob, beta=beta) # Mask 
                inputmat[counter, :length, :] = mutvec # If the length is < 64, the rest is padding
                outputmat[counter, :length, :] = cropvec # If the length is < 64, the rest is padding
                notPadded[counter] = end - start
                protnames.append(protname)
                counter += 1
        self.beta = beta
        self.protnames = protnames
        self.inputmat = inputmat.to(device)
        self.outputmat = outputmat.to(device)
        self.notPadded = notPadded.to(device)
        self.device = device
        
    def __len__(self):
        return len(self.inputmat)
    def __getitem__(self, idx):
        inputprot = self.inputmat[idx]
        outputprot = self.outputmat[idx]
        realProt = torch.tensor(self.notPadded[idx].clone().detach(), dtype=torch.long)
        realProtMarker = torch.zeros(inputprot.shape[0]).to(self.device)
        realProtMarker[:realProt] = 1
        realProtMarker = realProtMarker == 0
        sample = {'input': inputprot, 
                  'output': outputprot, 
                  'mask' : realProtMarker,
                 }
        return sample