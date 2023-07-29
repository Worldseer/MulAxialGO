import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
AADICT = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10,
    'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19, 'V':20}
    

def effectivelen(seq,W):
    for i in range(3,W+1,1):
        if i*i < len(seq) and i!=W:
            continue
        else:
            break
    return i

def sequence2num(sequence):
    data = []
    for acid in sequence:
        data.append(int(AADICT.get(acid,'0')))
    return torch.tensor(data) 

def expand_sequence(sequence,expand_len):
    if len(sequence) > expand_len:
        return sequence[:expand_len]
    else:
        return sequence+"*"*(expand_len-len(sequence))
        
def genimatrix(sequence,out_size): #x,y are the coordinates of the current data, the default value is (0,0), and the sequence is the sequence that has been converted to a numerical representation.
    W,H = effectivelen(sequence,out_size),effectivelen(sequence,out_size)
    matrix = torch.zeros((out_size,out_size))
    if len(sequence) >W*H: 
        sequence = sequence[0:W*H]
    seqs = sequence2num(sequence)
    x,y = 0,0
    step = 0
    for seq in seqs:
        if step<W:
            matrix[x,y] = seq
            y += 1
            step += 1             
        if step == W:
            step = 0
            y = 0
            x += 1
            continue  
    return torch.LongTensor(matrix.numpy())    

    
def token2num(seq):#output torch.out_size([2304])
    data = torch.zeros(len(seq),dtype=torch.int32)
    for idx,acid in enumerate(seq):
        data[idx] = int(AADICT.get(acid,'0'))
    return data    

    
class dataset_loder(Dataset):
    def __init__(self,protein_df,go_df,out_size):
        super(dataset_loder,self).__init__()
        self.protein_df = protein_df
        if "terms" in go_df:
            self.terms  = go_df['terms'].values.flatten()
        if "gos" in go_df:
            self.terms  = go_df['gos'].values.flatten()
        self.go_dict = {v: i for i, v in enumerate(self.terms)}
        self.length = len(protein_df)
        self.out_size = out_size

        
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        labels = torch.zeros(len(self.go_dict), dtype=torch.float32)
        esm_1b = self.protein_df.iloc[idx].esm1b
        seq = self.protein_df.iloc[idx].sequences# Extraction of the corresponding protein sequence
        imatrix  = genimatrix(seq,self.out_size)
        if "prop_annotations" in self.protein_df:
            for go in self.protein_df.iloc[idx].prop_annotations:
                if go in self.go_dict:
                    index = self.go_dict[go]
                    labels[index] = 1 
        if "annotations" in self.protein_df:
            for go in self.protein_df.iloc[idx].annotations:
                if go in self.go_dict:
                    index = self.go_dict[go]
                    labels[index] = 1
        return imatrix,esm_1b,labels
        


def datasetloader(data_df,go_df,batch_size,Msize,shuffle):# input df data, output DataLoader
    dataloader = DataLoader(dataset=dataset_loder(data_df,go_df,Msize),shuffle=shuffle,batch_size=batch_size,num_workers=6)
    return dataloader

  
