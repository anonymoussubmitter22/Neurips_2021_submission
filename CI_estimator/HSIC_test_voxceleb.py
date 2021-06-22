import os 
import sys 
from sklearn.preprocessing import normalize
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
import os
from tqdm import tqdm
from collections import Counter
from scipy.io.wavfile import read
import time
import pandas as pd
from functools import partial 
melfs_dir = sys.argv[1]
gemap_dir = sys.argv[2]
gemap_value = sys.argv[3]
K_matrices = sys.argv[4]
outdir=sys.argv[5]
results_file= open(os.path.join(outdir,"results_per_speaker_"+gemap_value+".txt") , "w")
def rbf_value(x1,x2, sigma=0.05) : 
    diff = np.abs(x1-x2)**2
    return np.exp(-diff / (2*sigma)) 
melfs_files = os.listdir(melfs_dir)
melfs_paths = [os.path.join(melfs_dir, x) for x in melfs_files]
speakers = list(np.unique([x.split("_")[0] for x in melfs_files]))
print(speakers)
def per_speaker_matrix(speaker, paths) : 
    print(f"started speaker :{speaker}")
    melfs_paths =[paths[x] for x in range(len(paths)) if melfs_files[x].split("_")[0]==speaker]
    N=len(melfs_paths)
    L_matrix = np.zeros((N,N))
    #print(f"size of the matrix : {N}")
    all_values = []
    for i in (range(N)):
        name = melfs_paths[i].split("/")[-1].split(".")[0]
        fname = os.path.join(gemap_dir, name+".csv")
        value = pd.read_pickle(fname)[gemap_value][0]
        for j in range(i+1): 
            name = melfs_paths[j].split("/")[-1].split(".")[0]
            fname = os.path.join(gemap_dir, name+".csv")
            value2 = pd.read_pickle(fname)[gemap_value][0]
            value_rbf = rbf_value(value,value2)
            L_matrix[i,j]=value_rbf

    for i in range(N):
        for j in range(i,N): 
            L_matrix[i,j] = L_matrix[j,i]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    H = np.zeros((N,N))
    for i in range(N): 
        for j in range(N):
            v= (i==j)
            H[i,j] =v -1/(N*N)
    K_matrix = np.load(os.path.join(K_matrices, "K_matrix_"+speaker+".npy"))
    print(K_matrix.shape)
    test_value = np.trace(K_matrix@H@L_matrix@H) /((N-1)**2)

    print(f"for speaker : {speaker} , and value  : {gemap_value } , HCIS_test = {test_value}")
    results_file.write(f"for speaker : {speaker} , HCIS_test = {test_value}")
    results_file.write("\n")
    return test_value

v = mp.cpu_count()
parallel=True
Ns = []
for speaker in speakers : 
    M =len([melfs_paths[x] for x in range(len(melfs_paths)) if  melfs_files[x].split("_")[0]==speaker])
    Ns.append(M)
part_func = partial(per_speaker_matrix,paths = melfs_paths) 
if parallel : 
    cpus_needed = min(len(speakers),v)
    print(f"using {cpus_needed} cpus")
    p = Pool(cpus_needed)
    p.map(part_func, speakers)
    results_here = list(p.imap(part_func, speakers))
else : 
    results_here =[]
    for speaker in tqdm(speakers): 

        results_here.append(part_func(speaker))

total_files = np.sum(Ns)
normal_mean = np.mean(results_here)
weighted_mean = np.sum([(results_here[ind]*Ns[ind] / total_files) for ind in range(len(speakers))])
results_file.write(f"mean for all speakers : {normal_mean}")
results_file.write("\n")
results_file.write(f"weighted_mean : {weighted_mean}")
