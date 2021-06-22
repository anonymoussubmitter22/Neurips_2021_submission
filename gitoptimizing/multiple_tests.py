import numpy as np 
import torch
import os
import sys
from sparsemax import Sparsemax
from tqdm import tqdm
import time
from datetime import datetime
K_dir = sys.argv[1]

L_dir = sys.argv[2]
verbose = int(sys.argv[3])
outdir = sys.argv[4]

# datetime object containing current date and time
now = datetime.now()

dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
dt_string=dt_string.replace(" ", "")
dt_string=dt_string.replace("/", "-")
dt_string=dt_string.replace(":", "-")
n_epochs = 1000
outfile = os.path.join(outdir, dt_string)
f = open(outfile, "w")
Ks = os.listdir(K_dir)
speakers = [x.split(".")[0].split("_")[-1] for x in Ks]

#Loading K matrices 
K_matrices = {}
for K in Ks : 
    sp = K.split(".")[0].split("_")[-1]
    K_matrices[sp] =torch.tensor( np.load(os.path.join(K_dir, K)))
#Loading L matrices
Lmatrices ={}
features = os.listdir(L_dir)
for sp in speakers : 
    Lmatrices[sp] = []
    for feat in features : 
        pathtodirmatrix = os.path.join(L_dir, feat) 
        pathtomatrix = os.path.join(pathtodirmatrix, "L_matrix_" +sp+"_"+feat+".npy")
        Lmatrices[sp].append(torch.tensor(np.load(pathtomatrix)))
print(features)
epsilon  = 0.05
#Function choice
def test():
    function=torch.nn.Softmax
    sparse = False 
    if sparse : 
        function = Sparsemax
    W = torch.nn.Parameter(torch.randn(1,7))
    if sparse : 
        v = torch.ones(1,7) + torch.randn(1,7) *epsilon
        W = torch.nn.Parameter(v)
    W.requires_grad = True

    optimizer = torch.optim.SGD([W], lr=0.01, momentum=0.9)
    function = function(dim=-1)

    sigma = 0 # Norm 2 penality

    for i in tqdm(range(n_epochs)) :
        optimizer.zero_grad()
        total_loss = torch.tensor([0.0])
        lambdas = function(W)
        
        if verbose : 
            print(lambdas)
            print(features)
        for speaker in (speakers) : 
            K = K_matrices[speaker] 
            Ls = Lmatrices[speaker]
            Lsum = torch.zeros(K.size()[0]) 
            for ind in range(len(Ls)) : 
                Ls[ind] = torch.clamp(Ls[ind], -10,0)
                Lsum  = Lsum +  lambdas[0,ind] *Ls[ind]

            Lsum = torch.exp(Lsum)
            sizeconsidered = K.size()[0]
            H= torch.eye(sizeconsidered) - (1/sizeconsidered**2)*torch.ones((sizeconsidered, sizeconsidered)).double()
            secondpart = torch.matmul(Lsum, H)
            firstpart = torch.matmul(K,H)
            score = (1/ (sizeconsidered**2)) * torch.trace( torch.matmul(firstpart, secondpart))
            total_loss +=score
        if verbose  :
            print(f"HCIS loss = {total_loss/len(speakers)}")
        total_loss = total_loss / len(speakers) 
        total_loss += torch.norm(lambdas) * sigma
        if verbose : 
            print(f" norm of lambdas: {torch.norm(lambdas)}")

        total_loss.backward()
        optimizer.step()
    return lambdas, total_loss
n_tests = 10

f.write(str(features))
f.write("\n")
for nt in range(n_tests):
    f.write(f"test number :{nt} \n")
    lambdas,  total_loss = test()
    f.write(str(lambdas))

    f.write("\n")
    f.write(str(total_loss))
    print(lambdas)
    print(f" end of training loss : {total_loss}")
f.close()
