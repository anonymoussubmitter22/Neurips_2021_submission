import numpy as np 
import torch
import random
import os
import sys
from sparsemax import Sparsemax
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


K_dir = sys.argv[1]
L_dir = sys.argv[2]

n_epochs = 20

Ks = os.listdir(K_dir)
speakers = [x.split(".")[0].split("_")[-1] for x in Ks]
#Loading K matrices 
K_matrices = {}
for K in Ks : 
    sp = K.split(".")[0].split("_")[-1]
    K_matrices[sp] =torch.tensor( np.load(os.path.join(K_dir, K)))

Lmatrices ={}
features = os.listdir(L_dir)
for sp in speakers : 
    Lmatrices[sp] = []
    for feat in features : 
        pathtodirmatrix = os.path.join(L_dir, feat) 
        pathtomatrix = os.path.join(pathtodirmatrix, "L_matrix_" +sp+"_"+feat+".npy")
        Lmatrices[sp].append(torch.tensor(np.load(pathtomatrix)))
print(features)
function=torch.nn.Softmax
#function = Sparsemax
W = torch.nn.Parameter(torch.zeros(1,7))
W.requires_grad = True
sparsemax = function(dim=-1)
def get_score(W) :
    total_loss = torch.tensor([0.0])
    for speaker in (speakers) : 
        K = K_matrices[speaker] 
        Ls = Lmatrices[speaker]

        sizeconsidered = K.size()[0]
        Lsum = torch.zeros((sizeconsidered, sizeconsidered)).double()
        for i in range(len(features)) : 

            Ls[i] = torch.clamp(Ls[i], -10,0)
            Lsum += Ls[i]* W[0,i]
        H= torch.eye(sizeconsidered) - (1/sizeconsidered**2)*torch.ones((sizeconsidered, sizeconsidered)).double()

        Lsum = torch.exp(Lsum)

        secondpart = torch.matmul(Lsum, H)
        firstpart = torch.matmul(K,H)
        score = (1/ (sizeconsidered**2)) * torch.trace( torch.matmul(firstpart, secondpart))
        total_loss +=score

    return total_loss/len(speakers)
Z = np.linspace(0,1,30)
Y=np.linspace(0,1,30)
X= []
Y=[]

X= []
values = []
Xs = []
Ys= []
Zs = []
number1=1
number2=4
number3=5
for i in tqdm(range(len(Z))):
    X=Z[i]
#['audspecRasta_lengthL1norm_sma', 'logHNR_sma', 'voicingFinalUnclipped_sma', 'pcm_RMSenergy_sma', 'alphaRatio_sma3', 'F0final_sma', 'pcm_zcr_sma']

    y = np.linspace(0, 1-X, 30)
    for f in range(len(y)):
        Ys.append(y[f])
        Xs.append(X)
        W[0,number1] = X
        W[0,number2] = y[f]
        W[0,number3] = 1-X- y[f]
        vv= get_score(W)
        values.append(vv.detach().numpy())
name = f"X_{features[number1]}_Y_{features[number2]}_remaining_{features[number3]}.jpg"

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X, Y = np.meshgrid(Xs, Ys)
Z = np.reshape(np.array(values), ( 1, len(values)))
print(Z)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title(name.split(".")[0])
plt.xlabel(features[number1])
plt.ylabel(features[number2])
plt.savefig(name)
plt.show()
