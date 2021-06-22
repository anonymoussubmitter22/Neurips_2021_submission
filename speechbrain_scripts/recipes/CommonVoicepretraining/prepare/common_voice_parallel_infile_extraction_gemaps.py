import os 
import pandas as pd
import time
from tqdm import tqdm
import opensmile
import numpy as np
import sys
import multiprocessing as mp
from multiprocessing import Pool
print(f" numbre of cpus : {mp.cpu_count()}")

indir = sys.argv[1]
outdir= sys.argv[2]
infile = sys.argv[3]
if not os.path.exists(outdir) : 
    os.makedirs(outdir) 
considered_file = pd.read_csv(infile)
audio_files = list( considered_file["ID"])
audio_files = [x+".wav" for x in audio_files]
print(audio_files[0:5])
#audio_files = os.listdir( indir)
audio_files_path = [os.path.join( indir,x) for x in audio_files]
print(audio_files_path[0:5])

time.sleep(5)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)



def f(ind) : 

    try :
        z=smile.process_file(audio_files_path[ind])

        outpath = os.path.join(outdir,audio_files[ind].split(".")[0]+".csv")
        z.to_pickle(outpath)

    except : 
        print("there was an error")


p = Pool(mp.cpu_count())
N= len(audio_files_path)
r = list(tqdm(p.imap(f, range( N)), total=N))

