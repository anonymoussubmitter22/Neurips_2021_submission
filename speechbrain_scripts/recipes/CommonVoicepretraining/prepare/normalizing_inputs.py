import os 
import numpy as np 
import csv 
import sys 
import multiprocessing as mp
from tqdm import tqdm 
import pandas as pd
from functools import partial
from multiprocessing import Pool
import pandas as pd

values = ['alphaRatio_sma3', 'F0final_sma_opts', "audspecRasta_lengthL1norm_sma", "pcm_RMSenergy_sma",
         "pcm_zcr_sma", "voicingFinalUnclipped_sma", "logHNR_sma"]


def file_treatment_lld(filepath, outdir) : 
    cv_points = pd.read_csv(filepath, sep=",")
    
    for value in (values) : 
        if value not in list(cv_points.columns):
            continue
        numbers = []
        for element in tqdm(list(cv_points[value])[0:min(len(cv_points[value]),20000)]) : 
            numbers = numbers + [float(x) for x in element.split()]
        print(len(numbers))
        mean_value = np.mean(numbers)
        std_value = np.std(numbers)
        new_list = []
        for element in tqdm(list(cv_points[value])) : 
            vector = np.array([float(x) for x in element.split()])
            vector = (vector -mean_value)/std_value
            new_el = ' '.join([str(x) for x in list(vector)])
            new_list.append(new_el)
        cv_points[value] = new_list
    cv_points.to_csv(outdir, sep=",", index=False)








if __name__=="__main__":
    preparation_dir = sys.argv[1]
    outdir = sys.argv[2]
    files = ["test.csv", "dev.csv", "train.csv"]
    for el in files : 
        pathin = os.path.join(preparation_dir, el)
        pathout = os.path.join(outdir, el)
        file_treatment_lld(pathin, pathout)
