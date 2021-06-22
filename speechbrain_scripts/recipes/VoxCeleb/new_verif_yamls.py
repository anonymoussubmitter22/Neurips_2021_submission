import yaml
import sys
import os
filein=sys.argv[1]
outdir = sys.argv[3]
results_folder=sys.argv[2]
overrides={}
with open(filein) as f : 
    lines = f.read().splitlines()
test_names =["jitterLocal_sma", "voicingFinalUnclipped_sma", "alphaRatio_sma3",
             "pcm_zcr_sma", "shimmerLocal_sma",
             "audspecRasta_lengthL1norm_sma", "pcm_RMSenergy_sma",
             "Loudness_sma3", "logHNR_sma"]
def get_checkpoint(name): 
    path = os.path.join(results_folder, name+"_freezed") 
    seed = os.listdir(path)[0]
    save_path = os.path.join(os.path.join(path,seed), "save")
    check_folder = [x for x in os.listdir(save_path) if "CKPT" in x ]
    return os.path.join(save_path, check_folder[0])
for name in test_names :
    print(get_checkpoint(name))

for name in test_names : 
    for ind, line in enumerate(lines) : 
        if line[0:15] == "embedding_param" : 
            print(line)
            start, end = line.split(":")

            lines[ind] =start + ": "+ os.path.join("testing_yamls/",
                                                   name+".yaml")
        if  line[0:13] == "output_folder" : 
            start, end = line.split(":") 
            lines[ind] = start + ": results/voxceleb1/verif_"+ name
    with open(os.path.join(outdir, name+".yaml"), "w") as f:
        f.write("\n".join(lines))
        f.close()

