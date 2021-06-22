import yaml
import sys
import os
filein=sys.argv[1]
outdir = sys.argv[2]
overrides={}
with open(filein) as f : 
    lines = f.read().splitlines()
test_names =["jitterLocal_sma", "voicingFinalUnclipped_sma", "alphaRatio_sma3",
             "pcm_zcr_sma", "shimmerLocal_sma",
             "audspecRasta_lengthL1norm_sma", "pcm_RMSenergy_sma",
            "F0final_sma", "Loudness_sma3", "logHNR_sma"]
for name in test_names : 
    for ind, line in enumerate(lines) : 
        if  line[0:13] == "output_folder" : 
            start, end = line.split(":") 
            lines[ind] = start + ": !ref results/" + name+"_freezed" +  "/<seed>"
    with open(os.path.join(outdir, name+".yaml"), "w") as f:
        f.write("\n".join(lines))
        f.close()


