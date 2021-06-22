# Pseudo-Label-Selection
This repository presents the code for the "Pretext Tasks Selection for Multitask Self-Supervised Speech Representation Learning" paper. The repo contains the 3 phase of the done computations. First, the computation of the CI estimate for Librispeech  and VoxCeleb, followed by the pretraining on Common Voice and the two downstream trainings. 
### Group selection and weighting example
An example is provided for  in the gitoptimizing/example.sh file, the K and L matrices are stored in the same folder. Another readme is in that folder for further explanations.

### Pretraining
An example for pretraining is also proposed. Steps to get a pretraining experiment : 

Download the Common Voice english dataset here : https://commonvoice.mozilla.org/en/datasets

use the prepare.sh script providing the path to the unzipped dataset. 

example.sh offers the example of pretraining using alpharatio as the target pseudo-label.

### Retraining 

##### LibriSpeech 


For Librispeech, we perform end-to-end retraining and testing in one step. You'll have to copy the folder resulting from the pretraining in the Librispeech retraining folder first.

Then example.sh provides an example for retraining. At the end of the retraining, the test WER is output. 


##### VoxCeleb

Speaker Recognition is a two-step action in our work. First, we train Xvectors, as stated in the training\_xvectors\_example.sh. Afterwards, a few changes have to be made to the retraining yamls, mainly links to the embedding model. An example is provided with AlphaRatio. An example for verification and final results computing is provided in verification.sh  


Pretrained models for the CommonVoice pretraining and the VoxCeleb embedder are available here : 
https://1drv.ms/u/s!AtZNOLRhbqF6aH0KE5qIbwzEf60?e=NTaFDL

CommonVoice pretrained models can be used for the downstream tasks, by copying the folder in the Downstream task folder

VoxCeleb pretrained models can be used for the verification phase, by running a command similar to the one given in verification.sh 

In both cases, you can run the retraining by following the examples given in the .sh files. 
