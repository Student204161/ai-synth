####################################################################################
################################## synthviz ########################################
####################################################################################
######  code adapted from https://pappubahry.com/misc/piano_diaries/synthesia/ #####
####################################################################################
####################################################################################
####################################################################################
####################################################################################
 
#edited to only render piano and not incoming note
import os
import copy
import subprocess 
import pretty_midi
import PIL
import PIL.Image
import numpy as np
import tqdm
from utils import create_video, save_to_pt, make_synthetic,synth_create_vids
from pathlib import Path
import torch
import os
import pretty_midi
import librosa
import matplotlib.pyplot as plt


# Example usage


if __name__ == '__main__':
    #torch set seed
    torch.manual_seed(0)
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0)

    all_files = []
    for root, dirs, files in os.walk('././data/raw'):
        for file in files:
            if file.endswith('.mid'):
                all_files.append(os.path.join(root, file))
    all_files =np.asarray(all_files)
    N_tot = len(all_files)

    segm_length = 2 #in sec
    N_train,N_val,N_test = 16000, 1000,100
    split = "train"
    make_synthetic(N_train,split)
    synth_create_vids(split,segm_length=segm_length)
    save_to_pt(f'././data/processed/{split}/midi', split=split)

    split = "val"
    make_synthetic(N_val,split)
    synth_create_vids(split)
    save_to_pt(f'././data/processed/{split}/midi', split=split)
    #print_note_pitches_in_directory(f'././data/processed/{split}/midi')




