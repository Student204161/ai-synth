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
from utils import create_video, count_files, check_video_files, save_to_pt
from pathlib import Path
 


if __name__ == '__main__':

    all_files = []
    for root, dirs, files in os.walk('././data/raw'):
        for file in files:
            if file.endswith('.mid'):
                all_files.append(os.path.join(root, file))
    all_files =np.asarray(all_files)
    N_tot = len(all_files)

    segm_length = 1 #in sec
    N_train,N_val,N_test = 20,2,2
    split = "val"

    if not os.path.exists(f'././data/processed/{split}/midi'):
        os.makedirs(f'././data/processed/{split}/midi')
    if not os.path.exists(f'././data/processed/{split}/videos'):
        os.makedirs(f'././data/processed/{split}/videos')
    if not os.path.exists(f'././data/processed/{split}/wavs'):
        os.makedirs(f'././data/processed/{split}/wavs')
    if not os.path.exists(f'././data/processed/{split}/spectrograms'):
        os.makedirs(f'././data/processed/{split}/spectrograms')
    if not os.path.exists(f'././data/processed/{split}/spectrograms_pt'):
        os.makedirs(f'././data/processed/{split}/spectrograms_pt')
    if not os.path.exists(f'././data/processed/{split}/frames_pt'):
        os.makedirs(f'././data/processed/{split}/frames_pt')
    #linear space from 0 to N, with 100 points but integers
    train_sample_indx = np.linspace(0, N_tot-1, N_train, dtype=int)

    #for sampling across midi files in different folders
    count = 0
    
    if split == "train":
        split_files = all_files[train_sample_indx]
    elif split == "val":
        val_cand_files = np.delete(all_files,train_sample_indx)
        valsample_indx = np.linspace(0, N_tot-N_train-1, N_val, dtype=int)
        split_files = val_cand_files[valsample_indx]
    elif split == "test":
        val_cand_files = np.delete(all_files,train_sample_indx)
        valsample_indx = np.linspace(0, N_tot-N_train-1, N_val, dtype=int)
        test_cand_files = np.delete(val_cand_files,valsample_indx)
        testsample_indx = np.linspace(0, N_tot-N_train-N_val-1, N_test, dtype=int)
        split_files = test_cand_files[testsample_indx]
    else:
        print('bad split chosen')
        assert False
        
    

    for file in split_files:
        print(f"Processing {file}")

        #for each midi file, seperate the notes into 5 second intervals
        try:
            midi_data = pretty_midi.PrettyMIDI(file)
        except:
            continue
        artist = file.split('\\')[-2] if os.name != "posix" else file.split('/')[-2]
        artist = artist.replace(' ','_')
        song_name = file.split('\\')[-1] if os.name != "posix" else file.split('/')[-1]
        song_name = song_name.replace(' ','_')

        notes = [
            { "note": n.pitch, "start": n.start, "end": n.end, "velocity": n.velocity}
            for n in midi_data.instruments[0].notes
        ]

        #probably should add 0.5 sec of silence in beginning and end. In case we work with frequency distr. & do fourier, we dont have to zero pad.
        #is buggy tho, so dont...
        silence = 0.0
        # create list of midi file segments
        segments = []
        seq_count=0
        t1=0.0
        while t1 < midi_data.get_end_time():
            segment = []
            t1, t2 = seq_count*segm_length, (seq_count+1)*segm_length
            for note in notes:
                add_note =  copy.deepcopy(note)                            
                #clip to interval
                add_note['start'] = max(note['start'], t1) 
                add_note['end'] = min(note['end'], t2)

                if t1 <= add_note['start'] <= t2 and t1 <= add_note['end'] <= t2:
                    add_note['start'] -= (t1 + silence)
                    add_note['end'] -= (t1 + silence)

                    segment.append(add_note)

            segments.append(segment)
            seq_count+=1

        #save segments as seperate midi files
        for i, segment in tqdm.tqdm(enumerate(segments)):
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(0)
            for note in segment:
                instrument.notes.append(pretty_midi.Note(
                    velocity=note['velocity'],
                    pitch=note['note'],
                    start=note['start'],
                    end=note['end']
                ))
            midi.instruments.append(instrument)
            # if segment not empty and file doesn't exist, write to file
            if len(instrument.notes) > 0 and not Path(f"././data/processed/{split}/midi/{artist}--{song_name.split('.')[0]}_{i}.mid").exists():
                
                midi.write(f"././data/processed/{split}/midi/{artist}--{song_name.split('.')[0]}_{i}.mid")
                create_video(
                    input_midi=f"././data/processed/{split}/midi/{artist}--{song_name.split('.')[0]}_{i}.mid",
                    image_width = 512,
                    img_resize = 360, #resize to vivit input size after using synthviz code                                
                    image_height = 16,
                    piano_height = 16,
                    fps = 16,
                    end_t=segm_length,
                    silence=silence,
                    sample_rate=16000,
                    split=split
                )

    count += 1

    check_video_files(f'././data/processed/{split}/midi',split=split)

    save_to_pt(f'././data/processed/{split}/midi',seq_len=segm_length,split=split)



