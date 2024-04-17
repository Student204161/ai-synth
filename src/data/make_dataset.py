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

    if not os.path.exists('././data/processed/midi'):
        os.makedirs('././data/processed/midi')
    if not os.path.exists('././data/processed/videos'):
        os.makedirs('././data/processed/videos')
    if not os.path.exists('././data/processed/wavs'):
        os.makedirs('././data/processed/wavs')
    N_tot = count_files('././data/raw')

    segm_length = 5 #in sec
    N = 10
    #linear space from 0 to N, with 100 points but integers
    sample_indx = np.linspace(0, N_tot, N, dtype=int)

    #for sampling across midi files in different folders
    count = 0
    for root, dirs, files in os.walk('././data/raw'):
        for file in files:
            if file.endswith('.mid'):
                if count in sample_indx:
                    print(f"Processing {root}/{file}")

                    #for each midi file, seperate the notes into 5 second intervals

                    midi_data = pretty_midi.PrettyMIDI(os.path.join(root, file))
                    artist = root.split('\\')[-1] if os.name != "posix" else root.split('/')[-1]
                    artist = artist.replace(' ','_')

                    notes = [
                        { "note": n.pitch, "start": n.start, "end": n.end, "velocity": n.velocity}
                        for n in midi_data.instruments[0].notes
                    ]

                    #add 0.5 sec of silence in beginning and end. In case we work with frequency distr. & do fourier, we dont have to zero pad.
                    #is buggy, so dont...
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
                        if len(instrument.notes) > 0 and not Path(f"././data/processed/midi/{artist}--{file.split('.')[0].replace(' ','_')}_{i}.mid").exists():
                            

                            midi.write(f"././data/processed/midi/{artist}--{file.split('.')[0].replace(' ','_')}_{i}.mid")
                            create_video(
                                input_midi=f"././data/processed/midi/{artist}--{file.split('.')[0].replace(' ','_')}_{i}.mid",
                                image_width = 360,
                                image_height = 32,
                                fps = 60,
                                end_t=segm_length,
                                silence=silence
                            )

                count += 1
    
    check_video_files('././data/processed/midi')

    save_to_pt('././data/processed/midi')


