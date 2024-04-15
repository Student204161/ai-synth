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
from utils import create_video, count_files, check_video_files
from pathlib import Path
 
# Helper function to create a new MIDI file from a segment
def create_segment_midi(segment, start_time, end_time, tempo_changes, midi_data):
    # Create a new MIDI object
    new_midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=midi_data.instruments[0].program)
    new_midi.instruments.append(instrument)
    
    # Add notes to the new instrument
    for note in segment:
        new_note = pretty_midi.Note(
            velocity=note['velocity'],
            pitch=note['note'],
            start=note['start'] - start_time,
            end=note['end'] - start_time
        )
        instrument.notes.append(new_note)
    
    # Handle tempo changes within the segment
    for tempo, time in zip(*tempo_changes):
        if start_time <= time <= end_time:
            new_midi.add_tempo_change(tempo, time - start_time)

    return new_midi

if __name__ == '__main__':
    main_dir = 'data_sanity'
    processed_midi_dir = f'{main_dir}/processed/midi'
    frames_dir = f'{main_dir}/processed/frames'
    raw_dir = f'{main_dir}/raw'
    dirs = [processed_midi_dir, frames_dir, raw_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    N_tot = count_files(raw_dir)

    segm_length = 5 #in sec
    N = 100
    #linear space from 0 to N, with 100 points but integers
    sample_indx = np.linspace(0, N_tot, N, dtype=int)

    #for sampling across midi files in different folders
    count = 0
    file_delim = os.path.sep
    for root, dirs, files in os.walk(raw_dir):
        print(f"in {raw_dir} found {len(files)} files with root {root} and dirs {dirs}")
        for file in files:
            if file.endswith('.mid'):
                if count in sample_indx:
                    midi_data_path = os.path.join(root, file)
                    print(f"Processing {midi_data_path}...")

                    # for each midi file, seperate the notes into 5 second intervals

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

                    # Example loop over segments
                    for i, segment in tqdm.tqdm(enumerate(segments)):
                        # Create a new PrettyMIDI object
                        midi = pretty_midi.PrettyMIDI()
                        instrument = pretty_midi.Instrument(program=0)  # Set the program if needed

                        # Append notes to the instrument
                        for note in segment:
                            midi_note = pretty_midi.Note(
                                velocity=note['velocity'],
                                pitch=note['note'],
                                start=note['start'],
                                end=note['end']
                            )
                            instrument.notes.append(midi_note)
                        
                        # Add the instrument to the MIDI object
                        midi.instruments.append(instrument)

                        # Build the file path and check if it needs to be written
                        processed_midi_path = f"{processed_midi_dir}/{artist}--{file.split('.')[0].replace(' ','_')}_{i}.mid"
                        if len(instrument.notes) > 0:
                            midi.write(processed_midi_path)
                            create_video(
                                input_midi=processed_midi_path,
                                image_width=360,
                                image_height=32,
                                fps=60,
                                end_t=segm_length,
                                silence=silence,
                                data_dir=f"{main_dir}/processed/"
                            )
                            print(f"Saved to {processed_midi_path}")
                            # also save piano roll matrix
                            #piano_roll = midi.get_piano_roll(fs=60)
            
                            #np.save(f"././data/processed/midi/{artist}--{file.split('.')[0]}_{i}.npy", piano_roll)
                        elif Path(processed_midi_path).exists():
                            print(f"File {processed_midi_path} already exists, skipping...")
                count += 1
    check_video_files(processed_midi_dir, frames_dir="././data_sanity/processed/frames")
