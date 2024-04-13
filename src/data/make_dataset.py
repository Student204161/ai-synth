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
 
<<<<<<< HEAD
=======
# Only used in the print-out of the notes; not relevant to the video:
accidentals = "flat"
 
white_notes = {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"}
sharp_notes = {1: "C#", 3: "D#", 6: "F#", 8: "G#", 10: "A#"}
flat_notes  = {1: "Bb", 3: "Eb", 6: "Gb", 8: "Ab", 10: "Bb"}
 
white_notes_scale = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
 
print_notes_to_stdout = False
 
def note_breakdown(midi_note):
    note_in_chromatic_scale = midi_note % 12
    octave = round((midi_note - note_in_chromatic_scale) / 12 - 1)
   
    return [note_in_chromatic_scale, octave]
 
def is_white_key(note):
    return (note % 12) in white_notes
 
def pixel_range(midi_note, image_width):
    # Returns the min and max x-values for a piano key, in pixels.
   
    width_per_white_key = image_width / 52
   
    if is_white_key(midi_note):
        [in_scale, octave] = note_breakdown(midi_note)
        offset = 0
        width = 1
    else:
        [in_scale, octave] = note_breakdown(midi_note - 1)
        offset = 0.5
        width = 0.5
   
    white_note_n = white_notes_scale[in_scale] + 7*octave - 5
   
    start_pixel = round(width_per_white_key*(white_note_n + offset)) + 1
    end_pixel    = round(width_per_white_key*(white_note_n + 1 + offset)) - 1
   
    if width != 1:
        mid_pixel = round(0.5*(start_pixel + end_pixel))
        half_pixel_width = 0.5*width_per_white_key
        half_pixel_width *= width
       
        start_pixel = round(mid_pixel - half_pixel_width)
        end_pixel    = round(mid_pixel + half_pixel_width)
   
    return [start_pixel, end_pixel]
 
def create_video(input_midi: str,
        image_width = 360,
        image_height = 32,
        black_key_height = 2/3,
        falling_note_color = [75, 105, 177],     # darker blue
        pressed_key_color = [220, 10, 10], # lighter blue
        vertical_speed = 1/4, # Speed in main-image-heights per second
        fps = 30,
        video_filename = "output.mp4",
    ):
    str_list = input_midi.split('/')[-2:]
    #midi_save_name = (str_list[0] + '~' + str_list[1]).split('.')[0]
    midi_save_name = str_list[1].split('.')[0]
    frames_folder = os.path.join(Path.cwd(), "/Users/ellemcfarlane/Documents/dtu/ai-synth/data/processed/frames", midi_save_name)
    piano_height = image_height
    main_height = image_height - piano_height
    pixels_per_frame = main_height*vertical_speed / fps # (pix/image) * (images/s) / (frames / s) =
 
 
    note_names = {}
 
    for note in range(21, 109):
        [note_in_chromatic_scale, octave] = note_breakdown(note)
       
        if note_in_chromatic_scale in white_notes:
            note_names[note] = "{}{:d}".format(
                white_notes[note_in_chromatic_scale], octave)
        else:
            if accidentals == "flat":
                note_names[note] = "{}{:d}".format(
                    flat_notes[note_in_chromatic_scale], octave)
            else:
                note_names[note] = "{}{:d}".format(
                    sharp_notes[note_in_chromatic_scale], octave)
 
    # The 'notes' list will store each note played, with start and end
    # times in seconds.
    print("Loading MIDI file:", input_midi)
    midi_data = pretty_midi.PrettyMIDI(input_midi)
>>>>>>> e3b6fdb (custom to me)


<<<<<<< HEAD
if __name__ == '__main__':

    if not os.path.exists('././data/processed/midi'):
        os.makedirs('././data/processed/midi')
    if not os.path.exists('././data/processed/videos'):
        os.makedirs('././data/processed/videos')
    N_tot = count_files('././data/raw')

    segm_length = 5 #in sec
    N = 100
    #linear space from 0 to N, with 100 points but integers
    sample_indx = np.linspace(0, N_tot, N, dtype=int)

    #for sampling across midi files in different folders
    count = 0
    for root, dirs, files in os.walk('././data/raw'):
=======
    
    print(f"Loaded {len(notes)} notes from MIDI file")
 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~ The rest of the code is about making the video. ~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    if not os.path.isdir(frames_folder):
        os.makedirs(frames_folder)
       
    # Delete all previous image frames:
    for f in os.listdir(frames_folder):
        os.remove("{}/{}".format(frames_folder, f))
 
    im_base = np.zeros((image_height, image_width, 3), dtype=np.uint8)
 
    # Draw the piano, and the grey lines next to the C's for the main area:
    key_start = 0
    white_key_end = image_height - 1
    black_key_end = round(image_height - (1-black_key_height)*piano_height)
 
    im_lines = im_base.copy()
 
    for i in range(21, 109):
        # draw white keys
        if is_white_key(i):
            [x0, x1] = pixel_range(i, image_width)
            im_base[key_start:white_key_end, x0:x1] = [255, 255, 255]
       
        # draw lines separating octaves
        if i % 12 == 0:
            im_lines[0:(key_start-1), (x0-2):(x0-1)] = [20, 20, 20]
 
    for i in range(21, 109):
        # draw black keys
        if not is_white_key(i):
            [x0, x1] = pixel_range(i, image_width)
            im_base[key_start:black_key_end, x0:x1] = [0, 0, 0]
 
    im_piano = im_base[key_start:white_key_end, :]
 
    im_frame = im_base.copy()
    im_frame += im_lines
 
    # Timidity (the old version that I have!) always starts the audio
    # at time = 0.  Add a second of silence to the start, and also
    # keep making frames for a second at the end:
    frame_start = notes[0]["start"] - 1
    end_t = max(note["end"] for note in notes) + 1
 
    # First frame:
    # for j in range(main_height):
    #   im_j = main_height - j - 1
    #   t = frame_start + time_per_pixel*j
    #   for note in notes:
    #       if note["start"] <= t <= note["end"]:
    #           [x0, x1] = pixel_range(note["note"], image_width)
    #           im_frame[im_j, x0:x1] = falling_note_color
    img = PIL.Image.fromarray(im_frame)
    img.save("{}/frame00000.png".format(frames_folder))
 
 
    # Rest of video:
    finished = False
    frame_ct = 0
    pixel_start = 0
    pixel_start_rounded = 0
 
    print("[Step 1/3] Generating video frames from notes")
 
 
    pbar = tqdm.tqdm(total=end_t, desc='Creating video')
    while not finished:
        frame_ct += 1
       
        prev_pixel_start_rounded = pixel_start_rounded
        pixel_start += pixels_per_frame
        pixel_start_rounded = round(pixel_start)
       
        pixel_increment = pixel_start_rounded - prev_pixel_start_rounded
       
        pbar.update(1/fps)
        frame_start += 1/fps
 
        pbar.set_description(f'Creating video [Frame count = {frame_ct}]')
       
        # # Copy most of the previous frame into the new frame:
        # im_frame[pixel_increment:main_height, :] = im_frame[0:(main_height - pixel_increment), :]
        # im_frame[0:pixel_increment, :] = im_lines[0:pixel_increment, :]
        im_frame[key_start:white_key_end, :] = im_piano
        # Which keys need to be colored?
        # TODO(jxm): put notes in some data structure or something to make this much faster
        keys_to_color = []
        for note in notes:
            if note["start"] <= frame_start <= note["end"]:
                keys_to_color.append(note["note"])
       
        # First color the white keys (this will cover some black-key pixels),
        # then re-draw the black keys either side,
        # then color the black keys.
        for note in keys_to_color:
            if is_white_key(note):
                [x0, x1] = pixel_range(note, image_width)
                im_frame[key_start:white_key_end, x0:x1] = pressed_key_color
       
        for note in keys_to_color:
            if is_white_key(note):
                if (not is_white_key(note - 1)) and (note > 21):
                    [x0, x1] = pixel_range(note - 1, image_width)
                    im_frame[key_start:black_key_end, x0:x1] = [0,0,0]
               
                if (not is_white_key(note + 1)) and (note < 108):
                    [x0, x1] = pixel_range(note + 1, image_width)
                    im_frame[key_start:black_key_end, x0:x1] = [0,0,0]
       
        for note in keys_to_color:
            if not is_white_key(note):
                [x0, x1] = pixel_range(note, image_width)
                im_frame[key_start:black_key_end, x0:x1] = pressed_key_color
       
       
        img = PIL.Image.fromarray(im_frame)
        img.save("{}/frame{:05d}.png".format(frames_folder, frame_ct))
       
        if frame_start >= end_t:
            finished = True
   
 
    pbar.close()
 
    print("[Step 2/3] Rendering MIDI to audio with Timidity")
    wav_path = Path('/Users/ellemcfarlane/Documents/dtu/ai-synth/data/raw/'+midi_save_name+'.wav')
    sound_file = os.path.join(Path.cwd(), wav_path)
    save_wav_cmd = f"timidity place1 -Ow --output-24bit -A120 -o place2"
    save_wav_cmd = save_wav_cmd.split()
    save_wav_cmd[1], save_wav_cmd[-1] = input_midi, sound_file

    subprocess.call(save_wav_cmd)
 

    print("Skipped - [Step 3/3] Rendering full video with ffmpeg")
    # Running from a terminal, the long filter_complex argument needs to
    # be in double-quotes, but the list form of subprocess.call requires
    # _not_ double-quoting.
    
    ffmpeg_cmd = f"ffmpeg -framerate {fps} -i /Users/ellemcfarlane/Documents/dtu/ai-synth/data/processed/frames/{midi_save_name}/frame%05d.png -i {input_midi.split('.')[0]}.wav -f lavfi -t 0.1 -i anullsrc -filter_complex [1]adelay=1000|1000[aud];[2][aud]amix -c:v libx264 -vf fps={fps} -pix_fmt yuv420p -y -strict -2 {video_filename}.mp4"
    print("> ffmpeg_cmd: ", ffmpeg_cmd)
    subprocess.call(ffmpeg_cmd.split())
    # remove temp.m4
    # os.remove("temp.mp4")
 
 
 
if __name__ == '__main__':
    names = {}
    dir_ = '/Users/ellemcfarlane/Documents/dtu/ai-synth/data/raw'
    assert os.path.isdir(dir_), "Directory not found!"
    for root, dirs, files in os.walk(dir_):
        assert files, "No MIDI files found in the directory"
>>>>>>> e3b6fdb (custom to me)
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


