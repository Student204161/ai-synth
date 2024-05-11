


import argparse
import os
import subprocess
import matplotlib.pyplot as plt
import pretty_midi
import PIL
import PIL.Image
import numpy as np
import tqdm
import torch 
from pathlib import Path
import torchaudio
import librosa
import shutil

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
        img_resize = 224, #resize to base2 number after using synthviz code
        image_height = 360,
        black_key_height = 2/3,
        piano_height=64,
        falling_note_color = [75, 105, 177],     # darker blue
        pressed_key_color = [220, 10, 10], # lighter blue
        vertical_speed = 1/4, # Speed in main-image-heights per second
        fps = 16,
        end_t = 0.0,
        silence = 0.0,
        sample_rate = 16000,
        split="train"
    ):

    midi_save_name = input_midi.split('/')[-1].split('.')[0]  #(str_list[0] + '~' + str_list[1]).split('.')[0]
    frames_folder = os.path.join( f"././data/processed/{split}/frames", midi_save_name)
    piano_height = piano_height
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
    #print("Loading MIDI file:", input_midi)
    midi_data = pretty_midi.PrettyMIDI(input_midi)
    
    try:
        notes = [
            { "note": n.pitch, "start": n.start, "end": n.end}
            for n in midi_data.instruments[0].notes
        ]
    except:
        return 

    
    
    #print(f"Loaded {len(notes)} notes from MIDI file")
 
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
    key_start = image_height - piano_height #0
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
    frame_start = 0 #min(note["start"] for note in notes) - silence
    end_t = max(note["end"] for note in notes) if end_t == 0.0 else end_t  #### set as constant
 
    #first frame init to start conditions
    img = PIL.Image.fromarray(im_frame)
    #img.save("{}/frame00000.png".format(frames_folder))
 
 
    # Rest of video:
    finished = False
    frame_ct = 0
    pixel_start = 0
    pixel_start_rounded = 0
 
    #print("[Step 1/3] Generating video frames from notes")
 
 
    #pbar = tqdm.tqdm(total=end_t, desc='Creating video')
    while not finished:       
        prev_pixel_start_rounded = pixel_start_rounded
        pixel_start += pixels_per_frame
        pixel_start_rounded = round(pixel_start)
       
        pixel_increment = pixel_start_rounded - prev_pixel_start_rounded
       
        #pbar.update(1/fps)
        frame_start += 1/fps
 
        #pbar.set_description(f'Creating video [Frame count = {frame_ct}]')
       
        # # Copy most of the previous frame into the new frame:
        im_frame[pixel_increment:main_height, :] = im_frame[0:(main_height - pixel_increment), :]
        im_frame[0:pixel_increment, :] = im_lines[0:pixel_increment, :]
        im_frame[key_start:white_key_end, :] = im_piano
        # Which keys need to be colored?
        # TODO(jxm): put notes in some data structure or something to make this much faster
        keys_to_color = []
        for note in notes:
            if (note["start"]) <= frame_start <= note["end"]:
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

        #img = img.resize((img_resize,img_resize))
        
        frame_ct += 1
        img.save("{}/frame{:05d}.png".format(frames_folder, frame_ct))

        if frame_start >= end_t:
            finished = True
    
 
    #pbar.close()
 
    # print("[Step 2/3] Rendering MIDI to audio with Timidity")
    print(sample_rate)
    wav_path = f'././data/processed/{split}/wavs/'+midi_save_name+'.wav'
    save_wav_cmd = f"timidity {input_midi} -OwM --preserve-silence -s {sample_rate} -A120 -o {wav_path} "
    save_wav_cmd = save_wav_cmd.split()
    # save_wav_cmd[1], save_wav_cmd[-1] = input_midi, sound_file
    subprocess.call(save_wav_cmd)

    #if wav length is longer than end_t * sample rate, cut the end of the wav off
    wav, sr = torchaudio.load(wav_path) # should be 25600 or anything base2.
    target_len = int(sr*(end_t))
    wav = pad_tensor(wav, (1,target_len))
    #save_wav
    torchaudio.save(wav_path, wav, sr)

    # print("Skipped - [Step 3/3] Rendering full video with ffmpeg")
    # #Running from a terminal, the long filter_complex argument needs to
    # #be in double-quotes, but the list form of subprocess.call requires
    # #_not_ double-quoting.

    spec = librosa.feature.melspectrogram(y=wav.numpy(),
                                        sr=sr, 
                                            n_fft=4096, 
                                            hop_length=512, 
                                            win_length=None, 
                                            window='hann', 
                                            center=False, 
                                            pad_mode='constant', 
                                            power=2.0,
                                            fmin=1,
                                            fmax=5000, #pitch 109 / piano key 88 has frequency centered around 4200~ so just let fmax be 5000  https://inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
                                            n_mels=88) #88 piano keys, means 88 bins
    #print(spec.shape)
    #assert spec.shape[1] == spec.shape[2] 
    spec = spec.squeeze()
    #spec = torch.nn.functional.interpolate(torch.tensor(spec).unsqueeze(0),size=(80,128)).squeeze().permute(1,0)
    #plt.imshow(spec)    
    #plt.savefig(f'././data/processed/{split}/spectrograms/{midi_save_name}.png')
    torch.save(spec,f'././data/processed/{split}/spectrograms_pt/{midi_save_name}.pt')


    mp4_path = os.path.join(f"././data/processed/{split}/videos", midi_save_name)
    ffmpeg_cmd = f"ffmpeg -loglevel quiet -framerate {fps} -i {frames_folder}/frame%05d.png -i {wav_path} -f lavfi -t {end_t} -i anullsrc -filter_complex [1]adelay={0}|{0}[aud];[2][aud]amix -c:v libx264 -vf fps={fps} -pix_fmt yuv420p -y -strict -2 {mp4_path}.mp4 "
    #print("> ffmpeg_cmd: ", ffmpeg_cmd)
    subprocess.call(ffmpeg_cmd.split())



def pad_tensor(tensor, target_shape):
    """
    Pad a tensor to a target shape with zeros if necessary.
    
    Args:
    - tensor: NumPy array, the input tensor
    - target_shape: tuple, the target shape
    
    Returns:
    - padded_tensor: Tensor
    """
    current_shape = tensor.shape
    if current_shape[1] < target_shape[1]:
        tensor = torch.nn.functional.pad(tensor, (0,target_shape[1] - current_shape[1]))
    elif current_shape[1] > target_shape[1]:
        tensor = tensor[:,:target_shape[1]]
    return tensor



# collect all frames into a torch.tensor .pt file of size (N,302,360,32,1)

def save_to_pt(path_to_midi,split="train"):

    for root, dirs, files in os.walk(path_to_midi):
        for file in tqdm.tqdm(files):
            if file.endswith('.mid'):

                frames_folder = f'././data/processed/{split}/frames/'+ file.split('.')[0]

                frames = []
                for img_file in os.listdir(frames_folder):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(frames_folder, img_file)
                        #load img
                        img = PIL.Image.open(img_path)
                        #to numpy
                        img = np.array(img)
                        ## to grey scale
                        ##img = img[:,:,0]
                        #to tensor
                        img_tensor = torch.from_numpy(img)
                        frames.append(img_tensor)

                frames = torch.stack(frames)
                torch.save(frames, f'././data/processed/{split}/frames_pt/'+file.split('.')[0]+'.pt')

def make_synthetic(N,split,max_notes=3):

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

    note_count = np.random.randint(1,max_notes + 1, N)
    tot_notes = np.sum(note_count)
    #midi piano starts at 21 to 109.
    lowest_pitch = 21
    highest_pitch = 109
    pitches  = (torch.randperm(tot_notes) % (highest_pitch-lowest_pitch)).numpy() + lowest_pitch
    #velocitys  = (torch.randperm(tot_notes) % 20).numpy()

    note_index = 0
    for i in range(N):
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(0)

        for note in range(note_count[i]):
            pitch = pitches[note_index]
            velocity = 100
            #clip starts at 0 and ends at sec 1
            min_duration = 0.15
            start = np.random.uniform(0.1,0.7*2)
            end = np.random.uniform(start+min_duration, 2 - 0.1)
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
            instrument.notes.append(note)
            note_index += 1
        midi.instruments.append(instrument)
        #save midi
        string_i = '{:0>5}'.format(str(i))
        midi.write(f'data/processed/{split}/midi/syn_{string_i}.mid')



import os
import tqdm
import threading

def synth_create_vids_worker(split, midi_files, image_width, image_height, piano_height, fps, segm_length, sample_rate, pbar):
    for midi_file in midi_files:
        if not os.path.exists(os.path.join(f'data/processed/{split}/spectrograms_pt', midi_file.replace('.mid','.pt'))):
            create_video(
                input_midi=os.path.join(f'data/processed/{split}/midi', midi_file),
                image_width=image_width,
                image_height=image_height,
                piano_height=piano_height,
                fps=fps,
                end_t=segm_length,
                sample_rate=sample_rate,
                split=split
            )
            pbar.update(1)

def synth_create_vids(split, image_width=512, image_height=16, piano_height=16, segm_length=2, sample_rate=16000, fps=16, num_threads=8):
    # Find paths to all midi files
    midi_files = [file for file in os.listdir(f'data/processed/{split}/midi') if file.endswith('.mid')]
    midi_files.sort()

    # Split midi_files into chunks for each thread
    chunks = [midi_files[i:i + len(midi_files)//num_threads] for i in range(0, len(midi_files), len(midi_files)//num_threads)]

    # Initialize progress bar
    total_files = sum(len(chunk) for chunk in chunks)
    with tqdm.tqdm(total=total_files) as pbar:
        # Create threads
        threads = []
        for chunk in chunks:
            t = threading.Thread(target=synth_create_vids_worker, args=(split, chunk, image_width, image_height, piano_height, fps, segm_length, sample_rate, pbar))
            threads.append(t)

        # Start threads
        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()



def print_note_pitches_in_directory(directory):
    count=0
    files = os.listdir(directory)
    files.sort()
    for filename in files:
        if count > 128:
            break
        count+=1
        if filename.endswith(".mid"):
            file_path = os.path.join(directory, filename)
            print(f"Notes in {filename}:")
            midi_data = pretty_midi.PrettyMIDI(file_path)
            for instrument in midi_data.instruments:
                print(f"Instrument: {instrument.name}")
                for note in instrument.notes:
                    print(f"Note: {pretty_midi.note_number_to_name(note.pitch)} ({note.pitch})")
        spec_path = file_path.replace('.mid','.pt').replace('midi','spectrograms_pt')                    

        spectrogram = torch.load(spec_path)        
        plot_filename = os.path.splitext(filename)[0] + ".png"  # Generate plot filename
        plot_folder = r'/home/khalil/dtu/ai-synth/plots'
        plot_path = os.path.join(plot_folder, plot_filename)

        #librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', x_axis='time')
        # plt.imshow(librosa.power_to_db(spectrogram, ref=np.max))
        plt.imshow(spectrogram)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()