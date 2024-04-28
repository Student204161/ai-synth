import os
import shutil
from glob import glob

# Set your dataset directory
dataset_dir = '/Users/ellemcfarlane/Documents/dtu/ai-synth/data/one_note_frames_cleaner'  # Update this to your actual path
dest_dir = '/Users/ellemcfarlane/Documents/dtu/ai-synth/data/one_note_frames_split_cleaner'  # Update this to your actual path
os.makedirs(dest_dir, exist_ok=True)
# Create directories for the splits if they don't exist
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(dest_dir, split)
    if not os.path.exists(split_dir):
        os.mkdir(split_dir)
        for note in ['A', 'Ab', 'B', 'Bb', 'C', 'D', 'E', 'Eb', 'F', 'G', 'Gb']:
            os.mkdir(os.path.join(split_dir, note))

# Function to copy data based on octave
def copy_data(note, octave, split, src_dir, dest_dir):
    note_dir = os.path.join(src_dir, note)
    # Updated pattern to match the new filename format
    images = glob(os.path.join(note_dir, f'{note}{octave}.jpg'))
    assert len(images) > 0, f'No images found for {note}{octave} in {note_dir}'
    for img in images:
        # Define the destination path
        dest_path = os.path.join(dest_dir, split, note, os.path.basename(img))
        # Copy the image to the destination path
        shutil.copy(img, dest_path)

# Define which octaves go to which split
train_octaves = ['0', '1', '3', '6']
val_octaves = ['4', '7']
test_octaves = ['2', '5']

# Iterate through each note and copy the data based on the defined octaves
for note in ['A', 'Ab', 'B', 'Bb', 'C', 'D', 'E', 'Eb', 'F', 'G', 'Gb']:
    for octave in train_octaves:
        if octave == '0' and (note != 'A' or note != 'B'):
            continue # only A & B have octave 0
        copy_data(note, octave, 'train', dataset_dir, dest_dir)
    for octave in val_octaves:
        copy_data(note, octave, 'val', dataset_dir, dest_dir)
    for octave in test_octaves:
        copy_data(note, octave, 'test', dataset_dir, dest_dir)

print(f"saved splits to {dest_dir}")