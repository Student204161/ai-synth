from utils import create_piano_image
import pretty_midi
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    # for every note in the range of 0 to 127
    imgs = []
    midi_notes = [i for i in range(21, 108)] # these are all the notes in view except like last 2
    note_names = [pretty_midi.note_number_to_name(midi_note) for midi_note in midi_notes]
    # display all images by note name in order
    class_names = sorted(list(set(note_name[:-1] for note_name in note_names)))
    dir = 'data/one_note_frames/'
    os.makedirs(dir, exist_ok=True)
    print(f"made dir: {dir}")
    # create directory for each class within 
    print(f"classes: {class_names}")
    for class_name in class_names:
        os.makedirs(os.path.join(dir, class_name), exist_ok=True)
    for midi_note, note_name in zip(midi_notes, note_names):
        note_class = note_name[:-1]
        save_path = f"{os.path.join(dir, note_class, note_name)}.jpg"
        img = create_piano_image([midi_note], save_path=save_path)
        imgs.append(img)
    
    # display them here if you want