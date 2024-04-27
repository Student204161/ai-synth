from utils import create_piano_image
import pretty_midi
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # for every note in the range of 0 to 127
    imgs = []
    midi_notes = [i for i in range(21, 108)] # these are all the notes in view except like last 2
    note_names = [pretty_midi.note_number_to_name(midi_note) for midi_note in midi_notes]
    # display all images by note name in order
    for midi_note, note_name in zip(midi_notes, note_names):
        img = create_piano_image([midi_note])
        imgs.append(img)
    
    # display them here if you want