import os
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from tqdm import tqdm
import torchaudio
from torchvision import transforms
#import wandb
import sys,os
import soundfile
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # self.max_waveform_length = max_waveform_length
        # self.max_num_frames = max_num_frames #
        self.frames = [file for file in os.listdir(os.path.join(root_dir, 'frames_pt')) if (file.endswith('.pt') )]
        self.frames.sort()
        self.transf = transforms.Compose([
            transforms.Resize((88, 55)),  # Resize the image if needed
        ])
        #self.feature_extractor = feature_extractor
        # checkpoint = "microsoft/speecht5_tts"
        # self.feature_extractor = SpeechT5FeatureExtractor(fmin=6,fmax=10000,do_normalize=True,num_mel_bins=128)


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frames_path = os.path.join(self.root_dir, 'frames_pt', self.frames[idx])
        spectrogram_path = os.path.join(self.root_dir, 'spectrograms_pt', self.frames[idx])


        frames = torch.load(frames_path).unsqueeze(0)
        frames = frames.float() / 255 #

        frames = frames.permute(0, 1, 4, 2, 3)
        spectrogram = torch.tensor(torch.load(spectrogram_path)).unsqueeze(0)
        spectrogram = (spectrogram/torch.max(spectrogram)) #normalize

        name = self.frames[idx].replace('.pt', '')


        return frames, spectrogram, name
def collate_fn(batch):
    frames, spectrogram, name = zip(*batch)
    dat = {'frames':torch.vstack(frames).to(device=device), 'spectrogram':  torch.vstack(spectrogram).to(device=device), 'name':name}
    return dat



