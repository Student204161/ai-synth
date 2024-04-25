
import os
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
import torchvision
import soundfile as sf
from tqdm import tqdm
import torchaudio
import wandb
import sys,os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
import argparse

from mm_diffusion.multimodal_unet import MultimodalUNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, root_dir, max_waveform_length, max_num_frames, transform=None):
        self.root_dir = root_dir
        self.max_waveform_length = max_waveform_length
        self.max_num_frames = max_num_frames
        self.wav_files = [file for file in os.listdir(os.path.join(root_dir, 'wavs')) if file.endswith('.wav')]
        self.wav_files.sort()
        self.graytransform = torchvision.transforms.Grayscale()

    def __len__(self):
        return len(self.wav_files)

    def load_frames(self, frame_folder):
        frame_files = sorted(os.listdir(frame_folder))
        frames = [self.graytransform(torchvision.io.read_image(os.path.join(frame_folder, file))) for file in frame_files]  # Convert to grayscale
        
        return torch.vstack(frames)


    def __getitem__(self, idx):
        wav_file = os.path.join(self.root_dir, 'wavs', self.wav_files[idx])
        folder_names = os.path.splitext(self.wav_files[idx])[0]  # Get folder name (without extension)
        frame_folder = os.path.join(self.root_dir,'frames', folder_names)

        waveform, sample_rate = torchaudio.load(wav_file)
        frames = self.load_frames(frame_folder)
        frames = frames.float() / 127.5 - 1 #0-1

        # Clip waveform if longer than max_waveform_length
        if len(waveform) > self.max_waveform_length:
            waveform = waveform[:self.max_waveform_length]

        # Limit number of frames if more than max_num_frames
        if len(frames) > self.max_num_frames:
            frames = frames[:self.max_num_frames]

        return waveform.unsqueeze(0), frames.unsqueeze(1).unsqueeze(0), folder_names

def collate_fn(batch):
    waveforms, frames, folder_names = zip(*batch)
    dat = {'video':torch.vstack(frames).to(device=device), 'audio':  torch.vstack(waveforms).to(device=device),'folder_names':folder_names}
    return dat


wandb.init(
    project="test_aisynth",
    name="vid2audio",
    job_type="training")


batch_size=2


#init model

optim = torch.optim.SGD(model.parameters(),lr=lr)

train_dataset = CustomDataset(root_dir='/home/khalil/dtu/ai-synth/data/processed/train', max_waveform_length=audio_size[1], max_num_frames=video_size[0])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataset = CustomDataset(root_dir='/home/khalil/dtu/ai-synth/data/processed/val', max_waveform_length=audio_size[1], max_num_frames=video_size[0])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# %% Fit the model
# Number of epochs
epochs = 3
train_losses = []
val_losses = []
step = 0
for epoch in range(epochs):

    for dat in tqdm(train_loader):
        
        #simplified code - not using any exponential moving average (EMA) in this code,

        wandb.log({"loss": loss.detach().cpu().item()}, step=step)
        train_losses.append(loss.detach().cpu().item())


        loss =  F.mse_loss(video_target, video_out)+F.mse_loss(audio_target, audio_out)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"loss:{loss} time:{time.time()-time_start}")

        if False:#step % args.val_interval == 0:
            with torch.no_grad():
                model.eval()
                model_kwargs = {}
                val_iter = iter(val_loader)
                val_dat = next(val_iter)

                gt_save_path = os.path.join(args.output_dir, f'step_{step}', "gt")
                reconstruct_save_path = os.path.join(args.output_dir, f'step_{step}', "reconstract")
                audio_save_path = os.path.join(args.output_dir, f'step_{step}', "audio")
                os.makedirs(gt_save_path, exist_ok=True)
                os.makedirs(reconstruct_save_path, exist_ok=True)
                os.makedirs(audio_save_path, exist_ok=True)

                # save gt
                idx = 0
                for video, audio, folder_name in zip(val_dat["video"], val_dat["audio"],val_dat['folder_names']):             
                    #video = video.permute(0, 2, 3, 1)
                    video = ((video.cpu() + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()  
                    audio = audio.cpu().numpy()  
                    video_output_path = os.path.join(gt_save_path, f"{args.sample_fn}_{folder_name}.mp4")
                    save_multimodal(video, audio, video_output_path, args)
                    idx += 1        

                model_kwargs["video"] = val_dat["video"].to(dist_util.dev())
            
                shape = {"video":(args.batch_size , *args.video_size), \
                        "audio":(args.batch_size , *args.audio_size)
                    }

                sample_fn = (
                    diffusion.conditional_p_sample_loop if  args.sample_fn=="ddpm" else diffusion.ddim_sample_loop
                )

                sample = sample_fn(
                    model,
                    shape=shape,
                    use_fp16 = args.use_fp16,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    class_scale=args.classifier_scale
                    
                )

                video = ((sample["video"] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                audio = sample["audio"]              
                video = video.permute(0, 1, 3, 4, 2) #should be batchsize,16,16,256,1 
                print(video.shape)
                video = video.contiguous()

                all_videos = video.cpu().numpy()
                all_audios = audio.cpu().numpy()

                
                for video, audio in zip(all_videos, all_audios):
                    video_output_path = os.path.join(reconstruct_save_path, f"{args.sample_fn}_{folder_name}.mp4")
                    audio_output_path = os.path.join(audio_save_path, f"{args.sample_fn}_{folder_name}.wav")
                    
                    save_multimodal(video, audio, video_output_path, args)
                    save_audio(audio, audio_output_path, args.audio_fps)
        
                #  # Store the training and validation accuracy and loss for plotting
                val_losses_both = diffusion.multimodal_training_losses(model, val_dat, t)

                val_loss = (val_losses_both["loss"] * weights).mean()
                val_losses.append(val_loss.detach().cpu().item())

                wandb.log({"val_loss": val_loss.detach().cpu().item()}, step=step)
                model.train()


