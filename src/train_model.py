
import os
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
import torchvision
import soundfile as sf
from tqdm import tqdm
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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

        waveform, sample_rate = sf.read(wav_file)
        frames = self.load_frames(frame_folder)

        waveform, sample_rate = sf.read(wav_file)

        frames = frames.float() / 127.5 - 1 #0-1

        return torch.tensor(waveform).unsqueeze(0).unsqueeze(0), frames.unsqueeze(1).unsqueeze(0), folder_names

def collate_fn(batch):
    waveforms, frames, folder_names = zip(*batch)
    dat = {'video':torch.vstack(frames).to(device=device), 'audio':  torch.vstack(waveforms).to(device=device),'folder_names':folder_names}
    return dat


# Learning rate scheduler
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

wandb.init(
    project="test_aisynth",
    name="suhdude",
    job_type="training")


import sys,os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
import argparse
from mm_diffusion import dist_util, logger
from mm_diffusion.multimodal_datasets import load_data
from mm_diffusion.resample import create_named_schedule_sampler
from mm_diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from mm_diffusion.multimodal_train_util import TrainLoop
from mm_diffusion.common import set_seed_logger_random, save_audio, save_img, save_multimodal, delete_pkl
from mm_diffusion.fp16_util import MixedPrecisionTrainer

def load_training_data(args):
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        video_size=args.video_size,
        audio_size=args.audio_size,
        num_workers=args.num_workers,
        video_fps=args.video_fps,
        audio_fps=args.audio_fps
    )
   
    for video_batch, audio_batch in data:
        gt_batch = {"video": video_batch, "audio":audio_batch}
      
        yield gt_batch

def create_argparser():
    defaults = dict(
        data_dir="/home/khalil/dtu/ai-synth/data/processed/train/videos",
        schedule_sampler="uniform",
        lr=1e-4,
        t_lr=1e-4,
        seed=42,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        num_workers=0,
        save_type="mp4",
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        devices=None,
        save_interval=10000,
        output_dir="/home/khalil/dtu/ai-synth/results/gt",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        use_db=False,
        sample_fn="ddpm",
        frame_gap=1,
        video_fps=16,
        audio_fps=25600,
        video_size="32,1,16,256",
        audio_size="1,51200",
        val_interval=10
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    

args = create_argparser().parse_args()
args.video_size = [int(i) for i in args.video_size.split(',')]
args.audio_size = [int(i) for i in args.audio_size.split(',')]
logger.configure(args.output_dir)
dist_util.setup_dist(args.devices)

args = set_seed_logger_random(args)

logger.log("creating model and diffusion...")

model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
)

model.to(dist_util.dev())

schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

logger.log("creating data loader...")

train_dataset = CustomDataset(root_dir='/home/khalil/dtu/ai-synth/data/processed/train')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_dataset = CustomDataset(root_dir='/home/khalil/dtu/ai-synth/data/processed/val')
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

mp_trainer = MixedPrecisionTrainer(
            model=model,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth
        )

opt = torch.optim.AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)


# %% Fit the model
# Number of epochs
epochs = 3
train_losses = []
val_losses = []
step = 0
print(args.lr)
for epoch in range(epochs):

    for dat in tqdm(train_loader):
        
        t, weights = schedule_sampler.sample(args.batch_size, dist_util.dev())
        
        mp_trainer.zero_grad()
        losses_both = diffusion.multimodal_training_losses(model, dat, t)

        loss = (losses_both["loss"] * weights).mean()
        mp_trainer.backward(loss)
        took_step = mp_trainer.optimize(opt)
        #simplified code - not using any exponential moving average (EMA) in this code,


        wandb.log({"loss": loss.detach().cpu().item()}, step=step)
        train_losses.append(loss.detach().cpu().item())

        if step % args.val_interval == 0:
            val_iter = iter(val_loader)
            val_dat = next(val_iter)
            with torch.no_grad():
                model.eval()

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

                model_kwargs["video"] = batch_data["video"].to(dist_util.dev())
            
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

                video = ((sample["video"] + 1) * 127.5).clamp(0, 255).to(th.uint8)
                audio = sample["audio"]              
                video = video.permute(0, 1, 3, 4, 2)
                video = video.contiguous()

                all_videos = video.cpu().numpy()
                all_audios = audio.cpu().numpy()

                
                idx = 0
                for video, audio in zip(all_videos, all_audios):
                    video_output_path = os.path.join(reconstruct_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}.mp4")
                    audio_output_path = os.path.join(audio_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}.wav")
                    
                    save_multimodal(video, audio, video_output_path, args)
                    save_audio(audio, audio_output_path, args.audio_fps)
        
                    idx += 1        
                    
                groups += 1
                dist.barrier()

 
                # Store the training and validation accuracy and loss for plotting
                val_losses_both = diffusion.multimodal_training_losses(model, val_dat, t)

                val_loss = (val_losses_both["loss"] * weights).mean()
                val_losses.append(val_loss.detach().cpu().item())

                wandb.log({"val_loss": val_loss.detach().cpu().item()}, step=step)
                model.train()


