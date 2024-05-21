# %%
# import av
import numpy as np
import random
from torchvision.utils import make_grid

from transformers import VivitImageProcessor, VivitModel, VivitConfig, TransfoXLLMHeadModel, TransfoXLConfig
from transformers import SpeechT5ForSpeechToText, SpeechT5Config
from datasets import load_metric
from huggingface_hub import hf_hub_download
# print(transformers.__version__)
import matplotlib.pyplot as plt
from torch import nn
# import py_midicsv as pm
import torch
import torch.nn.functional as F

from PIL import Image
import os
from miditok import REMI, TokenizerConfig  # here we choose to use REMI
import miditok
from pathlib import Path
from torch.cuda.amp import autocast

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from torch import optim

from tqdm import tqdm
import wandb

# %%
# elle
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def cfg_to_dict(cfg):
    return {attr: getattr(cfg, attr) for attr in dir(cfg) if not attr.startswith("__") and not callable(getattr(cfg, attr))}

def read_frames_from_path(frames_path, indices, rgb=False):
    '''
    Read specific frames from a directory containing image files of video frames.
    Args:
        frames_path (str): Path to the directory containing frame images.
        indices (List[int]): List of frame indices to read.
    Returns:
        result (np.ndarray): numpy array of frames of shape (num_frames, height, width, 3).
    '''
    # List all files in the directory and sort them to maintain order
    all_files = sorted(os.listdir(frames_path))
    frames = []

    # Process only files at specific indices
    color_mode = 'RGB' if rgb else 'L'
    for idx in indices:
        if idx < len(all_files):
            file_path = os.path.join(frames_path, all_files[idx])
            with Image.open(file_path) as img:
                # Convert image to RGB to ensure consistency
                img = img.convert(color_mode)
                # Calculate differences to make the image square
                width, height = img.size
                max_side = max(width, height)
                # Create a new image with a black background
                new_img = Image.new(color_mode, (max_side, max_side))
                # Paste the original image onto the center of the new image
                new_img.paste(img, ((max_side - width) // 2, (max_side - height) // 2))
                frame_array = np.array(new_img)
                if color_mode == 'L':
                    # Expand dims to add the channel dimension, resulting in (H, W, 1)
                    frame_array = np.expand_dims(frame_array, axis=-1)
                frames.append(frame_array)

    stacked_frames = np.stack(frames, axis=0)
    return stacked_frames

# def read_video_pyav(container, indices):
#     '''
#     Decode the video with PyAV decoder.
#     Args:
#         container (`av.container.input.InputContainer`): PyAV container.
#         indices (`List[int]`): List of frame indices to decode.
#     Returns:
#         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
#     '''
#     frames = []
#     container.seek(0)
#     start_index = indices[0]
#     end_index = indices[-1]
#     for i, frame in enumerate(container.decode(video=0)):
#         if i > end_index:
#             break
#         if i >= start_index and i in indices:
#             frames.append(frame)
#     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# %%
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

class Video2MIDIDataset(Dataset):
    def __init__(self, root_dir, tokenizer, image_processor, transform=None, color_mode='gray'):
        self.root_dir = root_dir
        self.frames_dir = os.path.join(root_dir, 'frames')
        self.midi_dir = os.path.join(root_dir, 'midi')
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.transform = transform
        self.piece_names = [d for d in os.listdir(self.frames_dir) if os.path.isdir(os.path.join(self.frames_dir, d))]
#         self.piece_names = self.piece_names[:2] # TODO: remove!!!
        assert self.piece_names, f"frame_dir at {self.frames_dir} is empty!"
        self.color_mode = 'L' if color_mode == 'gray' else 'RGB'
        # for when we want to train on different sampling rate than what data has
        # e.g. if we have 64 frames but want only 32 of them every 2 frames
        self.frame_indices = set(sample_frame_indices(32, frame_sample_rate=2, seg_len=65))
        assert len(self.frame_indices) == 32

    def __len__(self):
        return len(self.piece_names)

    def __getitem__(self, idx):
        piece_name = self.piece_names[idx]
        frames_path = os.path.join(self.frames_dir, piece_name)
        midi_path = os.path.join(self.midi_dir, f'{piece_name}.mid')

        midi_token_ids = self.load_midi(midi_path)
        frames = self.load_frames(frames_path, rgb=True)
        assert len(frames) > 0, f"no frames found at {frames_path}"
#         img_side_width = 64
#         img_size = (img_side_width,img_side_width)
#         print(frames.shape, type(frames))
        # NOTE (elle): MUST convert frames to list for some reason otherwise it complains!!!
#         processed_frames = self.image_processor(list(frames), return_tensors="pt", do_center_crop=False, do_resize=True, size=img_size)
        processed_frames = image_processor(list(frames), return_tensors="pt")
#         print(processed_frames)

        sample = {'frames': processed_frames['pixel_values'], 'midi_tokens': midi_token_ids}
        return sample

#     def load_frames(self, frames_path):
#         frame_files = sorted(os.listdir(frames_path))
#         frames = [Image.open(os.path.join(frames_path, f)).convert(self.color_mode) for f in frame_files]
#         print(f"img shape: {print(frames[0].size)}")
#         if self.transform:
#             frames = [self.transform(frame) for frame in frames]
#         return frames

    def load_frames(self, frames_path, rgb=False):
        '''
        Read specific frames from a directory containing image files of video frames.
        Args:
            frames_path (str): Path to the directory containing frame images.
            indices (List[int]): List of frame indices to read.
        Returns:
            result (np.ndarray): numpy array of frames of shape (num_frames, height, width, 3).
        '''
        # List all files in the directory and sort them to maintain order
        frame_names = sorted(os.listdir(frames_path))
        frames = []

        # Process only files at specific indices
        color_mode = 'RGB' if rgb else 'L'
        for idx, frame_name in enumerate(frame_names):
            if self.frame_indices == {} or (self.frame_indices != {} and idx in self.frame_indices):
                file_path = f"{frames_path}/{frame_name}"
                with Image.open(file_path) as img:
                    # Convert image to RGB to ensure consistency
                    img = img.convert(color_mode)
                    # Calculate differences to make the image square
                    width, height = img.size
                    max_side = max(width, height)
                    # Create a new image with a black background
                    new_img = Image.new(color_mode, (max_side, max_side))
                    # Paste the original image onto the center of the new image
                    new_img.paste(img, ((max_side - width) // 2, (max_side - height) // 2))
                    frame_array = np.array(new_img)
                    if color_mode == 'L':
                        # Expand dims to add the channel dimension, resulting in (H, W, 1)
                        frame_array = np.expand_dims(frame_array, axis=-1)
                    frames.append(frame_array)
        stacked_frames = np.stack(frames, axis=0)
        return stacked_frames

    def load_midi(self, midi_path):
#         midi_token_ids_numpy = np.array(self.tokenizer(midi_path)[0].ids)
        midi_tokens = self.tokenizer(midi_path)
        midi_token_ids = torch.tensor(midi_tokens[0].ids, dtype=torch.long)
        return midi_token_ids

def custom_collate_fn(batch, tokenizer):
    # Extract frames and midi_tokens from the batch
    frames = [item['frames'] for item in batch]
    midi_tokens = [item['midi_tokens'] for item in batch]
    # Pad the midi_tokens
    # Assuming tokenizer provides PAD token index via tokenizer['PAD_None']
    pad_token_index = tokenizer['PAD_None']  # Ensure this is the correct index for your PAD token
    # print length before padding
#     print("Length before padding: ", [len(midi_token) for midi_token in midi_tokens])
    midi_tokens_padded = pad_sequence(midi_tokens, batch_first=True, padding_value=pad_token_index)
#     print("Length after padding: ", [len(midi_token) for midi_token in midi_tokens_padded])

    # Collate frames normally (assuming they are tensors of the same shape)
    frames = default_collate(frames)
    # Return a new dictionary with padded midi_tokens and frames
    return {'frames': frames, 'midi_tokens': midi_tokens_padded}

def show_images_and_midi(dataloader):
    for i, batch in enumerate(dataloader):
        frames = batch['frames']  # Assuming frames are tensors of shape (batch_size, channels, extra_dim, another_channel_like, height, width)
        midi_tokens = batch['midi_tokens']  # MIDI tokens

        print(f"Batch {i + 1}")

        # Calculate the number of rows and columns for the subplots
        batch_size = frames.size(0) # torch.Size([4, 1, 302, 1, 64, 64])
#         n_igms_in_batch = frames.size(2)
        cols = int(np.ceil(np.sqrt(batch_size)))
        rows = int(np.ceil(batch_size / cols))

        # Displaying images in a grid that's as square as possible
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust size as needed
        axs = axs.flatten()  # Flatten the array of axes to make indexing easier

        for j in range(batch_size):
            # Select the first image from the video sequence and remove the singleton dimensions
            img = frames[j, 0, 0, 0]  # Reduces to (64, 64)
            axs[j].imshow(img.numpy())
            axs[j].axis('off')  # Hide axes
            axs[j].set_title(f'MIDI: {midi_tokens[j]}')  # Optionally print MIDI token IDs

        # Hide any unused axes if the total number of subplots exceeds the batch size
        for k in range(batch_size, len(axs)):
            axs[k].axis('off')

        plt.show()

        # Optional: stop after first batch for demonstration
        if i == 0:
            break

def show_frames(frames):
    print("First Batch")

    # Calculate the total number of images to display
    total_images = len(frames)  # Assuming 'extra_dim' holds 32 images

    cols = int(np.ceil(np.sqrt(total_images)))
    rows = int(np.ceil(total_images / cols))

    # Displaying images in a grid that's as square as possible
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust size as needed
    axs = axs.flatten()  # Flatten the array of axes to make indexing easier

    for j in range(total_images):
        # Select the image from the video sequence for each frame in the batch
        img = frames[j]  # Adjust indexing based on your data's shape, using the first item in batch
        axs[j].imshow(img.numpy())
        axs[j].axis('off')  # Hide axes
        axs[j].set_title(f'MIDI: {midi_tokens[0]}')  # Optionally print MIDI token IDs for the first item in batch

    # Hide any unused axes if the total number of subplots exceeds the total images
    for k in range(total_images, len(axs)):
        axs[k].axis('off')

    plt.show()

def show_images_and_midi_one_batch(dataloader, tokenizer):
    # Fetch the first batch from the dataloader
    token_id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    batch = nextclas(iter(dataloader))
    batch_i = 0
    frames = batch['frames']  # Assuming frames are tensors of shape (batch_size, channels, extra_dim, another_channel_like, height, width)
    # ^ [4, 1, 32, 3, 224, 224]
    midi_tokens = batch['midi_tokens']  # MIDI tokens
    print(f"miditokens shape {midi_tokens.shape}")
    print("First Batch")

    # Calculate the total number of images to display
    total_images = frames.size(2)  # Assuming 'extra_dim' holds 32 images

    cols = int(np.ceil(np.sqrt(total_images)))
    rows = int(np.ceil(total_images / cols))

    # Displaying images in a grid that's as square as possible
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust size as needed
    axs = axs.flatten()  # Flatten the array of axes to make indexing easier

    midi_translations = []
    midis = []
    tokenss = []
    for j in range(total_images):
        # Select the image from the video sequence for each frame in the batch
        img = frames[0, 0, j] # Adjust indexing based on your data's shape, using the first item in batch
        axs[j].imshow(img.permute(1, 2, 0).numpy())
        axs[j].axis('off')  # Hide axes
        tokens = midi_tokens[batch_i]
        tokenss.append(tokens)
        midi = tokenizer([tokens])
        midis.append(midi)
        midi_translation = [token_id_to_token[int(id_.detach().numpy())] for id_ in tokens]
        midi_translations.append(midi_translation)
#         axs[j].set_title(f'MIDI: {midi_tokens_translated}')  # Optionally print MIDI token IDs for the first item in batch

    # Hide any unused axes if the total number of subplots exceeds the total images
    for k in range(total_images, len(axs)):
        axs[k].axis('off')

    title = f"{midis[batch_i]}\n{midi_translations[batch_i]}\n{tokenss[batch_i]}"
    fig.suptitle(title, fontsize=16)
    print(title)

    plt.show()

def collate_fn(batch):
    return custom_collate_fn(batch, tokenizer)

def compute_accuracy(outputs, labels):
    logits = outputs.logits
    prediction_ids = torch.argmax(logits, dim=-1)
    # Flatten the tensors to compare each token
    prediction_ids = prediction_ids.view(-1)
    labels = labels.view(-1)
    
    # Compare predictions with labels
    correct = (prediction_ids == labels).sum().item()
    total = labels.size(0)

    accuracy = correct / total
    return accuracy, correct, total

# %% [markdown]
# # TODO
# * investigate why this training loop (using real data but just twinkle sequence 0, one second) seems to work now -- is it AdamW, the LR, or probably just that getting a transformer to overfit on random labels won't work because not properly padded with start/end token? although idk bc I feel like I tried it with non-random at first and it didn't work so truly I have no idea
# * why does the loss not quite go to 0 though? It hovers around .03 and then is a bit unstable and goes to .3 even sometimes e.g. a continuation of it although btw here https://wandb.ai/elles/video2music/runs/kmfa7au7?nw=nwuserellesummer I would think it should go to .0005ish which is what happened when I overfit to a single label sequence of the same numbers although at one point it did get to 0.0086 (not tracked in wandb). Just feel like it should be trivially easy/stable to memorize one one-sec sequence idk
# * BUT good news: despite loss not being ~0, the output of the model is still the given sequence, so all is good heh
# * ^^ take it further and track wer metric in general AH or just the accuracy tbh like per token (if it doesn't match for example)
# 
# btw for original model/dataset (not ours! aka the audio one)
# shapes:
# labels: torch.Size([1, 67]). encoder_outputs: torch.Size([1, 112, 768]). labels: torch.Size([1, 22]). encoder_outputs: torch.Size([1, 113, 768]).
# 
# our model tho is like: encoder_outputs: torch.Size([1, 3137, 768]) labels: torch.Size([1, 16])  
# with 2 batch size  
# label shape torch.Size([2, 20])  
# frame shape torch.Size([2, 33, 3, 224, 224])  
# encoder_outputs shape torch.Size([2, 3137, 768])  

# %% [markdown]
# ## NEXT
# * last progress: able to fully overfit on twinkle 1 second clips but really slowly (converged to 100% accuracy after an hour yikes)
# * is slowness bc of the size of the encoder? is this worth the trade off of not having to train our own encoder? or is it bc the image quality has a lot of noise when resizing to fit the encoder needs? maybe worth using Khalil's modified encoder then...
# * will the model scale well to e.g. 2-5 second clips? (dimitrios suggested 5 sec) and across different songs?
# * try creating validation set just within twinkle twinkle? how to create a good one for this task in general? e.g. what should be "allowed" to overlap? We should see much less overlap in both inp/output when using longer sequences e.g. due to permutation diversity
# * debug why the original training didn't work for your own sanity/learning

# %%
# frames_path = f"{ds_dir}/processed/frames/raw--twinke_twinkle_0"
# # indices = sample_frame_indices(clip_len=32, frame_sample_rate=1)
# indices = range(32)
# video = read_frames_from_path(frames_path=frames_path, indices=indices, rgb=True)
# inputs = image_processor(list(video), return_tensors="pt")
# print(inputs["pixel_values"].shape)
# fig, ax = plt.subplots(2, 6, figsize=(12, 8))
# for i in range(12):
#     ax[i // 3, i % 3].imshow(inputs["pixel_values"][0, i].permute(1, 2, 0).numpy())
#     ax[i // 3, i % 3].axis("off")

# %%
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %%
def show_data_samples(data_loader, title, nrow=3, figsize=(15, 6)):
    # Get a batch of training data
#     idx2class = {0: 'A', 1: 'Ab', 2: 'B', 3: 'Bb', 4: 'C', 5: 'D', 6: 'E', 7: 'Eb', 8: 'F', 9: 'G', 10: 'Gb'}
    dataiter = iter(data_loader)
    batch = next(dataiter)
    # just show first seq
    frames, labels = batch["frames"][0][0], batch["midi_tokens"][0]
    # subset does not have class_to_idx so remove this line
    # idx2class = {v: k for k, v in data_loader.dataset.class_to_idx.items()}
#     print(idx2class)
    print(labels)
    label_list = [label.item() for label in labels]
    print(labels)
#     notes = [idx2class[label] for label in label_list]

    plt.figure(figsize=figsize)
    plt.title(f"{title} {label_list}")
    print(frames.shape)
    imshow(make_grid(frames[:10], nrow=nrow))

# %%
# show_data_samples(DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn), 'Training Data Samples', figsize=(20,20))

# %%

# ds_dir = "../input/data-32fps/data_32fps"
# ds_dir = "/kaggle/input/twinkle-32fps-5s/Users/ellemcfarlane/Documents/dtu/ai-synth/data_32fps_5s"
# ds_dir = "/kaggle/input/data-13fps-5s/data_13fps_5s"
# ds_dir = "/kaggle/input/pop-70/data_pop"
ds_dir = "/work3/s222376/ai-synth/data/data_pop"
# frames_path = f"{ds_dir}/processed/frames/somebody--twinke_twinkle_0"
# frames_path = f"{ds_dir}/processed/frames/raw--twinke_twinkle_0"
# Our parameters
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4}, # TODO: finetune beat res?
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}
tok_config = TokenizerConfig(**TOKENIZER_PARAMS)
# wer_metric = load_metric("wer")
# Creates the tokenizer
tokenizer = REMI(tok_config)
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
train_dataset = Video2MIDIDataset(
    root_dir=f"{ds_dir}/processed/train",
    tokenizer=tokenizer,
    image_processor=image_processor
    # transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
)
val_dataset = Video2MIDIDataset(
    root_dir=f"{ds_dir}/processed/val",
    tokenizer=tokenizer,
    image_processor=image_processor
    # transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
)
print(f"Training on {len(train_dataset.piece_names)}, validating on {len(val_dataset.piece_names)}")

config = SpeechT5Config(
    vocab_size=328,
    d_model=768,
    max_length=450
)

model = SpeechT5ForSpeechToText(config)
model = model.to('cuda')  # Ensure your model is on GPU if available
model_enc = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
model_enc = model_enc.to('cuda')
model.speecht5.encoder = None

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 43
    
    LOG_EVERY_X_EPOCHS = 1
#     SAVE_EVERY_X_EPOCHS = 10

    LR = 0.0001 #5e-5 # 4.786300923226385e-05 # 0.0001
    EPOCHS = 100
    BATCH_SIZE = 12 # 6 for 16GB gpu

    USE_WANDB = True
    WANDB_PROJECT = "video2music"
    WANDB_ENTITY = "elles"
    WANDB_GROUP = f"hpc_pop70_bsz{BATCH_SIZE}_lr{LR}"
    EXPERIMENT = WANDB_GROUP
    # FPS = 32

cfg = Config()

# print which device running on 
print(f"Running on {cfg.DEVICE}")

train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

if cfg.SEED:
    set_seed(cfg.SEED)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
# Training loop
model.train()

if cfg.USE_WANDB:
    # convert cfg class to dict
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict.update(config.to_dict())
    print(cfg_dict)
    assert cfg_dict != {}, "cfg_dict is empty"
    wandb_id = wandb.util.generate_id()
    wandb.init(
        project=cfg.WANDB_PROJECT,
        name=cfg.EXPERIMENT,
        entity=cfg.WANDB_ENTITY,
        config=cfg_dict,
        id=wandb_id,
        resume="allow",
        group=cfg.WANDB_GROUP
    )
    wandb.watch(model, log="all", log_freq=10)

epochs = cfg.EPOCHS

best_acc = 0
for epoch in range(epochs):
    total_correct = 0
    total_tokens = 0
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
    model.train()
    for batch in progress_bar:
  
        optimizer.zero_grad()
        frames = batch['frames'].to('cuda')
        labels = batch['midi_tokens'].to('cuda')
        with torch.no_grad():
#             frames = batch['frames']

            # labels = rand_labels.unsqueeze(0)
#             labels = batch['midi_tokens']
#             frames = frames.to('cuda')
            # squeeze only if batch-size is > 1
            frames = frames.squeeze(1)
#             labels = labels.to('cuda')
            outputs = model_enc(frames)
            last_hidden_states = outputs.last_hidden_state
#         new_encoder_outputs = last_hidden_states.to('cuda')
        new_encoder_outputs = (last_hidden_states,)
        inputs = {
            "encoder_outputs": new_encoder_outputs,
            "labels": labels # labels.unsqueeze(0) only unsqueeze if batch size is 1?
        }
        outputs = model(**inputs)

        # Loss computation
        loss = outputs.loss
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        accuracy, batch_corr, batch_tot = compute_accuracy(outputs, labels)

        # Accumulate results
        # TODO fix: this is not accumulating per batch..
        total_correct += batch_corr
        total_tokens += batch_tot

        progress_bar.set_postfix({'loss': loss.item()})

    # Logging to wandb
    average_loss_train = total_loss / len(train_dataloader)
    overall_train_accuracy = total_correct / total_tokens
    stats = {'epoch': epoch, 'train_loss': average_loss_train, 'train_accuracy': overall_train_accuracy}
    if epoch % cfg.LOG_EVERY_X_EPOCHS == 0:
        total_correct_val = 0
        total_tokens_val = 0
        total_loss_val = 0
        model.eval()
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader)
            for batch in progress_bar:
                frames = batch['frames'].to('cuda')
                labels = batch['midi_tokens'].to('cuda')
                frames = frames.squeeze(1)
                outputs = model_enc(frames)
                last_hidden_states = outputs.last_hidden_state
                new_encoder_outputs = (last_hidden_states,)
                inputs = {
                    "encoder_outputs": new_encoder_outputs,
                    "labels": labels # labels.unsqueeze(0) only unsqueeze if batch size is 1?
                }
                outputs = model(**inputs)
                loss = outputs.loss
                total_loss_val += loss.item()
                _accuracy_val, batch_corr, batch_tot = compute_accuracy(outputs, labels)

                # Accumulate results
                # TODO fix: this is not accumulating per batch..
                total_correct_val += batch_corr
                total_tokens_val += batch_tot
            average_loss_val = total_loss_val / len(val_dataloader)
            overall_accuracy_val = total_correct_val / total_tokens_val
            stats['val_loss'] = average_loss_val
            stats['val_accuracy'] = overall_accuracy_val
#             stats = {'epoch': epoch, 'val_loss': average_loss_val, 'val_accuracy': overall_accuracy_val}
            print(stats)
            if overall_accuracy_val > best_acc:
                print("saving best model so far")
                torch.save(model.state_dict(), f"{cfg.EXPERIMENT}.pt")
            if wandb.run:
                wandb.log(stats)
    del frames, labels, outputs, last_hidden_states, new_encoder_outputs, inputs
    torch.cuda.empty_cache()

# End of training
if wandb.run:
  wandb.finish()

# %%
# model.eval()
# with torch.no_grad():
#   inputs = {
#         "encoder_outputs": last_hidden_states.to('cuda'),
#         "decoder_input_ids": labels.unsqueeze(0)
#         # "attention_mask": batch["attention_mask"].to('cuda'),
#         # "labels": batch["decoder_input_ids"].to('cuda'), # TODO: this for sure gets shifted automatically by the library no?
#   }
#   outputs = model(**inputs)

# # %% [markdown]
# # # Check that the trained model is outputting correct sequence

# # %%
# compute_accuracy(outputs, labels)

# # %%
# logits = outputs.logits
# predicted_ids = torch.argmax(logits, dim=-1)

# # %%
# predicted_ids

# # %%
# labels

# %%
# tokens = midi_tokens[batch_i]
# tokenss.append(tokens)
# midi = tokenizer([tokens])
# midis.append(midi)
# midi_translation = [token_id_to_token[int(id_.detach().numpy())] for id_ in tokens]
# midi_translations.append(midi_translation)


