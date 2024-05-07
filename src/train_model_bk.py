import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
from tqdm import tqdm
import wandb
import sys, os
#sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from tqdm import tqdm

from transformers.models.vivit.modeling_vivit import VivitModel, VivitConfig, VivitLayer, VivitEncoder
from transformers.models.speecht5.modeling_speecht5 import SpeechT5Decoder, SpeechT5Config, SpeechT5SpeechDecoderPostnet, SpeechT5HifiGan, SpeechT5HifiGanConfig

from models.model import AiSynthModel
from data.data_loader import CustomDataset, collate_fn



device = 'cuda' if torch.cuda.is_available() else 'cpu'



#init model
batch_size=1
train_dataset = CustomDataset(root_dir='data/processed/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
val_dataset = CustomDataset(root_dir='data/processed/val')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


wandb.init(
    project="aisynth_pls",
    name="vid2audio_pls",
    job_type="training")


config_vivit = VivitConfig()

conf_dict = {
"activation_dropout": 0.1,
"attention_dropout": 0.1,
"decoder_attention_heads": 8,
"decoder_ffn_dim": 3137,
"decoder_layerdrop": 0.1,
"decoder_layers": 4,
"decoder_start_token_id": 2,
"hidden_act": "gelu",
"hidden_dropout": 0.1,
"hidden_size": 768,
"is_encoder_decoder": True,
"layer_norm_eps": 1e-05,
"mask_feature_length": 4,
"mask_feature_min_masks": 0,
"mask_feature_prob": 0.0,
"mask_time_length": 4,
"mask_time_min_masks": 2,
"positional_dropout": 0.1,
"transformers_version": "4.40.1",
"use_guided_attention_loss": True,
}
vocoder_dict = {
"initializer_range": 0.01,
"leaky_relu_slope": 0.1,
"model_in_dim": 80,
"model_type": "hifigan",
"normalize_before": True,
"resblock_dilation_sizes": [
    [
    1,
    3,
    5
    ],
    [
    1,
    3,
    5
    ],
    [
    1,
    3,
    5
    ]
],
"resblock_kernel_sizes": [
    3,
    7,
    11
],
"sampling_rate": 16000,
"transformers_version": "4.40.0",
"upsample_initial_channel": 512,
"upsample_kernel_sizes": [
    8,
    8,
    8,
    8
],
"upsample_rates": [
    2,
    2,
    2,
    2
]
}


config_speecht5 = SpeechT5Config(**conf_dict)

encoder = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", add_pooling_layer=False)
decoder = SpeechT5Decoder(config_speecht5)
decoder_postnet = SpeechT5SpeechDecoderPostnet(config_speecht5)


config_vocoder = SpeechT5HifiGanConfig(**vocoder_dict)
vocoder = SpeechT5HifiGan(config_vocoder)

for param in encoder.parameters():
    param.requires_grad = False

model = AiSynthModel(encoder, decoder, decoder_postnet, vocoder, image_size=(224,224),tubelet_size=[2,8,32], num_frames = 64, dim = 512, num_layers=4).cuda()


parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3, weight_decay=1e-2)

#loss fun
criterion = nn.CrossEntropyLoss()


# %% Fit the model
# Number of epochs
epochs = 20
train_losses = []
val_losses = []
step = 0
val_interval=10
#use tqdm to print train loss and val loss as updating instead of constantly printing


val_loss = 10000
for epoch in range(epochs):

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} - Train") as pbar:

        for dat in train_loader:

            model.train()

            out = model(dat['frames'])

            loss = criterion(out, dat['wav'])
            wandb.log({"train_loss": loss.detach().cpu().item()}, step=step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'train_loss': f'{loss:.4f}','val_loss': f'{val_loss:.4f}'})
            pbar.update()
            if step % val_interval == 0:
                with torch.no_grad():
                    model.eval()

                    val_dat = next(iter(val_loader))

                    val_out = model(val_dat['frames'])


                    #cross entropy loss for reconstructed wav and "ground truth" wav
                    val_loss = criterion(torch.tensor(val_out), val_dat['wav'].detach().cpu())
                    wandb.log({"val_loss": val_loss.detach().cpu().item()}, step=step)


            step += 1



