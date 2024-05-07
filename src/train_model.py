import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
from tqdm import tqdm
import wandb
import sys, os
#sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
from torch import nn

from transformers.models.vivit.modeling_vivit import VivitConfig, VivitEncoder


from models.model import SimpleModel, VivitEmbeddings
from data.data_loader import CustomDataset, collate_fn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
#init model

vivit_dict = {
    "image_size": (16,512),
    "num_frames": 32,
    "tubelet_size": [4, 16, 32],
    "num_channels": 3,
    "hidden_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 256,
    "intermediate_size": 512,
    "hidden_act": 'gelu_fast',
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-06,
    "qkv_bias": True
}
config_vivit = VivitConfig(**vivit_dict)

embeddings = VivitEmbeddings(config_vivit)
encoder = VivitEncoder(config_vivit)

model = SimpleModel(embeddings, encoder).cuda()

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)


batch_size=16
train_dataset = CustomDataset(root_dir='data/processed/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataset = CustomDataset(root_dir='data/processed/val')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print('Train data N:',len(train_dataset))
print('Val data N:',len(val_dataset))
day = iter(train_dataset)
fram, spe, name = next(day)

optimizer = torch.optim.AdamW(model.parameters(),weight_decay=1e-6, lr = 1e-4)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

#loss fun
criterion = nn.L1Loss() #nn.L1Loss() #nn.CrossEntropyLoss() # no reason to use crossentropy if we dont work with classes...

wandb.init(
    project="aisynth_works",
    name="vid2audio",
    job_type="training",
    reinit=True)

# %% Fit the model
# Number of epochs
epochs = 20
train_losses = []
val_losses = []
step = 0
val_interval=100
save_interval=1000
lr_update=1000
# plot_interval = 100
#use tqdm to print train loss and val loss as updating instead of constantly printing

val_loss = 10000
for epoch in range(epochs):
    train_epoch_loss =[]
    val_epoch_loss =[]
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} - Train") as pbar:
        for dat in train_loader:
            model.train()
            optimizer.zero_grad()
            out = model(dat['frames'])
            loss = criterion(out, dat['spectrogram'])
            wandb.log({"train_loss": loss.detach().cpu().item()}, step=step)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.detach().cpu())
            pbar.set_postfix({'train_loss': f'{loss:.4f}','val_loss': f'{val_loss:.4f}'})
            pbar.update()

            # if step % plot_interval == 0:
            #   fig, axes = plt.subplots(1, 2)
            #   h = out.shape[2]
            #   w = out.shape[1]

            #   viz_out = out[0].detach().clone().cpu().numpy()
            #   viz_ref = dat['spectrogram'][0].detach().clone().cpu().numpy()

            #   # Plot the first image on the first subplot
            #   axes[0].imshow(viz_out,aspect='auto', extent=(0, h, 0, w))
            #   axes[0].set_title('predicted')

            #   # Plot the second image on the second subplot
            #   axes[1].imshow(viz_ref,aspect='auto', extent=(0, h, 0, w))
            #   axes[1].set_title('GT')

            #   plt.show()

              # fig, axes = plt.subplots(1, 2)
              # h = out.shape[2]
              # w = out.shape[1]


              # viz_out = np.exp(out[0].detach().clone().cpu().numpy())
              # viz_ref = np.exp(dat['spectrogram'][0].detach().clone().cpu().numpy())

              # # Plot the first image on the first subplot
              # axes[0].imshow(viz_out,aspect='auto', extent=(0, h, 0, w))
              # axes[0].set_title('predicted_log')

              # # Plot the second image on the second subplot
              # axes[1].imshow(viz_ref,aspect='auto', extent=(0, h, 0, w))
              # axes[1].set_title('GT_log')

            #   plt.show()

            if step % val_interval == 0:
                with torch.no_grad():
                    model.eval()

                    val_dat = next(iter(val_loader))

                    val_out = model(val_dat['frames'])

                    val_loss = criterion(val_out, val_dat['spectrogram'])

                    val_epoch_loss.append(val_loss.detach().cpu())


                    # fig, axes = plt.subplots(1, 2)

                    # viz_out_val = val_out[0].detach().clone().cpu().numpy()
                    # viz_ref_val = val_dat['spectrogram'][0].detach().clone().cpu().numpy()

                    # # Plot the first image on the first subplot
                    # axes[0].imshow(viz_out_val,aspect='auto', extent=(0, h, 0, w))
                    # axes[0].set_title('validation predicted')

                    # # Plot the second image on the second subplot
                    # axes[1].imshow(viz_ref_val,aspect='auto', extent=(0, h, 0,w))
                    # axes[1].set_title('validation GT')

                    # plt.show()

                    wandb.log({"val_loss": val_loss.detach().cpu().item()}, step=step)


            step += 1
            if step % save_interval == 0:
              torch.save(model.state_dict(),f'models/ai_synth_model_works_6l_{step}.pth')

            if step % lr_update == 0:
              scheduler.step()

    wandb.log({"avg_train_loss": np.mean(train_epoch_loss)})
    wandb.log({"avg_val_loss": np.mean(val_epoch_loss)})

