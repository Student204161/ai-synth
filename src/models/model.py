import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

from transformers.models.vivit.modeling_vivit import VivitModel, VivitConfig, VivitLayer, VivitEncoder
#from transformers.models.speecht5.modeling_speecht5 import SpeechT5Decoder, SpeechT5Config, SpeechT5SpeechDecoderPostnet

class VivitTubeletEmbeddings(nn.Module):
    """
    Construct Vivit Tubelet embeddings.

    This module turns a batch of videos of shape (batch_size, num_frames, num_channels, height, width) into a tensor of
    shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size[0]) * (height // tubelet_size[1]) *
    (width // tubelet_size[2]).
    """

    def __init__(self, config):
        super().__init__()
        self.num_frames = config.num_frames
        self.image_size = config.image_size
        self.patch_size = config.tubelet_size
        # print(self.image_size)
        # print(self.patch_size)
        self.num_patches = (
            (self.image_size[1] // self.patch_size[2]) # 256/16
            * (self.image_size[0] // self.patch_size[1]) # 16/4
            * (self.num_frames // self.patch_size[0]) # 32/2
        )
        self.embed_dim = config.hidden_size

        self.projection = nn.Conv3d(
            config.num_channels, config.hidden_size, kernel_size=config.tubelet_size, stride=config.tubelet_size
        )

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height},{width}) doesn't match model ({self.image_size},{self.image_size})."
            )

        # permute to (batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        x = self.projection(pixel_values)
        # out_batch_size, out_num_channels, out_num_frames, out_height, out_width = x.shape
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class VivitEmbeddings(nn.Module):
    """
    Vivit Embeddings.

    Creates embeddings from a video using VivitTubeletEmbeddings, adds CLS token and positional embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = VivitTubeletEmbeddings(config)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embeddings.num_patches, config.hidden_size) #used to be +1 patch for cls tokens
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        #cls_tokens = self.cls_token.tile([batch_size, 1, 1])
        #embeddings = torch.cat((cls_tokens, embeddings), dim=1) #we dont do classificaiton, we do prediction so no cls tokens

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class SimpleModel(nn.Module):
    def __init__(self, embeddings, encoder):
        super().__init__()

        self.embeddings = embeddings
        self.encoder = encoder
        # self.decoder = decoder
        # self.decoder_postnet = decoder_postnet

        #self.pool = torch.nn.MaxPool2d((2,1))
        self.pool1 = torch.nn.MaxPool2d((1,4))
        #self.mlp = torch.nn.Linear(128* 88, 88 * 55)
        self.mlp1 = torch.nn.Linear(256 * 64, 88 * 55)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = self.embeddings(x)
        #from second last dimension, drop the class tokens bcs we aren't doing classification
        x = self.encoder(x)#.last_hidden_state#[:,1:]
        #x = self.decoder(x.last_hidden_state)
        x = self.pool1(x.last_hidden_state).flatten(1)
        # x , _ , _ = self.decoder_postnet(x.last_hidden_state)

        # x = self.pool(x).flatten(1)
        x = self.mlp1(x).reshape(B, 88, 55)
        #x = self.activation(x)
        return x #torch.clip(x,max=10)



if __name__ == "__main__":

    img = torch.ones([2, 32, 3, 16, 512]).cuda()

    vivit_dict = {
        "image_size": (16,512),
        "num_frames": 32,
        "tubelet_size": [4, 16, 32],
        "num_channels": 3,
        "hidden_size": 512,
        "num_hidden_layers": 4,
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

    conf_dict = {
      "num_mel_bins": 88,
      "activation_dropout": 0.1,
      "attention_dropout": 0.1,
      "decoder_attention_heads": 32,
      "decoder_ffn_dim": 512,
      "decoder_layerdrop": 0.1,
      "decoder_layers": 2,
      "decoder_start_token_id": 2,
      "hidden_act": "gelu",
      "hidden_dropout": 0.1,
      "hidden_size": 512,
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
    embeddings = VivitEmbeddings(config_vivit)
    encoder = VivitEncoder(config_vivit)
    #config_speecht5 = SpeechT5Config(**conf_dict)

    # decoder = SpeechT5Decoder(config_speecht5)
    # decoder_postnet = SpeechT5SpeechDecoderPostnet(config_speecht5)

    model = SimpleModel(embeddings, encoder).cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out = model(img)

    print("Shape of out :", out.shape)      # [B, num_classes]
    print("dtype of out :", out.dtype)      # float32

