"""
Originally inspired by impl at https://github.com/facebookresearch/DiT/blob/main/models.py

Modified by Haoyu Lu, for video diffusion transformer
"""
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# 
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange, reduce, repeat
from einops_exts import check_shape, rearrange_many
from list_rotary import RotaryEmbedding
from torch import einsum
import torch.nn.functional as F


def modulate(x, shift, scale, T):

    N, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
    return x

def exists(x):
    return x is not None

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
class SParaEmbedder(nn.Module):
    """
    Embeds s parameters into vector representations (per-frame conditioning). Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, frames, time_hidden_size, hidden_size, dropout_prob):
        super().__init__()
        self.sign_emb = nn.Linear(1, hidden_size)
        # self.sign_emb = nn.Linear(2, hidden_size)
        self.cond_token_to_hidden = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, time_hidden_size)
                )
        self.dropout_prob = dropout_prob

        # conditional guidance
        # self.null_text_token = nn.Parameter(torch.randn(1, frames, hidden_size))
        # self.null_text_hidden = nn.Parameter(torch.randn(1, time_hidden_size))
        

    def token_drop(self, labels, null_labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, null_labels, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        # s-para embbeding used for per-frame conditioning
        # add dimension to generate tokens per frame [batch x frames x 1] -> [batch x frames x embedding]
        labels_s = labels[:,:,0].unsqueeze(-1).to(labels.device)
        # labels_f = labels[:,:,1].unsqueeze(-1).to(labels.device)
        labels_null_s = torch.ones(labels_s.shape).to(labels.device)
        # labels_null = torch.cat([labels_null_s,labels_f],dim=-1).to(labels.device)
        # this acts on the last dimension, gives token embedding for attention
        label_emb_token = self.sign_emb(labels_s)
        null_emb_token = self.sign_emb(labels_null_s)
        # average over frames dim to get hidden embedding
        mean_pooled_text_tokens = label_emb_token.mean(dim = -2)
        mean_pooled_null_tokens = null_emb_token.mean(dim = -2)
        # convert hidden averaged token embedding to hidden embedding
        label_emb_hidden = self.cond_token_to_hidden(mean_pooled_text_tokens)
        null_emb_hidden = self.cond_token_to_hidden(mean_pooled_null_tokens)

        # drop-out in batch dimensions for cfg
        # forced drop ids for consistency between time embedding and token embedding
        drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        token_drop_ids = rearrange(drop_ids, 'b -> b 1 1')
        hidden_drop_ids = rearrange(drop_ids, 'b -> b 1')
       
        if (train and use_dropout):
            label_emb_token = self.token_drop(label_emb_token, null_emb_token, token_drop_ids)
            label_emb_hidden = self.token_drop(label_emb_hidden, null_emb_hidden, hidden_drop_ids)
        return label_emb_token, label_emb_hidden


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

#################################################################################
#                                 Core VDT Model                                #
#################################################################################

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None, 
        cond_attention = None, 
        cond_dim = 64,
        per_frame_cond = True,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias=False)
        # self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.cond_attention = cond_attention # none, stacked self-attention or cross-attention
        
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

        self.per_frame_cond = per_frame_cond

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None, # we do not care about this
        label_emb_mm = None,
        rotary_emb_list = None,
        if_temperal = False,
        b = 4,
    ):
        # b is batch, b2 is either (h w) or f which will be treated as batch, n is the token, c the dim from which we build the heads and dim_head
        x = rearrange(x, '(b n) t d -> b n t d', b=b)
        b, b2, n, c = x.shape

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)
        if exists(self.rotary_emb) and if_temperal:
            k = self.rotary_emb.rotate_queries_or_keys(t=k, seq_list = rotary_emb_list)
        ek = self.to_k(label_emb_mm) # this acts on last dim
        ev = self.to_v(label_emb_mm) # this acts on last dim
        if if_temperal:  
            # repeat ek and ev along frames/pixels for agnostic attention, also holds for per-frame-conditioning in temporal attention (f=n), where we broadcast time signal to all pixels
            ek, ev = map(lambda t: repeat(t, 'b n c -> b b2 n c', b2 = b2), (ek, ev))
        else:
            # indicator of spatial attention -> we do per-frame-conditioning, otherwise we condition on the whole video with positional bias
            # in spatial attention, we get in b, 11, 1, c, i.e., we align the 11 frames of x [b, b2, n, c] - 'b2' with the 11 frames of label_emb_mm [batch x frames x embedding] - 'frames'
            # add single token (n=1) add correct dimension
            ek, ev = map(lambda t: repeat(t, 'b f c -> b f 1 c'), (ek, ev))
        # rearrange so that linear layer without bias corresponds to head and head_dim
        ek, ev = map(lambda t: rearrange(t, 'b b2 n (h d) -> b b2 h n d', h = self.heads), (ek, ev))
            
        # add rotary embedding to ek if we have temporal attention and per-frame-conditioning since we want to encode the temporal information in the conditioning
        if exists(self.rotary_emb) and if_temperal:
            ek = self.rotary_emb.rotate_queries_or_keys(t=ek, seq_list = rotary_emb_list)

        k = torch.cat([ek, k], dim=-2)
        v = torch.cat([ev, v], dim=-2)

        # scale
        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb) and if_temperal:
            q = self.rotary_emb.rotate_queries_or_keys(t=q, seq_list = rotary_emb_list)

        # similarity
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b n t d -> (b n) t d')
        return out

class VDTBlock(nn.Module):
    """
    A VDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, mode='video', num_frames=19, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        rotary_emb = RotaryEmbedding(hidden_size//num_heads)
        self.attn = Attention(dim=hidden_size, heads=num_heads, dim_head=hidden_size//num_heads, cond_dim = hidden_size, rotary_emb=rotary_emb)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.num_frames = num_frames
        
        self.mode = mode
        
        ## Temporal Attention Parameters
        if self.mode == 'video':
            
            self.temporal_norm1 = nn.LayerNorm(hidden_size)
            self.temporal_attn = Attention(dim=hidden_size, heads=num_heads, dim_head=hidden_size//num_heads, cond_dim = hidden_size, rotary_emb=rotary_emb)
            self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c4t, c, f):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c4t).chunk(6, dim=1)
        T = self.num_frames
        K, N, M = x.shape
        B = K // T
        if self.mode == 'video':
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_attn(self.temporal_norm1(x), label_emb_mm = c, rotary_emb_list=f, if_temperal = True, b=B)
            res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            res_temporal = self.temporal_fc(res_temporal)
            x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T,n=N,m=M)
            x = x + res_temporal

        attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa, self.num_frames), label_emb_mm = c, rotary_emb_list=f, if_temperal = False, b=B)
        attn = rearrange(attn, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        attn = gate_msa.unsqueeze(1) * attn
        attn = rearrange(attn, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + attn

        mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, self.num_frames))
        mlp = rearrange(mlp, '(b t) n m-> b (t n) m',b=B,t=T,n=N,m=M)
        mlp = gate_mlp.unsqueeze(1) * mlp
        mlp = rearrange(mlp, 'b (t n) m-> (b t) n m',b=B,t=T,n=N,m=M)
        x = x + mlp

        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class FinalLayer(nn.Module):
    """
    The final layer of VDT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, num_frames):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames)
        x = self.linear(x)
        return x


class VDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=6,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mode='video',
        num_frames=10,
        num_freqs=19
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.s_embedder = SParaEmbedder(num_frames, hidden_size, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.mode = mode
        if self.mode == 'video':
            self.num_frames = num_frames
            self.num_freqs = num_freqs
            self.time_embed = nn.Parameter(torch.zeros(1, num_freqs, hidden_size), requires_grad=False)
            self.time_drop = nn.Dropout(p=0)
        else:
            self.num_frames = 1

        self.blocks = nn.ModuleList([
            VDTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, mode=mode, num_frames=self.num_frames) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.num_frames)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.mode == 'video':
            grid_num_frames = np.arange(self.num_freqs, dtype=np.float32)
            time_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid_num_frames)
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.s_embedder.sign_emb.weight, std=0.02)
        # nn.init.normal_(self.s_embedder, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in VDT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, f):
        """
        Forward pass of VDT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        B, T, C, W, H = x.shape # b 10 4 32 32 
        x = x.contiguous().view(-1, C, W, H)
        # y = torch.zeros(B).long().to(x.device)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if self.mode == 'video':
            # Temporal embed
            x = rearrange(x, '(b t) n m -> n b t m',b=B,t=T)
            num_f = f.shape[1]
            temp_time_embed = self.time_embed[:,0:num_f,:]
            ## Resizing time embeddings in case they don't match
            for bb in torch.arange(0,B):
                bb_f = f[bb,:]
                bb_time_embed = self.time_embed[:,bb_f,:]
                temp_time_embed = torch.cat([temp_time_embed,bb_time_embed],dim=0)
            temp_time_embed = temp_time_embed[1:B+1,:,:]
            temp_time_embed = temp_time_embed.unsqueeze(0)#1*B*num_freqs*hidden_dims
            x = x + temp_time_embed
            x = self.time_drop(x)
            x = rearrange(x, 'n b t m -> (b t) n m',b=B,t=T)
        
        t = self.t_embedder(t)                   # (N, D)
        y, y4t = self.s_embedder(y, self.training)    # (N, D)
  
        c4t = t + y4t                             # (N, D)
        c=y                                        # (N,f,D)

        for block in self.blocks:
            x = block(x, c4t, c, f)                      # (N, T, D)
        x = self.final_layer(x, c4t)                # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        x = x.view(B, T, x.shape[-3], x.shape[-2], x.shape[-1])
        return x

    def forward_with_cfg(self, x, t, y, f, cfg_scale):
        """
        Forward pass of VDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, f)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :, :self.in_channels], model_out[:, :, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=2)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   VDT Configs                                  #
#################################################################################

def VDT_L_2(**kwargs):
    return VDT(depth=28, hidden_size=1152, num_frames=19, num_freqs=19, patch_size=2, num_heads=16, **kwargs)

def VDT_S_2(**kwargs):
    return VDT(depth=12, hidden_size=256, num_frames=19, num_freqs=19, patch_size=2, num_heads=8, **kwargs)

def VDT_M_2(**kwargs):
    return VDT(depth=18, hidden_size=768, num_freqs=19, patch_size=2, num_heads=12, **kwargs)

def VDT_MS_2(**kwargs):
    return VDT(depth=16, hidden_size=384, num_frames=19, num_freqs=19, patch_size=2, num_heads=12, **kwargs)

VDT_models = {
    'VDT-L/2':  VDT_L_2,
    'VDT-S/2':  VDT_S_2,   
    'VDT-M/2':  VDT_M_2,  
    'VDT-MS/2':  VDT_MS_2,
}
