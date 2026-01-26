import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
import numpy as np
from .model_base import Fast_rFFT2d_GPU_batch, Measurement_Render

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type:(Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn      #HS_MSA or FFN
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class SS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),   #(8,8)
            dim_head=28,
            heads=8,              #heads=dim_scale//dim  1,2,4
            only_local_branch=False
    ):
        super().__init__()

        self.dim = dim                        
        self.heads = heads                    # 1,2,4
        self.scale = dim_head ** -0.5         # 1/sqrt(28) = 0.189
        self.window_size = window_size        # (8,8)
        self.only_local_branch = only_local_branch

        # position embedding
        if only_local_branch:
            seq_l = window_size[0] * window_size[1]       
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l*4, seq_l*4))   
            trunc_normal_(self.pos_emb)   
        else:
            seq_l1 = window_size[0] * window_size[1]       
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l1, seq_l1))  
            h, w = 256 // self.heads, 256 // self.heads   
            seq_l2 = h*w//seq_l1                          
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l2, seq_l2)) 
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads  
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)


        # learning_shift_steps
        if self.dim==28:
            in_dim=28
        elif self.dim==56:
            in_dim=28
        elif self.dim==112:
            in_dim=56
        else:
            in_dim=28

        hidden_dim = window_size[0] * window_size[1]
        self.conv1_local = nn.Conv2d(in_dim, hidden_dim//4, kernel_size=3, padding=1)
        self.conv2_local = nn.Conv2d(hidden_dim//4, hidden_dim//4, kernel_size=3, padding=1)
        self.pool_local = nn.AdaptiveAvgPool2d((1,1))
        self.fc1_local = nn.Linear(hidden_dim//4, 2)
        self.fc2_local = nn.Linear(hidden_dim // 4, 1)

    def learning_shift_steps(self, q, k, v):
        '''
        Args:
            q: [bs, 256, 256, 28]
            k: [bs, 256, 256, 28]
            v: [bs, 256, 256, 28]
        Returns: [bs, 2]
        '''
        q, k, v = q.permute(0, 3, 1, 2), k.permute(0, 3, 1, 2), v.permute(0, 3, 1, 2)
        [bs, nC, H, W] = q.shape
        x = q + k + v
        x = F.relu(self.conv1_local(x))  
        x = F.relu(self.conv2_local(x))
        x = self.pool_local(x)
        x = x.view(x.size(0), -1)  # [bs, hidden_dim]
        shift_para = self.fc1_local(x)
        shift_range = self.fc2_local(x)
        shift_steps = shift_range * torch.tanh(shift_para)
        return shift_steps

    def conduct_float_shift_grid(self, x, shift_steps):
        '''
        Args:
            x: [bs, 256, 1, 256, 256]
            shift_steps: [bs, 2]
        Returns: [bs, 256, 1, 256, 256]
        '''

        bs, channels, _, H, W = x.shape  # 获取形状
        device = x.device
        dtype = x.dtype

        x_fix_shifted = torch.roll(x, shifts=(1, 1), dims=(3, 4))

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype)
        )
        base_grid = torch.stack((grid_x, grid_y), 2)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).repeat(bs, 1, 1, 1)  # [bs, H, W, 2]

        shift_norm = torch.zeros_like(base_grid)  # [bs, H, W, 2]
        shift_norm[:, :, :, 0] = shift_steps[:, 0].view(bs, 1, 1) * 2 / W  # x
        shift_norm[:, :, :, 1] = shift_steps[:, 1].view(bs, 1, 1) * 2 / H  # y

        shifted_grid = base_grid + shift_norm  # [bs, H, W, 2]

        x_reshaped = x.view(bs * channels, _, H, W)  # [bs*256, 1, 256, 256]

        shifted_grid = shifted_grid.view(bs, 1, H, W, 2).repeat(1, channels, 1, 1, 1)
        shifted_grid = shifted_grid.view(bs * channels, H, W, 2)  # [bs*256, 256, 256, 2]

        shifted_x = F.grid_sample(
            x_reshaped,
            shifted_grid,
            mode='bilinear',
            padding_mode='reflection',
            # padding_mode='zeros',
            # padding_mode='border',
            align_corners=True
        )  

        shifted_x = shifted_x.view(bs, channels, _, H, W)  # [bs, 256, 1, 256, 256]

        shifted_x = x_fix_shifted + shifted_x
        return shifted_x

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape           
        w_size = self.window_size     
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'

        if self.only_local_branch:
            #x: 5 256 256 28
            q = self.to_q(x)                           
            k, v = self.to_kv(x).chunk(2, dim = -1)    
            q, k, v = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0]*2, b1=w_size[1]*2), (q, k, v))  
            shift_steps = self.learning_shift_steps(q, k, v)
            q, k, v = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=1),
                             (q, k, v))  
            q *= self.scale
            sim = einsum('b n h i d, b n h j d -> b n h i j', q, k)  
            sim = sim + self.pos_emb
            sim = self.conduct_float_shift_grid(sim, shift_steps)


            attn = sim.softmax(dim=-1)  
            out = einsum('b n h i j, b n h j d -> b n h i d', attn, v)  
            out = rearrange(out, 'b n h mm d -> b n mm (h d)')  

        else:    
            q = self.to_q(x)    
            k, v = self.to_kv(x).chunk(2, dim=-1)  
            q1, q2 = q[:,:,:,:c//2], q[:,:,:,c//2:]
            k1, k2 = k[:,:,:,:c//2], k[:,:,:,c//2:]
            v1, v2 = v[:,:,:,:c//2], v[:,:,:,c//2:]  

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q1, k1, v1))    
            shift_steps1 = self.learning_shift_steps(q1, k1, v1)
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q1, k1, v1))  
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1) 
            sim1 = sim1 + self.pos_emb1                                      
            sim1 = self.conduct_float_shift_grid(sim1, shift_steps1)                 
            attn1 = sim1.softmax(dim=-1) 
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)       
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')               

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))                              
            shift_steps2 = self.learning_shift_steps(q2, k2, v2)
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))                      
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q2, k2, v2)) 
            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            sim2 = sim2 + self.pos_emb2    
            sim2 = self.conduct_float_shift_grid(sim2, shift_steps2)
            attn2 = sim2.softmax(dim=-1)   
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)  
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')           
            out2 = out2.permute(0, 2, 1, 3)                                

            out = torch.cat([out1,out2],dim=-1).contiguous() 
            out = self.to_out(out)                           
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])                     
        return out

class SSAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,        
            num_blocks=2,    
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, SS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, only_local_branch=(heads==1))),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)    #[b,h,w,c]
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class SST(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, num_blocks=[1,1,1]):
        super(SST, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim    # 28
        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                SSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
            ]))
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = SSAB(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                SSAB(dim=dim_scale // 2, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim,
                     heads=(dim_scale // 2) // dim),
            ]))
            dim_scale //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb   # 0
        pad_w = (wb - w_inp % wb) % wb   # 0
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')   

        # Embedding
        fea = self.embedding(x)   
        x = x[:,:28,:,:]          

        # Encoder
        fea_encoder = []
        for (SSAB, FeaDownSample) in self.encoder_layers:
            fea = SSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, SSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales-2-i]], dim=1))
            fea = SSAB(fea)

        # Mapping
        out = self.mapping(fea) + x
        return out[:, :, :h_inp, :w_inp]


class HyPaNet(nn.Module):
    def __init__(self, in_nc, out_nc, channel=256):
        super(HyPaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):
        x = self.down_sample(self.relu(self.fution(x)))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        return x[:, :self.out_nc // 3, :, :], x[:, self.out_nc // 3:2 * self.out_nc // 3, :, :], x[:,2 * self.out_nc // 3:,:, :]


def Phi_t(y, ff_batch, psf_batch):
    bs, C, H, W = ff_batch.shape
    y = y / C * 5.0
    y_expanded = y.repeat(1, C, 1, 1)
    filtered_HSI = y_expanded * ff_batch
    psf_batch_t = torch.flip(psf_batch, dims=(2,3))
    Phi_t_y = Fast_rFFT2d_GPU_batch(filtered_HSI, psf_batch_t)
    return Phi_t_y
'''
2024-12-5 Krito
需要排查是不是Phi_t的问题，需要改写
'''
def Phi_t_new(y, ff_batch, psf_batch):
    bs, C, H, W = ff_batch.shape
    # y = y / C * 5.0
    y_expanded = y.repeat(1, C, 1, 1)
    filtered_HSI = y_expanded * ff_batch
    psf_batch_t = torch.flip(psf_batch, dims=(2,3))
    Phi_t_y = Fast_rFFT2d_GPU_batch(filtered_HSI, psf_batch_t)
    return Phi_t_y


class ODAUVST(nn.Module):
    def __init__(self, num_iterations=1):   #num_iterations = 5
        super(ODAUVST, self).__init__()
        self.Render = Measurement_Render()
        self.para_estimator = HyPaNet(in_nc=28, out_nc=num_iterations*3)
        self.ff_down = nn.Conv2d(28, 3, 1, stride=1, padding=0)
        # 512
        self.psf_down_512 = nn.Conv2d(28, 3, 2, stride=2, padding=0)
        # 1024
        self.psf_down_1024 = nn.Conv2d(28, 3, 4, stride=4, padding=0)
        self.fution = nn.Conv2d(7, 28, 1, padding=0, bias=True)
        self.num_iterations = num_iterations
        self.denoisers = nn.ModuleList([])
        for _ in range(num_iterations):
            self.denoisers.append(
                SST(in_dim=29, out_dim=28, dim=28, num_blocks=[1, 1, 1]),
            )

    def initial(self, y, ff_batch, psf_batch):
        ff_feature = self.ff_down(ff_batch)
        if psf_batch.shape[3] == 512:
            psf_feature = self.psf_down_512(psf_batch)
        elif psf_batch.shape[3] == 1024:
            psf_feature = self.psf_down_1024(psf_batch)
        else:
            raise AttributeError('ukonwn psf shape')
        z = self.fution(torch.cat([y, ff_feature, psf_feature], dim=1))                               
        alphas, betas, deltas = self.para_estimator(self.fution(torch.cat([y, ff_feature, psf_feature], dim=1)))       
        return z, alphas, betas, deltas, ff_feature, psf_feature

    def forward(self, input_cube, ff_batch, psf_batch, y=None):
        """
        :param input_cube: [bs, 28, 512, 512]
        :param ff_batch:   [bs, 28, 256, 256]
        :param psf_batch:  [bs, 28, 512, 512]
        :return: z:        [bs, 28, 256, 256]
        """
        if y == None:    
            y = self.Render(input_cube=input_cube, ff_batch=ff_batch, psf_batch=psf_batch)
        z, alphas, betas, deltas, ff_feature, psf_feature = self.initial(y, ff_batch, psf_batch)
        z, alphas, betas, deltas = z.contiguous(), alphas.contiguous(), betas.contiguous(), deltas.contiguous()
        x = z
        Phi_t_y = Phi_t_new(y, ff_batch, psf_batch)
        for i in range(self.num_iterations):
            alpha, beta, delta = alphas[:,i:i+1,:,:], betas[:,i:i+1,:,:], deltas[:,i:i+1,:,:]
            # x = torch.mul((1 - delta - alpha * delta), x) + torch.mul(alpha * delta, v) + torch.mul(delta, y)
            x = torch.mul((1 - delta - alpha), z) + torch.mul(delta, Phi_t_y) + torch.mul(alpha, x)
            # denoise
            beta_repeat = beta.repeat(1, 1, x.shape[2], x.shape[3])
            z = self.denoisers[i](torch.cat([x, beta_repeat], dim=1))
        recon_pred = z
        return recon_pred[:, :, :y.shape[2], :y.shape[3]]

