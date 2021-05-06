import copy
import cv2
import math
import numpy as np
import random
import torch
from dnn.models.StyleGan import StyleGan
from dnn.models.ALAE import StyleALAE
from utils.progress_bar import progress_bar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {
    "z_dim": 256,
    "w_dim": 256,
    "image_dim": 64,
    "mapping_layers": 8,
    "resolutions": [4, 8, 16, 32, 64, 64],
    "channels": [256, 256, 128, 128, 64, 32, 16],
    "learning_rates": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "phase_lengths": [600_000, 600_000, 600_000, 600_000, 600_000, 1000_000],
    "batch_sizes": [128, 128, 128, 64, 32, 32],
    "n_critic": 1,
    "dump_imgs_freq": 5000,
    "checkpoint_freq": 10000
}

def slerp(z1, z2, t):
    z1_norm = z1.norm(dim=-1, keepdim=True)
    z2_norm = z2.norm(dim=-1, keepdim=True)
    cosa = (z1*z2)/(z1_norm*z2_norm)
    a = torch.acos(cosa)
    w1 = torch.sin((1-t)*a)/torch.sin(a)
    w2 = torch.sin(t*a) / torch.sin(a)
    sl = w1*z1 + w2+z2
    return sl

class PPLNet(torch.nn.Module):
    def __init__(self, G, F, config, batch_size, res_idx, alpha, epsilon, space, sampling, vgg16):
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super().__init__()
        self.G = copy.deepcopy(G)
        self.F = copy.deepcopy(F)
        self.config = config
        self.batch_size = batch_size
        self.res_idx = res_idx
        self.alpha = alpha
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.vgg16 = copy.deepcopy(vgg16)

    def forward(self):
        with torch.no_grad():
            # Generate random latents and interpolation t-values.
            t = torch.rand([self.batch_size], device=device) * (1 if self.sampling == 'full' else 0)
            z0, z1 = torch.randn([self.batch_size * 2, 256], device=device).chunk(2)
            # Interpolate in W or Z.
            if self.space == 'w':
                w0 = self.F(z0)
                w1 = self.F(z1)
                wt0 = torch.lerp(w0, w1, t.unsqueeze(1))
                wt1 = torch.lerp(w0, w1, t.unsqueeze(1) + self.epsilon)
            elif self.space == 'z':
                zt0 = slerp(z0, z1, t.unsqueeze(1))
                zt1 = slerp(z0, z1, t.unsqueeze(1) + self.epsilon)
                wt0, wt1 = self.F(torch.cat([zt0, zt1])).chunk(2)
            else:
                print("Please enter a correct space")
                return

            # Generate images.
            img = self.G(torch.cat([wt0,wt1]), self.res_idx, self.alpha)
            img = torch.nn.functional.interpolate(img, size=self.config['resolutions'][-1])

            # Downsample to 256x256.
            generator_img_resolution = 64
            factor = generator_img_resolution // 256
            if factor > 1:
                img = img.reshape([-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]).mean([3, 5])

            # Scale dynamic range from [-1,1] to [0,255].
            img = img*0.5 + 0.5

            ##For Debugging the Generated Image
            # cvimg = img[0].cpu().numpy()
            # cvimg = cvimg.transpose(1, 2, 0)
            # cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
            # cvimg = cv2.resize(cvimg, dsize=(128, 128))
            # cv2.imshow("Generated Image", cvimg)
            # cv2.waitKey(1000)

            img = img*255
            generator_img_channels = 3
            if generator_img_channels == 1:
                img = img.repeat([1, 3, 1, 1])

            # Evaluate differential LPIPS.
            lpips_t0, lpips_t1 = self.vgg16(img, resize_images=True, return_lpips=True).chunk(2)
            dist = (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon ** 2
        return dist

if __name__ == '__main__':
    batch_size = 16
    test_size = 100000
    total_iterations = math.ceil(test_size/batch_size)
    vgg = torch.jit.load("./score_calculator/vgg16.pt").eval().to(device)
    model = StyleALAE(model_config=config, device=device)
    model.load_train_state('./archived/FFHQ/StyleALAE-z-256_w-256_prog-(4,256)-(8,256)-(16,128)-(32,128)-(64,64)-(64,32)/checkpoints/ckpt_gs-120000_res-5=64x64_alpha-0.40.pt')
    model.G.eval()
    model.F.eval()
    for space in ['w', 'z']:
        for path_len in ['full','end']:
            sampler = PPLNet(G=model.G, F=model.F, config=config, batch_size=batch_size, res_idx=model.res_idx, alpha=0.4, epsilon=0.0001, space=space, sampling=path_len, vgg16=vgg)
            sampler.eval().requires_grad_(False).to(device)
            dists = []
            print("Calculating on ",space," latent space , path length is ", path_len )
            for itr in range(total_iterations):
                dist = sampler.forward()
                dists.append(dist)
                progress_bar(itr, total_iterations)
            print("Finished")
            dists = torch.cat(dists)[:test_size].cpu().numpy()
            lo = np.percentile(dists, 1, interpolation='lower')
            hi = np.percentile(dists, 99, interpolation='higher')
            ppl = np.extract(np.logical_and(dists >= lo, dists <= hi), dists).mean()
            print("The PPL is ", ppl)

