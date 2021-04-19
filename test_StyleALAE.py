import torch
import os
import torch.nn as nn
from tqdm import tqdm
import cv2
from dnn.models.ALAE import StyleALAE
from utils.common_utils import find_latest_checkpoint

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
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

if __name__ == '__main__':
    model = StyleALAE(model_config=config, device=device)
    model.load_train_state('./Training_dir-test/StyleALAE-z-256_w-256_prog-(4,256)-(8,256)-(16,128)-(32,128)-(64,64)-(64,32)/checkpoints/ckpt_gs-120000_res-5=64x64_alpha-0.40.pt')

    batch_size = 32
    batchs_in_phase = config['phase_lengths'][model.res_idx] // batch_size
    alpha = 64/(config['phase_lengths'][model.res_idx])
    test_samples_z = torch.randn(batch_size, config['z_dim'], dtype=torch.float32).to(device)
    with torch.no_grad():
        generated_images = model.generate(test_samples_z,final_resolution_idx=model.res_idx, alpha=0)
        generated_images = generated_images * 0.5 + 0.5
        generated_images = inverse_normalize(tensor=generated_images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


    for cvimg in generated_images.cpu().numpy():
        cvimg = cvimg.transpose(1,2,0)
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
        cv2.imshow("Img", cvimg)
        cv2.waitKey(0)

    print("Finished")

