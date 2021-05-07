import torch
import os
import math
from utils.progress_bar import progress_bar
import numpy as np
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import cv2
from dnn.models.StyleGan import StyleGan
from datasets import get_dataset, get_dataloader
from utils.common_utils import find_latest_checkpoint
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {
    'z_dim': 512,
    'w_dim': 512,
    'image_dim':64,
    'lr': 0.002,
    'g_penalty_coeff':10.0,
    'mapping_layers':8,
    "resolutions": [4, 8, 16, 32, 64],
    "channels": [256, 256, 128, 64, 32],
    "learning_rates": [0.001, 0.001, 0.001, 0.001, 0.001],
    "phase_lengths": [400_000, 600_000, 800_000, 1_000_000, 2_000_000],
    "batch_sizes": [128, 128, 128, 128, 128],
    "n_critic": 1,
    "dump_imgs_freq": 5000
}
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def single_tensor_to_img(tensor, w_size, h_size):
    cvimg = tensor.cpu().numpy()
    cvimg = cvimg.transpose(1, 2, 0)
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
    cvimg = cv2.resize(cvimg, dsize=(w_size, h_size))
    return cvimg

def generate_img(model, config, saved_path, show, generation_size, alpha=0.4, batch_size=32):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    cnt = 0
    for i in range(math.ceil(generation_size/batch_size)):
        test_samples_z = torch.randn(batch_size, config['z_dim'], dtype=torch.float32).to(device)
        with torch.no_grad():
            generated_images = model.generate(test_samples_z, final_resolution_idx=model.res_idx, alpha=alpha)
            generated_images = generated_images * 0.5 + 0.5
            # generated_images = inverse_normalize(tensor=generated_images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        for tensor in generated_images:
            cvimg = single_tensor_to_img(tensor, 128, 128)
            image_name = str(cnt) + ".jpg"
            cnt += 1
            file_name = os.path.join(saved_path, image_name)
            if show == True:
                cv2.imshow("restored_image", cvimg)
                cv2.waitKey(0)
            cvimg = cvimg * 255
            cv2.imwrite(file_name, cvimg)
        progress_bar(i, math.ceil(generation_size/batch_size))
    cv2.destroyAllWindows()
    print("Finished")

if __name__ == '__main__':
    generation_saved_path = "./data/FFHQ-thumbnails/stylegan_generated_thumbnails128x128"
    model = StyleGan(model_config=config, device=device)
    model.load_train_state('')
