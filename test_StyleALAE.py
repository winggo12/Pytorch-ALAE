import torch
import os
import numpy as np
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import cv2
from dnn.models.ALAE import StyleALAE
from datasets import get_dataset, get_dataloader
from utils.common_utils import find_latest_checkpoint
import glob

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

def generate_img(model, config, batch_size=32):
    test_samples_z = torch.randn(batch_size, config['z_dim'], dtype=torch.float32).to(device)
    with torch.no_grad():
        generated_images = model.generate(test_samples_z,final_resolution_idx=model.res_idx, alpha=0)
        generated_images = generated_images * 0.5 + 0.5
        # generated_images = inverse_normalize(tensor=generated_images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for cvimg in generated_images.cpu().numpy():
        cvimg = cvimg.transpose(1,2,0)
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
        cvimg = cv2.resize( cvimg, dsize=(256,256) )
        cv2.imshow("Img", cvimg)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("Finished")

def reconstruct_img(model, config, path, saved_path, show):
    test_samples_z = torch.randn(batch_size, config['z_dim'], dtype=torch.float32).to(device)
    image_paths = []
    rename = []
    image_extensions = ["png", "jpg"]
    for ext in image_extensions:
        print("Looking for images in", os.path.join(path, "*.{}".format(ext)))
        for impath in glob.glob(os.path.join(path, "*.{}".format(ext))):
            image_paths.append(impath)
            rename.append(impath[-9:-4])

    with torch.no_grad():
        for i in range(len(image_paths)):
            tran = transforms.ToTensor()
            img = cv2.imread(image_paths[i])
            cv2.imshow("orig", img)
            img = cv2.resize(img, dsize=(64,64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/127.5 - 1
            img = img.transpose(2, 0, 1)
            img_tensor = torch.tensor(img, dtype=torch.float32).to(device).unsqueeze(0)
            # img_tensor = tran(img).to(device).unsqueeze(0)
            restored_images = model.decode(model.encode(img_tensor, final_resolution_idx=model.res_idx, alpha=0.08),
                                          final_resolution_idx=model.res_idx, alpha=0.4)
            restored_images = restored_images * 0.5 + 0.5
            restored_image = restored_images[0].cpu().numpy().transpose(1,2,0)
            restored_image = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)
            restored_image = cv2.resize( restored_image, dsize=(128,128) )

            image_name = rename[i] + ".jpg"
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            file_name = os.path.join(saved_path, image_name)
            if show == True:
                cv2.imshow("restored_image", restored_image)
                cv2.waitKey(0)
            restored_image = restored_image * 255
            cv2.imwrite(file_name, restored_image)

    if show==True:
        cv2.destroyAllWindows()
    print("Finished Reconstruction")

if __name__ == '__main__':
    # dataset_name = "FFHQ"
    # train_dataset, test_dataset = get_dataset("data", dataset_name, dim=config['resolutions'][-1])
    path = "./data/FFHQ-thumbnails/thumbnails128x128"
    saved_path = "./data/FFHQ-thumbnails/reconstructed_thumbnails128x128"
    model = StyleALAE(model_config=config, device=device)
    model.load_train_state('./archived/FFHQ/StyleALAE-z-256_w-256_prog-(4,256)-(8,256)-(16,128)-(32,128)-(64,64)-(64,32)/checkpoints/ckpt_gs-120000_res-5=64x64_alpha-0.40.pt')

    batch_size = 32
    batchs_in_phase = config['phase_lengths'][model.res_idx] // batch_size
    alpha = 64/(config['phase_lengths'][model.res_idx])
    # generate_img(model, config, batch_size)
    reconstruct_img(model, config, path, saved_path, False)


