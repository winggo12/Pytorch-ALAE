import torch
import os
import math
from utils.progress_bar import progress_bar
from torchvision import transforms
import cv2
from dnn.models.ALAE import StyleALAE
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

def generate_style_mixing_img(model, config, alpha=0.4):
    z_main = torch.randn(1, config['z_dim'], dtype=torch.float32).to(device)
    z_copy = torch.randn(1, config['z_dim'], dtype=torch.float32).to(device)
    with torch.no_grad():
        main_images = model.generate(z_main, final_resolution_idx=model.res_idx, alpha=alpha)
        main_images = main_images * 0.5 + 0.5

        copy_images = model.generate(z_copy, final_resolution_idx=model.res_idx, alpha=alpha)
        copy_images = copy_images * 0.5 + 0.5

        combined_images = model.generate_style_mixing(z_main, z_copy, copystylefrom=[4,8], final_resolution_idx=model.res_idx, alpha=alpha)
        combined_images = combined_images * 0.5 + 0.5

        main_image = single_tensor_to_img(main_images[0], 128, 128)
        copy_image = single_tensor_to_img(copy_images[0], 128, 128)
        combined_image = single_tensor_to_img(combined_images[0], 128, 128)

        cv2.imshow("Main Image", main_image)
        cv2.imshow("Copy Image", copy_image)
        cv2.imshow("Combined Image", combined_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("Finished")

def generate_img_with_truncation(model, config, alpha=0.4):
    batch_size = 1
    test_samples_z = torch.randn(batch_size, config['z_dim'], dtype=torch.float32).to(device)
    style = 3
    while style >= 0.1:
        with torch.no_grad():
            generated_images = model.generate_with_truncation(test_samples_z, style=style, final_resolution_idx=model.res_idx, alpha=alpha)
            generated_images = generated_images * 0.5 + 0.5
            # generated_images = inverse_normalize(tensor=generated_images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        for cvimg in generated_images.cpu().numpy():
            cvimg = cvimg.transpose(1,2,0)
            cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
            cvimg = cv2.resize( cvimg, dsize=(256,256) )
            cv2.imshow("Img", cvimg)
            cv2.waitKey(20)
            style = style - 0.01

    cv2.destroyAllWindows()
    print("Finished")

def reconstruct_img(model, config, path, saved_path, show, alpha=0.4):
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
            restored_images = model.decode(model.encode(img_tensor, final_resolution_idx=model.res_idx, alpha=alpha),
                                          final_resolution_idx=model.res_idx, alpha=alpha)
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
    path = "./data/FFHQ-thumbnails/thumbnails128x128"
    reconstruction_saved_path = "./data/FFHQ-thumbnails/reconstructed_thumbnails128x128"
    generation_saved_path = "./data/FFHQ-thumbnails/generated_thumbnails128x128"
    model = StyleALAE(model_config=config, device=device)
    model.load_train_state('./archived/FFHQ/StyleALAE-z-256_w-256_prog-(4,256)-(8,256)-(16,128)-(32,128)-(64,64)-(64,32)/checkpoints/ckpt_gs-120000_res-5=64x64_alpha-0.40.pt')
    # model.load_train_state('./archived/FFHQ/StyleALAE-z-256_w-256_prog-(4,256)-(8,256)-(16,128)-(32,128)-(64,64)-(64,32)/checkpoints/ckpt_gs-120000_res-5=64x64_alpha-0.40.pt')
    batch_size = 32

    generate_img(model, config, generation_saved_path, False, 70000)
    # generate_img_with_truncation(model, config)
    # reconstruct_img(model, config, path, reconstruction_saved_path, False)
    # generate_style_mixing_img(model, config)


