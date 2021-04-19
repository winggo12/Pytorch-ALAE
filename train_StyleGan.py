import torch
from datasets import get_dataset
from dnn.models.StyleGan import StyleGan
from utils.common_utils import get_config_str
import argparse
import os
from pprint import pprint

parser = argparse.ArgumentParser(description='Train arguments')
parser.add_argument("--output_root", type=str, default="Training_dir-test")
parser.add_argument("--dataset_name", type=str, default="LFW", help='FFHQ/CelebA/LFW')
parser.add_argument("--num_debug_images", type=int, default=36)
parser.add_argument("--print_model", action='store_true', default=False)
parser.add_argument("--print_config", action='store_true', default=False)
parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0/cpu")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

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

if __name__ == '__main__':
    # Create datasets

    train_dataset, test_dataset = get_dataset("data", args.dataset_name, config['image_dim'])

    # Create model
    model = StyleGan(model_config=config, device=device)

    if args.print_config:
        print("Model config:")
        pprint(config)

    if args.print_model:
        print(model)

    test_samples_z = torch.randn(args.num_debug_images, config['z_dim'], dtype=torch.float32).to(device)

    config_descriptor = get_config_str(config)
    output_dir = os.path.join(args.output_root, f"StyleGan-{config_descriptor}")
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    model.train(train_dataset, test_samples_z, output_dir)