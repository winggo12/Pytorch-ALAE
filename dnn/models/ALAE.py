import os
from tqdm import tqdm
from torchvision.utils import save_image
from dnn.sub_modules.AlaeModules import *
from dnn.sub_modules.StyleGanGenerator import StylleGanGenerator, MappingFromLatent
from utils.tracker import LossTracker
from dnn.costume_layers import compute_r1_gradient_penalty
from datasets import get_dataloader, EndlessDataloader

COMMON_DEFAULT = {"g_penalty_coeff": 10,
                  'descriminator_layers': 3,
                  'mapping_lr_factor': 0.01, # StyleGan paper: "... reduce the learning rate by two orders of magnitude for the mapping network..."
                  'discriminator_lr_factor':0.1 # Found in The ALAE official implemenation.
                  }


class ALAE:
    """
    A generic (ASTRACT) Implementation of https://arxiv.org/abs/2004.04467
    """
    def __init__(self, model_config, architecture_mode, device):
        self.device = device
        self.cfg = COMMON_DEFAULT
        self.cfg.update(model_config)

        if architecture_mode == "MLP":

            self.G = GeneratorMLP(latent_dim=self.cfg['w_dim'], output_img_dim=self.cfg['image_dim']).to(device).train()
            self.E = EncoderMLP(input_img_dim=self.cfg['image_dim'], latent_dim=self.cfg['w_dim']).to(device).train()
        else:
            progression = list(zip(self.cfg['resolutions'], self.cfg['channels']))
            self.G = StylleGanGenerator(latent_dim=self.cfg['w_dim'], progression=progression).to(device).train()
            self.E = AlaeEncoder(latent_dim=self.cfg['w_dim'], progression=progression).to(device).train()

        self.F = MappingFromLatent(input_dim=self.cfg['z_dim'], out_dim=self.cfg['w_dim'], num_layers=self.cfg['mapping_layers']).to(device).train()
        self.D = DiscriminatorMLP(input_dim=self.cfg['w_dim'], num_layers=self.cfg['descriminator_layers']).to(device).train()

        self.ED_optimizer = torch.optim.Adam([{'params': self.D.parameters(), 'lr_mult': self.cfg['discriminator_lr_factor']},
                                      {'params': self.E.parameters(),}],
                                      betas=(0.0, 0.99), weight_decay=0)
        self.FG_optimizer = torch.optim.Adam([{'params': self.F.parameters(), 'lr_mult': self.cfg['mapping_lr_factor']},
                                      {'params': self.G.parameters()}],
                                      betas=(0.0, 0.99), weight_decay=0)

    def __str__(self):
        return f"F\n{self.F}\nG\n{self.G}\nE\n{self.E}\nD\n{self.D}\n"

    def set_optimizers_lr(self, new_lr):
        """
        resets the learning rate of the optimizers.
        lr_mult allows rescaling specifoic param groups.
        The StyleGan paper describes the lr scale of theMapping layers:
        "We thus reduce the learning rate by two orders of magnitude for the mapping network, i.e., λ = 0.01 ·λ"
        The decrease of the Discriminator D is just a parameter found in the official implementation.
        """
        for optimizer in [self.ED_optimizer, self.FG_optimizer]:
            for group in optimizer.param_groups:
                mult = group.get('lr_mult', 1)
                group['lr'] = new_lr * mult

    def get_ED_loss(self, batch_real_data, **ae_kwargs):
        """
        Computes a standard adverserial loss for the dictriminator D(E( * )):
          how much  D(E( * )) can differentiate between real images and images generated by G(F( * ))
         """
        batch_z = torch.randn(batch_real_data.shape[0], self.cfg['z_dim'], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            batch_fake_data = self.G(self.F(batch_z), **ae_kwargs)
        fake_images_dicriminator_outputs = self.D(self.E(batch_fake_data, **ae_kwargs))
        real_images_dicriminator_outputs = self.D(self.E(batch_real_data, **ae_kwargs))
        loss = F.softplus(fake_images_dicriminator_outputs) + F.softplus(-real_images_dicriminator_outputs)

        r1_penalty = compute_r1_gradient_penalty(real_images_dicriminator_outputs, batch_real_data)

        loss += self.cfg['g_penalty_coeff'] * r1_penalty
        loss = loss.mean()

        return loss

    def get_FG_loss(self, batch_real_data, **ae_kwargs):
        """
        Computes a standard adverserial loss for the generator:
            how much  G(F( * )) can fool D(E ( * ))
        """
        batch_z = torch.randn(batch_real_data.shape[0], self.cfg['z_dim'], dtype=torch.float32).to(self.device)
        batch_fake_data = self.G(self.F(batch_z), **ae_kwargs)
        fake_images_dicriminator_outputs = self.D(self.E(batch_fake_data, **ae_kwargs))
        loss = F.softplus(-fake_images_dicriminator_outputs).mean()

        return loss

    def get_EG_loss(self, batch_real_data, **ae_kwargs):
        """
        Compute a reconstruction loss in the w latent space for the auto encoder (E,G):
            || F(X) - E(G(F(x))) || = || W - E(G(W)) ||
        """
        batch_z = torch.randn(batch_real_data.shape[0], self.cfg['w_dim'], dtype=torch.float32).to(self.device)
        batch_w = self.F(batch_z)
        batch_reconstructed_w = self.E(self.G(batch_w, **ae_kwargs), **ae_kwargs)
        return torch.mean(((batch_reconstructed_w - batch_w.detach())**2))

    def perform_train_step(self, batch_real_data, tracker, **ae_kwargs):
        """
        Optimizes the model with a batch of real images:
             optimize :Disctriminator, Generator and reconstruction loss of the autoencoder
        """
        # Step I. Update E, and D: optimizer the discriminator D(E( * ))
        self.ED_optimizer.zero_grad()
        L_adv_ED = self.get_ED_loss(batch_real_data, **ae_kwargs)
        L_adv_ED.backward()
        self.ED_optimizer.step()
        tracker.update(dict(L_adv_ED=L_adv_ED))

        # Step II. Update F, and G: Optimize the generator G(F( * )) to fool D(E ( * ))
        self.FG_optimizer.zero_grad()
        L_adv_FG = self.get_FG_loss(batch_real_data, **ae_kwargs)
        L_adv_FG.backward()
        self.FG_optimizer.step()
        tracker.update(dict(L_adv_FG=L_adv_FG))

        # Step III. Update E, and G: Optimize the reconstruction loss in the Latent space W
        self.ED_optimizer.zero_grad()
        self.FG_optimizer.zero_grad()
        # self.EG_optimizer.zero_grad()
        L_err_EG = self.get_EG_loss(batch_real_data, **ae_kwargs)
        L_err_EG.backward()
        # self.EG_optimizer.step()
        self.ED_optimizer.step()
        self.FG_optimizer.step()
        tracker.update(dict(L_err_EG=L_err_EG))

    def train(self, train_dataset, test_data, output_dir):
        raise NotImplementedError

    def generate(self, z_vectors, **ae_kwargs):
        raise NotImplementedError

    def encode(self, img, **ae_kwargs):
        raise NotImplementedError

    def decode(self, latent_vectorsz, **ae_kwargs):
        raise NotImplementedError

    def save_sample(self, dump_path, samples_z, samples, **ae_kwargs):
        """
        Create debug image of images and their reconstruction alongside images generated from random noise
        """
        with torch.no_grad():
            restored_image = self.decode(self.encode(samples, **ae_kwargs), **ae_kwargs)
            generated_images = self.generate(samples_z, **ae_kwargs)

            resultsample = torch.cat([samples, restored_image, generated_images], dim=0).cpu()

            # Normalize images from -1,1 to 0, 1.
            # Eventhough train samples are in this range (-1,1), the generated image may not. But this should diminish as
            # raining continues or else the discriminator can detect them. Anyway save_image clamps it to 0,1
            resultsample = resultsample * 0.5 + 0.5

            save_image(resultsample, dump_path, nrow=len(samples))


class StyleALAE(ALAE):
    """
    Implements the Style version of ALAE. Use the StyleGenerator (including the mapping from latent) for StyleGan
    and use a Style-Encoder depicted in the paper.
    """
    def __init__(self, model_config, device):
        super().__init__(model_config, 'cnn', device)
        self.res_idx = 0
        self.train_step = 0

    def generate(self, z_vectors, **ae_kwargs):
        self.G.eval()
        self.F.eval()
        generated_images = self.G(self.F(z_vectors), **ae_kwargs)
        return torch.nn.functional.interpolate(generated_images, size=self.cfg['resolutions'][-1])

    def generate_with_truncation(self, z_vectors, style = 1, **ae_kwargs):
        self.G.eval()
        self.F.eval()
        w = self.F(z_vectors)
        w_new = torch.mean(w) + style*(w - torch.mean(w))
        generated_images = self.G(w_new, **ae_kwargs)
        return torch.nn.functional.interpolate(generated_images, size=self.cfg['resolutions'][-1])

    def generate_style_mixing(self, z_main, z_copy, copystyleto,**ae_kwargs):
        self.G.eval()
        self.F.eval()
        w_main = self.F(z_main)
        w_copy = self.F(z_copy)
        w = [w_main, w_copy]
        generated_images = self.G(w, copystyleto = copystyleto, **ae_kwargs)
        return torch.nn.functional.interpolate(generated_images, size=self.cfg['resolutions'][-1])

    def encode(self, img, **ae_kwargs):
        desired_resolution = self.cfg['resolutions'][ae_kwargs['final_resolution_idx']]
        downscaled_images = torch.nn.functional.interpolate(img, size=desired_resolution)
        self.E.eval()
        w_vectors = self.E(downscaled_images, **ae_kwargs)
        self.E.train()
        return w_vectors

    def decode(self, w_vectors, **ae_kwargs):
        self.G.eval()
        generated_images = self.G(w_vectors, **ae_kwargs)
        self.G.train()
        return torch.nn.functional.interpolate(generated_images, size=self.cfg['resolutions'][-1])

    def train(self, train_dataset, test_data, output_dir):
        tracker = LossTracker(output_dir)
        while self.res_idx < len(self.cfg['resolutions']):
            res = self.cfg['resolutions'][self.res_idx]
            self.set_optimizers_lr(self.cfg['learning_rates'][self.res_idx])
            batch_size = self.cfg['batch_sizes'][self.res_idx]
            batchs_in_phase = self.cfg['phase_lengths'][self.res_idx] // batch_size
            dataloader = EndlessDataloader(get_dataloader(train_dataset, batch_size, resize=res, device=self.device))
            progress_bar = tqdm(range(batchs_in_phase * 2))
            for i in progress_bar:
                # first half of the batchs are fade in phase where alpha < 1. in the second half alpha =1
                alpha = min(1.0, i / batchs_in_phase)
                batch_real_data = dataloader.next()
                self.perform_train_step(batch_real_data, tracker, final_resolution_idx=self.res_idx, alpha=alpha)

                self.train_step += 1
                progress_tag = f"gs-{self.train_step}_res-{self.res_idx}={res}x{res}_alpha-{alpha:.2f}"
                progress_bar.set_description(progress_tag)

                if self.train_step % self.cfg['dump_imgs_freq'] == 0:
                    tracker.plot()
                    dump_path = os.path.join(output_dir, 'images', f"{progress_tag}.jpg")
                    self.save_sample(dump_path, test_data[0], test_data[1], final_resolution_idx=self.res_idx, alpha=alpha)

                if self.train_step % self.cfg['checkpoint_freq'] == 0:
                    self.save_train_state(os.path.join(output_dir, 'checkpoints', f"ckpt_{progress_tag}.pt"))
            self.res_idx += 1
        self.save_train_state(os.path.join(output_dir, 'checkpoints', f"ckpt_final.pt"))

    def load_train_state(self, checkpoint_path):
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.F.load_state_dict(checkpoint['F'])
            self.G.load_state_dict(checkpoint['G'])
            self.E.load_state_dict(checkpoint['E'])
            self.D.load_state_dict(checkpoint['D'])
            self.ED_optimizer.load_state_dict(checkpoint['ED_optimizer'])
            self.FG_optimizer.load_state_dict(checkpoint['FG_optimizer'])
            self.res_idx = checkpoint['last_uncompleted_res_idx']
            self.train_step = checkpoint['global_step']
            print(f"Checkpoint {os.path.basename(checkpoint_path)} loaded. Starting from resolution "
                  f"number {self.res_idx} and global_step {self.train_step}")
        else:
            print("Starting training from scratch ")

    def save_train_state(self, save_path):
        torch.save(
            {
                'F': self.F.state_dict(),
                'G': self.G.state_dict(),
                'E': self.E.state_dict(),
                'D': self.D.state_dict(),
                'ED_optimizer': self.ED_optimizer.state_dict(),
                'FG_optimizer': self.FG_optimizer.state_dict(),
                'last_uncompleted_res_idx': self.res_idx,
                'global_step': self.train_step,
            },
            save_path
        )


class MLP_ALAE(ALAE):
    """
    Implements the MLP version of ALAE. all submodules here are composed of MLP layers
    """
    def __init__(self, model_config, device):
        super().__init__(model_config, 'MLP', device)

    def generate(self, z_vectors, **ae_kwargs):
        return self.G(self.F(z_vectors))

    def encode(self, img, **ae_kwargs):
        return self.E(img)

    def decode(self, latent_vectors, **ae_kwargs):
        return self.G(latent_vectors)

    def train(self, train_dataset, test_data, output_dir):
        train_dataloader = get_dataloader(train_dataset, self.cfg['batch_size'], resize=None, device=self.device)
        tracker = LossTracker(output_dir)
        self.set_optimizers_lr(self.cfg['lr'])
        for epoch in range(self.cfg['epochs']):
            for batch_real_data in tqdm(train_dataloader):
                self.perform_train_step(batch_real_data, tracker)

            tracker.plot()
            dump_path = os.path.join(output_dir, 'images', f"epoch-{epoch}.jpg")
            self.save_sample(dump_path, test_data[0], test_data[1])

            self.save_train_state(os.path.join(output_dir, "last_ckp.pth"))

    def load_train_state(self, checkpoint_path):
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.F.load_state_dict(checkpoint['F'])
            self.G.load_state_dict(checkpoint['G'])
            self.E.load_state_dict(checkpoint['E'])
            self.D.load_state_dict(checkpoint['D'])
            print(f"Checkpoint {os.path.basename(checkpoint_path)} loaded.")

    def save_train_state(self, save_path):
        torch.save(
            {
                'F': self.F.state_dict(),
                'G': self.G.state_dict(),
                'E': self.E.state_dict(),
                'D': self.D.state_dict(),
            },
            save_path
        )