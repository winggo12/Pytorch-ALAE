from dnn.costume_layers import *

"""
All the sub_modules here are descibed in https://arxiv.org/pdf/1812.04948.pdf
"""


class MappingFromLatent(nn.Module):
    """A mapping from the z latent space to the w space that should be run vefore the style generator"""
    def __init__(self, num_layers=5, input_dim=256, out_dim=256):
        super(MappingFromLatent, self).__init__()
        layers = [LREQ_FC_Layer(input_dim, out_dim), nn.LeakyReLU(0.2)]
        for i in range(num_layers - 1):
            layers += [LREQ_FC_Layer(out_dim, out_dim), nn.LeakyReLU(0.2)]
        self.mapping = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = pixel_norm(x)
        x = self.mapping(x)
        return x


class StyleGeneratorBlock(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, is_first_block=False, upscale=False):
        super(StyleGeneratorBlock, self).__init__()
        assert not (is_first_block and upscale), "You should not upscale if this is the first block in the generator"
        self.is_first_block = is_first_block
        self.upscale = upscale
        if is_first_block:
            self.const_input = nn.Parameter(torch.randn(1, out_channels, STARTING_DIM, STARTING_DIM))
        else:
            self.blur = LearnablePreScaleBlur(out_channels)
            self.conv1 = Lreq_Conv2d(in_channels, out_channels, 3, padding=1)

        self.style_affine_transform_1 = StyleAffineTransform(latent_dim, out_channels)
        self.style_affine_transform_2 = StyleAffineTransform(latent_dim, out_channels)
        self.noise_scaler_1 = NoiseScaler(out_channels)
        self.noise_scaler_2 = NoiseScaler(out_channels)
        self.adain = AdaIn(in_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = Lreq_Conv2d(out_channels, out_channels, 3, padding=1)

        self.name = f"StyleBlock({latent_dim}, {in_channels}, {out_channels}, is_first_block={is_first_block}, upscale={upscale})"

    def __str__(self):
        return self. name

    def forward(self, input, latent_w, noise, noise_scaling_factor=None):
        if self.is_first_block:
            assert(input is None)
            result = self.const_input.repeat(latent_w.shape[0], 1, 1, 1)
        else:
            if self.upscale:
                input = upscale_2d(input)
            result = self.conv1(input)
            result = self.blur(result)

        if noise_scaling_factor is None:
            result += self.noise_scaler_2(noise)
        else:
            result += (noise_scaling_factor * self.noise_scaler_2(noise))

        # result += self.noise_scaler_1(noise)
        result = self.adain(result, self.style_affine_transform_1(latent_w))
        result = self.lrelu(result)

        result = self.conv2(result)
        if noise_scaling_factor is None:
            result += self.noise_scaler_2(noise)
        else:
            result += (noise_scaling_factor * self.noise_scaler_2(noise))
        # result += self.noise_scaler_2(noise)
        result = self.adain(result, self.style_affine_transform_2(latent_w))
        result = self.lrelu(result)

        return result


class StylleGanGenerator(nn.Module):
    """
    The style generator.
    """
    def __init__(self, latent_dim, progression):
        """
        progression: A list of tuples (<out_res>, <out_channels>) that describes the Generator blocks of this module
        """
        super(StylleGanGenerator, self).__init__()
        self.latent_dim = latent_dim
        assert progression[0][0] == STARTING_DIM, f"First module should note upscale so first out_dim should be {STARTING_DIM}"
        self.to_rgb = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])
        self.progression = progression
        # Parse the module description given in "progression"
        for i in range(len(progression)):
            self.to_rgb.append(Lreq_Conv2d(progression[i][1], 3, 1, 0))
            if i == 0:
                self.conv_blocks.append(StyleGeneratorBlock(latent_dim, STARTING_CHANNELS, progression[i][1],
                                                            is_first_block=True))
            else:
                upscale = (progression[i - 1][0] * 2 == progression[i][0])
                self.conv_blocks.append(StyleGeneratorBlock(latent_dim, progression[i - 1][1], progression[i][1],
                                                            upscale=upscale))

    def __str__(self):
        name = "Style-Generator:\n"
        name += "\ttoRgbs\n"
        for i in range(len(self.conv_blocks)):
            name += f"\t {self.to_rgb[i]}\n"
        name += "\tStyleGeneratorBlocks\n"
        for i in range(len(self.conv_blocks)):
            name += f"\t {self.conv_blocks[i]}\n"
        return name

    def forward(self, w, final_resolution_idx, alpha, copystylefrom=None):
        generated_img = None
        feature_maps = None
        for i, block in enumerate(self.conv_blocks):
            # Separate noise for each block
            if type(w) != type([]) :
                noise = torch.randn((w.shape[0], 1, 1, 1), dtype=torch.float32).to(w.device)
                prev_feature_maps = feature_maps
                feature_maps = block(feature_maps, w, noise, noise_scaling_factor=None)
            else:
                if self.progression[i][0] in copystylefrom:
                    noise = torch.randn((w[1].shape[0], 1, 1, 1), dtype=torch.float32).to(w[1].device)
                    prev_feature_maps = feature_maps
                    feature_maps = block(feature_maps, w[1], noise, noise_scaling_factor=None)
                else:
                    noise = torch.randn((w[0].shape[0], 1, 1, 1), dtype=torch.float32).to(w[0].device)
                    prev_feature_maps = feature_maps
                    feature_maps = block(feature_maps, w[0], noise, noise_scaling_factor=None)

            if i == final_resolution_idx:
                generated_img = self.to_rgb[i](feature_maps)

                # If there is an already stabilized last previous resolution layer. alpha blend with it
                if i > 0 and alpha < 1:
                    generated_img_without_last_block = self.to_rgb[i - 1](prev_feature_maps)
                    if block.upscale:
                        generated_img_without_last_block = upscale_2d(generated_img_without_last_block)
                    generated_img = alpha * generated_img + (1 - alpha) * generated_img_without_last_block
                break

        return generated_img
