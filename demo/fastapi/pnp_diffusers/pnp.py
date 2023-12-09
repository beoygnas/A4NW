import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from .pnp_utils import *
from compel import Compel




# suppress partial model loading warning
logging.set_verbosity_error()

def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start

class PNP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.sd_version = config["sd_version"]
        
        self.output_path = 'data/results'
        self.latents_path = 'data/latents'
        self.img_path = 'data/image.jpg'
        
        if self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"

        print(f'[INFO] loading stable diffusion...')
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", revision="fp16",
                                                 torch_dtype=torch.float16).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
                                                          torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
                                                         torch_dtype=torch.float16).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        
        # self.compel_proc = Compel(tokenizer=self.tokenizer, text_encoder=self.text_encoder)
        
        vae_params = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        text_encoder_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        unet_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)    
        total_params = vae_params + text_encoder_params + unet_params
        print(f'[INFO] loaded stable diffusion!')
        print(f'\t VAE\t : ' + format(vae_params, ',') + f' / {vae_params * 2 / 1000000:.2f}MB' )
        print(f'\t TEXT_ENC: ' + format(text_encoder_params, ',') + f' / {text_encoder_params * 2 / 1000000:.2f}MB' )
        print(f'\t UNET\t : ' + format(unet_params, ',') + f' / {unet_params * 2 / 1000000:.2f}MB' )
        print(f'\t TOTAL\t : ' + format(total_params, ',') + f' / {total_params * 2 / 1000000:.2f}MB' )

        self.inversion_func = self.ddim_inversion
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=2):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt[0], padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        text_input_0 = self.tokenizer(prompt[1], padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings_0 = self.text_encoder(text_input_0.input_ids.to(self.device))[0]

        uncond_input_0 = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings_0 = self.text_encoder(uncond_input_0.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings] + [uncond_embeddings_0] + [text_embeddings] + [text_embeddings_0])
        return text_embeddings
    
    def get_text_embeds_inversion(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings
    
    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img
    
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self):
        # # load image
        # image = Image.open(self.img_path).convert('RGB') 
        # image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        # image = T.ToTensor()(image).to(self.device)
        # get noise
        latents_path = os.path.join(self.latents_path, os.path.splitext(os.path.basename(self.img_path))[0], f'noisy_latents_{self.scheduler.timesteps[0]}.pt')
        noisy_latent = torch.load(latents_path).to(self.device)
        noisy_latent = noisy_latent.repeat(2, 1, 1, 1)
        # print(noisy_latent.shape)
        return noisy_latent
        # return image, noisy_latent

    @torch.no_grad()
    def denoise_step(self, x, t):
        # register the time step and features in pnp injection modules
        source_latents = load_source_latents_t(t, os.path.join(self.latents_path, os.path.splitext(os.path.basename(self.img_path))[0]))
        source_latents = source_latents.repeat(2, 1, 1, 1)
        # source_latents = self.latents[idx]
        # batch_size = x.shape[0]
        # latent_model_input_list = []
        # for b in range(batch_size) : 
        #     latent_model_input_list += [source_latents] + ([x[b:b+1]] * 2)
        # latent_model_input = torch.cat(latent_model_input_list)
        latent_model_input = torch.cat([source_latents] + ([x] * 2))
        register_time(self, t.item())
        # compute text embeddings
        # text_embed_input_list = []
        # for b in range(batch_size) : 
        #     text_embed_input_list += [self.pnp_guidance_embeds] + [self.text_embeds[2*b:2*(b+1)]]
        # text_embed_input = torch.cat(text_embed_input_list)
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)
        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self, pnp_config):
        pnp_f_t = int(pnp_config["n_timesteps"] * pnp_config["pnp_f_t"])
        pnp_attn_t = int(pnp_config["n_timesteps"] * pnp_config["pnp_attn_t"])
        
        self.scheduler.set_timesteps(pnp_config["n_timesteps"], device=self.device)
        self.eps = self.get_data()
        self.text_embeds = self.get_text_embeds(pnp_config["prompt"], pnp_config["negative_prompt"])
        # self.pnp_guidance_embeds = self.get_text_embeds_inversion("", "").chunk(2)[0]
        self.pnp_guidance_embeds = self.get_text_embeds(["", ""], "").chunk(2)[0]
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        save_path = os.path.join(self.latents_path, os.path.splitext(os.path.basename(self.img_path))[0])
        edited_img = self.sample_loop(self.eps, pnp_config["prompt"], save_path)
        return edited_img

    def sample_loop(self, x, prompt, save_path):
        
        # pre_timesteps = self.latents.shape[0]
        # pnp_timesteps = len(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                # idx = (pre_timesteps - 1) - (i * (pre_timesteps//pnp_timesteps) + 1)
                x = self.denoise_step(x, t)
                # torch.save(x, os.path.join(save_path, f'denoised_latents_{t}.pt'))
            print(x.shape)
            # decoded_latent = self.decode_latent(x)
            decoded_latent_1 = self.decode_latent(x[0:1])
            decoded_latent_2 = self.decode_latent(x[1:2])
            output_path = os.path.join(self.output_path, os.path.splitext(os.path.basename(self.img_path))[0])
            os.makedirs(output_path, exist_ok=True)
            # generated_img = T.ToPILImage()(decoded_latent[0]).save(f'{output_path}/output-{prompt}.png')
            generated_img_1 = T.ToPILImage()(decoded_latent_1[0]).save(f'{output_path}/output-{prompt[0]}.png')
            generated_img_2 = T.ToPILImage()(decoded_latent_2[0]).save(f'{output_path}/output-{prompt[1]}.png')
        return f'{output_path}/output-{prompt[0]}.png', f'{output_path}/output-{prompt[1]}.png'
    
    ## preprocess function
    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=True,
                                timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if save_latents:
                    torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return latent
    
    @torch.no_grad()
    def extract_latents(self, num_steps, inversion_prompt='', extract_reverse=False):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds_inversion(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(self.img_path)
        latent = self.encode_imgs(image)

        save_path = os.path.join(self.latents_path, os.path.splitext(os.path.basename(self.img_path))[0])
        os.makedirs(save_path, exist_ok=True)
        
        inverted_x = self.inversion_func(cond, latent, save_path, save_latents=not extract_reverse)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='pnp-configs/config-horse.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(config["output_path"], exist_ok=True)
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    seed_everything(config["seed"])
    pnp = PNP(config)
    pnp.run_pnp()