from transformers import logging
from diffusers import IFPipeline, DDPMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class IF(nn.Module):
    def __init__(self, device, vram_O, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device

        print(f'[INFO] loading DeepFloyd IF-I-XL...')

        model_key = "DeepFloyd/IF-I-XL-v1.0"

        is_torch2 = torch.__version__[0] == '2'

        # Create model
        pipe = IFPipeline.from_pretrained(model_key, variant="fp16", torch_dtype=torch.float16)
        # if not is_torch2:
        #     pipe.enable_xformers_memory_efficient_attention()

        if vram_O:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.unet = pipe.unet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        self.pipe = pipe
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.time_prior = [800, 500, 300, 100]
        m1, m2, s1, s2 = self.time_prior
        weights = torch.cat(
            (
                torch.exp(
                    -(torch.arange(self.num_train_timesteps, m1, -1) - m1)
                        / (2 * s1)
                    ),
                torch.ones(m1 - m2 + 1),
                torch.exp(
                        -(torch.arange(m2 - 1, 0, -1) - m2) / (2 * s2)
                    ),
            )
        )
        weights = weights / torch.sum(weights)
        self.time_prior_acc_weights = torch.cumsum(weights, dim=0)
        self.iters = 6000
        self.t_choice = self.t_choice_nonlinear(self.iters)
        print(f'[INFO] loaded DeepFloyd IF-I-XL!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        # TODO: should I add the preprocessing at https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py#LL486C10-L486C28
        prompt = self.pipe._text_preprocessing(prompt, clean_caption=False)
        inputs = self.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, grad_scale=1):

        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)

            # pred noise
            model_input = torch.cat([images_noisy] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(model_input, tt, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # TODO: how to use the variance here?
            # noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(images, grad)

        return loss

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, guidance_scale=100, grad_scale=1, global_step=0, max_step=0):

        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts        

        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=self.device)
        if global_step <= 5000:
            t = self.t_choice[global_step - 1]
        else:

            t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=self.device)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(images)
            # noise = torch.randn_like(images) + 0.01 * torch.randn(
            #             images.shape[0], images.shape[1], 1, 1, device=images.device)
            images_noisy = self.scheduler.add_noise(images, noise, t)

            # pred noise
            model_input = torch.cat([images_noisy] * (1 + K))
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * (1 + K))
            unet_output = self.unet(model_input, tt, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)



        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(images, grad)

        return loss

    @torch.no_grad()
    def produce_imgs(self, text_embeddings, height=64, width=64, num_inference_steps=50, guidance_scale=7.5):

        images = torch.randn((1, 3, height, width), device=text_embeddings.device, dtype=text_embeddings.dtype)
        images = images * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            model_input = torch.cat([images] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)

            # predict the noise residual
            noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            images = self.scheduler.step(noise_pred, t, images).prev_sample

        images = (images + 1) / 2

        return images


    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img
        imgs = self.produce_imgs(text_embeds, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs
    
    def t_choice_linear(self, global_step: int):
        value = [0, 0.98, 0.5, 5000]
        #[0, 0.98, 0.5, 5000]
        start_step, start_value, end_value, end_step = value
        current_step = global_step
        # 0.98 - 0.48 * min(1,  1000 / 5000)
        value = start_value + (end_value - start_value) * min(1.0, 
            (current_step - start_step) / (end_step - start_step))

        return torch.tensor([int(value * self.num_train_timesteps)], dtype=torch.long, device=self.device)
    
    def t_choice_nonlinear(self, max_step: int):
        t_choice = []
        for i in range(0, max_step):
            current_step_ratio = i / max_step
            time_index = torch.where(
                        (self.time_prior_acc_weights - current_step_ratio) > 0
                    )[0][0]
            if time_index == 0 or torch.abs(
                self.time_prior_acc_weights[time_index] - current_step_ratio
            ) < torch.abs(
                self.time_prior_acc_weights[time_index - 1] - current_step_ratio
            ):
                t = self.num_train_timesteps - time_index
            else:
                t = self.num_train_timesteps - time_index + 1
            t = torch.clip(t, self.min_step, self.max_step + 1)
            t = torch.full((1,), t, dtype=torch.long, device=self.device)
            t_choice.append(t)
        return t_choice

if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=64)
    parser.add_argument('-W', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = IF(device, opt.vram_O)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()



