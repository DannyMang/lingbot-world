import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint, torch_dtype=torch.bfloat16)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint, torch_dtype=torch.bfloat16)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def generate(self,
                 input_prompt,
                 img,
                 action_path=None,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy")) # opencv coordinate
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]

        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # cam preparation (only if action_path is provided)
        dit_cond_dict = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()

            # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
            Ks = get_Ks_transformed(Ks,
                                    height_org=480,
                                    width_org=832,
                                    height_resize=h,
                                    width_resize=w,
                                    height_final=h,
                                    width_final=w)
            Ks = Ks[0]
            
            len_c2ws = len(c2ws)
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...] # [b, f*h*w, c]
            c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            dit_cond_dict = {
                "c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0),
            }

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
                'dit_cond_dict': dit_cond_dict,
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
                'dit_cond_dict': dit_cond_dict,
            }

            if offload_model:
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                noise_pred_cond = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def generate_fast(self,
                      input_prompt,
                      img,
                      action_path=None,
                      max_area=320 * 576,
                      frame_num=33,
                      shift=3.0,
                      sampling_steps=6,
                      seed=-1):
        r"""
        Fast single-expert generation: uses only high_noise_model, no CFG,
        6 denoising steps, 320x576 resolution. ~30x faster than generate().

        Args:
            input_prompt (`str`): Text prompt for content generation.
            img (PIL.Image.Image): Input image.
            action_path (`str`, *optional*): Path to poses/intrinsics .npy files.
            max_area (`int`, *optional*, defaults to 320*576): Pixel area for 320p.
            frame_num (`int`, *optional*, defaults to 33): Frames (must be 4n+1).
            shift (`float`, *optional*, defaults to 3.0): Noise schedule shift for 320p.
            sampling_steps (`int`, *optional*, defaults to 6): Denoising steps.
            seed (`int`, *optional*, defaults to -1): Random seed.

        Returns:
            torch.Tensor: Generated video (C, N, H, W) or None.
        """
        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]

        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, lat_f, lat_h, lat_w,
            dtype=torch.float32, generator=seed_g, device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # Encode text (T5)
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]

        # Plucker embeddings
        dit_cond_dict = None
        if action_path is not None:
            Ks = torch.from_numpy(
                np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            Ks = get_Ks_transformed(Ks, height_org=480, width_org=832,
                                    height_resize=h, width_resize=w,
                                    height_final=h, width_final=w)
            Ks = Ks[0]
            len_c2ws = len(c2ws)
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1))
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)
            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h), c2=int(w // lat_w))
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
                f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

        # VAE encode first frame
        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w)
            ], dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        # Ensure high_noise_model is on GPU
        model = self.high_noise_model
        if next(model.parameters()).device.type == 'cpu':
            model.to(self.device)

        with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps

            latent = noise
            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
                'dit_cond_dict': dit_cond_dict,
            }

            for _, t in enumerate(tqdm(timesteps, desc='fast-gen')):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

                # No CFG: single forward pass (guide_scale=1.0)
                noise_pred = model(latent_model_input, t=timestep, **arg_c)[0]

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                    return_dict=False, generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

            videos = self.vae.decode([latent])

        del noise, latent, sample_scheduler
        return videos[0]


class WanI2VCausal:
    """
    Causal (autoregressive, KV-cached) image-to-video generation.

    Loads only the high-noise expert and keeps it on GPU permanently.
    Manages a KV cache across chunks so that each new chunk only needs
    to attend to its own tokens + the cached past, giving ~2-4s per chunk
    after the first.

    Two-phase per chunk:
      1. Denoise with read-only cache (N denoising steps).
      2. One cache-fill pass with the clean latent (read_write), to persist
         this chunk's K/V for future chunks.
    """

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        t5_cpu=True,
        max_cache_chunks=4,
        sampling_steps=20,
        max_area=320 * 576,
        guide_scale=None,
        single_expert=False,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.t5_cpu = t5_cpu
        self.param_dtype = config.param_dtype
        self.num_train_timesteps = config.num_train_timesteps
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.max_cache_chunks = max_cache_chunks
        self.sampling_steps = sampling_steps
        self.max_area = max_area
        self.boundary = config.boundary * config.num_train_timesteps
        self.guide_scale = guide_scale or config.sample_guide_scale  # (low, high)
        self.sample_neg_prompt = config.sample_neg_prompt

        # T5 text encoder
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )

        # VAE
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        # Load both experts (both fit on A100-80GB at bf16: ~14GB each)
        if single_expert:
            logging.info(f"WanI2VCausal: loading high-noise expert only from {checkpoint_dir}")
            self.high_noise_model = WanModel.from_pretrained(
                checkpoint_dir, subfolder=config.high_noise_checkpoint,
                torch_dtype=torch.bfloat16)
            self.high_noise_model.eval().requires_grad_(False)
            self.high_noise_model.to(self.device)
            self.low_noise_model = None
        else:
            logging.info(f"WanI2VCausal: loading both experts from {checkpoint_dir}")
            self.low_noise_model = WanModel.from_pretrained(
                checkpoint_dir, subfolder=config.low_noise_checkpoint,
                torch_dtype=torch.bfloat16)
            self.low_noise_model.eval().requires_grad_(False)
            self.low_noise_model.to(self.device)

            self.high_noise_model = WanModel.from_pretrained(
                checkpoint_dir, subfolder=config.high_noise_checkpoint,
                torch_dtype=torch.bfloat16)
            self.high_noise_model.eval().requires_grad_(False)
            self.high_noise_model.to(self.device)

        # Streaming state
        self.frame_offset = 0  # global latent frame counter
        self._cache_initialized = False
        self._context = None  # cached text embeddings
        self._context_null = None  # cached negative text embeddings

        # Precompute resolution params (fixed across chunks)
        aspect_ratio = 480 / 832  # default dashcam aspect
        self.lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        self.lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        self.h = self.lat_h * self.vae_stride[1]
        self.w = self.lat_w * self.vae_stride[2]

        # Tokens per latent frame
        self._tokens_per_lat_frame = (
            self.lat_h * self.lat_w // (self.patch_size[1] * self.patch_size[2]))

    def _get_model_for_timestep(self, t):
        """Return the correct expert for this timestep."""
        if self.low_noise_model is None:
            return self.high_noise_model
        if t.item() >= self.boundary:
            return self.high_noise_model
        return self.low_noise_model

    def _ensure_cache(self, lat_f_chunk):
        """Initialize or verify KV cache is allocated.

        Only caches the low_noise_model (handles ~95% of timesteps).
        The high_noise_model runs uncached on its 1-2 high-noise steps
        (past context matters less at high noise levels).
        """
        if self._cache_initialized:
            return
        max_lat_frames = self.max_cache_chunks * lat_f_chunk
        max_tokens = max_lat_frames * self._tokens_per_lat_frame
        # Only cache the dominant expert (low_noise handles 95% of steps)
        cache_model = self.low_noise_model if self.low_noise_model is not None else self.high_noise_model
        cache_model.init_kv_caches(
            max_tokens=max_tokens, batch_size=1,
            device=self.device, dtype=self.param_dtype)
        self._cache_initialized = True

    def _encode_text(self, prompt):
        """Encode text + negative prompt (cached across chunks)."""
        if self._context is not None:
            return self._context, self._context_null
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([prompt], self.device)
            context_null = self.text_encoder([self.sample_neg_prompt], self.device)
            self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = self.text_encoder([self.sample_neg_prompt], torch.device('cpu'))
            context_null = [t.to(self.device) for t in context_null]
        self._context = context
        self._context_null = context_null
        return context, context_null

    def _compute_plucker(self, c2ws, intrinsics, lat_f):
        """Compute Plucker embedding dict for a chunk."""
        Ks = torch.from_numpy(intrinsics).float()
        Ks = get_Ks_transformed(Ks, height_org=480, width_org=832,
                                height_resize=self.h, width_resize=self.w,
                                height_final=self.h, width_final=self.w)
        Ks = Ks[0]

        len_c2ws = len(c2ws)
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1))
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(self.device)
        Ks = Ks.to(self.device)
        plucker = get_plucker_embeddings(c2ws_infer, Ks, self.h, self.w)
        plucker = rearrange(
            plucker, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(self.h // self.lat_h), c2=int(self.w // self.lat_w))
        plucker = plucker[None, ...]
        plucker = rearrange(
            plucker, 'b (f h w) c -> b c f h w',
            f=lat_f, h=self.lat_h, w=self.lat_w).to(self.param_dtype)
        return {"c2ws_plucker_emb": plucker.chunk(1, dim=0)}

    def generate_chunk(self, img, prompt, c2ws, intrinsics,
                       frame_num=17, shift=5.0, seed=42):
        """
        Generate one video chunk with KV-cached causal inference.

        Args:
            img: PIL.Image.Image — starting frame for this chunk.
            prompt: str — text prompt.
            c2ws: np.ndarray [N, 4, 4] — camera-to-world matrices.
            intrinsics: np.ndarray [N, 4] — [fx, fy, cx, cy] per frame.
            frame_num: int — frames in this chunk (must be 4n+1).
            shift: float — noise schedule shift.
            seed: int — random seed.

        Returns:
            (video_tensor, last_frame_pil) — video is (C, N, H, W) in [-1,1].
        """
        import torchvision.transforms.functional as TF_local

        F_chunk = min(frame_num, ((len(c2ws) - 1) // 4) * 4 + 1)
        c2ws = c2ws[:F_chunk]
        intrinsics = intrinsics[:F_chunk]

        lat_f = (F_chunk - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * self._tokens_per_lat_frame

        self._ensure_cache(lat_f)

        # Text encoding (cached)
        context, context_null = self._encode_text(prompt)

        # Plucker embeddings (chunk-local)
        dit_cond_dict = self._compute_plucker(c2ws, intrinsics, lat_f)

        # Prepare image conditioning
        img_t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        msk = torch.ones(1, F_chunk, self.lat_h, self.lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, self.lat_h, self.lat_w)
        msk = msk.transpose(1, 2)[0]

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img_t[None].cpu(), size=(self.h, self.w),
                    mode='bicubic').transpose(0, 1),
                torch.zeros(3, F_chunk - 1, self.h, self.w)
            ], dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        # Noise
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, lat_f, self.lat_h, self.lat_w,
            dtype=torch.float32, generator=seed_g, device=self.device)

        with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
            # Scheduler
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            scheduler.set_timesteps(self.sampling_steps, device=self.device, shift=shift)

            # Phase 1: Denoise with read-only cache + MoE routing + CFG
            latent = noise
            base_args = {
                'seq_len': max_seq_len,
                'y': [y],
                'dit_cond_dict': dit_cond_dict,
                'frame_offset': self.frame_offset,
                'use_cache': 'read_only',
            }

            for _, t in enumerate(tqdm(scheduler.timesteps, desc='denoise', leave=False)):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

                # Select expert based on timestep
                model = self._get_model_for_timestep(t)
                scale = self.guide_scale[1] if t.item() >= self.boundary else self.guide_scale[0]

                # Conditional forward pass
                noise_pred_cond = model(
                    latent_model_input, t=timestep,
                    context=[context[0]], **base_args)[0]

                # Unconditional forward pass (CFG)
                noise_pred_uncond = model(
                    latent_model_input, t=timestep,
                    context=context_null, **base_args)[0]

                # CFG combination
                noise_pred = noise_pred_uncond + scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = scheduler.step(
                    noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                    return_dict=False, generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

            # Phase 2: Cache-fill pass with clean latent (single forward, read_write)
            # Use low-noise expert for cache fill (clean signal is in its domain)
            cache_model = self.low_noise_model if self.low_noise_model is not None else self.high_noise_model
            t_zero = torch.zeros(1, device=self.device)
            _ = cache_model(
                [latent.to(self.device)], t=t_zero,
                context=[context[0]],
                seq_len=max_seq_len, y=[y],
                dit_cond_dict=dit_cond_dict,
                frame_offset=self.frame_offset,
                use_cache='read_write')

            # Advance global frame offset
            self.frame_offset += lat_f

            # VAE decode
            videos = self.vae.decode([latent])

        video = videos[0]  # (C, N, H, W)

        # Extract last frame as PIL
        last = video[:, -1, :, :]
        last = ((last + 1.0) / 2.0).clamp(0, 1).cpu()
        last_pil = TF_local.to_pil_image(last)

        del noise, scheduler
        return video, last_pil

    def reset(self):
        """Reset streaming state for a new sequence."""
        self.frame_offset = 0
        self._context = None
        self._context_null = None
        if self._cache_initialized:
            cache_model = self.low_noise_model if self.low_noise_model is not None else self.high_noise_model
            cache_model.clear_kv_caches()

    def free(self):
        """Free all cache memory."""
        self.frame_offset = 0
        self._context = None
        self._context_null = None
        if self._cache_initialized:
            cache_model = self.low_noise_model if self.low_noise_model is not None else self.high_noise_model
            cache_model.free_kv_caches()
            self._cache_initialized = False
