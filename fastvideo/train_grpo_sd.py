# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [MIT License] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [MIT License], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
from fastvideo.models.stable_diffusion.pipeline_with_logprob import pipeline_with_logprob
from fastvideo.models.stable_diffusion.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import torch.distributed as dist
import functools
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import save_file

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

class PromptDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            self.prompts = [line.strip() for line in f if line.strip()]
            
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int((config.sample.num_steps-1) * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    if config.reward_fn == 'hpsv3':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from hpsv3 import HPSv3RewardInferencer
        reward_model = HPSv3RewardInferencer(device=device)
    
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="grpo-sd",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        #config.pretrained.model, revision=config.pretrained.revision
        "./data/stable-diffusion-v1-4"
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        pipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model,
                revision=config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model
    
    def gather_tensor(tensor):
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    if config.reward_fn == 'hpsv2':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # prepare prompt and reward fn
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from typing import Union
        import huggingface_hub
        from hpsv2.utils import root_path, hps_version_map
        def initialize_model():
            model_dict = {}
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                './hps_ckpt/open_clip_pytorch_model.bin',
                precision='amp',
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict
        model_dict = initialize_model()
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        #cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
        cp = "./hps_ckpt/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(cp, map_location=f'cuda')
        model.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(device)
        reward_model.eval()

    #prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.num_generations
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    import torch.nn.functional as F

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    dataset = PromptDataset("./assets/prompts.txt")
    
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        seed = 123543,
        shuffle=True  
    )
    
    
    loader = DataLoader(
        dataset,
        batch_size=config.sample.batch_size,
        sampler=sampler,
        pin_memory=True,  
        drop_last=True    
    )

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0
    import torch.distributed as dist
    global_step = 0
    for epoch, prompts in enumerate(loader):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []

        if epoch > config.num_epochs:
            break

        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * config.num_generations)

        all_latents = []
        all_log_probs = []
        all_rewards = []
        all_prompts_embed = []

        ###for the sake of convenience, we use the same latents for all prompts in a batch.
        global_input_latents = torch.randn(
                    (1, 4, 64, 64),
                    device=accelerator.device,
                    dtype=torch.bfloat16,
                )

        batch_size = config.train.batch_size  
        for i in range(0, len(expanded_prompts), batch_size):
            current_batch = expanded_prompts[i:i+batch_size]
            
            prompt_ids = pipeline.tokenizer(
                current_batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            if i%config.num_generations == 0:
                input_latents = global_input_latents.repeat(batch_size,1,1,1).clone()

            with torch.no_grad():
                with autocast():
                    images, _, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                        latents=input_latents
                    )
            rewards = []
            tuwen_rewards = []
            for j, image in enumerate(images):
                pil = Image.fromarray(
                    (image.to(torch.float32).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((512, 512))
                image_path = os.path.join("./images_same", f"image-{i}-{j}-rank-{dist.get_rank()}.jpg")
                pil.save(image_path)
                if config.reward_fn == "hpsv2":
                    image = preprocess_val(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device=device, non_blocking=True)
                    # Process the prompt
                    text = processor([current_batch[j]]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        if config.reward_fn == "hpsv2":
                            outputs = reward_model(image, text)
                            image_features, text_features = outputs["image_features"], outputs["text_features"]
                            logits_per_image = image_features @ text_features.T
                            hps_score = torch.diagonal(logits_per_image)
                            rewards.append(hps_score)
                        elif config.reward_fn == "hpsv3":
                            hps_score = reward_model.reward([image_path], [current_batch[j]])
                            if hps_score.ndim == 2:
                                hps_score = hps_score[:,0]
                            rewards.append(hps_score)

            latents = torch.stack(latents, dim=1).detach().clone()     # (4, num_steps+1, ...)
            log_probs = torch.stack(log_probs, dim=1).detach().clone()   # (4, num_steps, ...)
            rewards = torch.cat(rewards, dim=0).clone()  
            

            all_latents.append(latents)
            all_log_probs.append(log_probs)
            all_rewards.append(rewards)
            all_prompts_embed.append(prompt_embeds)

            torch.cuda.empty_cache()


        all_latents = torch.cat(all_latents, dim=0)
        all_log_probs = torch.cat(all_log_probs, dim=0)
        all_rewards = torch.cat(all_rewards, dim=0).to(torch.float32)
        all_prompts_embed = torch.cat(all_prompts_embed, dim=0)
        timesteps = pipeline.scheduler.timesteps.repeat(
            config.sample.batch_size*config.num_generations, 1
        ) 

        # compute rewards asynchronously
        #rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
        # yield to to make sure reward computation starts
        time.sleep(0)

        samples={
                "prompt_embeds": all_prompts_embed,
                "timesteps": timesteps[:, :-1],
                "latents": all_latents[
                    :, :-1
                ][:, :-1],  # each entry is the latent before timestep t
                "next_latents": all_latents[
                    :, 1:
                ][:, :-1],  # each entry is the latent after timestep t
                "log_probs": all_log_probs[:, :-1],
                "rewards": all_rewards,
            }

        # gather rewards across processes
        all_rewards_world = gather_tensor(all_rewards)

        # log rewards and images
        accelerator.log(
            {
                "reward": all_rewards_world,
                "epoch": epoch,
                "reward_mean": all_rewards_world.mean(),
                "reward_std": all_rewards_world.std(),
            },
            step=global_step,
        )

        if dist.get_rank()==0:
            print("gathered_reward", all_rewards_world)
            with open('./reward.txt', 'a') as f:  # 'a'模式表示追加到文件末尾
                f.write(f"{all_rewards_world.mean().item()}\n")

        #samples = {k: v.cuda() for k, v in samples.items()}  # 假设原始数据在GPU
        #samples = process_samples(samples, config)
        n = len(samples["rewards"]) // (config.num_generations)
        advantages = torch.zeros_like(samples["rewards"])

        for i in range(n):
            start_idx = i * config.num_generations
            end_idx = (i + 1) * config.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        samples["advantages"] = advantages.clone()

        samples["final_advantages"] = advantages.clone()
        

        total_batch_size, num_timesteps = samples["timesteps"].shape
        #assert (
        #    total_batch_size
        #    == config.sample.batch_size * config.sample.num_batches_per_epoch
        #)
        #assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond
                                    + config.sample.guidance_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["final_advantages"],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                unet.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if dist.get_rank()%8==0:
                    print("reward", sample["rewards"])
                    print("ratio", ratio)
                    print("final advantage", advantages)
                    print("hps_advantage", sample["advantages"])
                    print("final loss", loss)
                dist.barrier()


            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % config.save_freq == 0: # 
        #if epoch % config.save_freq == 0: 
            if accelerator.is_main_process:
                base_checkpoint_dir = "./my_checkpoints"
                # Create a unique directory for this specific checkpoint
                checkpoint_epoch_dir = os.path.join(base_checkpoint_dir, f"checkpoint_epoch_{epoch}")
                os.makedirs(checkpoint_epoch_dir, exist_ok=True)

                # Define paths for the UNet weights (safetensors) and config (json)
                unet_safetensors_path = os.path.join(checkpoint_epoch_dir, "diffusion_pytorch_model.safetensors")
                config_json_path = os.path.join(checkpoint_epoch_dir, "config.json")

                unwrapped_unet = accelerator.unwrap_model(pipeline.unet)

                # 1. Save the UNet state_dict using safetensors
                try:
                    # The state_dict should contain only the model weights
                    model_state_dict = unwrapped_unet.state_dict()
                    save_file(model_state_dict, unet_safetensors_path)
                    accelerator.print(f"Manually saved UNet weights to {unet_safetensors_path}")
                except Exception as e:
                    accelerator.print(f"Error saving UNet weights with safetensors: {e}")

            # Barrier for distributed training, if initialized
            # Ensures all processes wait until the main process has finished saving
            if dist.is_initialized():
                dist.barrier()


if __name__ == "__main__":
    app.run(main)

