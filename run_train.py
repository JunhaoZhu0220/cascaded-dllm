import datetime
import os
import os.path
import gc
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import data_constituency
import data_keybert_no_stopword

import losses
import sampling
import graph_lib
import noise_lib
import utils
from utils import log_and_save_samples, calculate_perplexity, calculate_per_sample_overlap
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from omegaconf import OmegaConf
import wandb
from sampling import get_sampling_fn_with_stage

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

from load_model import load_model


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # setup wandb
    if cfg.wandb and rank == 0:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Modify wandb name to include prefix_mask_ratio
        wandb_name = cfg.wandb.name
        if hasattr(cfg, 'pretrained') and cfg.pretrained.use_pretrained:
            # Add finetuning suffix if using pretrained model
            wandb_name = f"{wandb_name}-finetuning"

        if hasattr(cfg, 'data') and cfg.data.dataloader:
            wandb_name = f"{wandb_name}-{cfg.data.dataloader}"
        
        wandb.init(
            project=cfg.wandb.project,
            config=cfg_dict,
            name=wandb_name,
            dir=work_dir,
            resume='allow'
        )


    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    else:
        logger = None
    def mprint(msg):
        if rank == 0 and logger is not None:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    mprint(f"--- Starting Training for Stage {cfg.stage} ---")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    score_model = SEDD(cfg).to(device)
    
    # Load pretrained weights if specified in config
    if hasattr(cfg, 'pretrained') and cfg.pretrained.use_pretrained:
        mprint(f"Loading pretrained model from {cfg.pretrained.model_path}")
        try:
            pretrained_model, pretrained_graph, pretrained_noise = load_model(cfg.pretrained.model_path, device)
            
            # Extract the state dict from the pretrained model
            if hasattr(pretrained_model, 'module'):
                pretrained_state_dict = pretrained_model.module.state_dict()
            else:
                pretrained_state_dict = pretrained_model.state_dict()
            
            # Load the pretrained weights into our model
            missing_keys, unexpected_keys = score_model.load_state_dict(pretrained_state_dict, strict=False)
            if missing_keys:
                mprint(f"Missing keys when loading pretrained model: {missing_keys}")
            if unexpected_keys:
                mprint(f"Unexpected keys when loading pretrained model: {unexpected_keys}")
            
            mprint(f"Successfully loaded pretrained weights from {cfg.pretrained.model_path}")
            
            # Clean up
            del pretrained_model, pretrained_graph, pretrained_noise
            torch.cuda.empty_cache()
            
        except Exception as e:
            mprint(f"Warning: Failed to load pretrained model from {cfg.pretrained.model_path}: {e}")
            mprint("Continuing with randomly initialized weights...")
    else:
        mprint("Training from scratch")
    
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # load stage 1 & stage 2 models if needed:
    # Initialize sampling functions as None, will be created when needed
    stage1_sampling_fn = None
    stage2_sampling_fn = None
    
    if cfg.training.get('enable_unconditional_sampling', True):
        if cfg.stage == 2:
            stage1_model, stage1_graph, stage1_noise = load_model(cfg.stage1_root_dir, device)
            mprint(f"Loaded Stage 1 model from {cfg.stage1_root_dir}")
            
            # Create stage 1 sampling function
            batch_size_per_gpu = cfg.eval.batch_size // (cfg.ngpus * cfg.training.accum)
            stage1_sampling_shape = (batch_size_per_gpu, cfg.target_lens.stage1)
            stage1_sampling_fn = get_sampling_fn_with_stage(cfg, stage1_graph, stage1_noise, sampling_eps, device, stage1_sampling_shape)
            mprint(f"Created Stage 1 sampling function with shape {stage1_sampling_shape}")
        
        if cfg.stage == 3:
            stage1_model, stage1_graph, stage1_noise = load_model(cfg.stage1_root_dir, device)
            stage2_model, stage2_graph, stage2_noise = load_model(cfg.stage2_root_dir, device)
            mprint(f"Loaded Stage 1 model from {cfg.stage1_root_dir}")
            mprint(f"Loaded Stage 2 model from {cfg.stage2_root_dir}")
            
            # Create stage 1 and stage 2 sampling functions
            batch_size_per_gpu = cfg.training.batch_size // (cfg.ngpus * cfg.training.accum)
            stage1_sampling_shape = (batch_size_per_gpu, cfg.target_lens.stage1)
            stage2_sampling_shape = (batch_size_per_gpu, cfg.target_lens.stage2)
            
            # Create stage 1 sampling function
            stage1_sampling_fn = get_sampling_fn_with_stage(cfg, stage1_graph, stage1_noise, sampling_eps, device, stage1_sampling_shape)
            mprint(f"Created Stage 1 sampling function with shape {stage1_sampling_shape}")
            
            # Create stage 2 sampling function
            stage2_sampling_fn = get_sampling_fn_with_stage(cfg, stage2_graph, stage2_noise, sampling_eps, device, stage2_sampling_shape)
            mprint(f"Created Stage 2 sampling function with shape {stage2_sampling_shape}")
    else:
        mprint("Skipping loading of previous stage models.")

    
    # load in tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Build data iterators
    if cfg.data.dataloader == 'constituency':
        train_ds, eval_ds = data_constituency.get_dataloaders(cfg)
    elif cfg.data.dataloader == 'keybert':
        train_ds, eval_ds = data_keybert.get_dataloaders(cfg)
    elif cfg.data.dataloader == 'keybert_no_stopword':
        train_ds, eval_ds = data_keybert_no_stopword.get_dataloaders(cfg)
    else:
        raise ValueError(f"Unknown dataloader: {cfg.preprocessor}. Must be one of 'constituency', 'keybert', 'keybert_no_stopword'.")

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)
    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn_with_stage(noise, graph, True, optimize_fn, cfg.training.accum, stage=cfg.stage)
    eval_step_fn = losses.get_step_fn_with_stage(noise, graph, False, optimize_fn, cfg.training.accum, stage=cfg.stage)


    if cfg.training.snapshot_sampling:
        # Determine the correct length based on the current stage
        if cfg.stage == 1:
            current_length = cfg.target_lens.stage1
        elif cfg.stage == 2:
            current_length = cfg.target_lens.stage2
        elif cfg.stage == 3:
            current_length = cfg.target_lens.stage3
        
        sampling_shape = (cfg.eval.batch_size // (cfg.ngpus * cfg.training.accum), current_length)
        sampling_fn = get_sampling_fn_with_stage(cfg, graph, noise, sampling_eps, device, sampling_shape)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")


    while state['step'] < num_train_steps + 1:
        step = state['step']
        batch = next(train_iter)
        input_ids = batch['input_ids'].to(device)
        input_ids_stage1 = batch["input_ids_stage1"].to(device)
        input_ids_stage2 = batch["input_ids_stage2"].to(device)
        
        loss = train_step_fn(state, input_ids_stage1, input_ids_stage2, input_ids)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
                
                # wandb logging
                if cfg.wandb and rank == 0:
                    wandb.log({
                        "global_step": step,
                        "training_loss": loss.item(),
                    })

            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                #TODO: change the cfg for baseline and non-baseline for validation
                eval_batch = next(eval_iter)

                input_ids_stage1 = eval_batch["input_ids_stage1"].to(device)
                input_ids_stage2 = eval_batch["input_ids_stage2"].to(device)
                input_ids = eval_batch['input_ids'].to(device)
                eval_loss = eval_step_fn(state, input_ids_stage1, input_ids_stage2, input_ids)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))

                # wandb logging
                if cfg.wandb and rank == 0:
                    wandb.log({
                        "global_step": step,
                        "evaluation_loss": eval_loss.item(),
                    })

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if cfg.training.snapshot_sampling:
                    current_step = step  # Use the step value at the time of checkpoint condition
                    mprint(f"Generating text at step: {current_step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(current_step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())

                    # --- Conditional Sampling ---
                    if cfg.training.get('enable_conditional_sampling', True) and cfg.stage != 1:
                        # First generate conditioned samples with input_ids_stage1 and input_ids_stage2 from dataloader
                        cond_batch = next(eval_iter)
                        if len(cond_batch['input_ids']) < sampling_shape[0]:
                            cond_batch = next(eval_iter)

                        input_ids_stage1_cond = cond_batch["input_ids_stage1"].to(device)
                        input_ids_stage2_cond = cond_batch["input_ids_stage2"].to(device)
                        input_ids_cond = cond_batch["input_ids"].to(device)
                    
                        if cfg.stage == 2:
                            # Log the conditional prefix from stage 1
                            log_and_save_samples(
                                samples=input_ids_stage1_cond,
                                file_name_prefix="sample_stage1_cond",
                                wandb_key="Samples Stage 1 (Conditional)",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )

                            # Generate stage 2 output based on the condition
                            with torch.no_grad():
                                input_ids_stage2 = sampling_fn(score_model, prefix=input_ids_stage1_cond)

                            # Calculate and log the overlap rate
                            stage1_stage2_overlap_cond = calculate_per_sample_overlap(input_ids_stage1_cond, input_ids_stage2)
                            if cfg.wandb and rank == 0:
                                wandb.log({"Stage 1 to Stage 2 Overlap (Conditional)": stage1_stage2_overlap_cond.item()}, step=current_step)
                                mprint(f"Stage 1 to Stage 2 Overlap (Conditional): {stage1_stage2_overlap_cond.item():.4f}")
                            
                            log_and_save_samples(
                                samples=input_ids_stage2,
                                file_name_prefix="sample_stage2_cond",
                                wandb_key="Samples Stage 2 (Conditional)",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )
                            
                            # Decode both generated and ground truth samples
                            generated = tokenizer.batch_decode(input_ids_stage2)
                            ground_truth = tokenizer.batch_decode(input_ids_stage2_cond)

                            # Log comparison table to wandb
                            if cfg.wandb and rank == 0:
                                comparison_table = wandb.Table(columns=["Sample ID", "Generated Text", "Ground Truth Text"])
                                for i, (generated, ground_truth) in enumerate(zip(generated, ground_truth)):
                                    comparison_table.add_data(i+1, generated, ground_truth)
                                wandb.log({"Comparison Table Stage 2 (Generated vs Ground Truth)": comparison_table}, step=current_step)

                            # Calculate conditional accuracy against ground truth
                            cond_acc_stage2 = calculate_per_sample_overlap(input_ids_stage2, input_ids_stage2_cond)
                            if cfg.wandb and rank == 0:
                                wandb.log({"Conditional Accuracy Stage 2": cond_acc_stage2.item()}, step=current_step)
                                mprint(f"Conditional Accuracy Stage 2: {cond_acc_stage2.item():.4f}")

                        elif cfg.stage == 3:
                            # Log the conditional prefixes from stage 1 and 2
                            log_and_save_samples(
                                samples=input_ids_stage1_cond,
                                file_name_prefix="sample_stage1_cond",
                                wandb_key="Samples Stage 1 (Conditional)",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )
                            log_and_save_samples(
                                samples=input_ids_stage2_cond,
                                file_name_prefix="sample_stage2_cond",
                                wandb_key="Samples Stage 2 (Conditional)",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )

                            # Generate stage 3 output based on the conditions
                            with torch.no_grad():
                                input_ids_stage3 = sampling_fn(score_model, prefix=input_ids_stage2_cond)
                            
                            log_and_save_samples(
                                samples=input_ids_stage3,
                                file_name_prefix="sample_stage3_cond",
                                wandb_key="Samples Stage 3 (Conditional)",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )

                            # Create comparison table between generated and ground truth
                            if rank == 0:
                                # Decode both generated and ground truth samples
                                generated = tokenizer.batch_decode(input_ids_stage3)
                                ground_truth = tokenizer.batch_decode(input_ids_cond)

                                # Log comparison table to wandb
                                if cfg.wandb and rank == 0:
                                    comparison_table = wandb.Table(columns=["Sample ID", "Generated Text", "Ground Truth Text"])
                                    for i, (generated, ground_truth) in enumerate(zip(generated, ground_truth)):
                                        comparison_table.add_data(i+1, generated, ground_truth)
                                    wandb.log({"Comparison Table Stage 3 (Generated vs Ground Truth)": comparison_table}, step=current_step)

                            # Calculate and log perplexity for conditional samples
                            cond_ppl = calculate_perplexity(input_ids_stage3, cfg, device, rank, world_size)
                            original_ppl = calculate_perplexity(input_ids_cond, cfg, device, rank, world_size)
                            if original_ppl is not None and cond_ppl is not None:
                                mprint(f"Original Perplexity at step: {current_step}. Perplexity: {original_ppl:.3f}.")
                                mprint(f"Conditional Generative Perplexity at step: {current_step}. Perplexity: {cond_ppl:.3f}.")
                                if cfg.wandb and rank == 0:
                                    wandb.log({"original_perplexity": original_ppl.item()}, step=current_step)
                                    wandb.log({"conditional_generative_perplexity": cond_ppl.item()}, step=current_step)

                            # Calculate conditional accuracy against ground truth
                            cond_acc_stage3 = calculate_per_sample_overlap(input_ids_stage3, input_ids_cond)
                            if cfg.wandb and rank == 0:
                                wandb.log({"Conditional Accuracy Stage 3": cond_acc_stage3.item()}, step=current_step)
                                mprint(f"Conditional Accuracy Stage 3: {cond_acc_stage3.item():.4f}")
                    else:
                        if cfg.stage != 1:
                            mprint("Conditional sampling is disabled via config.")

                    # --- Unconditional Sampling ---
                    if cfg.training.get('enable_unconditional_sampling', True):
                        mprint("--- Starting Unconditional Sampling ---")
                        if cfg.stage == 1:
                            with torch.no_grad():
                                input_ids_stage1 = sampling_fn(score_model, prefix=None)
                            log_and_save_samples(
                                samples=input_ids_stage1,
                                file_name_prefix="sample_stage1",
                                wandb_key="Samples Stage 1",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )
                        
                        elif cfg.stage == 2:
                            # Use pre-created stage 1 sampling function
                            with torch.no_grad():
                                input_ids_stage1 = stage1_sampling_fn(stage1_model)
                                input_ids_stage2 = sampling_fn(score_model, prefix=input_ids_stage1)

                            # Log stage 1 and stage 2 unconditional samples
                            log_and_save_samples(
                                samples=input_ids_stage1,
                                file_name_prefix="sample_stage1_uncond",
                                wandb_key="Samples Stage 1 (Unconditional)",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )
                            log_and_save_samples(
                                samples=input_ids_stage2,
                                file_name_prefix="sample_stage2_uncond",
                                wandb_key="Samples Stage 2 (Unconditional)",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )

                            # Calculate and log overlap rate
                            stage1_stage2_overlap = calculate_per_sample_overlap(input_ids_stage1, input_ids_stage2)
                            if cfg.wandb and rank == 0:
                                wandb.log({"Stage 1 to Stage 2 Overlap (Unconditional)": stage1_stage2_overlap.item()}, step=current_step)
                                mprint(f"Stage 1 to Stage 2 Overlap (Unconditional): {stage1_stage2_overlap.item():.4f}")

                        
                        elif cfg.stage == 3:
                            # Generate stage 1 samples
                            with torch.no_grad():
                                input_ids_stage1 = stage1_sampling_fn(stage1_model)
                            log_and_save_samples(
                                samples=input_ids_stage1,
                                file_name_prefix="sample_stage1_uncond_prefix",
                                wandb_key="Samples Stage 1 (Unconditional Prefix)",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )

                            # Generate stage 2 samples
                            with torch.no_grad():
                                input_ids_stage2 = stage2_sampling_fn(stage2_model, prefix=input_ids_stage1)
                            log_and_save_samples(
                                samples=input_ids_stage2,
                                file_name_prefix="sample_stage2_uncond_prefix",
                                wandb_key="Samples Stage 2 (Unconditional Prefix)",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )

                            with torch.no_grad():
                                input_ids_stage3 = sampling_fn(score_model, prefix=input_ids_stage2)
                            log_and_save_samples(
                                samples=input_ids_stage3,
                                file_name_prefix="sample_stage3_uncond",
                                wandb_key="Samples Stage 3 (Unconditional)",
                                step=current_step, rank=rank, this_sample_dir=this_sample_dir,
                                tokenizer=tokenizer, cfg=cfg, logger=logger
                            )

                            # Calculate and log perplexity for unconditional samples
                            uncond_ppl = calculate_perplexity(input_ids_stage3, cfg, device, rank, world_size)
                            if uncond_ppl is not None:
                                mprint(f"Unconditional Generative Perplexity at step: {current_step}. Perplexity: {uncond_ppl:.3f}.")
                                if cfg.wandb and rank == 0:
                                    wandb.log({"unconditional_generative_perplexity": uncond_ppl.item()}, step=current_step)

                            # Calculate and log overlap rates
                            stage1_stage2_overlap = calculate_per_sample_overlap(input_ids_stage1, input_ids_stage2)
                            stage1_stage3_overlap = calculate_per_sample_overlap(input_ids_stage1, input_ids_stage3)
                            stage2_stage3_overlap = calculate_per_sample_overlap(input_ids_stage2, input_ids_stage3)

                            if cfg.wandb and rank == 0:
                                wandb.log({
                                    "Stage 1 to Stage 2 Overlap (Unconditional)": stage1_stage2_overlap.item(),
                                    "Stage 1 to Stage 3 Overlap (Unconditional)": stage1_stage3_overlap.item(),
                                    "Stage 2 to Stage 3 Overlap (Unconditional)": stage2_stage3_overlap.item()
                                }, step=current_step)
                                mprint(f"Stage 1 to Stage 2 Overlap (Unconditional): {stage1_stage2_overlap.item():.4f}")
                                mprint(f"Stage 1 to Stage 3 Overlap (Unconditional): {stage1_stage3_overlap.item():.4f}")
                                mprint(f"Stage 2 to Stage 3 Overlap (Unconditional): {stage2_stage3_overlap.item():.4f}")
                    else:
                        mprint("Unconditional sampling is disabled via config.")

                    ema.restore(score_model.parameters())
                    dist.barrier()
