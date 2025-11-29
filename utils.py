import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import GPT2LMHeadModel, AutoModel
import wandb
import os
import logging
from omegaconf import OmegaConf, open_dict


def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def log_and_save_samples(samples, file_name_prefix, wandb_key, step, rank, this_sample_dir, tokenizer, cfg, logger):
    """Decodes, saves, and logs samples to wandb."""
    # Decode samples to text
    sentences = tokenizer.batch_decode(samples, skip_special_tokens=True)

    # Log to wandb if it's the main process
    if cfg.wandb and rank == 0:
        table = wandb.Table(columns=["Step", "Generated Text"])
        for sentence in sentences:
            table.add_data(step, sentence)
        wandb.log({wandb_key: table}, step=step)


def calculate_perplexity(samples, cfg, device, rank, world_size):
    """Calculates perplexity for a given set of samples."""
    if not cfg.eval.perplexity:
        return None
    with torch.no_grad():
        eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
        batch_size = cfg.eval.perplexity_batch_size
        batches = samples.shape[0] // batch_size
        if batches == 0:
            print("Warning: Not enough samples to calculate perplexity.")
            del eval_model
            torch.cuda.empty_cache()
            return None

        total_perplexity = 0
        for i in range(batches):
            s = samples[i * batch_size:(i + 1) * batch_size]
            loss, logits = eval_model(s, labels=s)[:2]
            logits = logits.transpose(-1, -2)
            perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
            total_perplexity += perplexity
        
        total_perplexity /= batches
        dist.all_reduce(total_perplexity, op=dist.ReduceOp.SUM)
        total_perplexity /= world_size
        
        del eval_model, logits, loss
        torch.cuda.empty_cache()
        return total_perplexity


def calculate_per_sample_overlap(source_samples, target_samples):
    """
    Calculates the overlap rate between source and target samples, sample by sample,
    and returns the average overlap rate across the batch.
    """
    batch_size = source_samples.shape[0]
    if batch_size == 0:
        return torch.tensor(0.0, device=source_samples.device)
    
    overlaps = []
    for i in range(batch_size):
        source = source_samples[i]
        target = target_samples[i]
        # For each token in the source, is it present in the target?
        overlap_mask = torch.isin(source, target)
        # The overlap rate for this sample is the mean of the boolean mask.
        sample_overlap_rate = overlap_mask.float().mean()
        overlaps.append(sample_overlap_rate)
    
    # Average the overlap rates across the batch.
    return torch.stack(overlaps).mean()