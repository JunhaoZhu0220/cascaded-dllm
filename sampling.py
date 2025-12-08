"""
Sampling Module for Discrete Diffusion Models

This module provides sampling algorithms for discrete diffusion models, including:
- Standard predictor-corrector sampling
- Priority-based sampling with mask gating
- First-hit sampling
- Logging and visualization utilities

The module supports various predictors including Euler, Analytic, and Priority-based predictors.
"""

import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils
from tqdm import tqdm
import numpy as np

# =============================================================================
# Predictor Registry
# =============================================================================

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    """Get a registered predictor by name."""
    return _PREDICTORS[name]


# =============================================================================
# Base Classes
# =============================================================================



class Predictor(abc.ABC):
    """The abstract base class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: Score function that returns model logits
            x: A PyTorch tensor representing the current state [B, L]
            t: A PyTorch tensor representing the current time step [B, 1]
            step_size: Step size for the update

        Returns:
            x: A PyTorch tensor of the next state
        """
        pass


class Denoiser:
    """Denoiser for final cleanup step in sampling."""
    
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        """
        Apply final denoising step.
        
        Args:
            score_fn: Score function
            x: Current state [B, L]
            t: Current timestep [B, 1]
            
        Returns:
            Denoised state
        """
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        
        # Truncate probabilities if absorbing state is used
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        return sample_categorical(probs)


# =============================================================================
# Standard Predictors
# =============================================================================


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, fhs=False):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, fhs=False):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)

        if fhs and self.graph.absorb:
            probs = probs[:, :, :-1]  # truncate probabilities if using first-hit sampling
        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, first_hit_sampler=False, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps


        if not first_hit_sampler:
            for i in tqdm(range(steps)):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
                x = projector(x)
                x = predictor.update_fn(sampling_score_fn, x, t, dt)
        else:
            t = torch.ones(x.shape[0], 1, device=device)
            mask_positions = (x == 50257)      # [B, L]
            total_masks = mask_positions.sum(dim=1).max().item()
            dt = 1 / total_masks
            for i in tqdm(range(total_masks)):
                u = np.random.rand()
                curr_masks = mask_positions.sum(dim=1)    # [B]
                t = t * (u ** (1.0 / curr_masks.float())).unsqueeze(-1)
                x = projector(x) # In most cases, the projector is the identity function, but it can be customized.
                px0 = predictor.update_fn(sampling_score_fn, x, t, dt, first_hit_sampler)

                # Randomly select a masked token to sample
                mask_positions = (x == 50257) 
                chosen_pos = mask_positions.float().multinomial(num_samples=1).squeeze(-1)
                
                batch_idx = torch.arange(x.size(0), device=device)
                x[batch_idx, chosen_pos] = px0[batch_idx, chosen_pos]
                mask_positions = (x == 50257)

        if denoise and not first_hit_sampler:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        return x
    
    return pc_sampler

def get_pc_sampler_with_logging(graph, noise, batch_dims, predictor, steps, fhs=False, denoise=True, 
                               eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x,
                               tokenizer=None, output_file=None):
    
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)
    
    def decode_with_mask(tokens, tokenizer, mask_value=50257):
        if torch.is_tensor(tokens):
            tokens = tokens.detach().cpu().long()
        
        mask_positions = (tokens == mask_value)
        
        tokens_copy = tokens.clone()
        tokens_copy[mask_positions] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        tokens_copy = torch.clamp(tokens_copy, 0, tokenizer.vocab_size - 1)
        
        if tokens_copy.dim() == 1:
            decoded = tokenizer.decode(tokens_copy, skip_special_tokens=False)
            if mask_positions.any():
                mask_count = mask_positions.sum().item()
                decoded = decoded.replace(tokenizer.decode([tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0]) * mask_count, '[MASK]' * mask_count)
        else:
            decoded = []
            for i in range(tokens_copy.shape[0]):
                text = tokenizer.decode(tokens_copy[i], skip_special_tokens=False)
                if mask_positions[i].any():
                    mask_indices = torch.where(mask_positions[i])[0]
                    original_tokens = tokens[i]
                    result_parts = []
                    for j, token_id in enumerate(original_tokens):
                        if token_id == mask_value:
                            result_parts.append('[MASK]')
                        else:
                            try:
                                token_text = tokenizer.decode([token_id.item()], skip_special_tokens=False)
                                result_parts.append(token_text)
                            except:
                                result_parts.append(f'[UNK_{token_id.item()}]')
                    text = ''.join(result_parts)
                decoded.append(text)
        
        return decoded
    
    @torch.no_grad()    
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps
        if output_file and tokenizer:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Sampling Progress Log - {steps} steps\n")
                f.write("="*80 + "\n\n")
        
        if not fhs:
            for i in tqdm(range(steps)):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
                x = projector(x)
                x = predictor.update_fn(sampling_score_fn, x, t, dt)
                if tokenizer:
                    current_samples = decode_with_mask(x, tokenizer)
                    
                    mask_count = (x == 50257).sum().item()
                    total_tokens = x.numel()
                    
                    
                    if output_file:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(f"Step {i+1}/{steps} [MASK: {mask_count}/{total_tokens}]:\n")
                            f.write("-" * 40 + "\n")
                            if isinstance(current_samples, list):
                                for j, sample in enumerate(current_samples):
                                    f.write(f"Batch {j+1}:\n{sample}\n\n")
                            else:
                                f.write(f"{current_samples}\n\n")
                            f.write("\n")
            
            if denoise:
                # denoising step
                x = projector(x)
                t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
                x = denoiser.update_fn(sampling_score_fn, x, t)
                
                if tokenizer:
                    denoised_samples = decode_with_mask(x, tokenizer)
                    
                    final_mask_count = (x == 50257).sum().item()
                    total_tokens = x.numel()
                    
                    
                    if output_file:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(f"Denoising Step [MASK: {final_mask_count}/{total_tokens}]:\n")
                            f.write("-" * 40 + "\n")
                            if isinstance(denoised_samples, list):
                                for j, sample in enumerate(denoised_samples):
                                    f.write(f"Batch {j+1}:\n{sample}\n\n")
                            else:
                                f.write(f"{denoised_samples}\n\n")
            
            return x
        
        else:
            t = torch.ones(x.shape[0], 1, device=device)
            mask_positions = (x == 50257)      # [B, L]
            total_masks = mask_positions.sum(dim=1).max().item()
            dt = 1 / total_masks
            for i in tqdm(range(total_masks)):
                u = np.random.rand()
                curr_masks = mask_positions.sum(dim=1)    # [B]
                t = t * (u ** (1.0 / curr_masks.float())).unsqueeze(-1)
                x = projector(x) # In most cases, the projector is the identity function, but it can be customized.
                px0 = predictor.update_fn(sampling_score_fn, x, t, dt, fhs)

                # Randomly select a masked token to sample
                mask_positions = (x == 50257) 
                chosen_pos = mask_positions.float().multinomial(num_samples=1).squeeze(-1)
                
                batch_idx = torch.arange(x.size(0), device=device)
                x[batch_idx, chosen_pos] = px0[batch_idx, chosen_pos]
                mask_positions = (x == 50257)

                if tokenizer:
                    current_samples = decode_with_mask(x, tokenizer)
                    
                    mask_count = (x == 50257).sum().item()
                    total_tokens = x.numel()
                    
                    
                    if output_file:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(f"Step {i+1}/{steps} [MASK: {mask_count}/{total_tokens}]:\n")
                            f.write("-" * 40 + "\n")
                            if isinstance(current_samples, list):
                                for j, sample in enumerate(current_samples):
                                    f.write(f"Batch {j+1}:\n{sample}\n\n")
                            else:
                                f.write(f"{current_samples}\n\n")
                            f.write("\n")
            return x
    
    return pc_sampler


def get_sampling_fn_with_stage(config, graph, noise, eps, device, sampling_shape):
    """
    Get sampling function for cascade discrete diffusion models.
    
    Args:
        config: Configuration object
        graph: Graph object for transitions
        noise: Noise schedule object
        eps: Minimum timestep
        device: torch device
        sampling_shape: Tuple of (batch_size, sequence_length)
        
    Returns:
        Sampling function
    """
    
    sampling_fn = get_pc_sampler_with_stage(graph=graph,
                                           noise=noise,
                                           batch_dims=sampling_shape,
                                           predictor=config.sampling.predictor,
                                           steps=config.sampling.steps,
                                           prefix=None,
                                           denoise=config.sampling.noise_removal,
                                           eps=eps,
                                           device=device)
    
    return sampling_fn



def get_pc_sampler_with_stage(graph, noise, batch_dims, predictor, steps, prefix=None,
                             denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x,
                             ):
    """
    Create a sampling function that automatically determines prefix based on provided inputs.
    
    Args:
        graph: Graph object for transitions
        noise: Noise schedule object  
        batch_dims: Tuple of (batch_size, sequence_length) for the suffix generation
        predictor: Predictor name (e.g., 'euler', 'analytic')
        steps: Number of sampling steps
        prefix: Prefix tokens (can be None for unconditional generation)
        denoise: Whether to apply denoising step
        eps: Minimum timestep
        device: torch device
        proj_fun: Optional projector function
        
    Returns:
        Sampling function that generates sequences based on provided prefix
        
    Usage:
        - Stage 1: prefix=None (unconditional generation)
        - Stage 2: prefix=stage1_tokens  
        - Stage 3: prefix=stage2_tokens (or concatenated stage1+stage2)
        
    Suffix is generated based on batch_dims shape.
    """
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler_with_stage(model, prefix=prefix):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        
        # Determine prefix length
        if prefix is not None:
            prefix_len = prefix.shape[1]
        else:
            # No prefix provided: unconditional generation
            prefix_len = 0

        batch_size = batch_dims[0]
        suffix_len = batch_dims[1]
        
        suffix = graph.sample_limit(batch_size, suffix_len).to(device)
        
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in tqdm(range(steps), desc="Sampling"):
            t = timesteps[i] * torch.ones(suffix.shape[0], 1, device=device)
            
            # Apply projector to suffix
            suffix = projector(suffix)
            
            # Create suffix-only score function
            def suffix_score_fn(x_suf, sigma):
                if prefix is not None:
                    full_input = torch.cat([prefix, x_suf], dim=1)
                else:
                    full_input = x_suf
                # Get full model prediction
                full_score = sampling_score_fn(full_input, sigma)
                # Return only suffix portion (matches log_score_suffix = log_score[:, prefix_len:])
                if prefix is not None:
                    suffix_score = full_score[:, prefix_len:]
                    return suffix_score
                else:
                    return full_score
            
            # Update only the suffix using predictor
            suffix = predictor.update_fn(suffix_score_fn, suffix, t, dt)

        if denoise:
            # Denoising step
            suffix = projector(suffix)
            t = timesteps[-1] * torch.ones(suffix.shape[0], 1, device=device)
            
            def suffix_score_fn(x_suf, sigma):
                if prefix is not None:
                    full_input = torch.cat([prefix, x_suf], dim=1)
                else:
                    full_input = x_suf
                full_score = sampling_score_fn(full_input, sigma)
                if prefix is not None:
                    return full_score[:, prefix_len:]
                else:
                    return full_score
            
            suffix = denoiser.update_fn(suffix_score_fn, suffix, t)
        return suffix
    
    return pc_sampler_with_stage