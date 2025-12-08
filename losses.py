import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils

def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """
        # inspect_perturbations(graph, noise, batch, sampling_eps=1e-5, num_steps=128)
        # exit()
        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps # (epsilon, 1]
        sigma, dsigma = noise(t) # noise and the derivative of noise w.r.t. t
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss
    return loss_fn


def get_loss_fn_with_stage(noise, graph, train, stage, sampling_eps=1e-3, lv=False):
    def loss_fn(model, input_ids_stage1, input_ids_stage2, input_ids, cond=None, t=None, perturbed_batch=None):

        prefix = None
        suffix = None
        prefix_len = 0 

        if stage == 1:
            prefix = None
            suffix = input_ids_stage1
            prefix_len = 0
        
        elif stage == 2:
            prefix = input_ids_stage1.clone()
            suffix = input_ids_stage2
            prefix_len = prefix.shape[1]
        
        elif stage == 3:
            prefix = input_ids_stage2.clone()
            suffix = input_ids
            prefix_len = prefix.shape[1]

        else:
            raise ValueError(f"Invalid stage: {stage}. Must be an integer 1, 2, or 3.")


        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(suffix.shape[0], device=suffix.device) + sampling_eps
        
        sigma, dsigma = noise(t)
        
        perturbed_suffix = graph.sample_transition(suffix, sigma[:, None])
        
        if prefix is not None:
            model_input = torch.cat([prefix, perturbed_suffix], dim=1)
        else:
            model_input = perturbed_suffix

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(model_input, sigma)

        log_score_suffix = log_score[:, prefix_len:]
        perturbed_suffix_for_loss = model_input[:, prefix_len:]

        loss = graph.score_entropy(
            log_score_suffix, 
            sigma[:, None], 
            perturbed_suffix_for_loss, 
            suffix
        )

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss
    
    return loss_fn

def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn




def get_step_fn_with_stage(noise, graph, train, optimize_fn, accum, stage):
    loss_fn = get_loss_fn_with_stage(noise, graph, train, stage=stage)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, input_ids_stage1, input_ids_stage2, input_ids, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, input_ids_stage1, input_ids_stage2, input_ids, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, input_ids_stage1, input_ids_stage2, input_ids, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn