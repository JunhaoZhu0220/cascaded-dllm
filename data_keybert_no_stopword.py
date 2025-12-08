import numpy as np
from datasets import load_from_disk, concatenate_datasets, logging
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
import torch
from functools import partial

def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data

def load_all_shards(root: str):
    """
    Load all sharded datasets.
    """
    datasets = []
    root_path = Path(root)
    shard_dirs = list(sorted(root_path.glob("shard_*")))
    
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found matching 'shard_*' in {root_path}")
    
    print(f"Found {len(shard_dirs)} shard directories.")
    
    for shard_dir in shard_dirs:
        part_dirs = list(sorted(shard_dir.glob("part_*")))
        for part_dir in part_dirs:
            datasets.append(load_from_disk(str(part_dir)))
    
    if not datasets:
        raise ValueError(f"No dataset parts found in any of the shard directories in {root_path}.")
    
    full_ds = concatenate_datasets(datasets)
    return full_ds

def docs_to_chunks(batch, chunk_size: int = 1024, drop_last: bool = True):
    '''
    Chunk documents into chunk_size tokens and scores.
    '''  
    out = {"input_ids": [], "scores": []}
    n = len(batch["input_ids"])
    for i in range(n):                           
        chunk_ids, chunk_scores = [], []

        j = i                                      
        while len(chunk_ids) < chunk_size and j < n:
            ids, scores = batch["input_ids"][j], batch["scores"][j]

            take = min(len(ids), chunk_size - len(chunk_ids))
            chunk_ids.extend(ids[:take])
            chunk_scores.extend(scores[:take])

            if len(chunk_ids) == chunk_size:
                break
            j += 1

        if len(chunk_ids) == chunk_size or (not drop_last and chunk_ids):
            out["input_ids"].append(chunk_ids)
            out["scores"].append(chunk_scores)

    return out




def map_fn(batch, chunk_size, drop_last=True):
    chunked_output = docs_to_chunks(
        batch,
        chunk_size=chunk_size,
        drop_last=drop_last,
    )
    return chunked_output


def filter_valid_samples(examples, stage1_len, stage2_len):
    """
    Filter samples based on a two-step validation process:
    1. Ensure there are enough key tokens for Stage 1.
    2. Ensure a full Stage 2 sequence can be constructed.
    """
    valid_mask = []
    
    for ids, scores in zip(examples["input_ids"], examples["scores"]):
        # Step 1: Check for Stage 1 key token capacity.
        high_score_indices = [i for i, score in enumerate(scores) if score > 1]
        unique_medium_tokens = {
            ids[i] for i, score in enumerate(scores) if -1 <= score <= 1
        }
        key_token_count = len(high_score_indices) + len(unique_medium_tokens)
        
        if key_token_count < stage1_len:
            valid_mask.append(False)
            continue

        # Step 2: Simulate Stage 2 selection to ensure it can be filled.
        all_indices = list(range(len(scores)))
        sorted_all_indices = sorted(all_indices, key=lambda i: scores[i], reverse=True)
        
        chosen_indices_stage2 = []
        selected_low_score_ids = set()

        for idx_candidate in sorted_all_indices:
            if len(chosen_indices_stage2) >= stage2_len:
                break
            
            score = scores[idx_candidate]
            token_id = ids[idx_candidate]
            
            if score > 1:
                chosen_indices_stage2.append(idx_candidate)
            else:
                if token_id not in selected_low_score_ids:
                    chosen_indices_stage2.append(idx_candidate)
                    selected_low_score_ids.add(token_id)
        
        # A sample is valid only if it can fill the Stage 2 sequence completely.
        is_valid = len(chosen_indices_stage2) == stage2_len
        valid_mask.append(is_valid)
    
    # Filter the examples based on the valid_mask
    filtered_examples = {key: [] for key in examples.keys()}
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            for key in examples.keys():
                filtered_examples[key].append(examples[key][i])
    
    return filtered_examples


def make_length_subsets(examples, indices, stage1_len, stage2_len):
    """Create stage1, stage2, and stage3 subsets from filtered examples."""
    
    out_stage1_ids = []
    out_stage2_ids = []
    out_stage3_ids = []

    for ids, scores, idx in zip(examples["input_ids"], examples["scores"], indices):
        # Step 1: Select stage 2 tokens based on score rules
        # - score > 1: can repeat
        # - score <= 1: must be unique
        
        all_indices = list(range(len(scores)))
        sorted_all_indices = sorted(all_indices, key=lambda i: scores[i], reverse=True)
        
        chosen_indices_stage2 = []
        selected_low_score_ids = set()  # Tracks unique tokens with score <= 1

        # First pass: iterate through all tokens sorted by score
        for idx_candidate in sorted_all_indices:
            if len(chosen_indices_stage2) >= stage2_len:
                break
            
            score = scores[idx_candidate]
            token_id = ids[idx_candidate]
            
            if score > 1:
                # High-score tokens can be repeated
                chosen_indices_stage2.append(idx_candidate)
            else:
                # Low-score tokens must be unique
                if token_id not in selected_low_score_ids:
                    chosen_indices_stage2.append(idx_candidate)
                    selected_low_score_ids.add(token_id)

        # Step 2: Derive Stage 1 tokens from the selected Stage 2 tokens
        # Sort the collected stage 2 indices by their scores to get the best ones
        stage2_by_score = sorted(chosen_indices_stage2, key=lambda i: scores[i], reverse=True)
        chosen_indices_stage1 = stage2_by_score[:stage1_len]

        # Step 3: Prepare the final output by sorting indices to maintain original order
        indices_stage1 = sorted(chosen_indices_stage1)
        indices_stage2 = sorted(chosen_indices_stage2)

        ids_stage1 = [ids[i] for i in indices_stage1]
        ids_stage2 = [ids[i] for i in indices_stage2]

        out_stage1_ids.append(ids_stage1)
        out_stage2_ids.append(ids_stage2)
        out_stage3_ids.append(ids)

    return {
        "input_ids_stage1": out_stage1_ids,
        "input_ids_stage2": out_stage2_ids,
        "input_ids": out_stage3_ids
    }


def get_dataset(name, mode, block_size=1024, config=None):
    logging.enable_progress_bar()
    root = Path(__file__).resolve().parents[0]
    data_dir = root / "data" / "preprocessed" / "keybert_no_stopword" / f"{name}_{mode}"
    print(f"Loading {mode} set from {data_dir}")
    dataset = load_all_shards(str(data_dir))
    
    print(f"Chunking dataset into blocks of size {block_size}...")
    chunked_dataset = dataset.map(
        partial(docs_to_chunks, chunk_size=block_size, drop_last=True),
        batched=True,
        batch_size=10000,
        remove_columns=dataset.column_names,
        num_proc=8,
        load_from_cache_file=True
    )

    stage1_len = config.target_lens.stage1 if config and hasattr(config, 'target_lens') else 16
    stage2_len = config.target_lens.stage2 if config and hasattr(config, 'target_lens') else 32
    
    print(f"Filtering samples with insufficient valid scores (need at least {stage1_len} for stage1 and {stage2_len} for stage2)...")
    filtered_dataset = chunked_dataset.map(
        partial(filter_valid_samples, stage1_len=stage1_len, stage2_len=stage2_len),
        batched=True,
        batch_size=10000,
        num_proc=8,
        load_from_cache_file=True
    )
    print(f"Dataset size after filtering: {len(filtered_dataset)} (was {len(chunked_dataset)})")
    
    print(f"Creating subsets (stage1: {stage1_len}, stage2: {stage2_len})...")
    final_dataset = filtered_dataset.map(
        partial(make_length_subsets, stage1_len=stage1_len, stage2_len=stage2_len),
        batched=True,
        with_indices=True,
        batch_size=10000,
        num_proc=8,
        remove_columns=filtered_dataset.column_names, 
        load_from_cache_file=True
    )
    return final_dataset

def collate_fn(batch):
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        list_of_values_for_key = [item[key] for item in batch]
        try:
            collated_batch[key] = torch.tensor(list_of_values_for_key)
        except Exception as e:
            print(f"Error converting key '{key}' to tensor: {e}")
    return collated_batch

def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.accum}.")

    block_size = config.target_lens.stage3 if config and hasattr(config, 'target_lens') and hasattr(config.target_lens, 'stage3') else config.model.length
    
    train_set = get_dataset(config.data.train, "train", block_size=block_size, config=config)
    valid_set = get_dataset(config.data.valid, "validation", block_size=block_size, config=config)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=8,
        pin_memory=False,
        shuffle=(train_sampler is None),
        persistent_workers=True,
        collate_fn=collate_fn
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=8,
        pin_memory=True,
        shuffle=(test_sampler is None),
        collate_fn=collate_fn
    ))
    return train_loader, valid_loader