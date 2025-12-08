import numpy as np
from datasets import load_from_disk, concatenate_datasets
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
    out = {"input_ids": [], "labels": []}

    n = len(batch["input_ids"])
    for i in range(n):                           
        chunk_ids, chunk_lbls = [], []

        j = i                                      
        while len(chunk_ids) < chunk_size and j < n:
            ids, lbls = batch["input_ids"][j], batch["labels"][j]

            take = min(len(ids), chunk_size - len(chunk_ids))
            chunk_ids.extend(ids[:take])
            chunk_lbls.extend(lbls[:take])

            if len(chunk_ids) == chunk_size:
                break
            j += 1

        if len(chunk_ids) == chunk_size or (not drop_last and chunk_ids):
            out["input_ids"].append(chunk_ids)
            out["labels"].append(chunk_lbls)

    return out




def map_fn(batch, chunk_size, drop_last=True):
    """
    Updated mapping function that calls the new chunker.
    """
    chunked_output = docs_to_chunks(
        batch,
        chunk_size=chunk_size,
        drop_last=drop_last,
    )
    return chunked_output


def make_length_subsets(examples, indices, stage1_len, stage2_len):
    out_stage1_ids, out_stage1_lab = [], []
    out_stage2_ids, out_stage2_lab = [], []
    out_stage3_ids = []

    for ids, labs, idx in zip(examples["input_ids"], examples["labels"], indices):
        
        valid_indices = [i for i, l in enumerate(labs) if l >= 0]
        
        sorted_valid_indices = sorted(valid_indices, key=lambda i: labs[i])
        
        chosen_indices_stage2 = []
        seen_ids_stage2 = set()

        for i in sorted_valid_indices:
            if len(chosen_indices_stage2) >= stage2_len:
                break
            
            token_id = ids[i]
            
            if token_id not in seen_ids_stage2:
                seen_ids_stage2.add(token_id)
                chosen_indices_stage2.append(i)

        if len(chosen_indices_stage2) < stage2_len:
            continue

        chosen_indices_stage1 = chosen_indices_stage2[:stage1_len]

        indices_stage1 = sorted(chosen_indices_stage1)
        indices_stage2 = sorted(chosen_indices_stage2)

        ids_stage1 = [ids[i] for i in indices_stage1]
        lab_stage1 = [labs[i] for i in indices_stage1]
        
        ids_stage2 = [ids[i] for i in indices_stage2]
        lab_stage2 = [labs[i] for i in indices_stage2]

        out_stage1_ids.append(ids_stage1)
        out_stage1_lab.append(lab_stage1)
        out_stage2_ids.append(ids_stage2)
        out_stage2_lab.append(lab_stage2)
        out_stage3_ids.append(ids)

    return {
        "input_ids_stage1": out_stage1_ids, "labels_stage1": out_stage1_lab,
        "input_ids_stage2": out_stage2_ids, "labels_stage2": out_stage2_lab,
        "input_ids": out_stage3_ids
    }


def get_dataset(name, mode, block_size=1024, config=None):
    root = Path(__file__).resolve().parents[0]
    #root = Path("/scratch/jz5770/Discrete-Diffusion")
    data_dir = root / "data" / "preprocessed" / "constituency" / f"{name}_{mode}"
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
    print(f"Creating subsets (stage1: {stage1_len}, stage2: {stage2_len})...")
    final_dataset = chunked_dataset.map(
        partial(make_length_subsets, stage1_len=stage1_len, stage2_len=stage2_len),
        batched=True,
        with_indices=True, # This is crucial to pass the `indices` argument
        batch_size=10000,
        num_proc=8,
        remove_columns=chunked_dataset.column_names, 
        load_from_cache_file=False
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
            # This error can happen if subsets have variable lengths, which shouldn't be the case here.
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