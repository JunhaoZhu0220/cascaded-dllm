import os
import re
import copy
from itertools import chain
from functools import partial
from concurrent.futures import ProcessPoolExecutor, \
     ThreadPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Pool

import numpy as np
import spacy
from transformers import GPT2TokenizerFast
from datasets import load_dataset, Dataset
import requests
import json
from tqdm import tqdm
from fire import Fire
# from nltk.parse.corenlp import CoreNLPDependencyParser
# from graphviz import Source
# import stanza


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset

def get_dataset(name, mode, cache_dir=None, num_proc=8):
    if name == "wikitext103":
        dataset = load_dataset(
            "wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset(
            "wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)

    if name == "lambada":
        data = dataset
    else:
        data = dataset[mode]

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    
    if name.startswith("wikitext"):
        # filter out '' and headers like ' = = = Diet = = = \n'
        header_pattern = r"^\s*=.+.=\s*?$"

        def lstrip_batch(batch):
            # Remove leading whitespace from each text entry
            return {"text": [t.lstrip() if t is not None else t for t in batch["text"]]}

        data = (
            data
            .map(lstrip_batch, batched=True, num_proc=num_proc, load_from_cache_file=True, desc="lstrip")
            .filter(lambda x: bool(x["text"].strip()) and not re.match(header_pattern, x["text"]),
                    num_proc=num_proc, load_from_cache_file=True)
            )
        
    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    def preprocess(example):
        if name == "ptb":
            text = example['sentence']
        else:
            text = example["text"]
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)
        return {"text": text}

    preprocessed_dataset = data.map(preprocess, batched=True, num_proc=num_proc, load_from_cache_file=True)

    return preprocessed_dataset


def get_parser():
    nlp = spacy.load('en_core_web_md')
    return nlp

def compute_word_labels(doc):
    word_labels = []
    n = len(doc)
    for i, tok in enumerate(doc):
        tag = tok.tag_
        
        if tag in ("NNP", "NNPS"):
            # Check if this is the last NNP/NNPS in a sequence
            is_last = (i == n - 1) or (doc[i + 1].tag_ not in ("NNP", "NNPS"))
            word_labels.append(1 if is_last else 2)
        elif tag in ("NN", "NNS"):
            word_labels.append(3)
        elif tag.startswith("VB") and not tok.is_stop:
            # VB* (Stop Word Excluded)
            word_labels.append(4)
        elif tag.startswith("JJ"):
            # JJ*
            word_labels.append(5)
        elif tag.startswith("RB"):
            # RB*
            word_labels.append(6)
        else:
            # Other (including VB stop words, punctuation, etc.)
            word_labels.append(7)
    return word_labels

parser = None

def process_text(text, tokenizer):
    """
    Worker function that processes a single text.
    It initializes its own parser instance lazily.
    """
    global parser
    if parser is None:
        spacy.require_cpu()
        # Each process creates its own parser, avoiding the pickling issue.
        parser = get_parser()

    try:
        doc = parser(text)

    except ValueError as e:
        if "maximum supported length" in str(e):
            input_ids = None
            token_labels = None
            return input_ids, token_labels
        raise
    except AssertionError:
        input_ids = None
        token_labels = None
        return input_ids, token_labels
    
    words = [tok.text for tok in doc]

    word_labels = compute_word_labels(doc)
    encoding = tokenizer(
        words,is_split_into_words=True,
        return_attention_mask=False, add_special_tokens=False
    )
    word_ids = encoding.word_ids()
    token_labels = [word_labels[w_id] if w_id is not None else -100 for w_id in word_ids]
    input_ids = encoding["input_ids"]

    ## Temporarily remove the EOS token and its label
    #input_ids.append(tokenizer.eos_token_id)
    #token_labels.append(0)

    assert len(input_ids) == len(token_labels), \
        f"Input IDs and token labels length mismatch: {len(input_ids)} vs {len(token_labels)}"
    
    return input_ids, token_labels

def parse_and_label_sentences(dataset, tokenizer, n_process=4):
    """
    Parses and labels sentences using a multiprocessing Pool to avoid pickling errors.
    """
    print("Parsing and labeling sentences using a multiprocessing Pool...")
    
    all_input_ids = []
    all_labels = []

    text_column = "sentence" if "sentence" in dataset.column_names else "text"
    texts = dataset[text_column]
    
    worker_func = partial(process_text, tokenizer=tokenizer)

    with Pool(processes=n_process) as pool:
        results = list(tqdm(pool.imap(worker_func, texts, chunksize=1000), total=len(texts)))

    for input_ids, labels in results:
        if input_ids is None or labels is None:
            print("Skipping a sample due to parsing error or length issue.")
            continue
        
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "labels": all_labels}


def save_in_chunks(processed_data, shard_dir, chunk_size=100000):
    os.makedirs(shard_dir, exist_ok=True)
    num_examples = len(processed_data["input_ids"])
    for i in tqdm(range(0, num_examples, chunk_size), desc="Saving to disk"):
        chunk_data = {key: val[i:i+chunk_size] for key, val in processed_data.items()}
        sub_dataset = Dataset.from_dict(chunk_data)
        sub_dataset.save_to_disk(f"{shard_dir}/part_{i//chunk_size}")
    del processed_data


def main(
    dataset_name="openwebtext",
    dataset_mode="train",
    save_root="data/preprocessed/constituency",
    n_process=8,
    num_shards=32,
    shard_idx=10,
):
    data = get_dataset(name=dataset_name, mode=dataset_mode, num_proc=n_process)
    print(f"Dataset {dataset_name} loaded with {len(data)} samples.")

    # Ensure save_root is relative to the project root, not the current working directory
    if not os.path.isabs(save_root):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_root = os.path.join(project_root, save_root)

    shard_dir = f"{save_root}/{dataset_name}_{dataset_mode}/shard_{shard_idx}"
    if num_shards > 1:
        if os.path.exists(shard_dir):
            print(f"Shard {shard_idx} already exists, skipping.")
            return
        os.makedirs(f"{save_root}/{dataset_name}_{dataset_mode}", exist_ok=True)
        print(f"Sharding dataset {dataset_name} into {num_shards} parts...")
        print(f"Current shard index: {shard_idx}")
        data = data.shard(num_shards=num_shards, index=shard_idx, contiguous=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
    
    processed_data = parse_and_label_sentences(
        data, tokenizer, n_process=n_process)

    save_in_chunks(processed_data, shard_dir, chunk_size=100000)

if __name__ == "__main__":
    try:
        # This is still necessary for CUDA safety
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    Fire(main)