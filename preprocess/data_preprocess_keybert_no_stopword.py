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
from keybert import KeyBERT


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
    text = text.replace(""", '"')
    text = text.replace(""", '"')
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


# Global variables for multiprocessing
parser = None
kw_model = None

def _ensure_models_initialized():
    global parser, kw_model
    if parser is None:
        spacy.require_cpu()
        parser = spacy.load('en_core_web_md')
    if kw_model is None:
        kw_model = KeyBERT()

def _get_token_scores_with_keybert_batch(tokens_list, spacy_docs):
    """
    Get KeyBERT-style scores for multiple documents in batch.
    
    Args:
        tokens_list: list of token lists, each representing a document
        spacy_docs: list of spaCy Doc objects for POS tagging
    
    Returns:
        list of (selected_tokens, token_scores) tuples
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    batch_data = []
    for tokens in tokens_list:
        if not tokens:
            batch_data.append(([], ""))
            continue
            
        selected_tokens = tokens
        doc_text = " ".join(selected_tokens)
        batch_data.append((selected_tokens, doc_text))
    
    doc_texts = [data[1] for data in batch_data if data[1]]
    selected_tokens_list = [data[0] for data in batch_data]
    
    # Get KeyBERT scores for all words (no stop_words filtering)
    batch_keyword_scores = kw_model.extract_keywords(
        doc_texts,
        keyphrase_ngram_range=(1, 1), 
        stop_words=None,
        top_n=max(len(tokens) for tokens in selected_tokens_list if tokens)
    )
        
    results = []
    doc_idx = 0
        
    for i, (selected_tokens, doc_text) in enumerate(batch_data):
        # Get keyword scores for this document
        if doc_idx < len(batch_keyword_scores):
            keyword_scores = batch_keyword_scores[doc_idx]
            spacy_doc = spacy_docs[doc_idx]
            doc_idx += 1
                
            # Create score mapping (case-insensitive)
            # KeyBERT returns lowercase keywords
            score_dict = {}
            for keyword, score in keyword_scores:
                score_dict[keyword.lower()] = score
            
            # Step 1: Assign scores to each token (case-insensitive matching)
            token_scores = []
            stop_word_scores = []
            
            for j, token in enumerate(selected_tokens):
                token_clean = token.rstrip('.,;:!?').lower()
                
                if token_clean and token_clean in score_dict:
                    score = score_dict[token_clean]
                    token_scores.append(score)

                    # Add stop word scores
                    if token_clean in ENGLISH_STOP_WORDS:
                        stop_word_scores.append((j, score))
                else:
                    # If not found, assign -1.2
                    token_scores.append(-1.2)
            
            # Step 2: Rescale stop word scores to [-1.1, -1.0]
            if stop_word_scores:
                indices, scores = zip(*stop_word_scores)
                min_score = min(scores)
                max_score = max(scores)
                
                for idx, original_score in stop_word_scores:
                    if max_score == min_score:
                        # All stop words have the same score
                        rescaled_score = -1.05  # Middle of [-1.1, -1.0]
                    else:
                        # Rescale from [min_score, max_score] to [-1.1, -1.0]
                        rescaled_score = -1.1 + ((original_score - min_score) / (max_score - min_score)) * 0.1
                    token_scores[idx] = rescaled_score
            
            # Step 3: Handle period
            for j, token in enumerate(selected_tokens):
                if token_scores[j] == -1.2:
                    # Check if it's a PUNCT period
                    if j < len(spacy_doc) and spacy_doc[j].pos_ == "PUNCT" and spacy_doc[j].text == ".":
                        token_scores[j] = 1.1
                
            results.append((selected_tokens, token_scores))
        
    return results
        
def parse_and_score_sentences(dataset, tokenizer, batch_size=32):
    """
    Parse and score sentences using KeyBERT with multiprocessing and batch processing.
    
    Args:
        dataset: Dataset to process
        tokenizer: GPT2 tokenizer
        n_process: Number of processes for multiprocessing
        batch_size: Number of documents to process in each KeyBERT batch
    """
    print(f"Processing sentences with KeyBERT scoring using batch size {batch_size}...")
    
    all_input_ids = []
    all_scores = []

    text_column = "sentence" if "sentence" in dataset.column_names else "text"
    texts = dataset[text_column]

    def process_text_batch(text_batch):
        """Process a batch of texts with spaCy and KeyBERT"""
        _ensure_models_initialized()
        
        # Step 1: Parse all texts with spaCy
        batch_words = []
        spacy_docs = []
        for text in text_batch:
            doc = parser(text)
            words = [tok.text for tok in doc]
            batch_words.append(words)
            spacy_docs.append(doc)
        
        # Step 2: Get KeyBERT scores in batch (returns word-level scores)
        # Pass spacy_docs for POS tagging
        batch_results = _get_token_scores_with_keybert_batch(batch_words, spacy_docs)
        
        # Step 3: Encode with GPT2 tokenizer
        final_results = []
        for i, (selected_words, word_scores) in enumerate(batch_results):
            encoding = tokenizer(
                selected_words,
                is_split_into_words=True,
                return_attention_mask=False, 
                add_special_tokens=False
            )
            word_ids = encoding.word_ids()
            input_ids = encoding["input_ids"]
                
            # Map word scores to subword token scores
            token_scores = []
            for word_id in word_ids:
                if word_id is not None and word_id < len(word_scores):
                    token_scores.append(word_scores[word_id])
                else:
                    token_scores.append(-1.2)
            
            assert len(input_ids) == len(token_scores), \
                f"Input IDs and token scores length mismatch: {len(input_ids)} vs {len(token_scores)}"
                
            final_results.append((input_ids, token_scores))
        
        return final_results

    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i+batch_size]
        batch_results = process_text_batch(batch_texts)
        
        for input_ids, scores in batch_results:
            if input_ids is None or scores is None:
                print("Skipping a sample due to processing error.")
                continue
            
            all_input_ids.append(input_ids)
            all_scores.append(scores)

    return {"input_ids": all_input_ids, "scores": all_scores}


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
    save_root="data/preprocessed/keybert_no_stopword",
    n_process=8,
    num_shards=32,
    shard_idx=1,
    batch_size=128,
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
    
    processed_data = parse_and_score_sentences(
        data, tokenizer, batch_size=batch_size
    )

    save_in_chunks(processed_data, shard_dir, chunk_size=100000)

if __name__ == "__main__":
    try:
        # This is still necessary for CUDA safety
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    Fire(main)