import torch
import argparse
from load_model import load_model
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch.nn.functional as F
import sampling
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--stage1_model_path", type=str, default="/scratch/jz5770/Discrete-Diffusion/exp_local/openwebtext/stage1-lr3e-4-16-no-stopword/2025.10.31/133956")
    parser.add_argument("--stage2_model_path", type=str, default="/scratch/jz5770/Discrete-Diffusion/exp_local/openwebtext/stage2_lr3e-4-32_64_128-no-stopword/2025.10.31/134228")
    parser.add_argument("--stage3_model_path", type=str, default="/scratch/jz5770/Discrete-Diffusion/exp_local/openwebtext/stage3-lr3e-4-32_64_128-no-stopword/2025.10.31/114107")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--predictor", type=str, default='analytic')
    parser.add_argument("--output_dir", default="./sampling_outputs", type=str)
    #parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--eval_perplexity", action='store_true', help="Evaluate perplexity of the generated samples.")
    args = parser.parse_args()
    
    device = torch.device('cuda')
    
    # Load models
    print("Loading models...")
    stage1_model, stage1_graph, stage1_noise = load_model(args.stage1_model_path, device)
    print("Stage 1 model loaded.")
    stage2_model, stage2_graph, stage2_noise = load_model(args.stage2_model_path, device)
    print("Stage 2 model loaded.")
    stage3_model, stage3_graph, stage3_noise = load_model(args.stage3_model_path, device)
    print("Stage 3 model loaded.")

    stage1_len = stage3_model.config.target_lens.stage1
    stage2_len = stage3_model.config.target_lens.stage2
    stage3_len = stage3_model.config.target_lens.stage3
    print(f"Target lengths - Stage 1: {stage1_len}, Stage 2: {stage2_len}, Stage 3: {stage3_len}")

    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"sample_cascade_steps_{args.steps}_len_{stage3_len}_{timestamp}"
    run_output_dir = os.path.join(args.output_dir, run_dir_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    stage1_sampling_shape = (args.batch_size, stage1_len)
    stage2_sampling_shape = (args.batch_size, stage2_len)
    stage3_sampling_shape = (args.batch_size, stage3_len)

    # --- Stage 1 Sampling ---
    print("\n--- Stage 1 Sampling ---")
    stage1_sampling_fn = sampling.get_pc_sampler_with_stage(
        stage1_graph, stage1_noise, 
        batch_dims=stage1_sampling_shape, predictor=args.predictor, 
        steps=args.steps, prefix=None,
        device=device)

    with torch.no_grad():
        stage1_ids = stage1_sampling_fn(stage1_model)

    texts_stage1 = tokenizer.batch_decode(stage1_ids)
    file_name_s1 = os.path.join(run_output_dir, f"sample_stage1.txt")
    with open(file_name_s1, 'w') as file:
        for sentence in texts_stage1:
            file.write(sentence + "\n")
            file.write("="*80 + "\n")
    print(f"Stage 1 sampling done. Samples saved to {file_name_s1}")

    # --- Stage 2 Sampling ---
    print("\n--- Stage 2 Sampling ---")
    stage2_sampling_fn = sampling.get_pc_sampler_with_stage(
        stage2_graph, stage2_noise, 
        batch_dims=stage2_sampling_shape, predictor=args.predictor, 
        steps=args.steps, prefix=stage1_ids,
        device=device)

    with torch.no_grad():
        stage2_ids = stage2_sampling_fn(stage2_model)

    texts_stage2 = tokenizer.batch_decode(stage2_ids)
    file_name_s2 = os.path.join(run_output_dir, f"sample_stage2.txt")
    with open(file_name_s2, 'w') as file:
        for sentence in texts_stage2:
            file.write(sentence + "\n")
            file.write("="*80 + "\n")
    print(f"Stage 2 sampling done. Samples saved to {file_name_s2}")

    # --- Stage 3 Sampling ---
    print("\n--- Stage 3 Sampling ---")
    stage3_sampling_fn = sampling.get_pc_sampler_with_stage(
        stage3_graph, stage3_noise, 
        batch_dims=stage3_sampling_shape, predictor=args.predictor, 
        steps=args.steps, prefix=stage2_ids,
        device=device)
    
    with torch.no_grad():
        final_samples = stage3_sampling_fn(stage3_model)

    text_samples = tokenizer.batch_decode(final_samples)
    
    output_file = os.path.join(run_output_dir, f"sampling_final.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("FINAL RESULTS:\n")
        f.write("="*80 + "\n")
        for i, text in enumerate(text_samples):
            f.write(f"Sample {i+1}:\n{text}\n")
            f.write("-" * 50 + "\n")
    
    print("\n--- Final Samples ---")
    for i in text_samples:
        print(i)
        print("="*50)
    print(f"All steps results saved to: {run_output_dir}")

    # --- Evaluation ---
    summary_lines = []
    if args.eval_perplexity:
        print("\n--- Evaluating Perplexity ---")
        with torch.no_grad():
            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
            loss, logits = eval_model(final_samples, labels=final_samples)[:2]
            logits = logits.transpose(-1, -2)
            perplexity = F.cross_entropy(logits[..., :-1], final_samples[..., 1:], reduction="none").mean(dim=-1).exp().mean()
            perplexity_str = f"Generative Perplexity: {perplexity:.3f}"
            print(perplexity_str)
            summary_lines.append(perplexity_str)

    # --- Overlap Calculation ---
    print("\n--- Calculating Overlap ---")
    final_samples_flat = final_samples.view(-1)
    input_ids_16_flat = input_ids_16.view(-1)
    input_ids_32_flat = input_ids_32.view(-1)

    stage1_overlap = torch.isin(input_ids_16_flat, final_samples_flat).float().mean()
    stage2_overlap = torch.isin(input_ids_32_flat, final_samples_flat).float().mean()

    stage1_overlap_str = f"Percentage of Stage 1 tokens in final result: {stage1_overlap.item() * 100:.2f}%"
    stage2_overlap_str = f"Percentage of Stage 2 tokens in final result: {stage2_overlap.item() * 100:.2f}%"
    print(stage1_overlap_str)
    print(stage2_overlap_str)
    summary_lines.append(stage1_overlap_str)
    summary_lines.append(stage2_overlap_str)

    summary_file_path = os.path.join(run_output_dir, "summary.txt")
    with open(summary_file_path, 'w') as f:
        for line in summary_lines:
            f.write(line + '\n')
    print(f"Summary saved to {summary_file_path}")

if __name__=="__main__":
    main()