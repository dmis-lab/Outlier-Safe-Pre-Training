from __future__ import annotations

import argparse

import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=128)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="bfloat16",
    )
    model = model.requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    full_text = ""
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    for text, _ in zip(dataset, tqdm.trange(1000)):
        full_text += text["text"] + "\n\n"

    tokens = tokenizer(full_text, return_tensors="pt").input_ids
    tokens = tokens[0, : args.max_length * args.num_samples]
    tokens = tokens.view(args.num_samples, args.max_length)

    kurtosis = []
    for seq in tqdm.tqdm(tokens.cuda()):
        hidden_states = model(seq.unsqueeze(1), output_hidden_states=True).hidden_states
        for hidden_state in hidden_states:
            x = hidden_state.float()
            kurtosis.append(((x**4).mean(-1) / x.std(-1) ** 4 - 3).relu().mean().item())
    print(f"[*] Excess Kurtosis: {sum(kurtosis) / len(kurtosis):.2f}")
