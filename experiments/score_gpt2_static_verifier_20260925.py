"""Run exact, untruncated GPT-2 mean-token-logprob scoring on the frozen set."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform


def score(input_path: Path, output_path: Path, revision: str = "main", batch_size: int = 8) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    input_data = json.loads(input_path.read_text())
    items = input_data["items"]
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cpu")
    model.to(device).eval()
    pad_id = tokenizer.eos_token_id

    output_rows = []
    with torch.inference_mode():
        for offset in range(0, len(items), batch_size):
            batch = items[offset:offset + batch_size]
            texts = [item["text"] for item in batch]
            encoded = tokenizer(texts, add_special_tokens=False, padding=True,
                                truncation=False, return_tensors="pt")
            ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            if ids.shape[1] > model.config.n_positions:
                raise ValueError(f"passage exceeds GPT-2 context: {ids.shape[1]} > {model.config.n_positions}")
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
            targets = ids[:, 1:]
            predicted_mask = mask[:, 1:]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            for index, item in enumerate(batch):
                valid = predicted_mask[index].bool()
                predicted_tokens = int(valid.sum().item())
                if predicted_tokens == 0:
                    raise ValueError(f"passage has fewer than two GPT-2 tokens: {item['id']}")
                total = float(selected[index][valid].sum().item())
                output_rows.append({
                    "id": item["id"],
                    "kind": item["kind"],
                    "pair_id": item["pair_id"],
                    "text_sha256": item["text_sha256"],
                    "token_count": int(mask[index].sum().item()),
                    "predicted_token_count": predicted_tokens,
                    "total_logprob": total,
                    "mean_token_logprob": total / predicted_tokens,
                    "perplexity": float(torch.exp(torch.tensor(-total / predicted_tokens)).item()),
                })

    resolved_revision = getattr(model.config, "_commit_hash", None) or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if not resolved_revision:
        raise RuntimeError("Hugging Face did not report a resolved model commit; refusing unpinned result")
    result = {
        "experiment": input_data["experiment"],
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "model": {
            "repo_id": model_name,
            "revision": resolved_revision,
            "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash", resolved_revision),
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "device": "CPU on NUC bench",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
        },
        "method": {
            "metric": "mean causal-token log probability; higher is a better fit under GPT-2",
            "tokenizer": "GPT-2 fast byte-level BPE; add_special_tokens=False",
            "context": "untruncated; fail if passage exceeds model context",
            "prediction_positions": "scores tokens 2..N from preceding context; first token is not scored",
            "padding": "right-padding uses EOS with attention mask; padding positions are excluded",
            "temperature": None,
            "random_seed": None,
            "batch_size": batch_size,
        },
        "scores": output_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="Hub revision; result records the resolved immutable commit")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = score(args.input, args.output, args.revision, args.batch_size)
    print(json.dumps({"output": str(args.output), "items": len(result["scores"]), "revision": result["model"]["revision"], "mean_candidate_logprob": sum(row["mean_token_logprob"] for row in result["scores"] if row["kind"] == "candidate") / sum(row["kind"] == "candidate" for row in result["scores"])}, indent=2))


if __name__ == "__main__":
    main()
