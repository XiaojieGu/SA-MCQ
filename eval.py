import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

TABLE13_LLM_JUDGE_PROMPT = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT"].
The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: Malia and Sasha Obama are the names of Barack Obama's children.
```
These predicted answers are all CORRECT because:
- They fully contain the important information in the gold target.
- They do not contain any information that contradicts the gold target.
The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Malia and Sasha, Malia and Sasha, Malia and Sasha, Malia and Sasha (repeated answer)
```
These predicted answers are all INCORRECT because:
- A factual statement in the answer contradicts the gold target or contain repeated answer.
Here is a sample. Simply reply with either CORRECT or INCORRECT.
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
According to the gold target, please grade the predicted answer of this question as one of:
A: CORRECT
B: INCORRECT
Just return the letters "A" or "B", with no text around it.
""".strip()

SPLITS = {
    "Eff.": ("src", "golden_answer"),
    "Gen.": ("rephrase", "golden_answer"),
    "Spe.": ("loc", "loc_ans"),
}

ALL_METRICS = {
    "exact_match_tf",
    "exact_match_wo_tf",
    "llm_as_judge",
    "likelihood_margin",
    "sa_mcq",
}
METRIC_ALIASES = {
    "all": "all",
    "em_tf": "exact_match_tf",
    "exact_match_tf": "exact_match_tf",
    "exact_match_w_tf": "exact_match_tf",
    "exact_match_wo_tf": "exact_match_wo_tf",
    "exact_match_without_tf": "exact_match_wo_tf",
    "llm_judge": "llm_as_judge",
    "llm_as_judge": "llm_as_judge",
    "likelihood": "likelihood_margin",
    "likelihood_margin": "likelihood_margin",
    "sa_mcq": "sa_mcq",
    "mcq": "sa_mcq",
}


def apply_chat_template(tok, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def build_sa_mcq_prompt(tok, record: Dict[str, Any], with_uncertain: bool) -> str:
    evidence = ""
    golden_evidence = record.get("golden_evidence")
    if isinstance(golden_evidence, dict):
        evidence = (golden_evidence.get("passage_1") or "").strip()

    question = (record.get("src") or "").strip()
    parametric_answer = (record.get("parametric_answer") or "").strip()
    golden_answer = (record.get("golden_answer") or record.get("ans") or "").strip()

    system_prompt = (
        "Based on the given information and your own knowledge, please select the "
        "option that best answers the question.\n"
        "Given information:\n"
        f"{evidence or '(none)'}"
    )
    if with_uncertain:
        system_prompt += "\nYou may choose C if you are truly uncertain; otherwise choose between A or B."
        user_prompt = (
            f"Question: {question}\n"
            f"A. {parametric_answer}\n"
            f"B. {golden_answer}\n"
            "C. I am uncertain / not sure\n"
            "Answer with only the letter (A, B, or C)."
        )
    else:
        user_prompt = (
            f"Question: {question}\n"
            f"A. {parametric_answer}\n"
            f"B. {golden_answer}\n"
            "Answer with only the letter (A or B)."
        )
    return apply_chat_template(tok, system_prompt, user_prompt)


def parse_choice(text: str) -> str:
    first_line = (text or "").strip().upper().splitlines()[0] if (text or "").strip() else ""
    match = re.search(r"\b([ABC])\b", first_line)
    if match:
        return match.group(1)
    match = re.search(r"\b([ABC])\b", (text or "").upper())
    return match.group(1) if match else "?"


def format_chat_prompt_new(tok, user_prompt: str) -> str:
    question = user_prompt.strip()
    messages = [
        {
            "role": "user",
            "content": (
                "Refer to your own knowledge to answer the following question.\n"
                "Only output the final answer. Do NOT include explanations, reasoning steps, or any additional text.\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            ),
        }
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (
            "User:\n"
            "Refer to your own knowledge to answer the following question.\n"
            "Only output the final answer. Do NOT include explanations, reasoning steps, or any additional text.\n\n"
            f"Question: {question}\n\n"
            "Answer:\nAssistant:\n"
        )


def load_records(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for row in raw:
        rec = dict(row)
        if not rec.get("golden_answer"):
            rec["golden_answer"] = (rec.get("ans") or "").strip()
        if not rec.get("src"):
            rec["src"] = (rec.get("question") or "").strip()
        records.append(rec)
    return records


def collect_split(records: List[Dict[str, Any]], split_name: str) -> Tuple[List[str], List[str]]:
    qk, ak = SPLITS[split_name]
    prompts, targets = [], []
    for r in records:
        q = (r.get(qk) or "").strip()
        a = (r.get(ak) or "").strip()
        if q and a:
            prompts.append(q)
            targets.append(a)
    return prompts, targets


def parse_metrics(metrics_arg: str) -> set[str]:
    requested = []
    for raw_name in metrics_arg.split(","):
        name = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
        if not name:
            continue
        if name not in METRIC_ALIASES:
            valid = ", ".join(sorted(METRIC_ALIASES))
            raise ValueError(f"Unknown metric '{raw_name}'. Valid values: {valid}")
        requested.append(METRIC_ALIASES[name])
    if not requested or "all" in requested:
        return set(ALL_METRICS)
    return set(requested)


def run_exact_match_wo_tf(model, tok, device, prompts, targets) -> float:
    model.eval()
    max_input_tokens = 4096
    sentence_ok = []
    with torch.no_grad():
        for p, t in zip(prompts, targets):
            p_fmt = format_chat_prompt_new(tok, p)
            p_ids = tok(p_fmt, add_special_tokens=False, truncation=True, max_length=max_input_tokens).input_ids
            pt_ids = tok(p_fmt + t, add_special_tokens=False, truncation=True, max_length=max_input_tokens).input_ids
            gold = pt_ids[len(p_ids) :]
            if not gold:
                continue
            inputs = tok(p_fmt, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
            outs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=len(gold),
                min_new_tokens=len(gold),
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
            pred = outs[0][inputs["input_ids"].shape[1] :].tolist()
            L = min(len(gold), len(pred))
            sentence_ok.append(1.0 if L > 0 and all(gold[i] == pred[i] for i in range(L)) else 0.0)
    return float(sum(sentence_ok) / max(1, len(sentence_ok))) * 100.0


def run_exact_match_with_tf(model, tok, device, prompts, targets) -> float:
    model.eval()
    max_input_tokens = 4096
    sentence_ok = []
    with torch.no_grad():
        for p, t in zip(prompts, targets):
            p_fmt = format_chat_prompt_new(tok, p)
            p_ids = tok(p_fmt, add_special_tokens=False, truncation=True, max_length=max_input_tokens).input_ids
            pt_ids = tok(p_fmt + t, add_special_tokens=False, truncation=True, max_length=max_input_tokens).input_ids
            if len(pt_ids) <= len(p_ids):
                continue
            input_ids = torch.tensor([pt_ids], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[0]
            pred = torch.argmax(logits, dim=-1)
            start = len(p_ids)
            gold = pt_ids[start:]
            pred_target = [int(pred[k - 1].item()) for k in range(start, len(pt_ids))]
            if not gold or not pred_target:
                continue
            L = min(len(gold), len(pred_target))
            sentence_ok.append(1.0 if all(gold[i] == pred_target[i] for i in range(L)) else 0.0)
    return float(sum(sentence_ok) / max(1, len(sentence_ok))) * 100.0


def generate_predictions(model, tok, device, prompts: List[str], max_new_tokens: int = 32) -> List[str]:
    preds: List[str] = []
    batch_size = 32
    model.eval()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        texts = [format_chat_prompt_new(tok, p) for p in batch]
        inputs = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
        for b in range(outs.shape[0]):
            cont = outs[b][inputs["input_ids"].shape[1] :]
            preds.append(tok.decode(cont, skip_special_tokens=True).strip())
    return preds


def run_sa_mcq(model, tok, device, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        r for r in records
        if (r.get("src") or "").strip()
        and (r.get("golden_answer") or r.get("ans") or "").strip()
        and (r.get("parametric_answer") or "").strip()
    ]
    results = {}
    for mode_name, with_uncertain in [
        ("Table 11 w/ uncertain", True),
        ("Table 12 w/o uncertain", False),
    ]:
        counts = {"A": 0, "B": 0, "C": 0, "?": 0}
        prompts = [build_sa_mcq_prompt(tok, r, with_uncertain=with_uncertain) for r in valid]
        batch_size = 32
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=8,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.eos_token_id,
                )
            for b in range(outs.shape[0]):
                cont = outs[b][inputs["input_ids"].shape[1] :]
                choice = parse_choice(tok.decode(cont, skip_special_tokens=True))
                counts[choice if choice in counts else "?"] += 1

        n = max(1, len(valid))
        results[mode_name] = {
            "A_parametric_rate": counts["A"] / n * 100.0,
            "B_golden_rate": counts["B"] / n * 100.0,
            "C_uncertain_rate": counts["C"] / n * 100.0,
        }
    return results


def run_llm_as_judge(api_key: str, prompts: List[str], targets: List[str], preds: List[str], max_workers: int) -> float:
    if not api_key:
        return float("nan")

    def judge_one(q: str, t: str, p: str) -> float:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        content = TABLE13_LLM_JUDGE_PROMPT.format(question=q, target=t, predicted_answer=p)
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
            )
            ans = (resp.choices[0].message.content or "").strip().upper()
            return 1.0 if ans == "A" else 0.0
        except Exception:
            return 0.0

    scores = [0.0] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(judge_one, q, t, p): i
            for i, (q, t, p) in enumerate(zip(prompts, targets, preds))
        }
        for future in as_completed(future_to_idx):
            scores[future_to_idx[future]] = future.result()
    return float(sum(scores) / max(1, len(scores))) * 100.0


def mean_logprob(model, tok, device, prompts: List[str], targets: List[str]) -> float:
    model.eval()
    max_length = 4096
    batch_logprobs = []
    with torch.no_grad():
        for p, t in zip(prompts, targets):
            p_fmt = format_chat_prompt_new(tok, p)
            p_ids = tok(p_fmt, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
            pt_ids = tok(p_fmt + t, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
            target_ids = pt_ids[len(p_ids) :]
            if not target_ids:
                continue
            input_ids = torch.tensor([pt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            token_logprobs = []
            for pos in range(len(p_ids), len(pt_ids)):
                logp = torch.log_softmax(logits[pos - 1], dim=-1)
                token_logprobs.append(float(logp[pt_ids[pos]].item()))
            if token_logprobs:
                batch_logprobs.append(sum(token_logprobs) / len(token_logprobs))
    return float(sum(batch_logprobs) / max(1, len(batch_logprobs)))


def evaluate_model_metrics(model, tok, device, records, api_key: str, judge_workers: int, metrics: set[str]) -> Dict[str, Any]:
    em_tf = {}
    em_wo = {}
    judge = {}
    for split in ["Eff.", "Gen.", "Spe."]:
        prompts, targets = collect_split(records, split)
        if "exact_match_tf" in metrics:
            em_tf[split] = run_exact_match_with_tf(model, tok, device, prompts, targets)
        if "exact_match_wo_tf" in metrics:
            em_wo[split] = run_exact_match_wo_tf(model, tok, device, prompts, targets)
        if "llm_as_judge" in metrics:
            preds = generate_predictions(model, tok, device, prompts, max_new_tokens=32)
            judge[split] = run_llm_as_judge(api_key, prompts, targets, preds, max_workers=judge_workers)

    result = {}
    if "exact_match_tf" in metrics:
        result["Exact Match w/ TF"] = em_tf
    if "exact_match_wo_tf" in metrics:
        result["Exact Match w/o TF"] = em_wo
    if "llm_as_judge" in metrics:
        result["LLM-as-judge"] = judge
    if "sa_mcq" in metrics:
        result["SA-MCQ"] = run_sa_mcq(model, tok, device, records)
    return result


def compute_likelihood_margin(vanilla_model, vanilla_tok, edited_model, edited_tok, device, records) -> Dict[str, float]:
    eff_p, eff_gold = collect_split(records, "Eff.")
    eff_base = generate_predictions(vanilla_model, vanilla_tok, device, eff_p, max_new_tokens=16)

    eff_margin = mean_logprob(edited_model, edited_tok, device, eff_p, eff_gold) - mean_logprob(
        edited_model, edited_tok, device, eff_p, eff_base
    )
    return {
        "Eff.": eff_margin,
        "definition": "log P_edited(golden_answer | src) - log P_edited(vanilla_answer | src)",
        "prompt_field": "src",
        "target_field": "golden_answer",
    }


def load_model(model_path: str, device: torch.device):
    model_path = normalize_model_path(model_path)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def normalize_model_path(model_path: str) -> str:
    prefixes = ("https://huggingface.co/", "http://huggingface.co/")
    for prefix in prefixes:
        if model_path.startswith(prefix):
            return model_path[len(prefix) :].strip("/")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 1 metrics with Eff/Gen/Spe mapping.")
    parser.add_argument("--vanilla_model", required=True, help="Vanilla model path or HF repo id.")
    parser.add_argument("--edited_model", required=True, help="Edited model path or HF repo id.")
    parser.add_argument("--data_path", default="./zsre_966.json", help="Dataset path.")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""), help="API key for LLM judge.")
    parser.add_argument("--judge_workers", type=int, default=100, help="Concurrent workers for LLM-as-judge.")
    parser.add_argument(
        "--metrics",
        default="all",
        help=(
            "Comma-separated metrics to run. Options: all, exact_match_tf, "
            "exact_match_wo_tf, llm_as_judge, likelihood_margin, sa_mcq."
        ),
    )
    parser.add_argument("--output_path", default="eval_compare_results.json", help="Output JSON path.")
    args = parser.parse_args()
    selected_metrics = parse_metrics(args.metrics)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script.")
    device = torch.device("cuda")
    records = load_records(args.data_path)

    vanilla_model, vanilla_tok = load_model(args.vanilla_model, device)
    edited_model, edited_tok = load_model(args.edited_model, device)

    vanilla_metrics = evaluate_model_metrics(
        vanilla_model, vanilla_tok, device, records, args.api_key, args.judge_workers, selected_metrics
    )
    edited_metrics = evaluate_model_metrics(
        edited_model, edited_tok, device, records, args.api_key, args.judge_workers, selected_metrics
    )
    if "likelihood_margin" in selected_metrics:
        vanilla_metrics["Likelihood Margin"] = {
            "Eff.": None,
            "definition": "Not defined for the vanilla baseline.",
            "prompt_field": "src",
            "target_field": "golden_answer",
        }
        edited_metrics["Likelihood Margin"] = compute_likelihood_margin(
            vanilla_model, vanilla_tok, edited_model, edited_tok, device, records
        )

    result = {
        "meta": {
            "data_path": args.data_path,
            "selected_metrics": sorted(selected_metrics),
            "mapping": {
                "Eff.": "src + golden_answer",
                "Gen.": "rephrase + golden_answer",
                "Spe.": "loc + loc_ans",
                "Likelihood Margin": "src + golden_answer",
                "SA-MCQ evidence": "golden_evidence.passage_1",
                "SA-MCQ question": "src",
            },
            "llm_judge_prompt_source": "Table 13, arXiv:2604.05995",
            "sa_mcq_template_source": "Table 11 and Table 12, arXiv:2604.05995",
        },
        "vanilla": vanilla_metrics,
        "edited": edited_metrics,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
