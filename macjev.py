"""MacJev portable FP32 inference, derived from Apache-2.0 Laya components.

See LICENSE and NOTICE. This module ranks candidates; it never executes actions.
No dependency on the training project, private files, or upstream git checkout.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

QTYPES = {"choice": 0, "score": 1, "noul": 2}


class InputBudgetError(ValueError):
    pass


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def criterion(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def options(question):
    kind = question.get("t")
    if kind not in QTYPES or not isinstance(question.get("ins"), str):
        raise ValueError("Use t=choice/score/noul and a string ins")
    criteria = question.get("crit")
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria or not all(isinstance(k, str) for k in criteria):
            raise ValueError("Choice crit must be a nonempty ordered mapping with string keys")
        return [key if value is None or value == "" else f"{key}: {criterion(value)}" for key, value in criteria.items()]
    if kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("Score crit must be a nonempty ordered list")
        return [f"level {i}: {criterion(c)}" for i, c in enumerate(criteria)]
    criteria = criteria or {}
    if not isinstance(criteria, dict):
        raise ValueError("Noul crit must be a mapping, or omitted")
    return ["false: " + (criterion(criteria.get("false")) if criteria.get("false") not in (None, "") else "no, the statement does not hold"),
            "true: " + (criterion(criteria.get("true")) if criteria.get("true") not in (None, "") else "yes, the statement holds")]


def build_sequence(tokenizer, state, question, max_len=4096, head_max_len=1024):
    rendered = options(question)
    clean = lambda value: str(value).replace(tokenizer.mask_token, " ")
    serialized = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    encoded = tokenizer([f'{question["t"]} question: {clean(question["ins"])}',
                         *[" " + clean(text) for text in rendered], clean(serialized)],
                        add_special_tokens=False)["input_ids"]
    pieces = [[tokenizer.mask_token_id] + values for values in encoded[1:-1]]
    header_size = len(encoded[0]) + sum(map(len, pieces))
    if header_size > head_max_len:
        raise InputBudgetError(f"Question and complete options need {header_size} tokens; budget={head_max_len}")
    ids = [tokenizer.cls_token_id] + encoded[0] + [tokenizer.sep_token_id]
    markers = []
    for piece in pieces:
        markers.append(len(ids)); ids.extend(piece)
    ids += [tokenizer.sep_token_id] + encoded[-1] + [tokenizer.sep_token_id]
    if len(ids) > max_len:
        raise InputBudgetError(f"Complete input needs {len(ids)} tokens; budget={max_len}; nothing was silently truncated")
    return ids, markers


class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, n_act=2):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4*d, .1, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d+4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=~attention_mask.bool())
        index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = self.scorer(torch.gather(h, 1, index)).squeeze(-1).float().masked_fill(~marker_mask, -1e4)
        p = logits.detach().softmax(-1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(p * p.clamp_min(1e-9).log()).sum(-1) / k.log()
        top = p.topk(min(2, p.size(-1)), -1).values
        if top.size(-1) == 1:
            top = torch.cat([top, torch.zeros_like(top)], -1)
        features = torch.stack([top[:, 0], top[:, 0]-top[:, 1], entropy, k/255.], -1)
        return logits, self.act_head(torch.cat([h[:, 0].float(), features], -1))


class MacJev:
    def __init__(self, model_directory, device="cpu"):
        self.directory = Path(model_directory).resolve(strict=True)
        self.device = torch.device(device)
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS is unavailable; no silent fallback")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable; no silent fallback")
        manifest = json.loads((self.directory / "manifest.json").read_text())
        for relative, expected in manifest["files_sha256"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid model manifest path")
            resolved = (self.directory / path).resolve(strict=True)
            if not resolved.is_relative_to(self.directory) or file_hash(resolved) != expected:
                raise ValueError("Model package integrity mismatch: " + relative)
        self.config = json.loads((self.directory / "rl_agent_config.json").read_text())
        self.max_len = self.config["max_len"]; self.head_max_len = self.config["head_max_len"]
        if (self.max_len, self.head_max_len) != (4096, 1024):
            raise ValueError("Unexpected model input budget")
        self.temperatures = manifest["temperatures"]
        if set(self.temperatures) != set(QTYPES) or any(not math.isfinite(v) or v <= 0 for v in self.temperatures.values()):
            raise ValueError("Invalid calibration temperatures")
        if self.config["temperature"] != [self.temperatures[k] for k in QTYPES]:
            raise ValueError("Serving configuration and temperatures differ")
        raw_config = json.loads((self.directory / "encoder/config.json").read_text())
        config = AutoConfig.from_pretrained(self.directory / "encoder", local_files_only=True, trust_remote_code=False)
        if config.rope_parameters != raw_config["rope_parameters"] or config.layer_types != raw_config["layer_types"]:
            raise RuntimeError("Encoder attention/RoPE configuration changed; use the tested dependencies")
        config.reference_compile = False
        encoder = AutoModel.from_config(config, attn_implementation="sdpa")
        self.model = DecisionModel(encoder, self.config["head_layers"], len(self.config["act_costs"])+1)
        self.model.load_state_dict(load_file(self.directory / "model.safetensors"), strict=True)
        if sum(p.numel() for p in self.model.parameters()) != 321908995:
            raise RuntimeError("Unexpected model parameter count")
        self.model.float().to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.directory / "tokenizer", local_files_only=True, trust_remote_code=False)

    @torch.inference_mode()
    def decide(self, state, question):
        started = time.monotonic()
        ids, markers = build_sequence(self.tokenizer, state, question, self.max_len, self.head_max_len)
        batch = {"input_ids": torch.tensor([ids], device=self.device),
                 "attention_mask": torch.ones((1,len(ids)), dtype=torch.long, device=self.device),
                 "marker_pos": torch.tensor([markers], device=self.device),
                 "marker_mask": torch.ones((1,len(markers)), dtype=torch.bool, device=self.device),
                 "qtype": torch.tensor([QTYPES[question["t"]]], device=self.device)}
        logits, _ = self.model(**batch)
        probs = (logits[0] / self.temperatures[question["t"]]).softmax(-1).cpu().tolist()
        names = (list(question["crit"]) if question["t"] == "choice" else
                 [str(i) for i in range(len(question["crit"]))] if question["t"] == "score" else ["false", "true"])
        best = max(range(len(probs)), key=probs.__getitem__)
        return {"answer": names[best], "probabilities": dict(zip(names, probs)), "top_probability": probs[best],
                "input_tokens": len(ids), "latency_ms": (time.monotonic()-started)*1000,
                "model": "MacJev-322M-4K-Laya", "actions_executed": False}


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="One JSON object per line: state and question. No Mac actions are executed.")
    parser.add_argument("--model", default=".")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    model = MacJev(args.model, args.device)
    for line in sys.stdin:
        if line.strip():
            request = json.loads(line)
            print(json.dumps(model.decide(request["state"], request["question"]), ensure_ascii=False), flush=True)
