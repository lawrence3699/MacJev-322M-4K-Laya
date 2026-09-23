---
license: apache-2.0
base_model: convaiinnovations/laya
base_model_relation: finetune
language:
- en
- zh
pipeline_tag: text-classification
tags:
- macos
- jev
- laya
- decision-model
- tool-routing
- local-agents
- safetensors
---

# MacJev-322M-4K-Laya

**A compact decision model for local Mac agents: one forward pass, calibrated probabilities, 4096-token inputs.**

MacJev takes an observed state and a question with candidate answers. In a single non-autoregressive forward pass it returns a probability for every candidate. Use it to rank candidate actions, route tool calls and check task state, in Chinese or English, entirely on your machine.

![MacJev vs. the Laya multilingual checkpoint it starts from](assets/macjev_vs_laya_multilingual.png)

## Highlights

Compared with the Laya multilingual checkpoint it was trained from, with the same 4096-token input budget, on held-out test sets:

| Task | Laya multilingual | MacJev |
|---|---:|---:|
| Yes/no checks on 2K–4K-token inputs | 31.0% | **89.1%** |
| Rule decisions on 2K–4K-token inputs | 23.7% | **44.7%** |
| Rule decisions on long inputs, all lengths | 23.2% | **39.7%** |
| Typed decisions | 35.2% | **42.2%** |
| Calibration error on long inputs, lower is better | 0.253 | **0.032** |

- **Reads long observations.** Yes/no task-state checks on 2K–4K-token inputs rise from 31% to 89% accuracy.
- **Probabilities you can threshold.** Calibration error on long inputs is about 8 times lower, so stated confidence closely tracks actual accuracy.
- **Better on public benchmarks too.** Across 11,600 public decisions from typed decisions, Emotion and AG News, accuracy rises by 1.5 points, with a 95% interval of 1.1 to 1.8.
- **Reproducible.** The long-input and public-benchmark gains appear in all three independently trained seeds. This release is the seed chosen in advance on development data.

## Why MacJev for agents

- **One pass, every candidate.** All options are scored together in one forward pass, with no token generation.
- **Your answer set, per request.** Choice, ordered Score and yes/no Noul questions take options you define at request time. No classifier head to retrain.
- **4× more room.** 4096 tokens of total input and 1024 for the question and options, four times the defaults of the checkpoint it starts from.
- **Nothing silently dropped.** Inputs over budget raise an error instead of quietly cutting candidates or evidence.
- **Calibrated per question type.** One temperature each for Choice, Score and Noul, fitted on 6,000 independent calibration decisions.
- **Trained for the Mac.** 20,000 Mac decisions built from verified trajectories across files, Chrome, Word, Excel and PowerPoint, including recovery cases, alongside general-language, typed and rule decisions.
- **Local and decision-only.** No API calls. MacJev returns decisions and never executes actions, so your agent keeps its own executor and confirmation step.

## Formats

| Format | Repository | Runs on |
|---|---|---|
| PyTorch FP32 | this repository | CPU, Apple silicon (MPS), CUDA |
| MLX FP32 | [MacJev-322M-4K-Laya-MLX](https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya-MLX) | Apple silicon, no PyTorch needed |
| GGUF F16 | [MacJev-322M-4K-Laya-GGUF](https://huggingface.co/chaoliangUNSW/MacJev-322M-4K-Laya-GGUF) | Stock llama.cpp `llama-server`, including the one bundled with LM Studio |

MacJev is a decision model rather than a chat model, so each format ships with a small runtime that runs the full model, including its decision head. To use MacJev from a chat assistant such as LM Studio, expose `decide()` as a tool or MCP server.

## Quickstart

```bash
hf download chaoliangUNSW/MacJev-322M-4K-Laya --local-dir MacJev
cd MacJev
python -m pip install -r requirements.txt
```

```python
from macjev import MacJev

model = MacJev(".", device="mps")  # Apple silicon; use "cpu" or "cuda" elsewhere
result = model.decide(
    state={
        "request": "打开浏览器",
        "available_actions": ["open_browser", "copy_file", "ask_user"],
    },
    question={
        "t": "choice",
        "ins": "Which available action best matches the user's request?",
        "crit": {
            "open_browser": "Open the user's browser",
            "copy_file": "Copy a file inside the approved folder",
            "ask_user": "Ask for clarification",
        },
    },
)
print(result["answer"], result["probabilities"])  # A decision only. Nothing is executed.
```

The result contains `answer`, `probabilities`, `top_probability`, `input_tokens`, `latency_ms`, and `actions_executed: False`. Keep one model object loaded across requests.

Other question shapes:

```python
score = {"t": "score", "ins": "How complete is the observed task?",
         "crit": ["Not started", "Partly complete", "Complete and verified"]}
boolean = {"t": "noul", "ins": "Does the observation prove that the requested file exists?"}
```

Score answers are ordered string indices (`"0"`, `"1"`, ...). Noul answers are `"false"` and `"true"`. Choice keys keep their insertion order.

The CLI reads one JSON object per line with `state` and `question`:

```bash
python macjev.py --model . --device mps
```

## Training

- **Base:** `convaiinnovations/laya`, `multilingual` subfolder, revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. Encoder lineage `jhu-clsp/mmBERT-base`.
- **Size:** 321,908,995 parameters; 44,861,185 adapted. The last six encoder layers, type embeddings, decision transformer and scorer were trained, starting from the official decision head.
- **Data:** 50,000 decisions split evenly between short and long inputs: 25,000 up to 1K tokens, 12,500 from 1K to 2K and 12,500 from 2K to 4K. A separate 3,000 development and 6,000 calibration decisions.
- **Objective:** supervised fine-tuning with an ordinal loss, replay of verified original-model outputs, and candidate-permutation consistency, trained in stages from 1K to 4K inputs.
- **Selection:** recipes were compared on development data, then three seeds were trained from the official weights. Seed 0 was designated in advance; test results were never used to choose it.
- **Hardware:** one NVIDIA H100, FP16 encoder arithmetic with an FP32 decision head and master weights. Released weights are FP32.

`manifest.json` lists every file's checksum, and the loader verifies them. `validation.json` records the loading and integrity checks.

## License and attribution

Apache-2.0, inherited from Laya; see `LICENSE` and `NOTICE`. The mmBERT-base backbone is MIT-licensed; see `MMBERT_LICENSE`. MacJev is an independent project. It is not affiliated with or endorsed by JEV or the Laya authors, and it contains no JEV weights.

- [Laya model](https://huggingface.co/convaiinnovations/laya) and [source](https://github.com/NandhaKishorM/laya)
- [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)

## Community

Interested in compact decision models, local agents, and practical macOS automation? Join our [Discord community](https://discord.gg/udfvMu2GM).

如果你也对轻量决策模型、本地 Agent 和 macOS 自动化感兴趣，欢迎加入我们的 [Discord 社区](https://discord.gg/udfvMu2GM)。
