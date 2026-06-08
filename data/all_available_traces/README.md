## Download the Traces

**Notes**

LMSYS and ShareChat require Hugging Face token and to accept the license to access the dataset and also to share your contact information:

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (Read access is enough)
3. Accept the LMSYS license at https://huggingface.co/datasets/lmsys/lmsys-chat-1m or https://huggingface.co/datasets/tucnguyen/ShareChat
4. Set it before running:
```powershell
$env:HF_TOKEN = "hf_xxxx"   # PowerShell
```

**Download datasets**

```powershell
# Download all datasets (from within the `all_available_traces` folder)
python download_traces.py
```


## Public LLM Inference Trace Datasets — Overview & Reference

**Last updated:** June 2026

This document summarizes publicly available datasets containing traces of Large Language Model (LLM) inference workloads. All datasets listed include, at minimum, the number of input and output tokens per request.

Datasets are grouped by whether token counts are natively provided or must be derived from conversation text via tokenization.

---

## 1. Datasets with Native Token Counts

*Ready to use.*

### 1.1 Azure LLM Inference Traces

**Releases:** 2023, 2024, 2025  
**Source:** Microsoft Azure / Microsoft Research  
**License:** CC-BY Attribution  
**Repository:** https://github.com/Azure/AzurePublicDataset

Production traces from Azure LLM inference services. Fully anonymized metadata only, with no conversation content.

All three releases share a common schema:

| Field | Description |
|---|---|
| `TIMESTAMP` | Request invocation time |
| `ContextTokens` | Number of input/context tokens |
| `GeneratedTokens` | Number of output tokens |

The 2025 multimodal trace adds:

| Field | Description |
|---|---|
| `NumImages` | Number of images in the request |

Each trace is split by workload type, coding vs. conversation, except the 2025 release, which covers a single multimodal service.

#### 2023 — Splitwise / ISCA’24

| Item | Details |
|---|---|
| Duration | ~20 minutes, collected November 11, 2023 |
| Files | `data/AzureLLMInferenceTrace_code.csv`<br>`data/AzureLLMInferenceTrace_conv.csv` |
| Description | https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md |
| Paper | Patel et al., “Splitwise: Efficient generative LLM inference using phase splitting”, ISCA 2024. |

#### 2024 — DynamoLLM / HPCA’25

| Item | Details |
|---|---|
| Duration | ~10 days, collected May 10–19, 2024 |
| Files | Hosted on Azure Blob Storage, not in the Git repo:<br>https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv<br>https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv |
| Description | https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md |
| Paper | Stojkovic et al., “DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency”, HPCA 2025. |

#### 2025 — ModServe / SoCC’25

*Multimodal.*

| Item | Details |
|---|---|
| Duration | ~1 week, collected October 15–22, 2024 |
| Files | `data/AzureLMMInferenceTrace_multimodal.csv.gz`<br>Unzip with: `gunzip -k AzureLMMInferenceTrace_multimodal.csv.gz` |
| Description | https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md |
| Paper | Qiu et al., “ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving”, SoCC 2025. |

**Note:** The 2025 trace covers a Large Multimodal Model (LMM), not a text-only LLM. The `ContextTokens` field includes both text and image tokens.

---

### 1.2 BurstGPT

**Source:** HPMLL Lab, HKUST Guangzhou / Harbin Institute of Technology  
**License:** MIT  
**Repository:** https://github.com/HPMLL/BurstGPT  
**HuggingFace:** https://huggingface.co/datasets/lzzmm/BurstGPT  
**Paper:** Wang et al., “BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems”, KDD 2025.

The largest publicly available LLM inference trace dataset by volume and duration. Contains real-world workload traces from Azure OpenAI GPT services, ChatGPT / GPT-3.5 and GPT-4. Fully anonymized, with no conversation content.

| Item | Details |
|---|---|
| Scale | ~10.31 million traces, ~188 MB |
| Duration | 213 consecutive days, 2023–2024 |
| Timestamp | Yes, seconds from 0:00:00 on the first day |

#### Schema

| Field | Description |
|---|---|
| `Timestamp` | Request submission time |
| `Model` | Model type, ChatGPT or GPT-4 |
| `Request tokens` | Number of input tokens |
| `Response tokens` | Number of output tokens |
| `Total tokens` | Sum of request + response tokens |
| `Log Type` | Service type, API or Conversation |

#### Files, Release v1.1

| File | Description |
|---|---|
| `BurstGPT_1.csv` | First 2 months, includes failures, ~1.43M rows |
| `BurstGPT_without_fails_1.csv` | First 2 months, failures removed, ~1.40M rows |
| `BurstGPT_2.csv` | Next 2 months, includes failures, ~3.86M rows |
| `BurstGPT_without_fails_2.csv` | Next 2 months, failures removed, ~3.78M rows |

**Note:** Traces with additional `SessionID` and `Elapsed time` columns are announced as forthcoming.

---

## 2. Datasets with Derivable Token Counts

*Tokenization required.*

These datasets contain full conversation text. Token counts can be computed by applying a tokenizer, such as `tiktoken` or a Llama tokenizer, to the input/output text fields.

---

### 2.1 WildChat

**Source:** Allen Institute for AI / University of Washington  
**License:** ODC-BY  
**HuggingFace:**  
https://huggingface.co/datasets/allenai/WildChat-1M — 1M, non-toxic  
https://huggingface.co/datasets/allenai/WildChat-4.8M — 4.8M, non-toxic  
Full versions with toxic content available with approval.  
**Paper:** Zhao et al., “WildChat: 1M ChatGPT Interaction Logs in the Wild”, NAACL 2024.

Real-world user-ChatGPT conversations collected with explicit user consent.

| Item | Details |
|---|---|
| Scale | ~1M conversations original; ~4.8M extended through Jul 2025 |
| Duration | April 2023 – July 2025, 4.8M version |
| Timestamp | Yes — per-conversation and per-turn timestamps, UTC |
| Models | GPT-3.5-Turbo, GPT-4 |

#### Key Fields

| Field | Description |
|---|---|
| `conversation_hash` | Content hash, not unique ID |
| `turn_identifier` | Unique per-turn identifier |
| `model` | OpenAI model name |
| `timestamp` | Timestamp of last turn, UTC |
| `conversation` | List of user/assistant turns with text content |
| `language` | Detected language |
| `country / state` | Inferred from IP address |
| `hashed_ip` | Anonymized IP for session linking |
| `toxicity flags` | OpenAI Moderation API and Detoxify annotations |

| Metric | Value |
|---|---|
| Avg. user prompt | ~295 tokens |
| Languages | 68+ detected |

---

### 2.2 LMSYS-Chat-1M

**Source:** LMSYS.org, UC Berkeley  
**License:** Custom, research + commercial use permitted  
**HuggingFace:** https://huggingface.co/datasets/lmsys/lmsys-chat-1m  
**Paper:** Zheng et al., “LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset”, ICLR 2024.

Conversations from the Chatbot Arena and Vicuna demo platforms.

| Item | Details |
|---|---|
| Scale | 1 million conversations, ~2.5 million turns |
| Duration | April – August 2023 |
| Timestamp | No, not included in public release |
| Models | 25 LLMs, including Vicuna, Llama, GPT-3.5-Turbo, GPT-4, Claude-2, etc. |

#### Key Fields

| Field | Description |
|---|---|
| `conversation_id` | Unique conversation identifier |
| `model` | Model name |
| `conversation` | JSON transcript, OpenAI Chat API format |
| `language` | Auto-detected language tag |
| `openai_moderation` | OpenAI Moderation API flags |
| `redacted` | Whether PII redaction was applied |

| Metric | Value |
|---|---|
| Avg. user prompt | ~69.5 tokens, Llama-2 tokenizer |
| Avg. model response | ~214.5 tokens |
| Languages | 150+ |

**Note:** Requires acceptance of license agreement on HuggingFace. Raw, unredacted data available by separate application.

---

### 2.3 ShareChat

**Source:** Indiana University Bloomington  
**License:** CC-BY-NC 4.0  
**Paper:** Yan et al., “ShareChat: A Dataset of Chatbot Conversations in the Wild”, arXiv:2512.17843.

Cross-platform corpus collected from publicly shared conversation URLs.

| Item | Details |
|---|---|
| Scale | ~142,808 conversations, ~660,000 turns |
| Duration | April 2023 – October 2025 |
| Timestamp | No |
| Platforms | ChatGPT, Claude, Gemini, Perplexity, Grok |

#### Key Fields

Conversation text with platform-native affordances preserved, including:

| Field / Feature | Description |
|---|---|
| Conversation text | Platform-native affordances preserved |
| Reasoning traces | Preserved where present |
| Source links | Preserved where present |
| Code artifacts | Preserved where present |
| Language detection | 101 languages |
| Toxicity annotations | Included |

| Metric | Value |
|---|---|
| Avg. model response | ~1,115 tokens |
| Languages | 101 |

**Note:** Non-commercial license (NC). Preserves platform-specific features such as reasoning traces and artifact blocks that other datasets strip out.