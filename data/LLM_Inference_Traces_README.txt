================================================================================
         PUBLIC LLM INFERENCE TRACE DATASETS — OVERVIEW & REFERENCE
================================================================================

Last updated: June 2026

This document summarizes publicly available datasets containing traces of
Large Language Model (LLM) inference workloads. All datasets listed include
at minimum the number of input and output tokens per request. Datasets are
grouped by whether token counts are natively provided or must be derived
from conversation text via tokenization.


================================================================================
 1. DATASETS WITH NATIVE TOKEN COUNTS (ready to use)
================================================================================

--------------------------------------------------------------------------------
 1.1  Azure LLM Inference Traces (2023, 2024, 2025)
--------------------------------------------------------------------------------

Source:       Microsoft Azure / Microsoft Research
License:      CC-BY Attribution
Repository:   https://github.com/Azure/AzurePublicDataset

Production traces from Azure LLM inference services. Fully anonymized metadata
only (no conversation content). All three releases share a common schema:

  TIMESTAMP        — Request invocation time
  ContextTokens    — Number of input (context) tokens
  GeneratedTokens  — Number of output tokens

The 2025 multimodal trace adds:
  NumImages        — Number of images in the request

Each trace is split by workload type (coding vs. conversation), except the
2025 release which covers a single multimodal service.

  [2023 — Splitwise / ISCA'24]
    Duration:     ~20 minutes, collected November 11, 2023
    Files:        data/AzureLLMInferenceTrace_code.csv
                  data/AzureLLMInferenceTrace_conv.csv
    Description:  https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md
    Paper:        Patel et al., "Splitwise: Efficient generative LLM inference
                  using phase splitting", ISCA 2024.

  [2024 — DynamoLLM / HPCA'25]
    Duration:     ~10 days, collected May 10-19, 2024
    Files:        Hosted on Azure Blob Storage (not in the Git repo):
                  https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv
                  https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv
    Description:  https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md
    Paper:        Stojkovic et al., "DynamoLLM: Designing LLM Inference Clusters
                  for Performance and Energy Efficiency", HPCA 2025.

  [2025 — ModServe / SoCC'25]  (multimodal)
    Duration:     ~1 week, collected October 15-22, 2024
    Files:        data/AzureLMMInferenceTrace_multimodal.csv.gz
                  (unzip with: gunzip -k AzureLMMInferenceTrace_multimodal.csv.gz)
    Description:  https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md
    Paper:        Qiu et al., "ModServe: Modality- and Stage-Aware Resource
                  Disaggregation for Scalable Multimodal Model Serving", SoCC 2025.

NOTE: The 2025 trace covers a Large Multimodal Model (LMM), not a text-only
LLM. The ContextTokens field includes both text and image tokens.


--------------------------------------------------------------------------------
 1.2  BurstGPT
--------------------------------------------------------------------------------

Source:       HPMLL Lab (HKUST Guangzhou / Harbin Institute of Technology)
License:      MIT
Repository:   https://github.com/HPMLL/BurstGPT
HuggingFace:  https://huggingface.co/datasets/lzzmm/BurstGPT
Paper:        Wang et al., "BurstGPT: A Real-world Workload Dataset to
              Optimize LLM Serving Systems", KDD 2025.

The largest publicly available LLM inference trace dataset by volume and
duration. Contains real-world workload traces from Azure OpenAI GPT services
(ChatGPT / GPT-3.5 and GPT-4). Fully anonymized, no conversation content.

  Scale:        ~10.31 million traces (~188 MB)
  Duration:     213 consecutive days (2023-2024)
  Timestamp:    Yes (seconds from 0:00:00 on the first day)

  Schema:
    Timestamp        — Request submission time
    Model            — Model type (ChatGPT or GPT-4)
    Request tokens   — Number of input tokens
    Response tokens  — Number of output tokens
    Total tokens     — Sum of request + response tokens
    Log Type         — Service type (API or Conversation)

  Files (Release v1.1):
    BurstGPT_1.csv                — First 2 months, includes failures (~1.43M rows)
    BurstGPT_without_fails_1.csv  — First 2 months, failures removed (~1.40M rows)
    BurstGPT_2.csv                — Next 2 months, includes failures (~3.86M rows)
    BurstGPT_without_fails_2.csv  — Next 2 months, failures removed (~3.78M rows)

  NOTE: Traces with additional SessionID and Elapsed time columns are
  announced as forthcoming.


================================================================================
 2. DATASETS WITH DERIVABLE TOKEN COUNTS (tokenization required)
================================================================================

These datasets contain full conversation text. Token counts can be computed
by applying a tokenizer (e.g., tiktoken, Llama tokenizer) to the input/output
text fields.


--------------------------------------------------------------------------------
 2.1  WildChat
--------------------------------------------------------------------------------

Source:       Allen Institute for AI / University of Washington
License:      ODC-BY
HuggingFace:  https://huggingface.co/datasets/allenai/WildChat-1M   (1M, non-toxic)
              https://huggingface.co/datasets/allenai/WildChat-4.8M  (4.8M, non-toxic)
              Full versions with toxic content available with approval.
Paper:        Zhao et al., "WildChat: 1M ChatGPT Interaction Logs in the
              Wild", NAACL 2024.

Real-world user-ChatGPT conversations collected with explicit user consent.

  Scale:        ~1M conversations (original), ~4.8M (extended through Jul 2025)
  Duration:     April 2023 – July 2025 (4.8M version)
  Timestamp:    Yes — per-conversation and per-turn timestamps (UTC)
  Models:       GPT-3.5-Turbo, GPT-4

  Key fields:
    conversation_hash  — Content hash (not unique ID)
    turn_identifier    — Unique per-turn identifier
    model              — OpenAI model name
    timestamp          — Timestamp of last turn (UTC)
    conversation       — List of user/assistant turns with text content
    language           — Detected language
    country / state    — Inferred from IP address
    hashed_ip          — Anonymized IP for session linking
    toxicity flags     — OpenAI Moderation API and Detoxify annotations

  Avg. user prompt:   ~295 tokens
  Languages:          68+ detected


--------------------------------------------------------------------------------
 2.2  LMSYS-Chat-1M
--------------------------------------------------------------------------------

Source:       LMSYS.org (UC Berkeley)
License:      Custom (research + commercial use permitted)
HuggingFace:  https://huggingface.co/datasets/lmsys/lmsys-chat-1m
Paper:        Zheng et al., "LMSYS-Chat-1M: A Large-Scale Real-World LLM
              Conversation Dataset", ICLR 2024.

Conversations from the Chatbot Arena and Vicuna demo platforms.

  Scale:        1 million conversations, ~2.5 million turns
  Duration:     April – August 2023
  Timestamp:    NO (not included in public release)
  Models:       25 LLMs (Vicuna, Llama, GPT-3.5-Turbo, GPT-4, Claude-2, etc.)

  Key fields:
    conversation_id    — Unique conversation identifier
    model              — Model name
    conversation       — JSON transcript (OpenAI Chat API format)
    language           — Auto-detected language tag
    openai_moderation  — OpenAI Moderation API flags
    redacted           — Whether PII redaction was applied

  Avg. user prompt:   ~69.5 tokens (Llama-2 tokenizer)
  Avg. model response: ~214.5 tokens
  Languages:          150+

  NOTE: Requires acceptance of license agreement on HuggingFace.
  Raw (unredacted) data available by separate application.


--------------------------------------------------------------------------------
 2.3  ShareChat
--------------------------------------------------------------------------------

Source:       Indiana University Bloomington
License:      CC-BY-NC 4.0
Paper:        Yan et al., "ShareChat: A Dataset of Chatbot Conversations in
              the Wild", arXiv:2512.17843.

Cross-platform corpus collected from publicly shared conversation URLs.

  Scale:        ~142,808 conversations, ~660,000 turns
  Duration:     April 2023 – October 2025
  Timestamp:    NO
  Platforms:    ChatGPT, Claude, Gemini, Perplexity, Grok

  Key fields:
    Conversation text with platform-native affordances preserved
    (reasoning traces, source links, code artifacts)
    Language detection (101 languages)
    Toxicity annotations

  Avg. model response: ~1,115 tokens
  Languages:           101

  NOTE: Non-commercial license (NC). Preserves platform-specific features
  such as reasoning traces and artifact blocks that other datasets strip out.