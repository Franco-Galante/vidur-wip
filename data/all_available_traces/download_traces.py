#!/usr/bin/env python3
"""
Download all publicly available LLM inference trace datasets into subfolders.

Usage:
    python download_traces.py
    python download_traces.py --datasets azure_2023 azure_2024 burstgpt

Requirements:
    pip install requests tqdm
    pip install datasets huggingface_hub   # for WildChat and LMSYS-Chat-1M

Subfolders created:
    azure_2023/     — Azure LLM Inference Traces 2023 (Splitwise / ISCA'24)
    azure_2024/     — Azure LLM Inference Traces 2024 (DynamoLLM / HPCA'25)
    azure_2025/     — Azure LMM Inference Trace 2025 multimodal (ModServe / SoCC'25)
    burstgpt/       — BurstGPT (KDD 2025)
    wildchat/       — WildChat-1M (NAACL 2024)
    lmsys_chat_1m/  — LMSYS-Chat-1M (ICLR 2024)  [requires HF_TOKEN + license acceptance]
    sharechat/      — ShareChat (arXiv:2512.17843)  [requires HF_TOKEN + license acceptance]
"""

import argparse
import gzip
import os
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

ALL_DATASETS = ["azure_2023", "azure_2024", "azure_2025", "burstgpt", "wildchat", "lmsys", "sharechat"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_http(url: str, dest: Path, label: str | None = None) -> None:
    """Stream-download url → dest, skipping if the file already exists."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("  [error] Missing dependencies. Run: pip install requests tqdm")
        sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip]  {dest.name} already exists")
        return

    desc = label or dest.name
    print(f"  Downloading {desc} …")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"  Saved  → {dest.relative_to(BASE_DIR)}")


def _decompress_gz(gz_path: Path) -> Path:
    """Decompress a .gz file in-place; returns the decompressed path."""
    out_path = gz_path.with_suffix("")  # strips .gz
    if out_path.exists():
        print(f"  [skip]  {out_path.name} already decompressed")
        return out_path
    print(f"  Decompressing {gz_path.name} …")
    with gzip.open(gz_path, "rb") as f_in, out_path.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"  Saved  → {out_path.relative_to(BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------

def download_azure_2023() -> None:
    """
    Azure LLM Inference Traces 2023 — Splitwise / ISCA 2024
    ~20-minute production trace (coding + conversation), ~2 small CSVs.
    Source: https://github.com/Azure/AzurePublicDataset
    """
    print("\n[azure_2023] Azure LLM Inference Traces 2023")
    out = BASE_DIR / "azure_2023"
    base = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data"
    for fname in [
        "AzureLLMInferenceTrace_code.csv",
        "AzureLLMInferenceTrace_conv.csv",
    ]:
        _download_http(f"{base}/{fname}", out / fname)


def download_azure_2024() -> None:
    """
    Azure LLM Inference Traces 2024 — DynamoLLM / HPCA 2025
    ~10-day production trace (coding + conversation), hosted on Azure Blob Storage.
    Source: https://github.com/Azure/AzurePublicDataset
    """
    print("\n[azure_2024] Azure LLM Inference Traces 2024")
    out = BASE_DIR / "azure_2024"
    base = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024"
    for fname in [
        "AzureLLMInferenceTrace_code_1week.csv",
        "AzureLLMInferenceTrace_conv_1week.csv",
    ]:
        _download_http(f"{base}/{fname}", out / fname)


def download_azure_2025() -> None:
    """
    Azure LMM Inference Trace 2025 — ModServe / SoCC 2025
    ~1-week multimodal trace (text + images), gzip-compressed CSV.
    Source: https://github.com/Azure/AzurePublicDataset
    """
    print("\n[azure_2025] Azure LMM Inference Trace 2025 (multimodal)")
    out = BASE_DIR / "azure_2025"
    gz_name = "AzureLMMInferenceTrace_multimodal.csv.gz"
    base = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data"
    gz_path = out / gz_name
    _download_http(f"{base}/{gz_name}", gz_path)
    if gz_path.exists():
        _decompress_gz(gz_path)


def download_burstgpt() -> None:
    """
    BurstGPT — KDD 2025
    ~10.3M traces over 213 days from Azure OpenAI (GPT-3.5 + GPT-4), ~188 MB total.
    Source: https://huggingface.co/datasets/lzzmm/BurstGPT
    """
    print("\n[burstgpt] BurstGPT")
    out = BASE_DIR / "burstgpt"
    base = "https://huggingface.co/datasets/lzzmm/BurstGPT/resolve/main/data"
    for fname in [
        "BurstGPT_1.csv",
        "BurstGPT_without_fails_1.csv",
        "BurstGPT_2.csv",
        "BurstGPT_without_fails_2.csv",
    ]:
        _download_http(f"{base}/{fname}", out / fname)


def download_wildchat() -> None:
    """
    WildChat-1M — NAACL 2024
    ~1M real user-ChatGPT conversations (April 2023 – July 2025, 4.8M extended).
    Downloaded via HuggingFace datasets library (Apache / ODC-BY licensed).
    Source: https://huggingface.co/datasets/allenai/WildChat-1M
    """
    print("\n[wildchat] WildChat-1M")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [skip] `datasets` not installed. Run: pip install datasets")
        return

    out = BASE_DIR / "wildchat"
    out.mkdir(parents=True, exist_ok=True)

    if (out / "dataset_dict.json").exists() or (out / "dataset_info.json").exists():
        print(f"  [skip] Already downloaded at {out.relative_to(BASE_DIR)}")
        return

    print("  Streaming WildChat-1M from HuggingFace (this may take a while) …")
    ds = load_dataset("allenai/WildChat-1M", cache_dir=str(out / "_hf_cache"))
    ds.save_to_disk(str(out))
    print(f"  Saved  → {out.relative_to(BASE_DIR)}")


def download_lmsys() -> None:
    """
    LMSYS-Chat-1M — ICLR 2024
    1M conversations from Chatbot Arena / Vicuna demo (25 LLMs).
    Requires prior license acceptance at https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    and a HuggingFace token exported as HF_TOKEN.
    Source: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    """
    print("\n[lmsys] LMSYS-Chat-1M")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [skip] `datasets` not installed. Run: pip install datasets")
        return

    out = BASE_DIR / "lmsys_chat_1m"
    out.mkdir(parents=True, exist_ok=True)

    if (out / "dataset_dict.json").exists() or (out / "dataset_info.json").exists():
        print(f"  [skip] Already downloaded at {out.relative_to(BASE_DIR)}")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "  [warn] HF_TOKEN not set. This dataset requires license acceptance at\n"
            "         https://huggingface.co/datasets/lmsys/lmsys-chat-1m\n"
            "         Set HF_TOKEN=<your_token> and re-run."
        )

    print("  Streaming LMSYS-Chat-1M from HuggingFace …")
    try:
        ds = load_dataset(
            "lmsys/lmsys-chat-1m",
            cache_dir=str(out / "_hf_cache"),
            token=token,
        )
        ds.save_to_disk(str(out))
        print(f"  Saved  → {out.relative_to(BASE_DIR)}")
    except Exception as exc:
        print(f"  [error] {exc}")
        print(
            "  Make sure you accepted the license at HuggingFace and that HF_TOKEN is valid."
        )


def download_sharechat() -> None:
    """
    ShareChat — arXiv:2512.17843
    142,808 real-world conversations across 5 AI platforms (ChatGPT, Claude,
    Gemini, Grok, Perplexity), ~4 GB of CSV/JSON files.
    Requires prior license acceptance at https://huggingface.co/datasets/tucnguyen/ShareChat
    and a HuggingFace token exported as HF_TOKEN.
    Source: https://huggingface.co/datasets/tucnguyen/ShareChat
    """
    print("\n[sharechat] ShareChat")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  [skip] `huggingface_hub` not installed. Run: pip install huggingface_hub")
        return

    out = BASE_DIR / "sharechat"
    out.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "  [warn] HF_TOKEN not set. This dataset requires license acceptance at\n"
            "         https://huggingface.co/datasets/tucnguyen/ShareChat\n"
            "         Set HF_TOKEN=<your_token> and re-run."
        )

    repo_id = "tucnguyen/ShareChat"
    for fname in [
        "chatgpt_results_final_language_filtered.csv",
        "claude_results_final_language_filtered.csv",
        "gemini_results_final_language_filtered.csv",
        "grok_results_final_language_filtered.csv",
        "perplexity_results_final_language_filtered.csv",
        "filtered_out_conversations_non_target_languages.json",
    ]:
        dest = out / fname
        if dest.exists():
            print(f"  [skip]  {dest.name} already exists")
            continue
        print(f"  Downloading {fname} …")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=fname,
                token=token,
                local_dir=str(out),
            )
            print(f"  Saved  → {Path(local_path).relative_to(BASE_DIR)}")
        except Exception as exc:
            print(f"  [error] {exc}")
            print(
                "  Make sure you accepted the license at HuggingFace and that HF_TOKEN is valid."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "azure_2023": download_azure_2023,
    "azure_2024": download_azure_2024,
    "azure_2025": download_azure_2025,
    "burstgpt": download_burstgpt,
    "wildchat": download_wildchat,
    "lmsys": download_lmsys,
    "sharechat": download_sharechat,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=ALL_DATASETS,
        metavar="DATASET",
        help=f"Datasets to download (default: all). Choices: {', '.join(ALL_DATASETS)}",
    )
    args = parser.parse_args()

    print(f"Output directory: {BASE_DIR}")
    for name in args.datasets:
        DOWNLOADERS[name]()

    print("\nDone.")


if __name__ == "__main__":
    main()
