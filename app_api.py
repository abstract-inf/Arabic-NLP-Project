import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    import pyarabic.araby as araby
except ImportError as exc:
    raise ImportError("pyarabic is required. Install with: pip install pyarabic") from exc

TRANSFORMERS_AVAILABLE = True
PEFT_AVAILABLE = True

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoModelForSequenceClassification = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    TaskType = None
    get_peft_model = None


# -----------------------------
# Global config
# -----------------------------
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
MAX_SRC_LEN = 40
MAX_TGT_LEN = 48

# QA transformer generation settings (same style as notebook)
GEN_MAX_NEW_TOKENS = 96
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.5
GEN_NO_REPEAT_NGRAM = 2
SYSTEM_PERSONA = "انت مساعد ذكي وموثوق. اجب عن السؤال التالي بدقة وبشكل مباشر."

# Translation settings (same approach as machine_translation.ipynb)
TRANSLATION_MODEL = os.getenv("HF_TRANSLATION_MODEL", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# Classification settings
CLASSIFIER_PREPROCESSING = "pyarabic_tashkeel_tatweel_preprocessed"
CLASSIFIER_MODEL_NAME = "aubmindlab/bert-base-arabertv02"
CLASSIFIER_CHECKPOINT_DIR = os.getenv(
    "CLASSIFIER_CHECKPOINT_DIR",
    os.path.join(WORKSPACE_ROOT, "classification_transformer_outputs", "best_arabert_tashkeel_tatweel"),
)
CLASSIFIER_MAX_LEN = 128

DATASET_FILE_MAP = {
    "Original_raw": os.path.join(WORKSPACE_ROOT, "AAFAQ_Dataset.csv"),
    "pyarabic_punctuation_only_preprocessed": os.path.join(
        WORKSPACE_ROOT, "preprocessed datasets", "pyarabic_punctuation_only_preprocessed.csv"
    ),
    "pyarabic_hamza_only_preprocessed": os.path.join(
        WORKSPACE_ROOT, "preprocessed datasets", "pyarabic_hamza_only_preprocessed.csv"
    ),
    "pyarabic_hamza_tashkeel_preprocessed": os.path.join(
        WORKSPACE_ROOT, "preprocessed datasets", "pyarabic_hamza_tashkeel_preprocessed.csv"
    ),
    "pyarabic_tashkeel_tatweel_preprocessed": os.path.join(
        WORKSPACE_ROOT, "preprocessed datasets", "pyarabic_tashkeel_tatweel_preprocessed.csv"
    ),
    "pyarabic_aggressive_preprocessed": os.path.join(
        WORKSPACE_ROOT, "preprocessed datasets", "pyarabic_aggressive_preprocessed.csv"
    ),
    "regex_aggressive_preprocessed": os.path.join(
        WORKSPACE_ROOT, "preprocessed datasets", "regex_aggressive_preprocessed.csv"
    ),
}

MODEL_SPECS = {
    "T5": {"hf_name": "google/mt5-small", "family": "seq2seq"},
    "GPT": {"hf_name": "aubmindlab/aragpt2-base", "family": "causal"},
    "QWEN": {"hf_name": "Qwen/Qwen2-0.5B", "family": "causal"},
}


# -----------------------------
# Pydantic models
# -----------------------------
class PredictRequest(BaseModel):
    question: str
    model_id: str


class ClassificationResult(BaseModel):
    label: Optional[str]
    confidence: Optional[float]
    source: str


# -----------------------------
# Text preprocessing
# -----------------------------
def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def preprocess_original_raw(text: str) -> str:
    return normalize_spaces(text)


def preprocess_pyarabic_hamza_only(text: str) -> str:
    text = normalize_spaces(text)
    return araby.normalize_hamza(text, method="tasheel")


def preprocess_pyarabic_hamza_tashkeel(text: str) -> str:
    text = normalize_spaces(text)
    text = araby.strip_tashkeel(text)
    text = araby.normalize_hamza(text, method="tasheel")
    return normalize_spaces(text)


def preprocess_pyarabic_tashkeel_tatweel(text: str) -> str:
    text = normalize_spaces(text)
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    return normalize_spaces(text)


def preprocess_pyarabic_punctuation_only(text: str) -> str:
    text = normalize_spaces(text)
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return normalize_spaces(text)


def preprocess_pyarabic_aggressive(text: str) -> str:
    text = normalize_spaces(text)
    text = araby.strip_tashkeel(text)
    text = araby.normalize_hamza(text, method="tasheel")
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return normalize_spaces(text)


def preprocess_regex_aggressive(text: str) -> str:
    text = normalize_spaces(text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"\u0640", "", text)
    text = re.sub(r"[أإآءئؤ]", "ي", text)
    text = re.sub(r"[^\w\s]", "", text)
    return normalize_spaces(text)


PREPROCESSOR_MAP = {
    "Original_raw": preprocess_original_raw,
    "pyarabic_hamza_only_preprocessed": preprocess_pyarabic_hamza_only,
    "pyarabic_hamza_tashkeel_preprocessed": preprocess_pyarabic_hamza_tashkeel,
    "pyarabic_tashkeel_tatweel_preprocessed": preprocess_pyarabic_tashkeel_tatweel,
    "pyarabic_punctuation_only_preprocessed": preprocess_pyarabic_punctuation_only,
    "pyarabic_aggressive_preprocessed": preprocess_pyarabic_aggressive,
    "regex_aggressive_preprocessed": preprocess_regex_aggressive,
}


# -----------------------------
# Seq2Seq model classes
# -----------------------------
class Vocabulary:
    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.itos = {idx: tok for tok, idx in stoi.items()}
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    def decode(self, ids: List[int]) -> str:
        out = []
        for idx in ids:
            tok = self.itos.get(int(idx), self.unk_token)
            if tok in {self.pad_token, self.sos_token}:
                continue
            if tok == self.eos_token:
                break
            out.append(tok)
        return " ".join(out)


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        input_mode: str,
        vocab_size: int,
        rnn_type: str,
        bidirectional: bool,
        dense_input_dim: Optional[int],
    ):
        super().__init__()
        self.input_mode = input_mode
        self.bidirectional = bidirectional

        if input_mode == "dense":
            if dense_input_dim is None:
                raise ValueError("dense_input_dim is required for dense input")
            self.dense_projection = nn.Linear(dense_input_dim, EMBED_DIM)
        else:
            self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]

        self.encoder = rnn_cls(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            bidirectional=bidirectional,
        )
        enc_out_dim = HIDDEN_DIM * (2 if bidirectional else 1)

        self.decoder = rnn_cls(
            input_size=EMBED_DIM,
            hidden_size=enc_out_dim,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.decoder_emb = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.output_head = nn.Linear(enc_out_dim, vocab_size)

    def forward(self, src, tgt_input):
        if self.input_mode == "dense":
            src_proj = self.dense_projection(src)
            src_emb = src_proj.unsqueeze(1).repeat(1, MAX_SRC_LEN, 1)
        else:
            src_emb = self.embedding(src)

        _, enc_state = self.encoder(src_emb)
        dec_emb = self.decoder_emb(tgt_input)

        if isinstance(enc_state, tuple):
            h, c = enc_state
            if self.bidirectional:
                h = torch.cat([h[0::2], h[1::2]], dim=2)
                c = torch.cat([c[0::2], c[1::2]], dim=2)
            dec_state = (h, c)
        else:
            h = enc_state
            if self.bidirectional:
                h = torch.cat([h[0::2], h[1::2]], dim=2)
            dec_state = h

        dec_out, _ = self.decoder(dec_emb, dec_state)
        logits = self.output_head(dec_out)
        return logits


def generate_greedy(model: Seq2SeqModel, src_tensor: torch.Tensor, sos_id: int) -> List[int]:
    model.eval()
    with torch.no_grad():
        tokens = torch.full((1, 1), sos_id, dtype=torch.long, device=src_tensor.device)
        for _ in range(MAX_TGT_LEN - 1):
            logits = model(src_tensor, tokens)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_tok], dim=1)
    return tokens[0].detach().cpu().tolist()


@dataclass
class Seq2SeqRuntime:
    model: Seq2SeqModel
    vocab: Vocabulary
    vectorizer: CountVectorizer


@dataclass
class TransformerQARuntime:
    model: Any
    tokenizer: Any
    model_key: str
    family: str
    preprocessing: str
    checkpoint_path: str


@dataclass
class ClassifierRuntime:
    tokenizer: Any
    model: Any
    source: str


@dataclass
class FallbackClassifierRuntime:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    source: str


# -----------------------------
# Registries
# -----------------------------
SEQ2SEQ_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "seq2seq_bigru_best": {
        "display_name": "Seq2Seq BiGRU (best)",
        "family": "seq2seq",
        "architecture": "BiGRU",
        "embedding_strategy": "bow",
        "preprocessing": "pyarabic_punctuation_only_preprocessed",
        "dataset_variant": "pyarabic_punctuation_only_preprocessed",
        "checkpoint_path": os.path.join(WORKSPACE_ROOT, "QA_seq2seq_outputs", "qa_seq2seq_checkpoints", "best_BiGRU.pt"),
        "bow_vectorizer_path": "",
    },
    "seq2seq_gru_best": {
        "display_name": "Seq2Seq GRU (best)",
        "family": "seq2seq",
        "architecture": "GRU",
        "embedding_strategy": "bow",
        "preprocessing": "Original_raw",
        "dataset_variant": "Original_raw",
        "checkpoint_path": os.path.join(WORKSPACE_ROOT, "QA_seq2seq_outputs", "qa_seq2seq_checkpoints", "best_GRU.pt"),
        "bow_vectorizer_path": "",
    },
    "seq2seq_lstm_best": {
        "display_name": "Seq2Seq LSTM (best)",
        "family": "seq2seq",
        "architecture": "LSTM",
        "embedding_strategy": "bow",
        "preprocessing": "pyarabic_hamza_only_preprocessed",
        "dataset_variant": "pyarabic_hamza_only_preprocessed",
        "checkpoint_path": os.path.join(WORKSPACE_ROOT, "QA_seq2seq_outputs", "qa_seq2seq_checkpoints", "best_LSTM.pt"),
        "bow_vectorizer_path": "",
    },
    "seq2seq_rnn_best": {
        "display_name": "Seq2Seq RNN (best)",
        "family": "seq2seq",
        "architecture": "RNN",
        "embedding_strategy": "bow",
        "preprocessing": "pyarabic_punctuation_only_preprocessed",
        "dataset_variant": "pyarabic_punctuation_only_preprocessed",
        "checkpoint_path": os.path.join(WORKSPACE_ROOT, "QA_seq2seq_outputs", "qa_seq2seq_checkpoints", "best_RNN.pt"),
        "bow_vectorizer_path": "",
    },
}

TRANSFORMER_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Runtime state
# -----------------------------
seq2seq_runtimes: Dict[str, Seq2SeqRuntime] = {}
transformer_runtimes: Dict[str, TransformerQARuntime] = {}
startup_errors: Dict[str, str] = {}

classifier_runtime: Optional[ClassifierRuntime] = None
fallback_classifier_runtime: Optional[FallbackClassifierRuntime] = None


# -----------------------------
# Helpers: loading and preprocess
# -----------------------------
def _fit_bow_from_dataset(dataset_variant: str, max_features: int) -> CountVectorizer:
    dataset_path = DATASET_FILE_MAP.get(dataset_variant)
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file for variant '{dataset_variant}' not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "QuestionText" not in df.columns:
        raise ValueError(f"QuestionText column missing in dataset: {dataset_path}")

    texts = df["QuestionText"].fillna("").astype(str).tolist()
    vec = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    vec.fit(texts)
    return vec


def _load_vectorizer(model_cfg: Dict[str, Any], expected_dim: int) -> CountVectorizer:
    vec_path = model_cfg.get("bow_vectorizer_path", "")
    if vec_path and os.path.exists(vec_path):
        try:
            obj = joblib.load(vec_path)
            if isinstance(obj, CountVectorizer):
                loaded_dim = len(obj.get_feature_names_out())
                if loaded_dim == expected_dim:
                    return obj
        except Exception:
            pass
    return _fit_bow_from_dataset(model_cfg["dataset_variant"], max_features=expected_dim)


def _load_seq2seq_model(model_id: str, model_cfg: Dict[str, Any]) -> Seq2SeqRuntime:
    ckpt_path = model_cfg["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    stoi = checkpoint.get("vocab")
    if not isinstance(stoi, dict):
        raise ValueError(f"Checkpoint vocab not found/invalid in {ckpt_path}")

    dense_weight = checkpoint.get("state_dict", {}).get("dense_projection.weight")
    if dense_weight is None or len(dense_weight.shape) != 2:
        raise ValueError(f"dense_projection.weight missing/invalid in {ckpt_path}")
    expected_dense_dim = int(dense_weight.shape[1])

    vocab = Vocabulary(stoi)
    vectorizer = _load_vectorizer(model_cfg, expected_dim=expected_dense_dim)
    dense_dim = len(vectorizer.get_feature_names_out())

    arch = model_cfg["architecture"]
    is_bigru = arch == "BiGRU"
    rnn_type = "GRU" if is_bigru else arch

    model = Seq2SeqModel(
        input_mode="dense",
        vocab_size=len(vocab.stoi),
        rnn_type=rnn_type,
        bidirectional=is_bigru,
        dense_input_dim=dense_dim,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return Seq2SeqRuntime(model=model, vocab=vocab, vectorizer=vectorizer)


def _resolve_transformer_ckpt_path(model_name: str, raw_path: str) -> str:
    normalized_name = model_name.upper().strip()

    # The manifest may include './outputs/...', while checkpoints exist under QA_transformer_outputs_final.
    candidate_1 = os.path.join(WORKSPACE_ROOT, raw_path.lstrip("./\\"))
    candidate_2 = os.path.join(
        WORKSPACE_ROOT,
        "QA_transformer_outputs_final",
        "qa_transformer_checkpoints",
        f"best_{normalized_name}.pt",
    )

    for path in [candidate_1, candidate_2, raw_path]:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path

    return os.path.abspath(candidate_2)


def _load_transformer_registry_from_manifest() -> Dict[str, Dict[str, Any]]:
    manifest_path = os.path.join(
        WORKSPACE_ROOT,
        "QA_transformer_outputs_final",
        "question_answering_transformers_best_manifest.json",
    )
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    registry: Dict[str, Dict[str, Any]] = {}
    for row in data:
        model_key = str(row.get("Model", "")).upper().strip()
        if model_key != "QWEN":
            continue
        if model_key not in MODEL_SPECS:
            continue

        model_id = f"transformer_{model_key.lower()}_best"
        ckpt_path = _resolve_transformer_ckpt_path(model_key, str(row.get("CheckpointPath", "")))
        registry[model_id] = {
            "display_name": f"Transformer {model_key} (best)",
            "family": "transformer",
            "preprocessing": "pyarabic_tashkeel_tatweel_preprocessed",
            "embedding_strategy": "lora",
            "checkpoint_path": ckpt_path,
            "model_key": model_key,
            "ready": os.path.exists(ckpt_path),
            "metrics": {
                "best_val_bertscore_f1": row.get("BestValBERTScore_F1"),
                "test_bertscore_f1": row.get("TestBERTScore_F1"),
            },
        }

    return registry


def _build_lora_config(model_key: str, family: str) -> Any:
    if family == "seq2seq":
        return LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"],
            bias="none",
        )

    if model_key == "GPT":
        target_modules = ["c_attn", "c_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )


def _load_transformer_qa_model(model_id: str, model_cfg: Dict[str, Any]) -> TransformerQARuntime:
    if not TRANSFORMERS_AVAILABLE or not PEFT_AVAILABLE:
        raise RuntimeError("transformers/peft dependencies are not installed")

    ckpt_path = model_cfg["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=DEVICE)
    model_key = model_cfg["model_key"]
    spec = MODEL_SPECS[model_key]
    family = spec["family"]
    model_name = (
        payload.get("model_name")
        or payload.get("pretrained_model_name_or_path")
        or payload.get("tokenizer_name")
        or spec["hf_name"]
    )

    hf_kwargs = {"trust_remote_code": True}
    if HF_TOKEN:
        # Use 'token' instead of 'use_auth_token'
        hf_kwargs["token"] = HF_TOKEN

    tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)
    if family == "seq2seq":
        tokenizer.padding_side = "right"
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **hf_kwargs)
    else:
        tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **hf_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.config.eos_token_id = tokenizer.eos_token_id

    lora_cfg = _build_lora_config(model_key, family)
    model = get_peft_model(base_model, lora_cfg).to(DEVICE)
    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"model_state_dict missing in checkpoint: {ckpt_path}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return TransformerQARuntime(
        model=model,
        tokenizer=tokenizer,
        model_key=model_key,
        family=family,
        preprocessing=model_cfg["preprocessing"],
        checkpoint_path=ckpt_path,
    )


def _load_classifier_runtime() -> Optional[ClassifierRuntime]:
    if not TRANSFORMERS_AVAILABLE:
        return None

    if not os.path.isdir(CLASSIFIER_CHECKPOINT_DIR):
        return None

    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_CHECKPOINT_DIR).to(DEVICE)
    model.eval()

    return ClassifierRuntime(
        tokenizer=tokenizer,
        model=model,
        source="arabert_finetuned_pyarabic_tashkeel_tatweel",
    )


def _load_fallback_classifier_runtime() -> Optional[FallbackClassifierRuntime]:
    dataset_path = DATASET_FILE_MAP.get(CLASSIFIER_PREPROCESSING)
    if not dataset_path or not os.path.exists(dataset_path):
        return None

    df = pd.read_csv(dataset_path)
    if "QuestionText" not in df.columns or "Category" not in df.columns:
        return None

    df = df.dropna(subset=["QuestionText", "Category"]).copy()
    df["QuestionText"] = df["QuestionText"].map(preprocess_pyarabic_tashkeel_tatweel)

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    x = vec.fit_transform(df["QuestionText"].astype(str).tolist())
    y = df["Category"].astype(str).tolist()

    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    clf.fit(x, y)

    return FallbackClassifierRuntime(
        vectorizer=vec,
        classifier=clf,
        source="tfidf_logreg_fallback_pyarabic_tashkeel_tatweel",
    )


def _build_input_text(question: str) -> str:
    q = normalize_spaces(question)
    return f"{SYSTEM_PERSONA}\nالسؤال: {q}\nالإجابة:"


def _decode_generated_text(tokenizer, output_ids, prompt_length: int, family: str) -> str:
    if family == "causal":
        generated_ids = output_ids[prompt_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "الإجابة:" in text:
            text = text.split("الإجابة:")[-1].strip()
        return text.strip()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _strip_question_echo(question: str, text: str) -> str:
    q_norm = normalize_spaces(question).rstrip("؟?.!،,:;")
    t_norm = normalize_spaces(text)
    if t_norm.startswith(q_norm):
        trimmed = t_norm[len(q_norm):].lstrip(" ؟?.!،,:;")
        if trimmed:
            return trimmed
    return t_norm


def _translate_text(text: str) -> Tuple[Optional[str], str]:
    if not text:
        return "", "empty"
    if not HF_TOKEN:
        return None, "missing_hf_token"

    payload = {
        "model": TRANSLATION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional Arabic-to-English translator. "
                    "Translate the given Arabic text into fluent English. "
                    "Return ONLY the English translation without explanations."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 512,
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        choices = result.get("choices", []) if isinstance(result, dict) else []
        if not choices:
            return None, "invalid_response"
        content = choices[0].get("message", {}).get("content")
        if not content:
            return None, "invalid_response"
        return str(content).strip(), "ok"
    except Exception:
        return None, "error"


# -----------------------------
# Inference helpers
# -----------------------------
def _classify_question(question: str) -> ClassificationResult:
    pre_fn = PREPROCESSOR_MAP.get(CLASSIFIER_PREPROCESSING, preprocess_pyarabic_tashkeel_tatweel)
    processed = pre_fn(question)

    global classifier_runtime, fallback_classifier_runtime

    if classifier_runtime is not None:
        tok = classifier_runtime.tokenizer(
            processed,
            truncation=True,
            padding="max_length",
            max_length=CLASSIFIER_MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = classifier_runtime.model(**tok).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        pred_id = int(np.argmax(probs))
        conf = float(probs[pred_id])
        id2label = classifier_runtime.model.config.id2label or {}
        label = str(id2label.get(pred_id, pred_id))
        return ClassificationResult(label=label, confidence=conf, source=classifier_runtime.source)

    if fallback_classifier_runtime is not None:
        x = fallback_classifier_runtime.vectorizer.transform([processed])
        pred = fallback_classifier_runtime.classifier.predict(x)[0]
        conf = None
        if hasattr(fallback_classifier_runtime.classifier, "predict_proba"):
            proba = fallback_classifier_runtime.classifier.predict_proba(x)[0]
            conf = float(np.max(proba))
        return ClassificationResult(label=str(pred), confidence=conf, source=fallback_classifier_runtime.source)

    return ClassificationResult(label=None, confidence=None, source="unavailable")


def _answer_seq2seq(model_id: str, question: str) -> str:
    rt = seq2seq_runtimes[model_id]
    features = rt.vectorizer.transform([question]).toarray().astype(np.float32)
    src = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    sos_id = rt.vocab.stoi.get(rt.vocab.sos_token, 1)
    pred_ids = generate_greedy(rt.model, src, sos_id=sos_id)
    return rt.vocab.decode(pred_ids)


def _answer_transformer(model_id: str, question: str) -> str:
    rt = transformer_runtimes.get(model_id)
    if rt is None:
        raise HTTPException(status_code=503, detail=f"Transformer model '{model_id}' is not ready.")

    prompt = _build_input_text(question)
    tokenizer = rt.tokenizer
    model = rt.model

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SRC_LEN,
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            repetition_penalty=GEN_REPETITION_PENALTY,
            no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    if rt.family == "causal":
        prompt_length = int(enc["attention_mask"].sum().item())
        text = _decode_generated_text(tokenizer, out[0], prompt_length, rt.family)
    else:
        text = tokenizer.decode(out[0], skip_special_tokens=True)

    return _strip_question_echo(question, text.strip())


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Arabic QA API", version="2.0.0")


@app.on_event("startup")
def startup() -> None:
    startup_errors.clear()
    seq2seq_runtimes.clear()
    transformer_runtimes.clear()

    global classifier_runtime, fallback_classifier_runtime, TRANSFORMER_MODEL_REGISTRY

    # Load seq2seq runtimes
    for model_id, cfg in SEQ2SEQ_MODEL_REGISTRY.items():
        try:
            seq2seq_runtimes[model_id] = _load_seq2seq_model(model_id, cfg)
        except Exception as exc:
            startup_errors[model_id] = str(exc)

    # Build + load QA transformer runtimes from manifest
    TRANSFORMER_MODEL_REGISTRY = _load_transformer_registry_from_manifest()
    for model_id, cfg in TRANSFORMER_MODEL_REGISTRY.items():
        try:
            transformer_runtimes[model_id] = _load_transformer_qa_model(model_id, cfg)
        except Exception as exc:
            startup_errors[model_id] = str(exc)

    # Classification runtime (best AraBERT if available) + fallback
    try:
        classifier_runtime = _load_classifier_runtime()
    except Exception as exc:
        print(f"Error loading classifier runtime: {exc}")
        classifier_runtime = None
        startup_errors["classifier_arabert_best"] = str(exc)

    if not TRANSFORMERS_AVAILABLE:
        startup_errors["classifier_arabert_best"] = "transformers dependency is not installed"

    try:
        fallback_classifier_runtime = _load_fallback_classifier_runtime()
    except Exception as exc:
        fallback_classifier_runtime = None
        startup_errors["classifier_fallback"] = str(exc)


@app.get("/health")
def health() -> Dict[str, Any]:
    models = []

    for model_id, cfg in SEQ2SEQ_MODEL_REGISTRY.items():
        ready = model_id in seq2seq_runtimes
        models.append(
            {
                "model_id": model_id,
                "family": cfg["family"],
                "ready": ready,
                "error": startup_errors.get(model_id),
            }
        )

    for model_id, cfg in TRANSFORMER_MODEL_REGISTRY.items():
        ready = model_id in transformer_runtimes
        models.append(
            {
                "model_id": model_id,
                "family": cfg["family"],
                "ready": ready,
                "error": startup_errors.get(model_id),
            }
        )

    classifier_ready = classifier_runtime is not None or fallback_classifier_runtime is not None
    classifier_source = (
        classifier_runtime.source
        if classifier_runtime is not None
        else (fallback_classifier_runtime.source if fallback_classifier_runtime is not None else "unavailable")
    )

    return {
        "status": "ok",
        "device": str(DEVICE),
        "classifier": {
            "ready": classifier_ready,
            "source": classifier_source,
            "error": startup_errors.get("classifier_arabert_best") or startup_errors.get("classifier_fallback"),
        },
        "models": models,
    }


@app.get("/models")
def list_models() -> List[Dict[str, Any]]:
    out = []

    for model_id, cfg in SEQ2SEQ_MODEL_REGISTRY.items():
        out.append(
            {
                "model_id": model_id,
                "display_name": cfg["display_name"],
                "family": cfg["family"],
                "preprocessing": cfg["preprocessing"],
                "embedding_strategy": cfg.get("embedding_strategy"),
                "ready": model_id in seq2seq_runtimes,
                "error": startup_errors.get(model_id),
            }
        )

    for model_id, cfg in TRANSFORMER_MODEL_REGISTRY.items():
        out.append(
            {
                "model_id": model_id,
                "display_name": cfg["display_name"],
                "family": cfg["family"],
                "preprocessing": cfg["preprocessing"],
                "embedding_strategy": cfg.get("embedding_strategy"),
                "ready": model_id in transformer_runtimes,
                "error": startup_errors.get(model_id),
            }
        )

    return out


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    model_id = req.model_id
    model_cfg: Optional[Dict[str, Any]] = None
    if model_id in SEQ2SEQ_MODEL_REGISTRY:
        model_cfg = SEQ2SEQ_MODEL_REGISTRY[model_id]
    elif model_id in TRANSFORMER_MODEL_REGISTRY:
        model_cfg = TRANSFORMER_MODEL_REGISTRY[model_id]
    else:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")

    preprocess_name = model_cfg["preprocessing"]
    preprocess_fn = PREPROCESSOR_MAP.get(preprocess_name)
    if preprocess_fn is None:
        raise HTTPException(status_code=500, detail=f"Unknown preprocessing variant: {preprocess_name}")

    cleaned_question = preprocess_fn(question)

    if model_cfg["family"] == "seq2seq":
        if model_id not in seq2seq_runtimes:
            raise HTTPException(status_code=503, detail=startup_errors.get(model_id, "Model not ready"))
        generated_answer_ar = _answer_seq2seq(model_id, cleaned_question)
    else:
        if model_id not in transformer_runtimes:
            raise HTTPException(status_code=503, detail=startup_errors.get(model_id, "Model not ready"))
        generated_answer_ar = _answer_transformer(model_id, cleaned_question)

    classification_result = _classify_question(question)

    translated_question, q_status = _translate_text(question)
    translated_answer, a_status = _translate_text(generated_answer_ar)

    if q_status == "ok" and a_status == "ok":
        translation_status = "ok"
    elif q_status == "missing_hf_token" or a_status == "missing_hf_token":
        translation_status = "missing_hf_token"
    elif q_status == "empty" and a_status == "empty":
        translation_status = "empty"
    else:
        translation_status = "partial_or_failed"

    return {
        "question": question,
        "selected_model": model_id,
        "classification_result": classification_result.model_dump(),
        "generated_answer_ar": generated_answer_ar,
        "translated_question": translated_question,
        "translated_answer": translated_answer,
        "translation_status": translation_status,
        "debug": {
            "preprocessed_question": cleaned_question,
            "preprocessing": preprocess_name,
            "family": model_cfg["family"],
            "embedding_strategy": model_cfg.get("embedding_strategy"),
            "question_translation_status": q_status,
            "answer_translation_status": a_status,
            "classifier_source": classification_result.source,
        },
    }
