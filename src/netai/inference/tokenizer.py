"""Native BPE tokenizer for HuggingFace models.

Loads tokenizer.json (HuggingFace tokenizers format) directly, with zero
dependencies on the 'transformers' library. Handles GPT-2 byte-level BPE,
LLaMA SentencePiece, and other common tokenizer formats.

Only dependency: the 'tokenizers' Rust library (Apache 2.0 licensed) for
performance. If unavailable, falls back to regex-based BPE encoding.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

HAS_TOKENIZERS = False
try:
    from tokenizers import Tokenizer as HFTokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HFTokenizer = None


class NativeBPEEncoder:
    """Encodes/decodes text using a HuggingFace tokenizer.json file.

    Uses the 'tokenizers' Rust library when available (fast, Apache 2.0), or
    falls back to a pure-Python regex-based BPE implementation.
    """

    def __init__(self, tokenizer_path: str) -> None:
        self.tokenizer_path = tokenizer_path
        self._hf_tokenizer: Any = None
        self._vocab: dict[str, int] = {}
        self._vocab_rev: dict[int, str] = {}
        self._merges: list[tuple[str, str]] = []
        self._byte_encoder: dict[int, str] = {}
        self._byte_decoder: dict[str, int] = {}
        self._cache_dir: str = ""
        self._bos_id: int = 0
        self._eos_id: int = 0
        self._unk_id: int = 0
        self._loaded = False

    def _build_byte_table(self) -> None:
        """Build GPT-2 byte-to-unicode mapping."""
        table: list[tuple[int, str]] = []
        for b in range(ord("!"), ord("~") + 1):
            table.append((b, chr(b)))
        for b in range(ord("¡"), ord("¬") + 1):
            table.append((b, chr(b)))
        for b in range(ord("®"), ord("ÿ") + 1):
            table.append((b, chr(b)))
        n = 0
        for b in range(256):
            if b not in [c[0] for c in table]:
                table.append((b, chr(256 + n)))
                n += 1
        self._byte_encoder = {b: s for b, s in table}
        self._byte_decoder = {s: b for b, s in table}

    def _bytes_to_unicode(self, text: bytes) -> str:
        result: list[str] = []
        for b in text:
            result.append(self._byte_encoder.get(b, chr(b)))
        return "".join(result)

    def _unicode_to_bytes(self, text: str) -> bytes:
        result: list[int] = []
        for ch in text:
            result.append(self._byte_decoder.get(ch, ord(ch)))
        return bytes(result)

    def _get_pairs(self, word: tuple[str, ...]) -> set[tuple[str, str]]:
        pairs = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def _bpe_encode(self, token: str) -> str:
        """Apply BPE merges to a single token."""
        word = tuple(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda p: self._merges.index(p) if p in self._merges else 1e9)
            if bigram not in self._merges:
                break

            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i + 1 < len(word) and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        return " ".join(word)

    def _pre_tokenize(self, text: str) -> list[str]:
        """GPT-2 pre-tokenization regex."""
        pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\W+|"""
            r"""\s+(?!\S)|\s+"""
        )
        return pat.findall(text)

    def load(self) -> bool:
        """Load tokenizer from tokenizer.json file."""
        if self._loaded:
            return True

        if not os.path.exists(self.tokenizer_path):
            logger.error("Tokenizer file not found: %s", self.tokenizer_path)
            return False

        try:
            self._build_byte_table()

            # Load using tokenizers Rust lib if available
            if HAS_TOKENIZERS:
                self._hf_tokenizer = HFTokenizer.from_file(self.tokenizer_path)
                self._vocab = self._hf_tokenizer.get_vocab()
                self._vocab_rev = {v: k for k, v in self._vocab.items()}
                self._bos_id = self._hf_tokenizer.token_to_id("<|endoftext|>") or 0
                self._eos_id = self._hf_tokenizer.token_to_id("<|endoftext|>") or 0
                self._unk_id = 0
                self._loaded = True
                logger.info("Loaded tokenizer via tokenizers lib (%d vocab)", len(self._vocab))
                return True

            # Fallback: manual JSON parse
            with open(self.tokenizer_path) as f:
                data = json.load(f)

            model = data.get("model", {})
            self._vocab = model.get("vocab", {})
            self._vocab_rev = {v: k for k, v in self._vocab.items()}

            merges_raw = model.get("merges", [])
            self._merges = []
            for m in merges_raw:
                parts = m.split()
                if len(parts) == 2:
                    self._merges.append((parts[0], parts[1]))

            # Special tokens
            for at in data.get("added_tokens", []):
                if at.get("special", False):
                    tid = at.get("id", 0)
                    content = at.get("content", "")
                    if "<|endoftext|>" in content:
                        self._bos_id = tid
                        self._eos_id = tid

            self._loaded = True
            logger.info("Loaded tokenizer manually (%d vocab, %d merges)",
                        len(self._vocab), len(self._merges))
            return True

        except Exception as e:
            logger.error("Failed to load tokenizer %s: %s", self.tokenizer_path, e)
            return False

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if not self._loaded and not self.load():
            logger.warning("Tokenizer not loaded, using char-based fallback")
            return [ord(c) % 50257 for c in text] if text else [0]

        # Use HF tokenizers lib if available
        if self._hf_tokenizer is not None:
            encoded = self._hf_tokenizer.encode(text)
            return encoded.ids

        # Manual BPE encoding
        tokens: list[int] = []
        for token in self._pre_tokenize(text):
            # Convert bytes to unicode
            unicode_token = self._bytes_to_unicode(token.encode("utf-8"))

            # Check if whole token is in vocab
            if unicode_token in self._vocab:
                tokens.append(self._vocab[unicode_token])
                continue

            # Apply BPE merges
            bpe_result = self._bpe_encode(unicode_token)
            for sub_token in bpe_result.split():
                if sub_token in self._vocab:
                    tokens.append(self._vocab[sub_token])
                else:
                    tokens.append(self._vocab.get(sub_token, 0))

        return tokens if tokens else [0]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        if not self._loaded and not self.load():
            return " ".join(str(i) for i in ids)

        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(ids, skip_special_tokens=False)

        # Manual decoding
        tokens: list[str] = []
        for tid in ids:
            t = self._vocab_rev.get(tid, "")
            if t:
                tokens.append(t)

        text = "".join(tokens)

        # GPT-2 byte-level: undo bytes-to-unicode mapping
        text = text.replace("</w>", " ")
        try:
            # Split by unicode private use area chars (byte mapping)
            result_bytes = bytearray()
            i = 0
            while i < len(text):
                ch = text[i]
                if ch in self._byte_decoder:
                    result_bytes.append(self._byte_decoder[ch])
                else:
                    result_bytes.extend(ch.encode("utf-8", errors="replace"))
                i += 1
            return result_bytes.decode("utf-8", errors="replace")
        except Exception:
            return text

    def get_vocab_size(self) -> int:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.get_vocab_size()
        return len(self._vocab)

    @property
    def bos_token_id(self) -> int:
        return self._bos_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_id


def get_tokenizer(model_dir: str) -> NativeBPEEncoder | None:
    """Get tokenizer for a model directory. Returns None if no tokenizer found."""
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        # Try other locations
        for name in ("tokenizer.json", "vocab.json", "spiece.model"):
            alt_path = os.path.join(model_dir, name)
            if os.path.exists(alt_path):
                tokenizer_path = alt_path
                break
        else:
            logger.warning("No tokenizer found in %s", model_dir)
            return None

    encoder = NativeBPEEncoder(tokenizer_path)
    if encoder.load():
        return encoder
    return None
