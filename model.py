"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import os
import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Fill this after uploading best_checkpoint.pt to Google Drive ──────────────
_GDRIVE_FILE_ID  = "1pSSmJpuVCMy-Binn-kiKfZRvAEXoBdJf"
_CHECKPOINT_PATH = "best_checkpoint.pt"


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
#  Exposed at module level so the autograder can import and test it
#  independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)

    # Raw dot-product scores: [..., seq_q, seq_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask: positions marked True receive -inf so softmax gives ~0
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Attention weight distribution over keys
    attn_w = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attn_w, V)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
#  Exposed at module level so they can be tested independently and
#  reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # [batch, src_len] → [batch, 1, 1, src_len]
    mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)
    device  = tgt.device

    # Padding mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal mask: upper-triangular, True for future positions
    # shape: [1, 1, tgt_len, tgt_len]
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)

    # Combine: True wherever either condition holds
    # Broadcasts to [batch, 1, tgt_len, tgt_len]
    combined = pad_mask | causal_mask
    return combined


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        # Projection matrices for queries, keys, values and output
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        # Store last attention weights for visualisation (W&B §2.3)
        self.attn_weights = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape [batch, seq, d_model] → [batch, heads, seq, d_k].
        """
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)   # [batch, heads, seq, d_k]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape [batch, heads, seq, d_k] → [batch, seq, d_model].
        """
        batch, _, seq, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        # Linear projections + split into heads
        Q = self._split_heads(self.W_Q(query))   # [B, H, seq_q, d_k]
        K = self._split_heads(self.W_K(key))     # [B, H, seq_k, d_k]
        V = self._split_heads(self.W_V(value))   # [B, H, seq_k, d_k]

        # Scaled dot-product attention per head
        attended, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        # attended : [B, H, seq_q, d_k]
        # attn_w   : [B, H, seq_q, seq_k]

        # Apply dropout to attention weights
        attended = self.dropout(attended)

        # Store weights for external W&B visualisation hooks
        self.attn_weights = attn_w.detach()

        # Merge heads and apply output projection
        merged = self._merge_heads(attended)      # [B, seq_q, d_model]
        output = self.W_O(merged)
        return output


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the positional encoding table once
        pe = torch.zeros(max_len, d_model)               # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()   # [max_len, 1]

        # div_term: 1 / 10000^(2i/d_model)  for each dimension pair
        dim_idx   = torch.arange(0, d_model, 2).float()  # [d_model/2]
        div_term  = torch.exp(dim_idx * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)     # even dims
        pe[:, 1::2] = torch.cos(position * div_term)     # odd dims

        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # register_buffer → saved with model state but NOT a trainable parameter
        # The autograder checks that this is a buffer, not nn.Parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer (Post-LayerNorm variant):
        x → Self-Attention → Add & Norm → FFN → Add & Norm

    Post-LayerNorm is chosen to match the original paper exactly.
    The normalisation is applied after the residual addition, which
    keeps gradient flow stable at the scale used in the base model.

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]
        """
        # Self-attention sub-layer with residual + Post-LayerNorm
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward sub-layer with residual + Post-LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer (Post-LayerNorm):
        x → Masked Self-Attn → Add & Norm
          → Cross-Attn(memory) → Add & Norm
          → FFN → Add & Norm

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn       = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn              = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1            = nn.LayerNorm(d_model)
        self.norm2            = nn.LayerNorm(d_model)
        self.norm3            = nn.LayerNorm(d_model)
        self.dropout          = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # 1. Masked self-attention
        self_attn_out = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # 2. Cross-attention: queries from decoder, keys/values from encoder
        cross_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        # Use deepcopy so each layer has its own independent parameters
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int   = None,
        tgt_vocab_size: int   = None,
        d_model:        int   = 256,
        N:              int   = 3,
        num_heads:      int   = 8,
        d_ff:           int   = 512,
        dropout:        float = 0.1,
    ) -> None:
        super().__init__()

        # ── Inference / autograder mode ───────────────────────────────────
        # When called with no vocab sizes, download checkpoint from Drive,
        # load vocabs + spaCy tokeniser, then restore weights.
        _state_dict = None
        if src_vocab_size is None:
            import gdown, spacy

            if not os.path.exists(_CHECKPOINT_PATH):
                gdown.download(
                    f"https://drive.google.com/uc?id={_GDRIVE_FILE_ID}",
                    _CHECKPOINT_PATH, quiet=False,
                )

            ckpt           = torch.load(_CHECKPOINT_PATH, map_location="cpu",
                                        weights_only=False)
            model_cfg      = ckpt["model_config"]
            src_vocab_size = model_cfg["src_vocab_size"]
            tgt_vocab_size = model_cfg["tgt_vocab_size"]
            d_model        = model_cfg.get("d_model",   256)
            N              = model_cfg.get("N",         3)
            num_heads      = model_cfg.get("num_heads", 8)
            d_ff           = model_cfg.get("d_ff",      512)
            dropout        = model_cfg.get("dropout",   0.1)

            self._src_vocab = ckpt["src_vocab"]
            self._tgt_vocab = ckpt["tgt_vocab"]
            try:
                self._nlp_de = spacy.load("de_core_news_sm")
            except OSError:
                import subprocess, sys
                subprocess.run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"], check=True)
                self._nlp_de = spacy.load("de_core_news_sm")
            _state_dict     = ckpt["model_state_dict"]
        else:
            self._src_vocab = None
            self._tgt_vocab = None
            self._nlp_de    = None

        # ── Architecture ──────────────────────────────────────────────────
        self.src_embed   = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed   = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc     = PositionalEncoding(d_model, dropout)
        enc_layer        = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer        = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder     = Encoder(enc_layer, N)
        self.decoder     = Decoder(dec_layer, N)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.d_model     = d_model
        self._init_weights()

        # Load trained weights after architecture is built
        if _state_dict is not None:
            self.load_state_dict(_state_dict)

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) as in §3.4
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        dec_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.output_proj(dec_out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, german_sentence: str) -> str:
        """
        End-to-end German → English translation (used by autograder).

        Args:
            german_sentence : Raw German text string.

        Returns:
            English translation string.
        """
        device  = next(self.parameters()).device
        sos_idx = 2
        eos_idx = 3

        # Tokenise with spaCy
        tokens = [tok.text.lower()
                  for tok in self._nlp_de.tokenizer(german_sentence.strip())]

        # Numericalize: [<sos>] + token_ids + [<eos>]
        ids      = [sos_idx] + self._src_vocab.encode(tokens) + [eos_idx]
        src      = torch.tensor([ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src)

        # Autoregressive greedy decode
        self.eval()
        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys     = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
            for _ in range(100):
                tgt_mask = make_tgt_mask(ys)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys       = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == eos_idx:
                    break

        # Detokenise — strip special tokens
        out_tokens = []
        for idx in ys[0].tolist():
            tok = self._tgt_vocab.lookup_token(idx)
            if tok == "<sos>":
                continue
            if tok in ("<eos>", "<pad>"):
                break
            out_tokens.append(tok)

        return " ".join(out_tokens)