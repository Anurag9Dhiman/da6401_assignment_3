"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sacrebleu.metrics import BLEU

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import build_dataloaders, Vocabulary


def _wandb_cfg(cfg, **extra):
    """Return a wandb-serializable config dict — strips non-primitive values."""
    merged = {**cfg, **extra}
    return {k: v for k, v in merged.items()
            if isinstance(v, (int, float, str, bool, type(None)))}


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth[correct] = 1 - eps + eps / vocab_size
        y_smooth[others]  = eps / vocab_size
        y_smooth[pad]     = 0   (pad position receives no signal)

    Implemented via KL-divergence between predicted log-probs and the
    smoothed one-hot distribution.

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        log_probs = F.log_softmax(logits, dim=-1)  # [N, V]

        # Build smoothed target distribution
        # Start with uniform smoothing across all classes
        smooth_val = self.smoothing / (self.vocab_size - 2)   # -2: correct + pad excluded
        with torch.no_grad():
            target_dist = torch.full_like(log_probs, smooth_val)
            target_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # Zero out the pad index — no signal from padding positions
            target_dist[:, self.pad_idx] = 0.0
            # Mask out rows where the target itself is pad
            pad_mask = (target == self.pad_idx)
            target_dist[pad_mask] = 0.0

        # KL divergence: sum(target * (log_target - log_pred))
        # Since target_dist is not log-space, use F.kl_div with log_input=True
        loss = F.kl_div(log_probs, target_dist, reduction="sum", log_target=False)

        # Normalise by number of non-pad tokens
        n_tokens = (~pad_mask).sum().clamp(min=1)
        return loss / n_tokens


# ══════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL GRADIENT LOGGING STATE  (§2.2)
#  Kept here so run_epoch signature stays exactly as the skeleton requires.
#  Set _grad_norm_log = True and reset _global_step before the scaling
#  ablation run; the training loop reads these automatically.
# ══════════════════════════════════════════════════════════════════════

_grad_norm_log: bool = False   # enable Q/K gradient norm logging
_global_step:   int  = 0       # counts optimizer steps across batches


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss   = 0.0
    total_tokens = 0
    pad_idx      = Vocabulary.PAD_IDX

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)  # [B, S]
            tgt = tgt.to(device)  # [B, T]

            # Split tgt into input (shifted right) and expected output
            tgt_in  = tgt[:, :-1]   # feed: <sos> tok1 tok2 ...
            tgt_out = tgt[:, 1:]    # expect:     tok1 tok2 ... <eos>

            src_mask = make_src_mask(src, pad_idx).to(device)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx).to(device)

            # Forward pass
            logits = model(src, tgt_in, src_mask, tgt_mask)
            # logits: [B, T-1, tgt_vocab_size]

            # Flatten for loss
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(B * T))

            # Count real (non-pad) tokens in this batch for normalisation
            n_tokens = (tgt_out != pad_idx).sum().item()

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for training stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # ── §2.2: Log Q/K gradient norms for first 1000 steps ──
                # Controlled by module-level _grad_norm_log flag so the
                # skeleton-required run_epoch signature stays unmodified.
                global _grad_norm_log, _global_step
                if _grad_norm_log and _global_step < 1000:
                    q_norms, k_norms = [], []
                    for enc_layer in model.encoder.layers:
                        wq = enc_layer.self_attn.W_Q.weight
                        wk = enc_layer.self_attn.W_K.weight
                        if wq.grad is not None:
                            q_norms.append(wq.grad.norm().item())
                        if wk.grad is not None:
                            k_norms.append(wk.grad.norm().item())
                    if q_norms and k_norms:
                        wandb.log({
                            "grad_norm_Q": float(np.mean(q_norms)),
                            "grad_norm_K": float(np.mean(k_norms)),
                            "global_step": _global_step,
                        })
                _global_step += 1

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)

    # W&B per-epoch logging
    split_tag = "train" if is_train else "val"
    wandb.log({
        f"{split_tag}_loss": avg_loss,
        f"{split_tag}_ppl" : math.exp(min(avg_loss, 20)),
        "epoch": epoch_num,
        "lr"   : optimizer.param_groups[0]["lr"] if (is_train and optimizer) else 0.0,
    })

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        # Encode source once
        memory = model.encode(src, src_mask)   # [1, src_len, d_model]

        # Initialise decoder input with <sos>
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=Vocabulary.PAD_IDX).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            # logits: [1, t, vocab_size]

            # Greedily pick the token with highest probability at last position
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            ys = torch.cat([ys, next_token], dim=1)                       # [1, t+1]

            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
        tgt_vocab       : Vocabulary object with lookup_token support.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    model.eval()
    sos_idx = Vocabulary.SOS_IDX
    eos_idx = Vocabulary.EOS_IDX
    pad_idx = Vocabulary.PAD_IDX

    hypotheses = []
    references  = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            src_mask = make_src_mask(src, pad_idx).to(device)

            # Greedy decode
            pred_ids = greedy_decode(
                model, src, src_mask, max_len,
                start_symbol=sos_idx,
                end_symbol=eos_idx,
                device=device,
            )
            pred_ids = pred_ids[0].tolist()

            # Strip special tokens from prediction
            pred_tokens = []
            for idx in pred_ids:
                tok = tgt_vocab.lookup_token(idx)
                if tok in ("<sos>",):
                    continue
                if tok == "<eos>":
                    break
                pred_tokens.append(tok)
            hypotheses.append(" ".join(pred_tokens))

            # Build reference from target (strip <sos> and <eos>)
            ref_ids = tgt[0].tolist()
            ref_tokens = []
            for idx in ref_ids:
                tok = tgt_vocab.lookup_token(idx)
                if tok in ("<sos>",):
                    continue
                if tok in ("<eos>", "<pad>"):
                    break
                ref_tokens.append(tok)
            references.append(" ".join(ref_tokens))

    bleu_metric = BLEU(effective_order=True)
    result      = bleu_metric.corpus_score(hypotheses, [references])
    return result.score   # float in range [0, 100]


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
    src_vocab=None,
    tgt_vocab=None,
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    src_vocab and tgt_vocab are saved when provided (needed by Transformer.infer).
    """
    cfg = {
        "src_vocab_size": model.src_embed.num_embeddings,
        "tgt_vocab_size": model.tgt_embed.num_embeddings,
        "d_model"       : model.d_model,
        "N"             : len(model.encoder.layers),
        "num_heads"     : model.encoder.layers[0].self_attn.num_heads,
        "d_ff"          : model.encoder.layers[0].ffn.linear1.out_features,
        "dropout"       : model.encoder.layers[0].dropout.p,
    }
    torch.save({
        "epoch"               : epoch,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config"        : cfg,
        "src_vocab"           : src_vocab,
        "tgt_vocab"           : tgt_vocab,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


# ══════════════════════════════════════════════════════════════════════
#  W&B EXPERIMENT HELPERS
# ══════════════════════════════════════════════════════════════════════

def log_attention_heatmaps(model: Transformer, src: torch.Tensor,
                           src_mask: torch.Tensor, src_vocab, device: str) -> None:
    """
    §2.3 — Extract and log one heatmap per head from the last encoder layer.

    Call this after a validation pass with a fixed English-like sentence.

    Args:
        model    : Trained Transformer in eval mode.
        src      : Single source sentence, shape [1, src_len].
        src_mask : shape [1, 1, 1, src_len].
        src_vocab: Vocabulary object for decoding token labels.
        device   : torch device string.
    """
    model.eval()
    with torch.no_grad():
        model.encode(src.to(device), src_mask.to(device))

    last_enc = model.encoder.layers[-1]
    attn_w   = last_enc.self_attn.attn_weights  # [1, num_heads, src_len, src_len]

    if attn_w is None:
        return

    tokens = [src_vocab.lookup_token(i.item()) for i in src[0]]
    num_heads = attn_w.size(1)

    for h in range(num_heads):
        head_w = attn_w[0, h].cpu().numpy()   # [src_len, src_len]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            head_w,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"Encoder self-attention — head {h + 1}")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        wandb.log({f"attn_head_{h + 1}": wandb.Image(fig)})
        plt.close(fig)


def run_noam_vs_fixed_experiment(
    cfg: dict,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: str,
) -> None:
    """
    §2.1 — Train two models: one with Noam scheduler, one with fixed LR.
    Logs train_loss and val_loss curves to W&B for comparison.
    """
    for lr_type in ("noam", "fixed"):
        wandb.finish()
        run = wandb.init(
            project=cfg["wandb_project"],
            name=f"lr_experiment_{lr_type}",
            config=_wandb_cfg(cfg, lr_type=lr_type),
            reinit="finish_previous",
        )
        model = Transformer(
            src_vocab_size, tgt_vocab_size,
            d_model=cfg["d_model"], N=cfg["N"],
            num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0,
            betas=(0.9, 0.98), eps=1e-9,
        )

        if lr_type == "noam":
            scheduler = NoamScheduler(optimizer, cfg["d_model"], cfg["warmup_steps"])
        else:
            # Fixed LR: override with constant 1e-4
            for g in optimizer.param_groups:
                g["lr"] = 1e-4
            scheduler = None

        loss_fn = LabelSmoothingLoss(tgt_vocab_size, Vocabulary.PAD_IDX, smoothing=0.1)

        for epoch in range(cfg.get("exp_epochs", 10)):
            run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                      epoch, is_train=True, device=device)
            run_epoch(val_loader, model, loss_fn, None, None,
                      epoch, is_train=False, device=device)

        run.finish()


def run_scaling_ablation(
    cfg: dict,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: str,
) -> None:
    """
    §2.2 — Train two models: with and without the 1/sqrt(d_k) scaling factor.
    Logs gradient norms of Q and K weight matrices for the first 1000 steps.

    The model without scaling is achieved by monkey-patching
    scaled_dot_product_attention to omit the sqrt(d_k) divisor.
    """
    import model as model_module

    original_sdpa = model_module.scaled_dot_product_attention

    def sdpa_no_scale(Q, K, V, mask=None):
        import torch.nn.functional as F_
        # Raw dot-product WITHOUT dividing by sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn_w = F_.softmax(scores, dim=-1)
        return torch.matmul(attn_w, V), attn_w

    for use_scaling in (True, False):
        tag = "with_scaling" if use_scaling else "no_scaling"
        if not use_scaling:
            model_module.scaled_dot_product_attention = sdpa_no_scale

        wandb.finish()
        run = wandb.init(
            project=cfg["wandb_project"],
            name=f"scaling_{tag}",
            config=_wandb_cfg(cfg, use_scaling=use_scaling),
            reinit="finish_previous",
        )
        mdl = Transformer(
            src_vocab_size, tgt_vocab_size,
            d_model=cfg["d_model"], N=cfg["N"],
            num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
        ).to(device)

        opt = torch.optim.Adam(mdl.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sched = NoamScheduler(opt, cfg["d_model"], cfg["warmup_steps"])
        loss_fn = LabelSmoothingLoss(tgt_vocab_size, Vocabulary.PAD_IDX, 0.1)

        # Enable grad norm logging via module-level flag for §2.2
        global _grad_norm_log, _global_step
        _grad_norm_log = True
        _global_step   = 0

        for epoch in range(10):
            run_epoch(train_loader, mdl, loss_fn, opt, sched,
                      epoch_num=epoch, is_train=True, device=device)
            run_epoch(val_loader, mdl, loss_fn, None, None,
                      epoch_num=epoch, is_train=False, device=device)

        if not use_scaling:
            model_module.scaled_dot_product_attention = original_sdpa

        # Turn off grad norm logging after this experiment
        _grad_norm_log = False

        run.finish()


def run_pe_ablation(
    cfg: dict,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: str,
) -> None:
    """
    §2.4 — Compare sinusoidal PE vs learned positional embeddings.
    Logs val_bleu after each epoch for both conditions.
    """
    import model as model_module
    from model import PositionalEncoding

    class LearnedPositionalEncoding(nn.Module):
        """Learned positional embeddings via nn.Embedding."""
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super().__init__()
            self.embed   = nn.Embedding(max_len, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            return self.dropout(x + self.embed(positions))

    for pe_type in ("sinusoidal", "learned"):
        wandb.finish()
        run = wandb.init(
            project=cfg["wandb_project"],
            name=f"pe_{pe_type}",
            config=_wandb_cfg(cfg, pe_type=pe_type),
            reinit="finish_previous",
        )
        mdl = Transformer(
            src_vocab_size, tgt_vocab_size,
            d_model=cfg["d_model"], N=cfg["N"],
            num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
        ).to(device)

        if pe_type == "learned":
            mdl.pos_enc = LearnedPositionalEncoding(cfg["d_model"], cfg["dropout"]).to(device)

        opt    = torch.optim.Adam(mdl.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sched  = NoamScheduler(opt, cfg["d_model"], cfg["warmup_steps"])
        loss_fn = LabelSmoothingLoss(tgt_vocab_size, Vocabulary.PAD_IDX, 0.1)

        for epoch in range(cfg.get("exp_epochs", 10)):
            run_epoch(train_loader, mdl, loss_fn, opt, sched,
                      epoch, is_train=True, device=device)
            run_epoch(val_loader, mdl, loss_fn, None, None,
                      epoch, is_train=False, device=device)
            val_bleu_loader = DataLoader(
                val_loader.dataset, batch_size=1, shuffle=False,
                collate_fn=val_loader.collate_fn,
            )
            val_bleu = evaluate_bleu(mdl, val_bleu_loader, cfg["tgt_vocab"], device)
            wandb.log({"val_bleu": val_bleu, "pe_type": pe_type, "epoch": epoch})

        run.finish()


def run_label_smoothing_ablation(
    cfg: dict,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: str,
) -> None:
    """
    §2.5 — Train with eps=0.1 and eps=0.0 (standard cross-entropy).
    Logs prediction confidence = softmax probability of the correct token.
    """
    for eps in (0.1, 0.0):
        tag = f"smoothing_{eps}"
        wandb.finish()
        run = wandb.init(
            project=cfg["wandb_project"],
            name=tag,
            config=_wandb_cfg(cfg, label_smoothing=eps),
            reinit="finish_previous",
        )
        mdl = Transformer(
            src_vocab_size, tgt_vocab_size,
            d_model=cfg["d_model"], N=cfg["N"],
            num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
        ).to(device)

        opt     = torch.optim.Adam(mdl.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sched   = NoamScheduler(opt, cfg["d_model"], cfg["warmup_steps"])
        loss_fn = LabelSmoothingLoss(tgt_vocab_size, Vocabulary.PAD_IDX, smoothing=eps)

        pad_idx = Vocabulary.PAD_IDX

        for epoch in range(cfg.get("exp_epochs", 10)):
            run_epoch(train_loader, mdl, loss_fn, opt, sched,
                      epoch, is_train=True, device=device)
            run_epoch(val_loader, mdl, loss_fn, None, None,
                      epoch, is_train=False, device=device)

            # Measure prediction confidence over val set
            mdl.eval()
            confidence_vals = []
            with torch.no_grad():
                for src_b, tgt_b in val_loader:
                    src_b  = src_b.to(device)
                    tgt_b  = tgt_b.to(device)
                    tin    = tgt_b[:, :-1]
                    tout   = tgt_b[:, 1:]
                    smask  = make_src_mask(src_b, pad_idx).to(device)
                    tmask  = make_tgt_mask(tin,   pad_idx).to(device)
                    logits = mdl(src_b, tin, smask, tmask)
                    probs  = F.softmax(logits, dim=-1)   # [B, T, V]
                    B, T, V = probs.shape
                    correct_probs = probs.gather(
                        2, tout.clamp(0, V - 1).unsqueeze(-1)
                    ).squeeze(-1)                         # [B, T]
                    valid = (tout != pad_idx)
                    confidence_vals.append(
                        correct_probs[valid].mean().item()
                    )
            avg_conf = float(np.mean(confidence_vals))
            wandb.log({
                "prediction_confidence": avg_conf,
                "label_smoothing"      : eps,
                "epoch"                : epoch,
            })

        run.finish()


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment plus all W&B ablations.

    Steps:
        1. Init W&B
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders
        4. Instantiate Transformer
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler
        7. Instantiate LabelSmoothingLoss (ε=0.1)
        8. Training loop with checkpointing
        9. Final BLEU on test set
        10. All five W&B experiments (§2.1–§2.5)
    """
    # ── Hyperparameters ───────────────────────────────────────────────
    cfg = {
        "d_model"        : 256,
        "N"              : 3,
        "num_heads"      : 8,
        "d_ff"           : 512,
        "dropout"        : 0.1,
        "batch_size"     : 128,
        "epochs"         : 50,    # main training — maximise BLEU for autograder
        "exp_epochs"     : 20,    # ablation experiments — clear comparison curves
        "warmup_steps"   : 4000,
        "label_smoothing": 0.1,
        "wandb_project"  : "da6401-a3",
        "min_freq"       : 2,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────
    print("Building dataloaders …")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=cfg["batch_size"], min_freq=cfg["min_freq"]
    )
    cfg["tgt_vocab"] = tgt_vocab   # needed inside ablation helpers

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"  src vocab: {src_vocab_size}  |  tgt vocab: {tgt_vocab_size}")

    # ── Main training run (Noam + label smoothing) ────────────────────
    wandb.finish()
    wandb.init(project=cfg["wandb_project"], name="main_run",
               config=_wandb_cfg(cfg), reinit="finish_previous")

    model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model=cfg["d_model"], N=cfg["N"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0,
        betas=(0.9, 0.98), eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, cfg["d_model"], cfg["warmup_steps"])
    loss_fn   = LabelSmoothingLoss(tgt_vocab_size, Vocabulary.PAD_IDX,
                                   smoothing=cfg["label_smoothing"])

    best_val_loss = float("inf")
    best_ckpt     = "best_checkpoint.pt"

    # Enable Q/K gradient norm logging for main run (§2.2 data collected here)
    global _grad_norm_log, _global_step
    _grad_norm_log = True
    _global_step   = 0

    print("Starting training …")
    for epoch in range(cfg["epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )
        print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_ckpt,
                            src_vocab=src_vocab, tgt_vocab=tgt_vocab)
            print(f"    ✓ Saved best checkpoint (val_loss={val_loss:.4f})")

    # ── §2.3 Attention heatmaps ───────────────────────────────────────
    # Use a fixed validation sentence for reproducible heatmaps
    first_src, _ = next(iter(val_loader))
    fixed_src    = first_src[:1].to(device)
    fixed_mask   = make_src_mask(fixed_src, Vocabulary.PAD_IDX).to(device)
    log_attention_heatmaps(model, fixed_src, fixed_mask, src_vocab, device)

    # ── Final BLEU on test set ────────────────────────────────────────
    load_checkpoint(best_ckpt, model)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device)
    print(f"\nTest BLEU (best checkpoint): {test_bleu:.2f}")
    wandb.log({"test_bleu": test_bleu})
    wandb.finish()
    _grad_norm_log = False   # stop gradient logging before ablation runs

    # ── W&B ablation experiments ──────────────────────────────────────
    print("\nRunning §2.1 Noam vs fixed-LR experiment …")
    run_noam_vs_fixed_experiment(cfg, train_loader, val_loader,
                                 src_vocab_size, tgt_vocab_size, device)

    print("Running §2.2 scaling-factor ablation …")
    run_scaling_ablation(cfg, train_loader, val_loader,
                         src_vocab_size, tgt_vocab_size, device)

    print("Running §2.4 PE vs learned embeddings …")
    run_pe_ablation(cfg, train_loader, val_loader,
                    src_vocab_size, tgt_vocab_size, device)

    print("Running §2.5 label smoothing ablation …")
    run_label_smoothing_ablation(cfg, train_loader, val_loader,
                                 src_vocab_size, tgt_vocab_size, device)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    run_training_experiment()