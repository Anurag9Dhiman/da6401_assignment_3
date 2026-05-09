"""
dataset.py  —  Multi30k Dataset, Vocabulary, and DataLoader utilities
DA6401 Assignment 3: Implementing a Transformer for Machine Translation

Covers:
    - Vocabulary construction with special tokens
    - Multi30k loading via HuggingFace datasets
    - spaCy tokenisation (de_core_news_sm / en_core_web_sm)
    - Integer encoding with <sos>/<eos> wrapping
    - Collation with dynamic padding for DataLoader
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
import spacy


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Token <-> integer index mapping with four reserved special symbols.

    Index assignments (fixed):
        0  <unk>   unknown / out-of-vocabulary token
        1  <pad>   padding token
        2  <sos>   start-of-sequence token
        3  <eos>   end-of-sequence token

    Args:
        min_freq (int): Minimum corpus frequency required to add a token.
    """

    UNK_IDX = 0
    PAD_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.itos = list(self.SPECIAL_TOKENS)           # index → token
        self.stoi = {t: i for i, t in enumerate(self.itos)}  # token → index

    def build_from_token_lists(self, token_lists):
        """
        Populate vocabulary from a list of tokenised sentences.
        Only tokens appearing >= min_freq times are added.

        Args:
            token_lists: list[list[str]]
        """
        freq = Counter()
        for tokens in token_lists:
            freq.update(tokens)
        for token in sorted(freq.keys()):
            if freq[token] >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, tokens):
        """Map a list of token strings to integer indices."""
        return [self.stoi.get(t, self.UNK_IDX) for t in tokens]

    def decode(self, indices, strip_special=True):
        """Reconstruct a sentence string from integer indices."""
        skip = set(self.SPECIAL_TOKENS) if strip_special else set()
        return " ".join(
            self.itos[i] for i in indices
            if 0 <= i < len(self.itos) and self.itos[i] not in skip
        )

    def lookup_token(self, idx):
        """Return token string for an index (used by evaluate_bleu)."""
        return self.itos[idx] if 0 <= idx < len(self.itos) else "<unk>"

    def __len__(self):
        return len(self.itos)


# ══════════════════════════════════════════════════════════════════════
#  MULTI30K DATASET  —  skeleton-compatible
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    PyTorch Dataset wrapping the HuggingFace bentrevett/multi30k dataset.

    Follows the skeleton signature exactly:
        __init__(self, split='train')

    After construction the instance exposes:
        .src_vocab   Vocabulary   German vocabulary
        .tgt_vocab   Vocabulary   English vocabulary
        .src_data    list[list[int]]
        .tgt_data    list[list[int]]

    Vocab sharing across splits is handled externally via
    set_vocab(src_vocab, tgt_vocab) before calling process_data()
    — see build_dataloaders() for the canonical usage pattern.
    """

    # Shared spaCy models — loaded once for all instances
    _nlp_de = None
    _nlp_en = None

    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split

        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        raw = load_dataset("bentrevett/multi30k", trust_remote_code=True)
        self._raw = raw[split]

        # Load spacy tokenizers for de and en
        Multi30kDataset._load_spacy()

        # Tokenise every sentence into lowercase string lists
        self._src_tok = [Multi30kDataset._tok_de(row["de"]) for row in self._raw]
        self._tgt_tok = [Multi30kDataset._tok_en(row["en"]) for row in self._raw]

        # Vocab and encoded data — populated by build_vocab() + process_data()
        self.src_vocab = None
        self.tgt_vocab = None
        self.src_data  = None
        self.tgt_data  = None

    # ------------------------------------------------------------------
    # spaCy helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load_spacy(cls):
        if cls._nlp_de is None:
            cls._nlp_de = spacy.load("de_core_news_sm")
        if cls._nlp_en is None:
            cls._nlp_en = spacy.load("en_core_web_sm")

    @classmethod
    def _tok_de(cls, sentence):
        return [tok.text.lower() for tok in cls._nlp_de.tokenizer(sentence.strip())]

    @classmethod
    def _tok_en(cls, sentence):
        return [tok.text.lower() for tok in cls._nlp_en.tokenizer(sentence.strip())]

    # ------------------------------------------------------------------
    # Skeleton-required public methods
    # ------------------------------------------------------------------

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_vocab = Vocabulary(min_freq=2)
        tgt_vocab = Vocabulary(min_freq=2)
        src_vocab.build_from_token_lists(self._src_tok)
        tgt_vocab.build_from_token_lists(self._tgt_tok)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        return src_vocab, tgt_vocab

    def set_vocab(self, src_vocab, tgt_vocab):
        """
        Assign pre-built vocabularies (used for val/test splits so they
        share the train vocabulary without rebuilding).

        Args:
            src_vocab : Vocabulary built from the training split (German).
            tgt_vocab : Vocabulary built from the training split (English).
        """
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.

        Each sequence is wrapped: [<sos>, tok1, tok2, ..., <eos>]
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "Vocabulary not set. Call build_vocab() or set_vocab() first."
            )
        sos = Vocabulary.SOS_IDX
        eos = Vocabulary.EOS_IDX
        src_data, tgt_data = [], []
        for src_toks, tgt_toks in zip(self._src_tok, self._tgt_tok):
            src_data.append([sos] + self.src_vocab.encode(src_toks) + [eos])
            tgt_data.append([sos] + self.tgt_vocab.encode(tgt_toks) + [eos])
        self.src_data = src_data
        self.tgt_data = tgt_data
        return src_data, tgt_data

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = torch.tensor(self.src_data[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_data[idx], dtype=torch.long)
        return src, tgt


# ══════════════════════════════════════════════════════════════════════
#  COLLATION AND DATALOADER FACTORY
# ══════════════════════════════════════════════════════════════════════

def build_collate_fn(pad_idx=Vocabulary.PAD_IDX):
    """
    Return a collate function that pads sequences to the longest in the batch.

    Returns:
        collate_fn(batch) -> (src_padded [B, S], tgt_padded [B, T])
    """
    def collate_fn(batch):
        src_list, tgt_list = zip(*batch)
        src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
        return src_padded, tgt_padded
    return collate_fn


def build_dataloaders(batch_size=128, num_workers=0, min_freq=2):
    """
    Build train / val / test DataLoaders sharing a single vocabulary.

    The vocabulary is built exclusively from the training split.
    Val and test instances receive the same vocab via set_vocab() to
    guarantee no token distribution leakage across splits.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    # ── Training split: build vocab from scratch ──────────────────────
    train_ds = Multi30kDataset(split="train")
    train_ds.build_vocab()
    train_ds.process_data()

    # ── Val / test: reuse train vocabulary ────────────────────────────
    val_ds = Multi30kDataset(split="validation")
    val_ds.set_vocab(train_ds.src_vocab, train_ds.tgt_vocab)
    val_ds.process_data()

    test_ds = Multi30kDataset(split="test")
    test_ds.set_vocab(train_ds.src_vocab, train_ds.tgt_vocab)
    test_ds.process_data()

    collate = build_collate_fn(Vocabulary.PAD_IDX)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate, num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, train_ds.src_vocab, train_ds.tgt_vocab


# ══════════════════════════════════════════════════════════════════════
#  QUICK SANITY CHECK  —  python dataset.py
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading Multi30k (train) …")
    train_ds = Multi30kDataset(split="train")
    train_ds.build_vocab()
    train_ds.process_data()

    print(f"  Samples       : {len(train_ds)}")
    print(f"  Source vocab  : {len(train_ds.src_vocab):,}  (German)")
    print(f"  Target vocab  : {len(train_ds.tgt_vocab):,}  (English)")

    src, tgt = train_ds[0]
    print(f"\nFirst pair (ids) : src={src[:8].tolist()}…  tgt={tgt[:8].tolist()}…")
    print(f"Decoded src      : {train_ds.src_vocab.decode(src.tolist())}")
    print(f"Decoded tgt      : {train_ds.tgt_vocab.decode(tgt.tolist())}")

    print("\nLoading validation split …")
    val_ds = Multi30kDataset(split="validation")
    val_ds.set_vocab(train_ds.src_vocab, train_ds.tgt_vocab)
    val_ds.process_data()
    print(f"  Samples : {len(val_ds)}")

    print("Loading test split …")
    test_ds = Multi30kDataset(split="test")
    test_ds.set_vocab(train_ds.src_vocab, train_ds.tgt_vocab)
    test_ds.process_data()
    print(f"  Samples : {len(test_ds)}")