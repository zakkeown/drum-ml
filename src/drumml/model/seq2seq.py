"""MT3 / T5-style encoder-decoder transcription model.

A standard "Attention Is All You Need" transformer adapted for automatic drum
transcription: a continuous feature sequence (e.g. log-mel or MERT frames) is
projected into the model dimension, given sinusoidal positional encodings and
run through a transformer encoder. A token decoder (MT3-style absolute-time
event vocabulary) attends to the encoder memory and predicts the next event
token autoregressively.

Conventions (see ``docs/step3_contracts.md``):
- float32 feature tensors, int64 token tensors.
- padding masks are ``bool`` with ``True == PAD`` (PyTorch ``key_padding_mask``
  semantics, where ``True`` positions are ignored by attention).
- everything must run on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Seq2SeqConfig:
    """Hyper-parameters for :class:`Seq2SeqADT`.

    ``feature_dim`` (encoder input width) and ``vocab_size`` (decoder token
    vocabulary) are required; the rest default to a T5-base-ish configuration.
    """

    feature_dim: int
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 2048


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encodings (non-learned), added to inputs.

    Shapes follow ``batch_first`` transformers: input/output ``(B, T, d_model)``.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # Guard against odd d_model: the cosine slice may be one element shorter.
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        # (1, max_len, d_model) so it broadcasts over the batch dimension.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.pe.size(1)}; "
                "increase Seq2SeqConfig.max_len"
            )
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class Seq2SeqADT(nn.Module):
    """T5/MT3-style transformer encoder-decoder for drum transcription."""

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model

        # Encoder side: continuous features -> d_model -> +pos -> transformer.
        self.input_proj = nn.Linear(config.feature_dim, d_model)
        self.encoder_pos = SinusoidalPositionalEncoding(
            d_model, config.max_len, config.dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.n_encoder_layers, enable_nested_tensor=False
        )

        # Decoder side: token embedding -> +pos -> transformer w/ causal + cross-attn.
        self.token_embedding = nn.Embedding(config.vocab_size, d_model)
        self.decoder_pos = SinusoidalPositionalEncoding(
            d_model, config.max_len, config.dropout
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.n_decoder_layers)

        # Output projection to vocabulary logits.
        self.output_head = nn.Linear(d_model, config.vocab_size)

        self._emb_scale = math.sqrt(d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Xavier init for projection weights gives the transformer a sane start.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -- building blocks --------------------------------------------------

    def encode(
        self,
        features: torch.Tensor,
        feature_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode features ``(B, T, feature_dim)`` -> memory ``(B, T, d_model)``."""
        x = self.input_proj(features)
        x = self.encoder_pos(x)
        return self.encoder(x, src_key_padding_mask=feature_padding_mask)

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode ``tgt_tokens`` (B, L) against ``memory`` -> logits (B, L, vocab)."""
        seq_len = tgt_tokens.size(1)
        y = self.token_embedding(tgt_tokens) * self._emb_scale
        y = self.decoder_pos(y)
        # Bool causal mask (True == not allowed to attend), matching the bool
        # key_padding_mask convention so attention sees a single mask dtype.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt_tokens.device),
            diagonal=1,
        )
        out = self.decoder(
            y,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output_head(out)

    # -- public API -------------------------------------------------------

    def forward(
        self,
        features: torch.Tensor,
        tgt_tokens: torch.Tensor,
        feature_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            features: ``(B, T, feature_dim)`` float32 encoder inputs.
            tgt_tokens: ``(B, L)`` int64 decoder input tokens.
            feature_padding_mask: ``(B, T)`` bool, ``True == PAD`` (optional).
            tgt_padding_mask: ``(B, L)`` bool, ``True == PAD`` (optional).

        Returns:
            ``(B, L, vocab_size)`` logits.
        """
        memory = self.encode(features, feature_padding_mask)
        return self.decode(
            tgt_tokens,
            memory,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=feature_padding_mask,
        )

    @torch.no_grad()
    def greedy_decode(
        self,
        features: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        feature_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive greedy decoding.

        Each sequence starts with ``bos_id`` and stops contributing new tokens
        once ``eos_id`` is produced. Decoding halts when every sequence has
        emitted EOS or when ``max_len`` total tokens have been generated.

        Returns:
            ``(B, <=max_len)`` int64 tensor; column 0 is always ``bos_id``.
        """
        was_training = self.training
        self.eval()
        try:
            device = features.device
            batch_size = features.size(0)
            memory = self.encode(features, feature_padding_mask)

            tokens = torch.full(
                (batch_size, 1), bos_id, dtype=torch.long, device=device
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max(0, max_len - 1)):
                logits = self.decode(
                    tokens, memory, memory_padding_mask=feature_padding_mask
                )
                next_token = logits[:, -1, :].argmax(dim=-1)
                # Sequences that already hit EOS keep emitting EOS as filler.
                next_token = torch.where(
                    finished, torch.full_like(next_token, eos_id), next_token
                )
                tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
                finished = finished | (next_token == eos_id)
                if bool(finished.all()):
                    break

            return tokens
        finally:
            if was_training:
                self.train()
