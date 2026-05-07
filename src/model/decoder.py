# The Decoder takes the encoded image features and the text generated so far, 
# and predicts the next word in the caption.

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    # Sinusoidal Positional Encoding: Adds information about the position of each word in the sequence.
    # Since Transformers don't have inherent sequential awareness, we add these "positional encodings" to the token embeddings.
    # Sinusoidal functions are used because they allow the model to generalize to sequence lengths longer than those seen during training.
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class CaptionDecoder(nn.Module):
    # vocab_size: Size of the vocabulary (number of unique words).
    # d_model: Dimension of the model (matches the output of the encoder).
    # nhead: Number of attention heads in the Transformer.
    # num_layers: Number of Transformer decoder layers.
    # dim_feedforward: Dimension of the feedforward network in the Transformer.
    # dropout: Dropout rate.
    # max_len: Maximum length of a caption.

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 200,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        captions: torch.Tensor,       # [B, T]  token indices
        visual_tokens: torch.Tensor,  # [B, N, d_model]  from encoder
        pad_mask: torch.Tensor = None,  # [B, T]  True at <pad> positions
    ) -> torch.Tensor:                # [B, T, vocab_size]
        T = captions.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=captions.device
        )

        # Scale embeddings so positional encoding doesn't dominate them
        x = self.token_embed(captions) * math.sqrt(self.d_model)  # [B, T, d_model]
        x = self.pos_encoding(x)

        x = self.transformer(
            tgt=x,
            memory=visual_tokens,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )
        return self.output_proj(x)  # [B, T, vocab_size]

    # ------------------------------------------------------------------
    # Greedy decoding
    # ------------------------------------------------------------------

    # Greedy Decoding: Generates the most likely next word at each step.
    # It simply picks the word with the highest probability and adds it to the sequence.
    # Fast but not always optimal.
    def generate_greedy(
        self,
        visual_tokens: torch.Tensor,  # [1, N, d_model]
        start_idx: int,
        end_idx: int,
        max_len: int = 50,
    ) -> list[int]:
        device = visual_tokens.device
        tokens = [start_idx]

        for _ in range(max_len):
            inp = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = self.forward(inp, visual_tokens)   # [1, T, vocab_size]
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            if next_token == end_idx:
                break

        return tokens

    # ------------------------------------------------------------------
    # Beam search decoding
    # ------------------------------------------------------------------

    # Beam Search: Explores multiple possible captions at each step and keeps the top 'beam_size' candidates.
    # It keeps the 'beam_size' most probable sequences at each step, rather than just the single most probable one.
    # This explores the search space more broadly and usually results in better captions.
    def generate_beam(
        self,
        visual_tokens: torch.Tensor,  # [1, N, d_model]
        start_idx: int,
        end_idx: int,
        beam_size: int = 3,
        max_len: int = 50,
    ) -> list[int]:
        device = visual_tokens.device

        # Each beam is (cumulative_log_prob, token_sequence)
        beams: list[tuple[float, list[int]]] = [(0.0, [start_idx])]
        completed: list[tuple[float, list[int]]] = []

        for _ in range(max_len):
            candidates: list[tuple[float, list[int]]] = []
        
            for score, seq in beams:
                if seq[-1] == end_idx:
                    completed.append((score, seq))
                    continue

                inp = torch.tensor([seq], dtype=torch.long, device=device)
                logits = self.forward(inp, visual_tokens)          # [1, T, vocab_size]
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)

                top_log_probs, top_tokens = log_probs.topk(beam_size)
                for log_prob, token in zip(top_log_probs.tolist(), top_tokens.tolist()):
                    candidates.append((score + log_prob, seq + [token]))

            if not candidates:
                break

            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        completed.extend(beams)
        _, best_seq = max(completed, key=lambda x: x[0])
        return best_seq
