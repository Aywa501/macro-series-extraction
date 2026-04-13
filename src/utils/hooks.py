"""PyTorch forward-hook utilities for activation injection into BERT.

Usage
-----
    from hooks import ActivationInjectionHook, inject_at_layer

    # One-shot: run a forward pass with injection at layer k, return CLS vector
    cls_vec = inject_at_layer(model, tokenizer, sentence, layer=k,
                              direction=temporal_dir_k, lambda_val=1.5)

    # Batch: process many sentences with the same (layer, lambda)
    cls_vecs = batch_inject_at_layer(model, tokenizer, sentences, layer=k,
                                     direction=temporal_dir_k, lambda_val=1.5,
                                     device=device)
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional


class ActivationInjectionHook:
    """Forward hook that adds ``lambda_val * direction`` to the CLS token's
    hidden state at the output of a BertLayer.

    BERT's BertLayer returns a tuple whose first element is the hidden-state
    tensor of shape ``(batch, seq_len, hidden_dim)``.  We perturb only the
    CLS position (index 0 in the sequence dimension).

    Parameters
    ----------
    direction : torch.Tensor
        Unit vector of shape ``(hidden_dim,)`` on the correct device.
    lambda_val : float
        Scaling factor (λ).  Positive = forward in temporal direction.
    """

    def __init__(self, direction: torch.Tensor, lambda_val: float) -> None:
        self.direction = direction   # (hidden_dim,)
        self.lambda_val = lambda_val
        self.handle: Optional[torch.utils.hooks.RemovableHook] = None

    def __call__(
        self,
        module: torch.nn.Module,
        input: tuple,
        output,
    ):
        # Newer transformers versions return a plain tensor from BertLayer
        # when attention outputs are not requested; older versions return a tuple.
        if isinstance(output, tuple):
            hidden = output[0].clone()               # (batch, seq, hidden)
            hidden[:, 0, :] += self.lambda_val * self.direction
            return (hidden,) + output[1:]
        else:
            hidden = output.clone()                  # (batch, seq, hidden)
            hidden[:, 0, :] += self.lambda_val * self.direction
            return hidden

    def register(self, layer: torch.nn.Module) -> "ActivationInjectionHook":
        self.handle = layer.register_forward_hook(self)
        return self

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def inject_at_layer(
    model: torch.nn.Module,
    tokenizer,
    sentence: str,
    layer: int,
    direction: np.ndarray,
    lambda_val: float,
    device: torch.device,
    max_length: int = 512,
) -> np.ndarray:
    """Single-sentence convenience wrapper.  Returns final CLS as numpy (768,)."""
    return batch_inject_at_layer(
        model, tokenizer, [sentence], layer=layer,
        direction=direction, lambda_val=lambda_val,
        device=device, max_length=max_length,
    )[0]


def batch_inject_at_layer(
    model: torch.nn.Module,
    tokenizer,
    sentences: list[str],
    layer: int,
    direction: np.ndarray,
    lambda_val: float,
    device: torch.device,
    max_length: int = 512,
) -> np.ndarray:
    """Batch forward pass with activation injection at one layer.

    Parameters
    ----------
    model : BertModel (already in eval mode on ``device``)
    tokenizer : BertTokenizerFast
    sentences : list of str
    layer : int, 0-indexed BERT encoder layer at whose *output* to inject
    direction : np.ndarray of shape ``(768,)``
    lambda_val : float
    device : torch.device

    Returns
    -------
    np.ndarray of shape ``(len(sentences), 768)`` — final-layer CLS vectors
    """
    dir_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

    hook = ActivationInjectionHook(dir_tensor, lambda_val)
    hook.register(model.encoder.layer[layer])

    try:
        enc = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model(**enc)

        # Take the final-layer CLS token
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()  # (batch, 768)
    finally:
        hook.remove()

    return cls


def embed_sentences(
    model: torch.nn.Module,
    tokenizer,
    sentences: list[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """Embed sentences with NO injection; return all-layer CLS tokens.

    Returns
    -------
    np.ndarray of shape ``(n_sentences, n_layers, 768)``
    """
    model.eval()
    all_hidden: list[np.ndarray] = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        # hidden_states is a tuple of (n_layers+1) tensors each (batch, seq, 768)
        # index 0 = embedding layer, indices 1..12 = transformer layers
        layer_cls = np.stack(
            [hs[:, 0, :].cpu().numpy() for hs in out.hidden_states[1:]],
            axis=1,
        )  # (batch, n_layers, 768)
        all_hidden.append(layer_cls)

    return np.concatenate(all_hidden, axis=0)  # (n_sentences, n_layers, 768)
