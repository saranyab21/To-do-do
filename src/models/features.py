"""Frozen features for the linear probe.

Qwen features come from one forward pass of the frozen thinker with the primary-pattern
prompt (output_hidden_states). For each requested layer l (0 = input embeddings, i.e.
the merged vision-tower output at the image positions):

    img_L<l>   mean hidden state over the image tokens. The image precedes the question in
               the chat and attention is causal, so these do not depend on the question.
    last_L<l>  hidden state at the last prompt token, where the answer would be produced
               (question-conditioned; skipped for l = 0, which is just a token embedding).

Colour statistics are a trivial-feature baseline: a probe on the big model has to beat
what stain colour alone predicts.
"""

from __future__ import annotations

import numpy as np


def image_token_id(model, processor) -> int:
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        for name in ("image_token_id", "image_token_index"):
            v = getattr(cfg, name, None) if cfg is not None else None
            if isinstance(v, int):
                return v
    return processor.tokenizer.convert_tokens_to_ids(processor.image_token)


def qwen_features(model, processor, messages, image, layers, img_tok: int) -> dict:
    import torch

    from src.models.zeroshot import prepare_inputs

    inputs = prepare_inputs(model, processor, messages, image)
    with torch.inference_mode():
        hs = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
    if max(layers) >= len(hs):
        raise ValueError(f"layer {max(layers)} requested, model has {len(hs) - 1} layers")
    is_img = inputs["input_ids"][0] == img_tok
    n_img = int(is_img.sum())
    expected = int(inputs["image_grid_thw"][0].prod()) // 4
    if n_img != expected:
        raise RuntimeError(f"{n_img} image tokens found, {expected} expected")
    out = {}
    for layer in layers:
        h = hs[layer][0].float()
        out[f"img_L{layer}"] = h[is_img].mean(0).cpu().numpy()
        if layer > 0:
            out[f"last_L{layer}"] = h[-1].cpu().numpy()
    return out


def colour_features(rgb: np.ndarray, bins: int = 8) -> np.ndarray:
    """Mean/std of RGB and HSV over stained pixels, a hue/saturation/value histogram
    (bins each) and the stained fraction: 12 + 3*bins + 1 numbers. Glass (saturation
    <= 0.07, the same threshold as the tissue crop) is excluded."""
    from matplotlib.colors import rgb_to_hsv

    arr = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    hsv = rgb_to_hsv(arr)
    tissue = hsv[:, 1] > 0.07
    if tissue.sum() < 100:
        tissue[:] = True
    a, h = arr[tissue], hsv[tissue]
    hists = [np.histogram(h[:, j], bins=bins, range=(0, 1))[0] / len(h) for j in range(3)]
    return np.concatenate([a.mean(0), a.std(0), h.mean(0), h.std(0), *hists, [tissue.mean()]])
