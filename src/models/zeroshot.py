"""Zero-shot answer-code probabilities from Qwen2.5-Omni (thinker only).

The assessor's base distribution q0(y | x, q_d) is read off the next-token distribution
right after the assistant turn starts: one forward pass, no sampling. Each answer code is
a single tokenizer token (Qwen splits numbers into single digits), so

    log q_full(y)  = log softmax(logits)[token(y)]          (full-vocabulary log-prob)
    mass           = sum_y q_full(y)                         (format adherence)
    log q0(y)      = log q_full(y) - log mass                 (renormalised over codes)

Only q_full is stored; renormalisation happens at evaluation time.

torch/transformers are imported inside functions so the rest of the package (and the
tests) runs without the model stack.
"""

from __future__ import annotations

from collections.abc import Sequence


def code_token_ids(tokenizer, codes: Sequence[int | str]) -> dict[str, int]:
    """Token id of each answer code; fails loudly if a code is not exactly one token."""
    ids = {}
    for c in codes:
        enc = tokenizer.encode(str(c), add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"answer code {c!r} is {len(enc)} tokens, expected 1")
        ids[str(c)] = enc[0]
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"answer codes share token ids: {ids}")
    return ids


def build_messages(prompts: dict, dimension: str, image) -> list[dict]:
    """Chat for one (image, question). prompts: dict with 'system', 'context' and one
    question per dimension under 'questions'. `image` is a path or PIL image."""
    msgs = []
    if prompts.get("system"):
        msgs.append({"role": "system", "content": [{"type": "text", "text": prompts["system"]}]})
    text = " ".join(t for t in (prompts.get("context", ""), prompts["questions"][dimension]) if t)
    msgs.append({"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": text}]})
    return msgs


# Modules kept in fp16 under quantisation. Recent transformers versions match these against
# full module paths, and the vision tower can sit at "visual", "model.visual" or
# "thinker.visual", so every form is listed. Runs before 2026-11-26 used only
# ["visual", "lm_head"], which left the vision tower quantised (found when 8-bit failed on
# its 3420-wide MLP); load_thinker now verifies the result instead of trusting the list.
FP16_MODULES = ["visual", "model.visual", "thinker.visual", "lm_head", "model.lm_head"]


def quantised_layers(model) -> list[str]:
    """Names of the bitsandbytes (4-bit or 8-bit) linear layers in the model."""
    return [n for n, m in model.named_modules()
            if type(m).__name__ in ("Linear4bit", "Linear8bitLt")]


def load_thinker(model_id: str, quant: str = "4bit", revision: str | None = None):
    """Thinker (text-output part) of Qwen2.5-Omni plus its processor.

    quant: '4bit' (NF4, fp16 compute; fits an 8 GB RTX 2080), '8bit', or 'fp16'. Only the
    language model (and the unused audio tower) is quantised; the vision tower and lm_head
    stay in fp16, which is checked after loading. Turing GPUs (RTX 20xx) have no bfloat16.
    """
    import torch
    from transformers import (
        BitsAndBytesConfig,
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    kw = {"dtype": torch.float16, "device_map": "auto", "revision": revision}
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=FP16_MODULES)
    elif quant == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_skip_modules=FP16_MODULES)
    elif quant != "fp16":
        raise ValueError(f"unknown quant {quant!r}")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, **kw).eval()
    q = quantised_layers(model)
    bad = [n for n in q if "visual" in n or "lm_head" in n]
    if bad:
        raise RuntimeError(f"{len(bad)} vision/lm_head layers were quantised, e.g. {bad[:3]}; "
                           "add their path prefix to FP16_MODULES")
    if quant != "fp16" and not q:
        raise RuntimeError("nothing was quantised; check the quantisation config")
    print(f"quantised {len(q)} linear layers ({quant}); vision tower and lm_head in fp16")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id, revision=revision)
    return model, processor


def prepare_inputs(model, processor, messages: list[dict], image):
    """Tokenised chat + pixel values on the model's device (pixels in fp16)."""
    import torch

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    return inputs


def score_codes(model, processor, messages: list[dict], image, code_ids: dict[str, int]) -> dict:
    """Full-vocabulary log-prob of each code at the first answer position."""
    import torch

    inputs = prepare_inputs(model, processor, messages, image)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    top = int(torch.argmax(logp))
    out = {f"lp_{c}": float(logp[i]) for c, i in code_ids.items()}
    out["mass"] = float(sum(torch.exp(logp[i]) for i in code_ids.values()))
    out["top_token"] = processor.tokenizer.decode([top])
    out["top_lp"] = float(logp[top])
    out["finite"] = bool(torch.isfinite(logits).all())
    out["n_input_tokens"] = int(inputs["input_ids"].shape[1])
    if "image_grid_thw" in inputs:
        out["n_vision_tokens"] = int(inputs["image_grid_thw"][0].prod()) // 4  # 2x2 merge
    return out
