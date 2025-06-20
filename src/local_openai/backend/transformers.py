from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TransformersBackend:
    def __init__(self, model_name: str, device: str | None = None, **hf_kw):
        self._name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._hf_kw = hf_kw
        self._cache: dict[str, tuple[Any, Any]] = {}

    # ---------------------------------------------------------------------
    def _get(self, name: str):
        if name not in self._cache:
            tok = AutoTokenizer.from_pretrained(name, **self._hf_kw)
            mdl = AutoModelForCausalLM.from_pretrained(
                name,
                device_map=self._device,
                torch_dtype=torch.float16 if "cuda" in str(self._device) else None,
                **self._hf_kw,
            )
            self._cache[name] = (tok, mdl)
        return self._cache[name]

    # ---------------------------------------------------------------------
    def completion(
        self,
        *,
        model: str | None = None,
        is_chat: bool,
        messages: List[Dict[str, str]] | None = None,
        prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **_: Any,
    ):
        model = model or self._name
        tok, mdl = self._get(model)

        if is_chat:
            if hasattr(tok, "apply_chat_template"):
                prompt_text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        else:
            prompt_text = prompt or ""

        ids = tok(prompt_text, return_tensors="pt").to(mdl.device)
        gen = mdl.generate(
            **ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )
        gen_ids = gen[0, ids.input_ids.shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        finish = "length" if gen_ids.shape[-1] >= max_tokens else "stop"

        if is_chat:
            msg = SimpleNamespace(role="assistant", content=text)
            choice = SimpleNamespace(message=msg, finish_reason=finish)
        else:
            choice = SimpleNamespace(text=text, finish_reason=finish)

        return SimpleNamespace(choices=[choice])
