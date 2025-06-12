from __future__ import annotations

"""Minimal drop-in replacement for `openai.OpenAI` that runs inference locally
with HuggingFace Transformers. It supports the subset of the OpenAI Python
client interface used by the `verifiers` library:

• `client.chat.completions.create(model=..., messages=[...], **sampling)`
  – returns an object whose first choice has a `.message.content` string and a
    `.finish_reason` field ("length" when `max_new_tokens` is reached).

• `client.completions.create(model=..., prompt=..., **sampling)`
  – returns an object whose first choice has a `.text` string and
    `.finish_reason`.

Only the following sampling kwargs are recognised: `max_tokens`,
`temperature`, `top_p`, `stop`. Extra kwargs are ignored.

Example
-------
>>> client = LocalOpenAI(model_name_or_path="Qwen/Qwen1.5-1.8B")
>>> resp = client.chat.completions.create(
...     model="Qwen/Qwen1.5-1.8B",
...     messages=[
...         {"role": "system", "content": "You are a poet."},
...         {"role": "user", "content": "Write a haiku about the moon."},
...     ],
...     max_tokens=64,
... )
>>> print(resp.choices[0].message.content)
Silvery night sky…
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["LocalOpenAI"]


@dataclass
class _ChatMessage:
    role: str
    content: str


@dataclass
class _Choice:
    # For chat completions
    message: Optional[_ChatMessage] = None
    # For plain completions
    text: Optional[str] = None
    finish_reason: str | None = None


@dataclass
class _Response:
    choices: List[_Choice]


class _CompletionsEndpoint:
    """Implements the `.completions.create` endpoint."""

    def __init__(self, parent: "LocalOpenAI"):
        self._parent = parent

    def create(self, model: str, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
               top_p: float = 1.0, stop: Optional[List[str]] = None, **kwargs: Any) -> _Response:
        tokenizer, model_ref = self._parent._get_model(model)
        input_ids = tokenizer(prompt, return_tensors="pt").to(model_ref.device)
        gen_out = model_ref.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = gen_out[0, input_ids.input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        finish_reason = "length" if generated_ids.shape[-1] >= max_tokens else "stop"
        choice = _Choice(text=text, finish_reason=finish_reason)
        return _Response(choices=[choice])


class _ChatCompletionsEndpoint:
    """Implements the `.chat.completions.create` endpoint."""

    def __init__(self, parent: "LocalOpenAI"):
        self._parent = parent

    def create(self, model: str, messages: List[Dict[str, str]],
               max_tokens: int = 512, temperature: float = 0.7, top_p: float = 1.0,
               stop: Optional[List[str]] = None, **kwargs: Any) -> _Response:
        tokenizer, model_ref = self._parent._get_model(model)
        # Use tokenizer's chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                        add_generation_prompt=True)
        else:
            # Fallback: simple concatenation
            prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        input_ids = tokenizer(prompt_text, return_tensors="pt").to(model_ref.device)
        gen_out = model_ref.generate(
            **input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = gen_out[0, input_ids.input_ids.shape[1]:]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True)
        finish_reason = "length" if generated_ids.shape[-1] >= max_tokens else "stop"
        msg = _ChatMessage(role="assistant", content=content)
        choice = _Choice(message=msg, finish_reason=finish_reason)
        return _Response(choices=[choice])


class _ChatEndpoint:
    def __init__(self, parent: "LocalOpenAI"):
        self.completions = _ChatCompletionsEndpoint(parent)


class LocalOpenAI:
    """Local replacement for the `openai.OpenAI` client using Transformers."""

    def __init__(self, model_name_or_path: str, device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
                 torch_dtype: torch.dtype | None = torch.float16 if torch.cuda.is_available() else None,
                 trust_remote_code: bool = True):
        self._default_model_name = model_name_or_path
        self._device = device
        self._dtype = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._model_cache: Dict[str, tuple[AutoTokenizer, AutoModelForCausalLM]] = {}

        # Pre-load the default model
        self._get_model(model_name_or_path)

        # Sub-endpoints
        self.completions = _CompletionsEndpoint(self)
        self.chat = _ChatEndpoint(self)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _get_model(self, name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        if name not in self._model_cache:
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=self._trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                device_map=self._device,
                torch_dtype=self._dtype,
                trust_remote_code=self._trust_remote_code,
            )
            self._model_cache[name] = (tokenizer, model)
        return self._model_cache[name] 