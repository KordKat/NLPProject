"""KeawGPT model adapter.

Separates the model's generation protocol (chat template, streaming, stop strings, reasoning
segmentation, context fitting) from application logic. A single model instance is loaded at startup
and shared by every stage (planning, filtering, gap analysis, synthesis) via one generation lock.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from app.agent.tool_parser import split_reasoning
from app.config import get_settings

log = logging.getLogger("stemos.model")
settings = get_settings()

STOP_STRINGS = ["</tools_call>", "</tool_call>", "\n# USER:", "\n# TOOL:", "\n# SYSTEM:"]
TURN_MARKERS = ("\n# USER:", "\n# TOOL:", "\n# SYSTEM:")


class ModelUnavailableError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


@dataclass
class GenerationParams:
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.9
    enable_reasoning: bool = False


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def apply_yarn(config: Any, factor: float) -> None:
    """Enable YaRN RoPE scaling on a loaded config: max positions × factor.

    transformers ≥ 5 keeps RoPE settings in `config.rope_parameters` (incl. rope_theta);
    4.x uses `config.rope_scaling`. Idempotent: a config that already uses YaRN is left alone."""
    current = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    if (current.get("rope_type") or current.get("type")) == "yarn":
        return
    native = int(getattr(config, "max_position_embeddings", 32768) or 32768)
    yarn = {"rope_type": "yarn", "factor": float(factor), "original_max_position_embeddings": native}
    if getattr(config, "rope_parameters", None) is not None:
        config.rope_parameters = {**current, **yarn}
    else:
        config.rope_scaling = yarn
    config.max_position_embeddings = int(native * factor)
    log.info("YaRN RoPE scaling ×%.1f: %d → %d positions", factor, native, config.max_position_embeddings)


class KeawEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.max_context_len = settings.llm_max_context_tokens
        self.status = "not_loaded"
        self.error = ""
        self.device = "cpu"
        self.quantization = "none"
        self._lock = threading.Lock()  # one generate() at a time on the shared model

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.status = "loading"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.truncation_side = "left"

            config = AutoConfig.from_pretrained(self.model_name)
            if settings.llm_rope_scaling_factor > 1.0:
                apply_yarn(config, settings.llm_rope_scaling_factor)
            # the fine-tune's config ships use_cache=false (training leftover); generation needs the KV cache
            config.use_cache = True
            model_max = int(getattr(config, "max_position_embeddings", 4096) or 4096)
            self.max_context_len = min(model_max, settings.llm_max_context_tokens)

            attn_impl = "sdpa"
            if _flash_attn_available() and torch.cuda.is_available():
                attn_impl = "flash_attention_2"
            kwargs: dict[str, Any] = dict(config=config, device_map="auto", attn_implementation=attn_impl, low_cpu_mem_usage=True)

            if torch.cuda.is_available():
                self.device = "cuda"
                if settings.llm_load_in_4bit:
                    try:
                        from transformers import BitsAndBytesConfig

                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                        )
                    except Exception as exc:  # bitsandbytes missing
                        log.warning("4-bit quantization unavailable (%s); loading bf16", exc)
                        kwargs["dtype"] = torch.bfloat16
                else:
                    kwargs["dtype"] = torch.bfloat16
            else:
                kwargs["dtype"] = torch.float32

            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
            except TypeError:
                if "dtype" in kwargs:
                    kwargs["torch_dtype"] = kwargs.pop("dtype")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
            self.model.eval()
            if getattr(self.model, "is_loaded_in_4bit", False):
                self.quantization = "4-bit NF4 (bitsandbytes)"
            else:
                self.quantization = str(next(self.model.parameters()).dtype).replace("torch.", "")
            self.status = "ready"
            log.info("KeawGPT loaded on %s, %s, %.2f GB (context %d)", self.device, self.quantization, self.model.get_memory_footprint() / 1e9, self.max_context_len)
        except Exception as exc:
            self.status = "error"
            self.error = str(exc)
            log.exception("Failed to load %s", self.model_name)

    @property
    def available(self) -> bool:
        return self.status == "ready" and self.model is not None

    # ------------------------------------------------------------------ prompts
    def build_prompt(self, messages: list[dict[str, str]], tools: Optional[list[dict]] = None, enable_reasoning: bool = False) -> str:
        assert self.tokenizer is not None
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_reasoning=enable_reasoning,
        )

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text) // 4
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if self.tokenizer is None:
            return text[: max_tokens * 4]
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True) + " …"

    def fit_prompt(self, messages: list[dict[str, str]], tools: Optional[list[dict]], enable_reasoning: bool, max_new_tokens: int) -> tuple[str, int]:
        """Drop the oldest history turns (never the system prompt or the latest user turn) until the
        prompt fits. Returns (prompt, token_count)."""
        msgs = list(messages)
        reserve = min(max_new_tokens, self.max_context_len // 2)
        budget = max(1024, self.max_context_len - reserve)
        while True:
            prompt = self.build_prompt(msgs, tools, enable_reasoning)
            n = self.count_tokens(prompt)
            if n <= budget:
                return prompt, n
            # find the first droppable message (after system, before the last user message)
            last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=len(msgs) - 1)
            start = 1 if msgs and msgs[0]["role"] == "system" else 0
            if start >= last_user:
                # nothing left to drop: shrink the system/user content as a last resort
                if msgs and msgs[0]["role"] == "system" and len(msgs[0]["content"]) > 4000:
                    # cut from the middle: the rules are at the start and the task guidance at the end
                    over = n - budget
                    sys_c = msgs[0]["content"]
                    keep = max(2000, len(sys_c) - over * 4)
                    head = keep * 2 // 3
                    msgs[0] = {"role": "system", "content": sys_c[:head] + "\n…\n" + sys_c[len(sys_c) - (keep - head):]}
                    continue
                return prompt, n
            del msgs[start]

    # ------------------------------------------------------------------ generation
    def _run_generate(self, streamer, error_box: list, cancel_event: threading.Event, **kwargs) -> None:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):  # type: ignore[override]
                return cancel_event.is_set()

        try:
            with self._lock:
                if cancel_event.is_set():
                    streamer.end()
                    return
                with torch.inference_mode():
                    self.model.generate(streamer=streamer, stopping_criteria=StoppingCriteriaList([_Cancel()]), **kwargs)
        except Exception as exc:  # e.g. CUDA OOM — make sure the reader is released
            error_box.append(exc)
            try:
                streamer.end()
            except Exception:
                pass

    async def stream(
        self,
        prompt: str,
        params: GenerationParams,
        cancel_event: Optional[threading.Event] = None,
    ) -> AsyncIterator[str]:
        if not self.available:
            raise ModelUnavailableError(f"KeawGPT is not available (status={self.status}). {self.error}")
        from transformers import TextIteratorStreamer

        cancel_event = cancel_event or threading.Event()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max(512, self.max_context_len - 256),
        ).to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        max_new = max(32, min(params.max_new_tokens, self.max_context_len - input_len - 8))

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=settings.llm_generation_timeout)
        error_box: list = []
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new,
            do_sample=params.temperature > 0,
            temperature=max(params.temperature, 1e-5),
            top_p=params.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            stop_strings=STOP_STRINGS,
            tokenizer=self.tokenizer,
        )

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        _END = object()

        def reader() -> None:
            try:
                for delta in streamer:
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _END)

        gen_thread = threading.Thread(target=self._run_generate, args=(streamer, error_box, cancel_event), kwargs=gen_kwargs, daemon=True)
        read_thread = threading.Thread(target=reader, daemon=True)
        gen_thread.start()
        read_thread.start()
        finished = False
        try:
            while True:
                item = await queue.get()
                if item is _END:
                    finished = True
                    break
                if isinstance(item, Exception):
                    raise ModelUnavailableError(f"Generation failed: {item}")
                yield item
        finally:
            # only a consumer that stopped EARLY (client disconnect / exception) stops generation;
            # after a normal end the generate thread may still be returning — that is not a cancel
            if not finished and gen_thread.is_alive():
                cancel_event.set()
        await asyncio.to_thread(gen_thread.join, 5)
        if error_box:
            raise ModelUnavailableError(f"Generation failed: {error_box[0]}")
        if cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")

    async def complete(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
        tools: Optional[list[dict]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Non-streaming helper: returns the final answer text (reasoning removed)."""
        params = params or GenerationParams(max_new_tokens=1024, temperature=0.2, enable_reasoning=False)
        prompt, _ = self.fit_prompt(messages, tools, params.enable_reasoning, params.max_new_tokens)
        raw = ""
        async for delta in self.stream(prompt, params, cancel_event):
            raw += delta
        raw = strip_turn_markers(raw)
        _, answer = split_reasoning(raw, final=True)
        return answer.strip()


def strip_turn_markers(raw: str) -> str:
    for stop in TURN_MARKERS:
        if stop in raw:
            raw = raw[: raw.index(stop)]
    return raw.rstrip()


_engine: Optional[KeawEngine] = None


def get_engine() -> KeawEngine:
    global _engine
    if _engine is None:
        _engine = KeawEngine(settings.llm_model_name)
    return _engine
