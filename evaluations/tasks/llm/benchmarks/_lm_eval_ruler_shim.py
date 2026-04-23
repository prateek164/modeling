"""
Shim that patches RULER for:
  1. OpenAI tiktoken tokenizer (when --apply_chat_template is present)
  2. Groq / Llama HF AutoTokenizer (when RULER_HF_TOKENIZER or
     RULER_TOKENIZER_PATH is set, in addition to --apply_chat_template).
     This runs AFTER the OpenAI block and overrides it when active, so
     hosted Llama-style models served via OpenAI-compatible APIs (Groq) get
     the correct tokenizer for RULER prompt-length accounting.
  3. Caching synthetic dataset generation across model runs

Called by run.py for all RULER evaluations.
"""
import functools
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

CACHE_DIR = Path(os.environ.get("RULER_CACHE_DIR", "cache/ruler_datasets"))

# ---------------------------------------------------------------------------
# Dataset generation cache — wraps RULER generation functions so synthetic
# data is built once and reused across model runs.
# ---------------------------------------------------------------------------
def _cache_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(**kwargs):
        cache_key_data = {
            "func": fn.__module__ + "." + fn.__qualname__,
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        }
        key = hashlib.sha256(
            json.dumps(cache_key_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        cache_path = CACHE_DIR / f"{fn.__name__}_{key}.pkl"

        if cache_path.exists():
            print(f"  [ruler-cache] Loading {fn.__name__} from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print(f"  [ruler-cache] Generating {fn.__name__} (will cache to {cache_path})")
        result = fn(**kwargs)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        return result
    return wrapper


def _patch_ruler_caching():
    """Wrap all RULER generation functions with disk caching."""
    modules_and_funcs = []

    from lm_eval.tasks.ruler import niah_utils
    for name in dir(niah_utils):
        if name.startswith("niah_"):
            obj = getattr(niah_utils, name)
            if callable(obj):
                modules_and_funcs.append((niah_utils, name, obj))

    try:
        from lm_eval.tasks.ruler import cwe_utils
        for name in ["get_cw_dataset"]:
            if hasattr(cwe_utils, name):
                modules_and_funcs.append((cwe_utils, name, getattr(cwe_utils, name)))
    except ImportError:
        pass

    try:
        from lm_eval.tasks.ruler import fwe_utils
        for name in ["fwe_download"]:
            if hasattr(fwe_utils, name):
                modules_and_funcs.append((fwe_utils, name, getattr(fwe_utils, name)))
    except ImportError:
        pass

    try:
        from lm_eval.tasks.ruler import vt_utils
        for name in ["get_vt_dataset"]:
            if hasattr(vt_utils, name):
                modules_and_funcs.append((vt_utils, name, getattr(vt_utils, name)))
    except ImportError:
        pass

    try:
        from lm_eval.tasks.ruler import qa_utils
        for name in ["get_squad", "get_hotpotqa"]:
            if hasattr(qa_utils, name):
                modules_and_funcs.append((qa_utils, name, getattr(qa_utils, name)))
    except ImportError:
        pass

    patched = 0
    for mod, name, func in modules_and_funcs:
        wrapped = _cache_wrapper(func)
        setattr(mod, name, wrapped)
        patched += 1

    # Ensure short-name imports (used by !function in YAML) also see our patches.
    # lm_eval resolves "!function niah_utils.func" via a relative import that may
    # create a separate sys.modules entry from "lm_eval.tasks.ruler.niah_utils".
    import sys as _sys
    for short, full in [
        ("niah_utils", "lm_eval.tasks.ruler.niah_utils"),
        ("cwe_utils", "lm_eval.tasks.ruler.cwe_utils"),
        ("fwe_utils", "lm_eval.tasks.ruler.fwe_utils"),
        ("vt_utils", "lm_eval.tasks.ruler.vt_utils"),
        ("qa_utils", "lm_eval.tasks.ruler.qa_utils"),
        ("common_utils", "lm_eval.tasks.ruler.common_utils"),
    ]:
        if full in _sys.modules:
            _sys.modules[short] = _sys.modules[full]

    print(f"  [ruler-cache] Patched {patched} generation functions with disk caching")


_patch_ruler_caching()

# ---------------------------------------------------------------------------
# OpenAI tiktoken patch (only when --apply_chat_template is present)
# ---------------------------------------------------------------------------
if "--apply_chat_template" in sys.argv:
    import tiktoken

    class _TiktokenWrapper:
        def __init__(self, encoding_name: str = "o200k_base"):
            self._enc = tiktoken.get_encoding(encoding_name)

        def __call__(self, text: str):
            class _Result:
                def __init__(self, input_ids):
                    self.input_ids = input_ids
            return _Result(self._enc.encode(text))

    _wrapper = _TiktokenWrapper("o200k_base")
    _tok_replacement = lambda *args, **kwargs: _wrapper  # noqa: E731

    from lm_eval.tasks.ruler import common_utils
    common_utils.get_tokenizer = _tok_replacement

    try:
        from lm_eval.tasks.ruler import qa_utils
        qa_utils.get_tokenizer = _tok_replacement
    except (ImportError, AttributeError):
        pass

# ---------------------------------------------------------------------------
# Groq / hosted Llama HF tokenizer patch.
#
# Activated when EITHER of these env vars is set (run.py sets RULER_HF_TOKENIZER
# automatically when provider == "groq"):
#   RULER_TOKENIZER_PATH  — local directory containing HF tokenizer files
#                           (tokenizer.json, tokenizer_config.json,
#                            special_tokens_map.json, chat_template.jinja)
#   RULER_HF_TOKENIZER    — HF model id, e.g. meta-llama/Llama-4-Scout-17B-16E-Instruct
#                           (gated; requires HF_TOKEN env var to be exported)
#
# Runs after the OpenAI block so it cleanly overrides the tiktoken wrapper
# when both are technically applicable. Also requires --apply_chat_template
# (RULER's get_tokenizer is only called in that path).
# ---------------------------------------------------------------------------
if "--apply_chat_template" in sys.argv:
    _hf_tok_source = (
        os.environ.get("RULER_TOKENIZER_PATH")
        or os.environ.get("RULER_HF_TOKENIZER")
    )
    if _hf_tok_source:
        from transformers import AutoTokenizer

        print(f"  [ruler-tokenizer] Loading HF tokenizer from {_hf_tok_source}")
        _hf_tok = AutoTokenizer.from_pretrained(_hf_tok_source)
        _hf_replacement = lambda *args, **kwargs: _hf_tok  # noqa: E731

        from lm_eval.tasks.ruler import common_utils as _cu
        _cu.get_tokenizer = _hf_replacement

        try:
            from lm_eval.tasks.ruler import qa_utils as _qu
            _qu.get_tokenizer = _hf_replacement
        except (ImportError, AttributeError):
            pass

        # ---------------------------------------------------------------
        # Groq strict-validates chat-completions payloads and rejects the
        # spurious top-level `"type": "text"` field that lm_eval injects
        # into every message dict (see TemplateAPI.apply_chat_template in
        # lm_eval/models/api_models.py). OpenAI silently ignores it; Groq
        # returns:  "property 'type' is unsupported".
        #
        # We re-bind apply_chat_template on the TemplateAPI base class so
        # that the message list is serialized verbatim — canonical OpenAI
        # format ({"role": ..., "content": ...} only). Scoped to the HF
        # tokenizer branch, so the OpenAI tiktoken path above is untouched.
        # ---------------------------------------------------------------
        from lm_eval.models.api_models import TemplateAPI, JsonChatStr

        def _apply_chat_template_groq(self, chat_history, add_generation_prompt=True):
            if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
                return self.tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=not add_generation_prompt,
                )
            if self.tokenizer_backend == "remote" and self.tokenized_requests:
                return chat_history
            return JsonChatStr(json.dumps(chat_history, ensure_ascii=False))

        TemplateAPI.apply_chat_template = _apply_chat_template_groq
        print("  [ruler-tokenizer] Patched TemplateAPI.apply_chat_template to drop spurious 'type' field (Groq compat)")

        # ---------------------------------------------------------------
        # Proactive TPM (tokens-per-minute) throttle.
        #
        # lm_eval's only rate-limit defense is `max_retries=5` with
        # exponential backoff, but each retry still costs an API round
        # trip and the request never *progresses* until the bucket has
        # room. At long contexts (32K-131K) we can exhaust max_retries
        # before the bucket refills, killing the run.
        #
        # Solution: maintain a sliding 60-second token-usage window
        # client-side and sleep BEFORE issuing each request if it would
        # breach the budget. This eliminates 429s entirely.
        #
        # Activated when RULER_TPM_LIMIT is set (run.py exports it for
        # the Groq path). Tunable via:
        #   RULER_TPM_LIMIT       — provider's TPM ceiling (e.g. 300000)
        #   RULER_TPM_SAFETY      — fraction of ceiling to actually use
        #                            (default 0.85; leaves 15% headroom
        #                             for token-count estimation error)
        # ---------------------------------------------------------------
        _tpm_limit_str = os.environ.get("RULER_TPM_LIMIT")
        if _tpm_limit_str:
            import time as _time
            from collections import deque

            _TPM_LIMIT = int(_tpm_limit_str)
            _TPM_SAFETY = float(os.environ.get("RULER_TPM_SAFETY", "0.85"))
            _TPM_BUDGET = int(_TPM_LIMIT * _TPM_SAFETY)
            _tpm_window: "deque[tuple[float, int]]" = deque()

            def _estimate_request_tokens(messages) -> int:
                """Cheap upper-bound: count chars in serialized payload / 3.5
                (≈ tiktoken bpe ratio) plus a fixed completion buffer.

                lm_eval passes `messages` as either a tuple/list of
                JsonChatStr (chat completions, our Groq case), a list of
                strings (text completions), a list of token-id lists
                (tokenized requests), or a single JsonChatStr/str.
                """
                total_chars = 0
                if isinstance(messages, (list, tuple)) and messages:
                    first = messages[0]
                    if isinstance(first, JsonChatStr):
                        total_chars = sum(
                            len(m.prompt) for m in messages
                            if isinstance(m, JsonChatStr)
                        )
                    elif isinstance(first, str):
                        total_chars = sum(
                            len(m) for m in messages if isinstance(m, str)
                        )
                    elif isinstance(first, (list, tuple)) and first and isinstance(first[0], int):
                        return sum(len(m) for m in messages) + 256  # token ids
                elif isinstance(messages, JsonChatStr):
                    total_chars = len(messages.prompt)
                elif isinstance(messages, str):
                    total_chars = len(messages)
                est = int(total_chars / 3.5) + 256
                # Hard floor of 1000 tokens — defends against any unexpected
                # input shape silently degrading throttling to a no-op.
                return max(est, 1000)

            def _wait_for_tpm_budget(needed: int) -> None:
                while True:
                    now = _time.monotonic()
                    while _tpm_window and _tpm_window[0][0] < now - 60.0:
                        _tpm_window.popleft()
                    used = sum(t for _, t in _tpm_window)
                    if used + needed <= _TPM_BUDGET:
                        _tpm_window.append((now, needed))
                        return
                    oldest_ts = _tpm_window[0][0]
                    sleep_for = max(0.2, 60.0 - (now - oldest_ts) + 0.2)
                    print(
                        f"  [ruler-throttle] TPM budget {used}/{_TPM_BUDGET}"
                        f" + {needed} requested; sleeping {sleep_for:.1f}s"
                    )
                    _time.sleep(sleep_for)

            _orig_model_call = TemplateAPI.model_call
            _first_call_seen = [False]

            def _throttled_model_call(self, *args, **kwargs):
                # lm_eval invokes this as `model_call(messages=req, ...)`.
                # Support both positional and keyword forms defensively.
                messages = kwargs.get("messages")
                if messages is None and args:
                    messages = args[0]
                est = _estimate_request_tokens(messages)
                if not _first_call_seen[0]:
                    _first_call_seen[0] = True
                    print(
                        f"  [ruler-throttle] First request: estimated {est} tokens"
                        f" (input type={type(messages).__name__},"
                        f" inner={type(messages[0]).__name__ if isinstance(messages, (list, tuple)) and messages else 'n/a'})"
                    )
                _wait_for_tpm_budget(est)
                return _orig_model_call(self, *args, **kwargs)

            TemplateAPI.model_call = _throttled_model_call
            print(
                f"  [ruler-throttle] Client-side TPM throttle active:"
                f" budget={_TPM_BUDGET}/{_TPM_LIMIT} tokens/min"
                f" (safety={_TPM_SAFETY})"
            )

            # ---------------------------------------------------------------
            # Safety guardrail: override lm_eval's retry wait policy.
            #
            # lm_eval defaults to:
            #   wait_exponential(multiplier=0.5, min=1, max=10)
            # i.e. retry after 0.5s, 1s, 2s, 4s, 8s, capped at 10s. This
            # respects Groq's "Please try again in Xs" hint when X<10s,
            # but if our token estimator ever undershoots and a 429 still
            # leaks through, a 10s wait may not be enough to clear the
            # 60-second TPM window — we'd just retry and fail again.
            #
            # Replace it with a fixed 60s wait (configurable via
            # RULER_RETRY_WAIT_SEC) so any 429 forces a full window
            # rollover before retrying. With the proactive throttle
            # above, this code path should rarely fire — it's pure
            # belt-and-suspenders.
            # ---------------------------------------------------------------
            import tenacity as _tenacity
            from lm_eval.models import api_models as _api_models_mod
            _RETRY_WAIT_SEC = int(os.environ.get("RULER_RETRY_WAIT_SEC", "60"))
            _api_models_mod.wait_exponential = (
                lambda **kw: _tenacity.wait_fixed(_RETRY_WAIT_SEC)
            )
            print(
                f"  [ruler-throttle] Retry wait overridden:"
                f" fixed {_RETRY_WAIT_SEC}s on 429 (was exp backoff capped at 10s)"
            )

# ---------------------------------------------------------------------------
# Hand off to lm_eval CLI
# ---------------------------------------------------------------------------
sys.argv = ["lm_eval"] + sys.argv[1:]

from lm_eval.__main__ import cli_evaluate  # noqa: E402

cli_evaluate()
