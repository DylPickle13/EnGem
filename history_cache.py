from google import genai
import os
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def _get_int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError):
        return default
    if parsed_value < minimum:
        return default
    return parsed_value


def _get_float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        parsed_value = float(raw_value)
    except (TypeError, ValueError):
        return default
    if parsed_value < minimum:
        return default
    return parsed_value


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_ttl_seconds(ttl: str, default_seconds: int = 900) -> int:
    if not isinstance(ttl, str):
        return default_seconds

    normalized = ttl.strip().lower()
    if not normalized:
        return default_seconds

    multiplier = 1
    numeric_portion = normalized
    suffix = normalized[-1]
    if suffix in {"s", "m", "h"}:
        numeric_portion = normalized[:-1].strip()
        multiplier = {"s": 1, "m": 60, "h": 3600}[suffix]

    if not numeric_portion:
        return default_seconds

    try:
        ttl_value = float(numeric_portion)
    except (TypeError, ValueError):
        return default_seconds

    if ttl_value <= 0:
        return default_seconds

    return max(1, int(ttl_value * multiplier))


def _history_digest(history_text: str) -> str:
    payload = (history_text or "").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_model_name(model: str) -> str:
    normalized = (model or "").strip().lower()
    if normalized.startswith("models/"):
        normalized = normalized[len("models/") :]
    return normalized


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


HISTORY_CACHE_TTL = os.getenv("HISTORY_CACHE_TTL", "900s")
MIN_HISTORY_CACHE_CHARS = _get_int_env("MIN_HISTORY_CACHE_CHARS", 4096, minimum=0)
DEFAULT_MIN_CACHE_TOKENS = _get_int_env("DEFAULT_MIN_CACHE_TOKENS", 1024, minimum=1)
TOKEN_ESTIMATE_LOWER_FACTOR = _get_float_env("TOKEN_ESTIMATE_LOWER_FACTOR", 0.75, minimum=0.05)
TOKEN_ESTIMATE_UPPER_FACTOR = _get_float_env("TOKEN_ESTIMATE_UPPER_FACTOR", 1.25, minimum=0.1)
CACHE_TTL_REFRESH_WINDOW_SECONDS = _get_int_env("CACHE_TTL_REFRESH_WINDOW_SECONDS", 90, minimum=1)
CACHE_TTL_REFRESH_MIN_INTERVAL_SECONDS = _get_int_env(
    "CACHE_TTL_REFRESH_MIN_INTERVAL_SECONDS",
    45,
    minimum=1,
)
ENABLE_CACHE_TTL_REFRESH = _get_bool_env("ENABLE_CACHE_TTL_REFRESH", default=True)
DELETE_REMOTE_CACHE_ON_RELEASE = _get_bool_env("DELETE_REMOTE_CACHE_ON_RELEASE", default=False)
CACHE_METRICS_ENABLED = _get_bool_env("CACHE_METRICS_ENABLED", default=False)
CACHE_METRICS_FILE = Path(__file__).parent / "logs" / "cache_metrics.jsonl"

MODEL_MIN_CACHE_TOKENS: tuple[tuple[str, int], ...] = (
    ("gemini-3.1-pro", 4096),
    ("gemini-3-pro", 4096),
    ("gemini-2.5-pro", 4096),
    ("gemini-2.5-flash-lite", 2048),
    ("gemini-3.1-flash", 1024),
    ("gemini-3-flash", 1024),
    ("gemini-2.5-flash", 1024),
)

_GLOBAL_CACHE_LOCK = threading.RLock()
_GLOBAL_CACHE_REGISTRY: dict[str, list["CachedContentEntry"]] = {}
_GLOBAL_TOKEN_COUNT_CACHE: dict[tuple[str, str], int] = {}
_GLOBAL_TOKEN_COUNT_FAILURES: set[tuple[str, str]] = set()
_GLOBAL_MODEL_MIN_CACHE_TOKEN_OVERRIDES: dict[str, int] = {}


def emit_cache_metric(event: str, **payload) -> None:
    if not CACHE_METRICS_ENABLED:
        return

    row = {
        "ts": time.time(),
        "event": event,
        "payload": _json_safe(payload),
    }
    try:
        CACHE_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_METRICS_FILE.open("a", encoding="utf-8") as metric_file:
            metric_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # Metric logging must never impact response generation.
        return


def _resolve_model_min_cache_tokens(model: str) -> int:
    normalized_model = _normalize_model_name(model)

    with _GLOBAL_CACHE_LOCK:
        dynamic_override = _GLOBAL_MODEL_MIN_CACHE_TOKEN_OVERRIDES.get(normalized_model)

    if dynamic_override is not None and dynamic_override > 0:
        return dynamic_override

    for model_fragment, minimum_tokens in MODEL_MIN_CACHE_TOKENS:
        if model_fragment in normalized_model:
            return minimum_tokens
    return DEFAULT_MIN_CACHE_TOKENS


def _set_model_min_cache_tokens_override(model: str, min_tokens: int) -> None:
    normalized_model = _normalize_model_name(model)
    if not normalized_model or min_tokens <= 0:
        return

    with _GLOBAL_CACHE_LOCK:
        current_value = _GLOBAL_MODEL_MIN_CACHE_TOKEN_OVERRIDES.get(normalized_model, 0)
        if min_tokens > current_value:
            _GLOBAL_MODEL_MIN_CACHE_TOKEN_OVERRIDES[normalized_model] = min_tokens
            emit_cache_metric(
                "history_cache_model_min_tokens_override",
                model=normalized_model,
                min_tokens=min_tokens,
            )


def _extract_cache_size_error_details(error: Exception) -> tuple[int | None, int | None]:
    raw_error = str(error or "")
    if "cached content is too small" not in raw_error.lower():
        return None, None

    min_token_match = re.search(r"min_total_token_count=(\d+)", raw_error, flags=re.IGNORECASE)
    total_token_match = re.search(r"(?<!min_)total_token_count=(\d+)", raw_error, flags=re.IGNORECASE)

    min_tokens = int(min_token_match.group(1)) if min_token_match else None
    total_tokens = int(total_token_match.group(1)) if total_token_match else None
    return min_tokens, total_tokens


def _estimate_tokens(history_text: str) -> int:
    return max(1, len(history_text) // 4)


def _make_registry_key(history_file: str, model: str, profile_key: str) -> str:
    return f"{history_file}|{model}|{profile_key}"


def _cleanup_global_registry(now_monotonic: float | None = None) -> None:
    timestamp = now_monotonic if now_monotonic is not None else time.monotonic()
    keys_to_delete: list[str] = []

    for key, entries in _GLOBAL_CACHE_REGISTRY.items():
        active_entries = [entry for entry in entries if entry.expire_at_monotonic > timestamp]
        if active_entries:
            _GLOBAL_CACHE_REGISTRY[key] = active_entries
        else:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        _GLOBAL_CACHE_REGISTRY.pop(key, None)


def _register_global_entry(entry: "CachedContentEntry") -> None:
    key = _make_registry_key(entry.history_file, entry.model, entry.profile_key)
    with _GLOBAL_CACHE_LOCK:
        _cleanup_global_registry()
        entries = _GLOBAL_CACHE_REGISTRY.setdefault(key, [])
        entries = [existing for existing in entries if existing.cache_name != entry.cache_name]
        entries.append(entry)
        _GLOBAL_CACHE_REGISTRY[key] = entries


def _remove_global_entry(cache_name: str) -> None:
    with _GLOBAL_CACHE_LOCK:
        keys_to_delete: list[str] = []
        for key, entries in _GLOBAL_CACHE_REGISTRY.items():
            retained_entries = [entry for entry in entries if entry.cache_name != cache_name]
            if retained_entries:
                _GLOBAL_CACHE_REGISTRY[key] = retained_entries
            else:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            _GLOBAL_CACHE_REGISTRY.pop(key, None)


def _find_best_prefix_entry(
    history_file: str,
    model: str,
    profile_key: str,
    current_history_text: str,
) -> "CachedContentEntry | None":
    key = _make_registry_key(history_file, model, profile_key)
    with _GLOBAL_CACHE_LOCK:
        _cleanup_global_registry()
        entries = _GLOBAL_CACHE_REGISTRY.get(key, [])

        best_entry = None
        for entry in entries:
            if not current_history_text.startswith(entry.base_history_text):
                continue

            if best_entry is None or len(entry.base_history_text) > len(best_entry.base_history_text):
                best_entry = entry

        if best_entry is not None:
            best_entry.last_used_monotonic = time.monotonic()

        return best_entry


@dataclass(frozen=True)
class CachedContentProfile:
    profile_key: str
    profile_label: str
    system_instruction: str
    tools: list[genai.types.Tool] | None = None
    tool_config: genai.types.ToolConfig | None = None


@dataclass
class CachedContentEntry:
    cache_name: str
    history_file: str
    model: str
    profile_key: str
    base_history_text: str
    expire_at_monotonic: float
    created_at_monotonic: float
    last_used_monotonic: float
    last_ttl_refresh_monotonic: float = 0.0
    token_count: int | None = None


@dataclass
class HistoryContextCache:
    history_file: str
    history_text: str
    client: genai.Client
    backoff_call: Callable[..., object]
    ttl: str = HISTORY_CACHE_TTL
    cache_entries_by_profile: dict[str, CachedContentEntry] = field(default_factory=dict)
    failed_profiles: set[str] = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _reference_count: int = field(default=1, init=False, repr=False)
    _released: bool = field(default=False, init=False, repr=False)
    _threshold_skip_logged: bool = field(default=False, init=False, repr=False)

    def _extract_token_count(self, count_tokens_result: object) -> int | None:
        for attr_name in ("total_tokens", "total_token_count", "token_count"):
            value = getattr(count_tokens_result, attr_name, None)
            if isinstance(value, int) and value >= 0:
                return value

        if isinstance(count_tokens_result, dict):
            for key in ("totalTokens", "total_tokens", "total_token_count", "tokenCount", "token_count"):
                value = count_tokens_result.get(key)
                if isinstance(value, int) and value >= 0:
                    return value

        return None

    def _get_history_token_count(self, model: str, history_text: str) -> int | None:
        if not history_text:
            return 0

        digest = _history_digest(history_text)
        cache_key = (model, digest)

        with _GLOBAL_CACHE_LOCK:
            if cache_key in _GLOBAL_TOKEN_COUNT_CACHE:
                return _GLOBAL_TOKEN_COUNT_CACHE[cache_key]
            if cache_key in _GLOBAL_TOKEN_COUNT_FAILURES:
                return None

        try:
            token_count_response = self.backoff_call(
                lambda: self.client.models.count_tokens(
                    model=model,
                    contents=history_text,
                ),
                description=f"Gemini count_tokens ({model})",
            )
        except Exception as e:
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_TOKEN_COUNT_FAILURES.add(cache_key)
            emit_cache_metric(
                "history_cache_count_tokens_error",
                model=model,
                history_file=self.history_file,
                error=str(e),
            )
            return None

        token_count = self._extract_token_count(token_count_response)
        if token_count is None:
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_TOKEN_COUNT_FAILURES.add(cache_key)
            emit_cache_metric(
                "history_cache_count_tokens_missing",
                model=model,
                history_file=self.history_file,
            )
            return None

        with _GLOBAL_CACHE_LOCK:
            _GLOBAL_TOKEN_COUNT_CACHE[cache_key] = token_count

        emit_cache_metric(
            "history_cache_count_tokens_success",
            model=model,
            history_file=self.history_file,
            token_count=token_count,
        )
        return token_count

    def _is_history_cacheable(self, model: str, history_text: str) -> bool:
        if not history_text or history_text == "No history available.":
            return False

        minimum_tokens = _resolve_model_min_cache_tokens(model)
        estimated_tokens = _estimate_tokens(history_text)

        if estimated_tokens < int(minimum_tokens * TOKEN_ESTIMATE_LOWER_FACTOR):
            if not self._threshold_skip_logged:
                self._threshold_skip_logged = True
            emit_cache_metric(
                "history_cache_skip_below_estimated_threshold",
                model=model,
                history_file=self.history_file,
                estimated_tokens=estimated_tokens,
                minimum_tokens=minimum_tokens,
            )
            return False

        if estimated_tokens >= int(minimum_tokens * TOKEN_ESTIMATE_UPPER_FACTOR):
            return True

        token_count = self._get_history_token_count(model=model, history_text=history_text)
        if token_count is None:
            # Fallback when count_tokens is unavailable.
            return len(history_text) > MIN_HISTORY_CACHE_CHARS

        return token_count >= minimum_tokens

    def _is_entry_valid(self, entry: CachedContentEntry, current_history_text: str) -> bool:
        if time.monotonic() >= entry.expire_at_monotonic:
            return False
        return current_history_text.startswith(entry.base_history_text)

    def _maybe_refresh_entry_ttl(self, entry: CachedContentEntry) -> None:
        if not ENABLE_CACHE_TTL_REFRESH:
            return

        now_monotonic = time.monotonic()
        seconds_remaining = entry.expire_at_monotonic - now_monotonic
        if seconds_remaining > CACHE_TTL_REFRESH_WINDOW_SECONDS:
            return

        if now_monotonic - entry.last_ttl_refresh_monotonic < CACHE_TTL_REFRESH_MIN_INTERVAL_SECONDS:
            return

        try:
            self.backoff_call(
                lambda: self.client.caches.update(
                    name=entry.cache_name,
                    config=genai.types.UpdateCachedContentConfig(ttl=self.ttl),
                ),
                description=f"Gemini update history cache TTL ({entry.cache_name})",
            )
        except Exception as e:
            emit_cache_metric(
                "history_cache_ttl_refresh_error",
                history_file=self.history_file,
                cache_name=entry.cache_name,
                error=str(e),
            )
            return

        ttl_seconds = _parse_ttl_seconds(self.ttl)
        entry.expire_at_monotonic = now_monotonic + ttl_seconds
        entry.last_ttl_refresh_monotonic = now_monotonic
        entry.last_used_monotonic = now_monotonic
        _register_global_entry(entry)

        emit_cache_metric(
            "history_cache_ttl_refreshed",
            history_file=self.history_file,
            cache_name=entry.cache_name,
            ttl=self.ttl,
        )

    def retain(self) -> "HistoryContextCache":
        with self._lock:
            if self._released:
                raise RuntimeError("Cannot retain a released history cache.")
            self._reference_count += 1
        return self

    def release(self) -> None:
        entries_to_delete: list[CachedContentEntry] = []
        with self._lock:
            if self._reference_count <= 0:
                return

            self._reference_count -= 1
            if self._reference_count > 0 or self._released:
                return

            self._released = True
            if DELETE_REMOTE_CACHE_ON_RELEASE:
                entries_to_delete = list(self.cache_entries_by_profile.values())
            self.cache_entries_by_profile.clear()
            self.failed_profiles.clear()

        if not DELETE_REMOTE_CACHE_ON_RELEASE:
            emit_cache_metric(
                "history_cache_release_deferred_delete",
                history_file=self.history_file,
            )
            return

        for entry in entries_to_delete:
            try:
                self.backoff_call(
                    lambda cache_name=entry.cache_name: self.client.caches.delete(name=cache_name),
                    description=f"Gemini delete history cache ({entry.cache_name})",
                )
            except Exception as e:
                print(f"Error deleting history cache '{entry.cache_name}': {e}")
                continue

            _remove_global_entry(entry.cache_name)
            emit_cache_metric(
                "history_cache_deleted_on_release",
                history_file=self.history_file,
                cache_name=entry.cache_name,
            )

    def get_cached_content_entry(
        self,
        model: str,
        profile: CachedContentProfile,
        current_history_text: str | None = None,
    ) -> CachedContentEntry | None:
        working_history_text = current_history_text or self.history_text

        with self._lock:
            if self._released:
                return None

        if not self._is_history_cacheable(model=model, history_text=working_history_text):
            return None

        with self._lock:
            cached_entry = self.cache_entries_by_profile.get(profile.profile_key)
            if cached_entry and not self._is_entry_valid(cached_entry, working_history_text):
                self.cache_entries_by_profile.pop(profile.profile_key, None)
                _remove_global_entry(cached_entry.cache_name)
                cached_entry = None

        if cached_entry:
            cached_entry.last_used_monotonic = time.monotonic()
            self._maybe_refresh_entry_ttl(cached_entry)
            emit_cache_metric(
                "history_cache_local_hit",
                model=model,
                history_file=self.history_file,
                cache_name=cached_entry.cache_name,
            )
            return cached_entry

        with self._lock:
            if profile.profile_key in self.failed_profiles:
                return None

        reused_entry = _find_best_prefix_entry(
            history_file=self.history_file,
            model=model,
            profile_key=profile.profile_key,
            current_history_text=working_history_text,
        )
        if reused_entry is not None:
            with self._lock:
                if self._released:
                    return None
                self.cache_entries_by_profile[profile.profile_key] = reused_entry

            self._maybe_refresh_entry_ttl(reused_entry)
            emit_cache_metric(
                "history_cache_registry_hit",
                model=model,
                history_file=self.history_file,
                cache_name=reused_entry.cache_name,
                cached_prefix_chars=len(reused_entry.base_history_text),
                current_history_chars=len(working_history_text),
            )
            return reused_entry

        if not working_history_text:
            return None

        cache_history_text = self.history_text
        if not working_history_text.startswith(cache_history_text):
            cache_history_text = working_history_text

        if not self._is_history_cacheable(model=model, history_text=cache_history_text):
            # If the original base snapshot is too small, try caching the current
            # expanded history text for this request profile.
            if (
                cache_history_text != working_history_text
                and self._is_history_cacheable(model=model, history_text=working_history_text)
            ):
                cache_history_text = working_history_text
            else:
                emit_cache_metric(
                    "history_cache_skip_candidate_prefix",
                    model=model,
                    history_file=self.history_file,
                    candidate_prefix_chars=len(cache_history_text),
                )
                return None

        token_count = self._get_history_token_count(model=model, history_text=cache_history_text)
        minimum_tokens = _resolve_model_min_cache_tokens(model)
        if token_count is not None and token_count < minimum_tokens:
            emit_cache_metric(
                "history_cache_skip_below_model_min_tokens",
                model=model,
                history_file=self.history_file,
                token_count=token_count,
                minimum_tokens=minimum_tokens,
                cached_prefix_chars=len(cache_history_text),
            )
            return None

        try:
            cache = self.backoff_call(
                lambda: self.client.caches.create(
                    model=model,
                    config=genai.types.CreateCachedContentConfig(
                        display_name=f"engem-history-{self.history_file}-{model}-{profile.profile_label}",
                        contents=cache_history_text,
                        system_instruction=profile.system_instruction,
                        tools=profile.tools,
                        tool_config=profile.tool_config,
                        ttl=self.ttl,
                    ),
                ),
                description=f"Gemini create history cache ({model})",
            )
        except Exception as e:
            min_tokens, total_tokens = _extract_cache_size_error_details(e)
            if min_tokens is not None:
                _set_model_min_cache_tokens_override(model=model, min_tokens=min_tokens)
                emit_cache_metric(
                    "history_cache_create_too_small",
                    model=model,
                    history_file=self.history_file,
                    token_count=token_count,
                    total_token_count=total_tokens,
                    min_total_token_count=min_tokens,
                    cached_prefix_chars=len(cache_history_text),
                )
                return None

            with self._lock:
                self.failed_profiles.add(profile.profile_key)
            print(f"Error creating history cache for model '{model}': {e}")
            emit_cache_metric(
                "history_cache_create_error",
                model=model,
                history_file=self.history_file,
                error=str(e),
            )
            return None

        cache_name = getattr(cache, "name", None)
        if not isinstance(cache_name, str) or not cache_name:
            with self._lock:
                self.failed_profiles.add(profile.profile_key)
            emit_cache_metric(
                "history_cache_create_missing_name",
                model=model,
                history_file=self.history_file,
            )
            return None

        now_monotonic = time.monotonic()
        cache_entry = CachedContentEntry(
            cache_name=cache_name,
            history_file=self.history_file,
            model=model,
            profile_key=profile.profile_key,
            base_history_text=cache_history_text,
            expire_at_monotonic=now_monotonic + _parse_ttl_seconds(self.ttl),
            created_at_monotonic=now_monotonic,
            last_used_monotonic=now_monotonic,
            token_count=token_count,
        )

        with self._lock:
            if self._released:
                try:
                    self.backoff_call(
                        lambda cache_name=cache_name: self.client.caches.delete(name=cache_name),
                        description=f"Gemini delete history cache after late release ({cache_name})",
                    )
                except Exception:
                    pass
                return None
            self.cache_entries_by_profile[profile.profile_key] = cache_entry

        _register_global_entry(cache_entry)
        emit_cache_metric(
            "history_cache_created",
            model=model,
            history_file=self.history_file,
            cache_name=cache_name,
            cached_prefix_chars=len(cache_history_text),
            token_count=token_count,
        )
        return cache_entry

    def get_cached_content_name(self, model: str, profile: CachedContentProfile) -> str | None:
        entry = self.get_cached_content_entry(model=model, profile=profile)
        if entry is None:
            return None
        return entry.cache_name


def create_history_context_cache(
    history_file: str,
    history_text: str,
    client: genai.Client,
    backoff_call: Callable[..., object],
    ttl: str = HISTORY_CACHE_TTL,
) -> HistoryContextCache:
    return HistoryContextCache(
        history_file=history_file,
        history_text=history_text,
        client=client,
        backoff_call=backoff_call,
        ttl=ttl,
    )


def create_cached_content_profile(
    model: str,
    system_instruction: str,
    tools: list[genai.types.Tool] | None = None,
    tool_config: genai.types.ToolConfig | None = None,
) -> CachedContentProfile:
    tool_names: list[str] = []
    for tool in tools or []:
        declarations = getattr(tool, "function_declarations", None) or []
        for declaration in declarations:
            name = getattr(declaration, "name", None)
            if isinstance(name, str) and name:
                tool_names.append(name)

    function_calling_config = getattr(tool_config, "function_calling_config", None)
    tool_signature = {
        "tool_names": sorted(tool_names),
        "mode": getattr(function_calling_config, "mode", None),
        "allowed_function_names": sorted(getattr(function_calling_config, "allowed_function_names", None) or []),
    }
    payload = {
        "model": model,
        "system_instruction": system_instruction,
        "tool_signature": tool_signature,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return CachedContentProfile(
        profile_key=digest,
        profile_label=digest[:12],
        system_instruction=system_instruction,
        tools=tools,
        tool_config=tool_config,
    )


def extract_uncached_history_delta(base_history_text: str, current_history_text: str) -> str:
    if not current_history_text:
        return ""

    if base_history_text and current_history_text.startswith(base_history_text):
        return current_history_text[len(base_history_text):].strip()

    if current_history_text == "No history available.":
        return ""

    return current_history_text.strip()


def compose_cached_history_prompt(
    base_history_text: str,
    dynamic_text: str,
    current_history_text: str | None = None,
) -> str:
    sections: list[str] = []

    delta = extract_uncached_history_delta(base_history_text, current_history_text or "")
    if delta:
        sections.append(f"Conversation updates after the cached history:\n{delta}")

    cleaned_dynamic_text = (dynamic_text or "").strip()
    if cleaned_dynamic_text:
        sections.append(cleaned_dynamic_text)

    if not sections:
        return "Use the conversation history as the primary context."
    return "\n\n".join(sections)


def compose_uncached_history_prompt(history_text: str, dynamic_text: str) -> str:
    cleaned_history = (history_text or "").strip()
    cleaned_dynamic_text = (dynamic_text or "").strip()

    if cleaned_history and cleaned_dynamic_text:
        return cleaned_history + "\n\n" + cleaned_dynamic_text
    if cleaned_history:
        return cleaned_history
    return cleaned_dynamic_text or "Use the latest request as the primary context."


def resolve_history_cached_prompt(
    history_cache: HistoryContextCache,
    profile: CachedContentProfile,
    model: str,
    dynamic_text: str,
    current_history_text: str | None = None,
) -> tuple[str, str | None]:
    working_history_text = current_history_text if current_history_text is not None else history_cache.history_text
    cached_entry = history_cache.get_cached_content_entry(
        model=model,
        profile=profile,
        current_history_text=working_history_text,
    )

    if cached_entry:
        return (
            compose_cached_history_prompt(
                base_history_text=cached_entry.base_history_text,
                dynamic_text=dynamic_text,
                current_history_text=working_history_text,
            ),
            cached_entry.cache_name,
        )

    return compose_uncached_history_prompt(working_history_text or "", dynamic_text), None