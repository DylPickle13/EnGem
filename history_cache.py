from google import genai
import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Callable

HISTORY_CACHE_TTL = "900s"
MIN_HISTORY_CACHE_CHARS = 4096


@dataclass(frozen=True)
class CachedContentProfile:
    profile_key: str
    profile_label: str
    system_instruction: str
    tools: list[genai.types.Tool] | None = None
    tool_config: genai.types.ToolConfig | None = None


@dataclass
class HistoryContextCache:
    history_file: str
    history_text: str
    client: genai.Client
    backoff_call: Callable[..., object]
    ttl: str = HISTORY_CACHE_TTL
    cache_names_by_profile: dict[str, str] = field(default_factory=dict)
    failed_profiles: set[str] = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _reference_count: int = field(default=1, init=False, repr=False)
    _released: bool = field(default=False, init=False, repr=False)
    _threshold_skip_logged: bool = field(default=False, init=False, repr=False)

    def retain(self) -> "HistoryContextCache":
        with self._lock:
            if self._released:
                raise RuntimeError("Cannot retain a released history cache.")
            self._reference_count += 1
        return self

    def release(self) -> None:
        caches_to_delete: list[str] = []
        with self._lock:
            if self._reference_count <= 0:
                return

            self._reference_count -= 1
            if self._reference_count > 0 or self._released:
                return

            self._released = True
            caches_to_delete = list(self.cache_names_by_profile.values())
            self.cache_names_by_profile.clear()
            self.failed_profiles.clear()

        for cache_name in caches_to_delete:
            try:
                self.backoff_call(
                    lambda cache_name=cache_name: self.client.caches.delete(name=cache_name),
                    description=f"Gemini delete history cache ({cache_name})",
                )
            except Exception as e:
                print(f"Error deleting history cache '{cache_name}': {e}")

    def get_cached_content_name(self, model: str, profile: CachedContentProfile) -> str | None:
        with self._lock:
            if self._released:
                return None

            if len(self.history_text) <= MIN_HISTORY_CACHE_CHARS:
                if not self._threshold_skip_logged:
                    self._threshold_skip_logged = True
                return None

            cached_name = self.cache_names_by_profile.get(profile.profile_key)
            if cached_name:
                return cached_name

            if profile.profile_key in self.failed_profiles:
                return None

            if not self.history_text or self.history_text == "No history available.":
                return None

            try:
                cache = self.backoff_call(
                    lambda: self.client.caches.create(
                        model=model,
                        config=genai.types.CreateCachedContentConfig(
                            display_name=f"engem-history-{self.history_file}-{model}-{profile.profile_label}",
                            contents=self.history_text,
                            system_instruction=profile.system_instruction,
                            tools=profile.tools,
                            tool_config=profile.tool_config,
                            ttl=self.ttl,
                        ),
                    ),
                    description=f"Gemini create history cache ({model})",
                )
            except Exception as e:
                self.failed_profiles.add(profile.profile_key)
                print(f"Error creating history cache for model '{model}': {e}")
                return None

            cache_name = getattr(cache, "name", None)
            if not isinstance(cache_name, str) or not cache_name:
                self.failed_profiles.add(profile.profile_key)
                return None

            self.cache_names_by_profile[profile.profile_key] = cache_name
            return cache_name


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
    history_cache: HistoryContextCache,
    dynamic_text: str,
    current_history_text: str | None = None,
) -> str:
    sections: list[str] = []

    delta = extract_uncached_history_delta(history_cache.history_text, current_history_text or "")
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
    cached_content_name = history_cache.get_cached_content_name(model, profile)
    if cached_content_name:
        delta = extract_uncached_history_delta(history_cache.history_text, current_history_text or "")
        return (
            compose_cached_history_prompt(
                history_cache=history_cache,
                dynamic_text=dynamic_text,
                current_history_text=current_history_text,
            ),
            cached_content_name,
        )

    return compose_uncached_history_prompt(current_history_text or "", dynamic_text), None