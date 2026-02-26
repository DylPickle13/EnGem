import memory
import llm
import json
import re
from pathlib import Path


MEMORY_RELATED = Path(__file__).parent.parent / "agent_instructions" / "memory_related.md"

def _select_related_memory_ids(topic: str, candidates: list[memory.MemoryItem]) -> set[str]:
    candidate_payload = [
        {"id": item.memory_id, "text": item.text}
        for item in candidates
    ]
    prompt = (
        f"Topic: {topic}\n\n"
        f"Candidates JSON:\n{json.dumps(candidate_payload, ensure_ascii=False)}"
    )
    result = llm._run_model_api(
        text=prompt,
        system_instructions=MEMORY_RELATED.read_text(encoding="utf-8"),
        tool_use_allowed=False,
        force_tool=False,
        temperature=0,
    )

    raw = (result or "").strip()
    if not raw:
        return set()

    def _try_parse(text: str):
        try:
            return json.loads(text)
        except Exception:
            return None

    parsed = _try_parse(raw)

    # If direct parse failed, attempt to extract a JSON array substring
    if parsed is None:
        m = re.search(r"(\[.*\])", raw, re.S)
        if m:
            parsed = _try_parse(m.group(1))

    # If still not parsed, retry once asking for strict JSON only
    if parsed is None:
        retry_system = (
            MEMORY_RELATED.read_text(encoding="utf-8")
            + "\n\nIf your previous output was not valid JSON, now respond with ONLY a strict JSON array of ids (e.g. [\"id1\", \"id2\"]). No explanation, no markdown."
        )
        try:
            retry = llm._run_model_api(
                text=prompt,
                system_instructions=retry_system,
                tool_use_allowed=False,
                force_tool=False,
                temperature=0,
            )
            retry_raw = (retry or "").strip()
            parsed = _try_parse(retry_raw)
            if parsed is None:
                m2 = re.search(r"(\[.*\])", retry_raw, re.S)
                if m2:
                    parsed = _try_parse(m2.group(1))
        except Exception:
            parsed = None

    if not isinstance(parsed, list):
        return set()

    valid_ids = {item.memory_id for item in candidates}
    return {
        memory_id
        for memory_id in parsed
        if isinstance(memory_id, str) and memory_id in valid_ids
    }


def forget_memories(topic: str) -> str:
    """
    Delete up to 5 memories semantically related to the provided topic.
    """
    normalized_topic = (topic or "").strip()
    if not normalized_topic:
        return "Please provide a memory topic to forget."

    try:
        store = memory.get_default_store()
        matches = store.search_memories(normalized_topic, limit=5)

        if not matches:
            return f"No memories found for topic: '{normalized_topic}'."

        related_ids = _select_related_memory_ids(normalized_topic, matches)
        related_matches = [item for item in matches if item.memory_id in related_ids]

        if not related_matches:
            return (
                f"Found {len(matches)} candidate memory item(s), but none were "
                f"LLM-confirmed as related to topic: '{normalized_topic}'."
            )

        ids_to_delete = [item.memory_id for item in related_matches]
        store.collection.delete(ids=ids_to_delete)

        deleted_preview = "\n".join(
            f"{index + 1}. [ID: {item.memory_id}] {item.text}"
            for index, item in enumerate(related_matches)
        )

        return (
            f"Forgotten {len(ids_to_delete)} LLM-confirmed memory item(s) for topic: '{normalized_topic}'.\n"
            f"{deleted_preview}"
        )
    except Exception as error:
        return f"Failed to forget memories: {error}"
