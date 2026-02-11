from typing import Any, Dict, List


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    parts.append(text)
            elif block_type == "tool_result":
                tool_id = block.get("tool_use_id", "unknown")
                tool_content = str(block.get("content", "")).strip()
                if tool_content:
                    parts.append(f"[Tool Result {tool_id}] {tool_content}")
        return "\n".join(parts).strip()
    return str(content).strip()


def build_compact_chat_prompt(system: str, messages: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    compact_lines: List[str] = []
    system_text = (system or "").strip()
    if system_text:
        compact_lines.append(f"System: {system_text}")

    compact_lines.append(
        "IMPORTANT: Use available tools whenever a tool can provide factual or filesystem data. "
        "Prefer tool calls over speculation."
    )

    normalized: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).capitalize()
        text = extract_message_text(msg.get("content", ""))
        if not text:
            continue
        normalized.append(f"{role}: {text}")

    if not normalized:
        compact_lines.append("User: (no message content)")
        compact_lines.append("Assistant:")
        return "\n\n".join(compact_lines)

    selected: List[str] = []
    consumed = sum(len(x) + 2 for x in compact_lines) + len("\n\nAssistant:")
    for line in reversed(normalized):
        cost = len(line) + 1
        if selected and consumed + cost > max_chars:
            break
        if not selected and consumed + cost > max_chars:
            selected.append(line[: max(200, max_chars - consumed - 20)])
            break
        selected.append(line)
        consumed += cost

    selected.reverse()
    omitted = max(0, len(normalized) - len(selected))
    if omitted:
        compact_lines.append(
            f"Context summary: {omitted} earlier message(s) omitted to stay within token budget."
        )

    compact_lines.extend(selected)
    compact_lines.append("Assistant:")
    return "\n".join(compact_lines)
