from chat_prompt_optimizer import build_compact_chat_prompt, extract_message_text


def test_extract_message_text_supports_blocks():
    content = [
        {"type": "text", "text": "hello"},
        {"type": "tool_result", "tool_use_id": "abc", "content": "file_a.py"},
    ]
    result = extract_message_text(content)
    assert "hello" in result
    assert "Tool Result abc" in result


def test_compact_prompt_keeps_recent_messages_with_budget():
    messages = [
        {"role": "user", "content": "old-1 " * 80},
        {"role": "assistant", "content": "old-2 " * 80},
        {"role": "user", "content": "recent request: list plugins"},
    ]
    prompt = build_compact_chat_prompt("system", messages, max_chars=350)
    assert "recent request: list plugins" in prompt
    assert "Context summary:" in prompt
    assert prompt.rstrip().endswith("Assistant:")


def test_compact_prompt_handles_empty_messages():
    prompt = build_compact_chat_prompt("", [], max_chars=200)
    assert "(no message content)" in prompt
