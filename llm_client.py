"""
DeepSeek client helpers.

DeepSeek exposes an OpenAI-compatible chat completions API, so the project uses
the official OpenAI Python SDK with a DeepSeek base URL.
"""

from typing import Any, Dict, Optional

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_THINKING


def create_deepseek_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Create a DeepSeek chat client.

    Args:
        api_key: DeepSeek API key. Defaults to DEEPSEEK_API_KEY from config.

    Returns:
        OpenAI-compatible client configured for DeepSeek.
    """
    resolved_key = api_key or DEEPSEEK_API_KEY
    return OpenAI(api_key=resolved_key, base_url=DEEPSEEK_BASE_URL)


def deepseek_chat_options(json_output: bool = True) -> Dict[str, Any]:
    """
    Build request options shared by DeepSeek chat calls.

    DeepSeek's JSON mode requires response_format={"type": "json_object"} and
    a prompt that explicitly requests JSON. The project prompts already do that.
    """
    options: Dict[str, Any] = {}

    if json_output:
        options["response_format"] = {"type": "json_object"}

    if DEEPSEEK_THINKING in {"enabled", "disabled"}:
        options["extra_body"] = {"thinking": {"type": DEEPSEEK_THINKING}}

    return options
