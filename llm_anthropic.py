"""The Claude backend for llm.py (LLM_BACKEND=anthropic), kept in its own file so llm.py stays provider-neutral.

Needs an Anthropic Console API key in CAPSTONE_ANTHROPIC_API_KEY (pay as you go; the plain ANTHROPIC_API_KEY
still works as a fallback). A Claude Code subscription login is not
an API key and cannot be used here. Model from ANTHROPIC_MODEL, default claude-opus-5-5; claude-haiku-5-5 costs
about 40 times less per token (Claude API price list, cached 2026-10-06), so it is the one to try for judges.

Differences from the OpenAI path, from the Claude API docs (2026-10-24):
- No temperature: since 2026-10-21 the Messages API rejects temperature, top_p and top_k, so answers can vary
  a little between runs. Repeat a measurement before trusting a small difference.
- Thinking is on by default on current models and its tokens count against max_tokens, so short limits are
  raised to MIN_MAX_TOKENS; the reply still comes back short when the prompt asks for short.
- A reply can be a refusal (stop_reason "refusal"). On Opus 5.5, Fable 5.1 and Sonnet 5.5 the API is asked to
  retry such a request on a fallback model ("fallbacks": "default"); a refusal that remains comes back as text.
"""
import os

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
# a name of its own, so this key is not picked up by other tools that read ANTHROPIC_API_KEY
API_KEY = os.environ.get("CAPSTONE_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
MIN_MAX_TOKENS = 16000      # room for thinking; non-streaming requests stay under the SDK's HTTP timeout
FALLBACK_MODELS = ("claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_client = None


def call(messages, max_tokens):
    """messages in the OpenAI shape ({"role", "content"}); returns (text, tokens in, tokens out, model)."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=API_KEY)
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    args = {"model": ANTHROPIC_MODEL, "max_tokens": max(max_tokens, MIN_MAX_TOKENS), "messages": chat}
    if system:
        args["system"] = system
    if ANTHROPIC_MODEL in FALLBACK_MODELS:
        r = _client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **args)
    else:
        r = _client.messages.create(**args)
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    if r.stop_reason == "refusal":
        text = text or f"(the model declined: {getattr(r.stop_details, 'category', None)})"
    return text, r.usage.input_tokens, r.usage.output_tokens, r.model


def chat_model(max_tokens=300):
    """ChatAnthropic for create_agent, one tool call per model turn (as with ChatOpenAI here)."""
    from langchain_anthropic import ChatAnthropic

    class OneToolAtATime(ChatAnthropic):
        # create_agent binds the tools itself, so the switch is set where it binds them
        def bind_tools(self, tools, **kwargs):
            # not with tool_choice "none" (the agent's last turn): the API rejects disable_parallel_tool_use there
            if kwargs.get("tool_choice") != {"type": "none"}:
                kwargs.setdefault("parallel_tool_calls", False)
            return super().bind_tools(tools, **kwargs)

    return OneToolAtATime(model=ANTHROPIC_MODEL, api_key=API_KEY, max_tokens=max(max_tokens, MIN_MAX_TOKENS),
                          default_request_timeout=120, max_retries=1)
