"""Every model call in the capstone goes through call_llm(), so the cohort's key changes this file only.

Backend, chosen once at start:
    LLM_BACKEND=openai     OpenAI chat completions (needs CAPSTONE_OPENAI_API_KEY, or OPENAI_API_KEY as a fallback;
                           model from OPENAI_MODEL, default gpt-4o-mini)
    LLM_BACKEND=anthropic  Claude (needs CAPSTONE_ANTHROPIC_API_KEY; model from ANTHROPIC_MODEL; see llm_anthropic.py)
    LLM_BACKEND=qwen       local Qwen2.5-0.5B-Instruct on the CPU, greedy (the default while there is no key)
With LLM_BACKEND unset, an OpenAI key picks openai, then an Anthropic key picks anthropic, otherwise qwen.
call_llm(..., backend="anthropic") uses another backend for one call: a judge from a different model family
than the answer it checks (self-preference, guide 2 Part 5).

Every call is appended to .store/llm_calls.jsonl (backend, model, tokens in and out, seconds, purpose),
so the bill and the speed of each step can be counted later, the way guide 3 Part 4 counts a loop.
Local Qwen shows how the pieces fit, not good answers: on this CPU it takes 1-2 minutes per answer.
"""
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, ".store", "llm_calls.jsonl")
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY = os.environ.get("CAPSTONE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
BACKEND = os.environ.get("LLM_BACKEND") or ("openai" if OPENAI_KEY else
                                            "anthropic" if os.environ.get("CAPSTONE_ANTHROPIC_API_KEY")
                                                         or os.environ.get("ANTHROPIC_API_KEY") else "qwen")


def model_name(backend=None):
    backend = backend or BACKEND
    if backend == "anthropic":
        from llm_anthropic import ANTHROPIC_MODEL
        return ANTHROPIC_MODEL
    return OPENAI_MODEL if backend == "openai" else QWEN_MODEL

_qwen = None
_openai = None


def _load_qwen():
    global _qwen
    if _qwen is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
        model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, dtype=torch.float32)
        model.eval()
        # greedy decoding: drop the sampling settings the model ships with, so generate() does not warn
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.temperature = None
        _qwen = (tok, model)
    return _qwen


def _call_qwen(messages, max_tokens):
    import torch
    tok, model = _load_qwen()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
    new = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), ids["input_ids"].shape[1], len(new), QWEN_MODEL


def _openai_settings(max_tokens):
    """GPT-5 models think before they reply, the thinking counts against the limit, and they reject temperature=0."""
    if OPENAI_MODEL.startswith(("gpt-5", "gpt-6")):
        return {"max_completion_tokens": max(max_tokens, 4000)}
    return {"max_completion_tokens": max_tokens, "temperature": 0}


def _call_openai(messages, max_tokens):
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI(api_key=OPENAI_KEY)
    r = _openai.chat.completions.create(model=OPENAI_MODEL, messages=messages, **_openai_settings(max_tokens))
    return (r.choices[0].message.content or "").strip(), r.usage.prompt_tokens, r.usage.completion_tokens, r.model


def _call_anthropic(messages, max_tokens):
    from llm_anthropic import call
    return call(messages, max_tokens)


def call_llm(messages, max_tokens=300, purpose="", backend=None):
    """messages: [{"role": "system" | "user" | "assistant", "content": "..."}]. backend: None = BACKEND.
    Returns {"text", "backend", "model", "prompt_tokens", "output_tokens", "seconds"}."""
    t = time.time()
    backend = backend or BACKEND
    run = {"openai": _call_openai, "anthropic": _call_anthropic}.get(backend, _call_qwen)
    text, n_in, n_out, model = run(messages, max_tokens)
    result = {"text": text, "backend": backend, "model": model, "prompt_tokens": n_in,
              "output_tokens": n_out, "seconds": round(time.time() - t, 1)}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({**{k: v for k, v in result.items() if k != "text"}, "purpose": purpose,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    return result


# --- the agent's model: LangChain wants a chat model object, not a function -------------------------
# With a key this is ChatOpenAI. Without one, LocalQwenChat stands in, the way the cohort lab's tests replace
# the model and keep the real graph. Qwen's chat template has its own tool format: the tool descriptions go
# into the system message, and the model writes <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
# This class turns that text into the tool_calls LangChain expects. It is the harness reading the slip.

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _to_qwen(messages):
    """LangChain messages -> the dicts Qwen's chat template reads."""
    out = []
    for m in messages:
        if m.type == "system":
            out.append({"role": "system", "content": m.content})
        elif m.type == "human":
            out.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            d = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                d["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
                                   for c in m.tool_calls]
            out.append(d)
        elif m.type == "tool":
            out.append({"role": "tool", "content": m.content if isinstance(m.content, str) else json.dumps(m.content)})
    return out


def _parse_tool_calls(text, n_call):
    """Every well-formed <tool_call> block becomes a tool call; a malformed one stays in the text."""
    calls = []
    for k, m in enumerate(TOOL_CALL.finditer(text)):
        try:
            d = json.loads(m.group(1))
            calls.append({"name": d["name"], "args": d.get("arguments") or {}, "id": f"qwen_{n_call}_{k}",
                          "type": "tool_call"})
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    content = TOOL_CALL.sub("", text).strip() if calls else text
    return content, calls


def chat_model(max_tokens=300):
    """The model object for create_agent: ChatOpenAI or ChatAnthropic with a key, LocalQwenChat without."""
    if BACKEND == "anthropic":
        from llm_anthropic import chat_model as claude
        return claude(max_tokens)
    if BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        s = _openai_settings(max_tokens)
        return ChatOpenAI(api_key=OPENAI_KEY, model=OPENAI_MODEL, temperature=s.get("temperature"),
                          max_tokens=s["max_completion_tokens"], timeout=120, max_retries=1,
                          # gpt-6 models refuse function tools on chat completions (400, tested 2026-10-31)
                          use_responses_api=OPENAI_MODEL.startswith("gpt-6"),
                          model_kwargs={"parallel_tool_calls": False})
    return _local_chat_class()(max_tokens=max_tokens)


_chat_class = None


def _local_chat_class():
    global _chat_class
    if _chat_class is not None:
        return _chat_class
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool

    class LocalQwenChat(BaseChatModel):
        max_tokens: int = 300
        tools: list = []
        n_calls: int = 0

        @property
        def _llm_type(self):
            return "local-qwen"

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self.model_copy(update={"tools": [convert_to_openai_tool(t) for t in tools]})

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            import torch
            t = time.time()
            tok, model = _load_qwen()
            prompt = tok.apply_chat_template(_to_qwen(messages), tools=self.tools or None, tokenize=False,
                                             add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=self.max_tokens, do_sample=False)
            new = out[0, ids["input_ids"].shape[1]:]
            text = tok.decode(new, skip_special_tokens=True).strip()
            self.n_calls += 1
            content, calls = _parse_tool_calls(text, self.n_calls)
            n_in, n_out = ids["input_ids"].shape[1], len(new)
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"backend": "qwen", "model": QWEN_MODEL, "prompt_tokens": n_in,
                                    "output_tokens": n_out, "seconds": round(time.time() - t, 1),
                                    "purpose": "agent", "tool_calls": [c["name"] for c in calls],
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
            msg = AIMessage(content=content, tool_calls=calls, response_metadata={"model_name": QWEN_MODEL},
                            usage_metadata={"input_tokens": n_in, "output_tokens": n_out,
                                            "total_tokens": n_in + n_out})
            return ChatResult(generations=[ChatGeneration(message=msg)])

    _chat_class = LocalQwenChat
    return LocalQwenChat


def __getattr__(name):
    # LocalQwenChat is built on first use, so importing llm.py does not import LangChain
    if name == "LocalQwenChat":
        return _local_chat_class()
    raise AttributeError(name)


if __name__ == "__main__":
    r = call_llm([{"role": "user", "content": "In one sentence: what is a contingent liability?"}],
                 max_tokens=60, purpose="smoke test")
    print(r)
