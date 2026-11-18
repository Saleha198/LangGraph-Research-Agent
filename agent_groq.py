"""LangGraph research agent on Groq's free tier (no Anthropic key needed).
Usage: GROQ_API_KEY=... python agent_groq.py "question"   |   --graph"""
import os, re, sys, json
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles (cp1252) crash on characters like U+2011
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LOG = os.path.join("runs", "last_run.txt")  # every run is also saved here, as UTF-8
os.makedirs("runs", exist_ok=True)

def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")
from types import SimpleNamespace
from typing import TypedDict, Optional
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from groq import Groq
from langgraph.graph import StateGraph, START, END

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
SMALL = os.getenv("SUMMARY_MODEL", "openai/gpt-oss-20b")
MAX_STEPS = int(os.getenv("MAX_STEPS", 8))
client = Groq()
SOURCES = {}

# ---------- tools ----------
def web_search(query):
    try:
        res = list(DDGS().text(query, max_results=4))
    except Exception as e:
        return f"ERROR: search failed ({e}). Try a different query."
    if not res:
        return "No results. Try rephrasing the query."
    return "\n".join(f"- {r['title']} | {r['href']}\n  {r['body'][:150]}" for r in res)

def fetch_page(url):
    url = url.split('#')[0]  # same page, different anchor = same source
    for sid, s in SOURCES.items():
        if s["url"] == url:
            return f"Already fetched as [{sid}]."
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "research-agent/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer"]):
            t.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()[:4000]  # small: free-tier token limits
    except Exception as e:
        return f"ERROR: could not fetch {url} ({e}). Try another URL."
    if len(text) < 200:
        return f"ERROR: {url} returned almost no text. Try another URL."
    sid = f"S{len(SOURCES)+1}"
    SOURCES[sid] = {"url": url, "text": text}
    return f"Fetched as [{sid}]. Start of page:\n{text[:800]}"

def summarise(source_id, focus):
    s = SOURCES.get(source_id)
    if not s:
        return f"ERROR: unknown source {source_id}. Fetched: {list(SOURCES) or 'none'}."
    r = client.chat.completions.create(model=SMALL, max_tokens=300, messages=[{"role": "user",
        "content": f"Summarise in <=100 words using only this text. Focus: {focus}\n\n{s['text']}"}])
    return r.choices[0].message.content

def _t(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in props}, "required": req}}}
TOOLS = [
    _t("web_search", "Search the web; snippets are NOT citable.", ["query"], ["query"]),
    _t("fetch_page", "Fetch a URL and register it as citable source [S#].", ["url"], ["url"]),
    _t("summarise", "Summarise a fetched source by id (e.g. S1).", ["source_id", "focus"], ["source_id", "focus"]),
    _t("final_answer", "Final answer. Every sentence must end with [S#] citations of fetched sources only.", ["answer"], ["answer"]),
]
IMPL = {"web_search": web_search, "fetch_page": fetch_page, "summarise": summarise}
SYSTEM = ("You are a research agent. Search, fetch 2+ pages, then call final_answer. "
          "Cite only fetched sources as [S#]. Never state facts not found in fetched pages. "
          f"If tools fail or evidence is thin, say so. Budget: {MAX_STEPS} steps.")

import unicodedata

def clean(text):
    """Normalise model output: odd spaces/invisible chars -> ' ', odd hyphens -> '-', any bracket style -> [S#]."""
    text = "".join(" " if unicodedata.category(c) in ("Zs", "Cf") else c for c in text)
    text = text.replace("\u2011", "-").replace("\u2010", "-")
    text = re.sub(r"[\u3010\uff3b\[(]\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*[\u3011\uff3d\])]",
                  lambda m: "".join(f"[{x}]" for x in re.findall(r"S\d+", m.group(1))), text)
    return re.sub(r"[ ]{2,}", " ", text).strip()

def check_citations(answer):
    probs = []
    for sent in filter(None, re.split(r"(?<=[.!?])\s+(?!\[S\d)", answer.strip())):
        ids = re.findall(r"\[(S\d+)\]", sent)
        if not ids:
            probs.append(f"No citation: '{sent[:60]}'")
        probs += [f"Unknown source {i}" for i in ids if i not in SOURCES]
    return probs

def render(answer):
    refs = sorted(set(re.findall(r"\[(S\d+)\]", answer)), key=lambda x: int(x[1:]))
    return answer + "\n\nSources:\n" + "\n".join(f"[{i}] {SOURCES[i]['url']}" for i in refs if i in SOURCES)

# ---------- graph ----------
class State(TypedDict):
    msgs: list
    step: int
    retried: bool
    pending: list      # tool calls from the latest model turn
    answer: Optional[str]

def write_final(question, chars):
    """Fresh conversation with NO tool history, so the model cannot attempt tool calls."""
    ctx = "\n\n".join(f"[{i}] {v['url']}\n{v['text'][:chars]}" for i, v in SOURCES.items()) or "(no pages were fetched)"
    r = client.chat.completions.create(model=MODEL, max_tokens=800, messages=[
        {"role": "system", "content": "You write short, factual answers using ONLY the provided sources. "
         "Every sentence must end with a citation like [S1] naming a provided source. If the sources do not "
         "cover something, say so. Reply with the answer text only."},
        {"role": "user", "content": f"Question: {question}\n\nSources:\n{ctx}"}])
    return (r.choices[0].message.content or "").strip()

def agent_node(s):
    step = s["step"] + 1
    last = step >= MAX_STEPS
    if last:  # budget used up: fresh plain-text write-up from the fetched pages
        say(f"[step {step}/{MAX_STEPS}] final_answer (forced write-up)")
        text, err = "", None
        for chars in (4000, 1500):  # retry once with less text if it fails
            try:
                text = write_final(s["msgs"][0]["content"], chars)
                text = re.sub(r"[\u00a0\u2007\u2009\u200a\u202f\u200b]", " ", text).replace("\u2011", "-")  # odd spaces/hyphens -> ASCII
                if text and not text.lstrip().startswith(("{", "<")):
                    break
                text = ""
            except Exception as e:
                err = e
        if not text:
            srcs = "\n".join(f"[{i}] {v['url']}" for i, v in SOURCES.items()) or "none"
            return {"step": step, "pending": [], "answer":
                    f"Stopped: could not write the final answer ({str(err)[:150]}).\nSources fetched:\n{srcs}"}
        fa = SimpleNamespace(id="forced", function=SimpleNamespace(
            name="final_answer", arguments=json.dumps({"answer": text})))
        return {"step": step, "pending": [fa]}
    try:
        m = client.chat.completions.create(model=MODEL, max_tokens=1000, tools=TOOLS,
                messages=[{"role": "system", "content": SYSTEM}] + s["msgs"]).choices[0].message
    except Exception as e:  # malformed tool call / rate limit: tell the model and retry within budget
        say(f"[step {step}] model error: {str(e)[:200]}")
        if any(k in str(e) for k in ("404", "401", "403", "model_not_found")):
            return {"step": step, "pending": [], "answer": f"Stopped: model call failed ({str(e)[:150]})."}
        return {"step": step, "pending": [], "msgs": s["msgs"] + [{"role": "user",
                "content": "Your last tool call was invalid. Try again with valid arguments."}]}
    calls = m.tool_calls or []
    msg = {"role": "assistant", "content": m.content or ""}
    if calls:
        msg["tool_calls"] = [{"id": c.id, "type": "function",
                              "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]
    msgs = s["msgs"] + [msg] + ([] if calls else [{"role": "user", "content": "Call a tool or final_answer."}])
    for c in calls:
        say(f"[step {step}/{MAX_STEPS}] {c.function.name}({c.function.arguments[:100]})")
    return {"msgs": msgs, "step": step, "pending": calls}

def route(s):
    if s.get("answer"):
        return END
    if any(c.function.name == "final_answer" for c in s["pending"]):
        return "verify"
    return "tools" if s["pending"] else "agent"

def tools_node(s):
    out = []
    for c in s["pending"]:
        try:
            r = IMPL[c.function.name](**json.loads(c.function.arguments))
        except Exception as e:
            r = f"ERROR: bad call ({e})"
        out.append({"role": "tool", "tool_call_id": c.id, "content": r})
    return {"msgs": s["msgs"] + out, "pending": []}

def verify_node(s):
    final = next(c for c in s["pending"] if c.function.name == "final_answer")
    try:
        text = clean(json.loads(final.function.arguments)["answer"])
    except Exception:
        text = ""
    probs = check_citations(text) if text else ["empty answer"]
    if not probs or s["retried"] or s["step"] >= MAX_STEPS:
        return {"answer": render(text) + (f"\n\n(Warning: unverified: {probs})" if probs else "")}
    res = [{"role": "tool", "tool_call_id": c.id,
            "content": ("Rejected, fix and call final_answer again: " + "; ".join(probs))
            if c.id == final.id else "Skipped."} for c in s["pending"]]
    return {"msgs": s["msgs"] + res, "retried": True, "pending": []}

g = StateGraph(State)
g.add_node("agent", agent_node); g.add_node("tools", tools_node); g.add_node("verify", verify_node)
g.add_edge(START, "agent")
g.add_conditional_edges("agent", route, ["tools", "verify", "agent", END])
g.add_edge("tools", "agent")
g.add_conditional_edges("verify", lambda s: END if s.get("answer") else "agent", [END, "agent"])
app = g.compile()

if __name__ == "__main__":
    open(LOG, "w", encoding="utf-8").close()  # start a fresh log
    if sys.argv[1:] == ["--graph"]:
        say(app.get_graph().draw_mermaid())
    else:
        q = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"
        out = app.invoke({"msgs": [{"role": "user", "content": q}], "step": 0, "retried": False,
                          "pending": [], "answer": None}, config={"recursion_limit": MAX_STEPS * 3 + 5})
        say("\n" + out["answer"])