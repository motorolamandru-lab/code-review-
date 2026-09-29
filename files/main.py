"""Code Review Agent (Java). Run: uvicorn main:app --reload
Free AI providers are tried in this order, whichever keys you set:
GROQ_API_KEY -> CEREBRAS_API_KEY -> GEMINI_API_KEY -> OPENROUTER_API_KEY -> ANTHROPIC_API_KEY. No key = offline demo checks."""
import os, json, re, time, hashlib, sqlite3, urllib.request, urllib.error
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()
UA = {"User-Agent": "code-review-agent/1.0"}
KEYS = {"groq": os.getenv("GROQ_API_KEY"), "cerebras": os.getenv("CEREBRAS_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"), "openrouter": os.getenv("OPENROUTER_API_KEY"), "claude": os.getenv("ANTHROPIC_API_KEY")}
BASES = {"groq": "https://api.groq.com/openai/v1", "cerebras": "https://api.cerebras.ai/v1",
         "openrouter": "https://openrouter.ai/api/v1"}
PREFER = {"groq": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"],
          "cerebras": ["gpt-oss-120b", "llama-3.3-70b", "llama3.1-8b"],
          "openrouter": ["meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-oss-120b:free", "qwen/qwen3-coder:free"]}
SKIP = ("whisper", "guard", "tts", "embed", "orpheus", "playai", "moderation", "safeguard", "image", "audio", "live")

def db():
    c = sqlite3.connect("memory.db"); c.row_factory = sqlite3.Row
    c.executescript("""CREATE TABLE IF NOT EXISTS rules(id INTEGER PRIMARY KEY, text TEXT UNIQUE, kind TEXT);
    CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY, title TEXT, verdict TEXT);""")
    return c

def stats(c):
    out = {}
    for r in c.execute("SELECT title, verdict, COUNT(*) n FROM feedback GROUP BY title, verdict"):
        out.setdefault(r["title"], {"accept": 0, "dismiss": 0})[r["verdict"]] = r["n"]
    return out

class Review(BaseModel):
    code: str
    language: str = "java"
class Feedback(BaseModel):
    title: str
    verdict: str
class Rule(BaseModel):
    text: str

def demo_review(code):
    """Offline Java checks, used when no AI provider is available."""
    f = []
    for i, l in enumerate(code.split("\n"), 1):
        s = l.strip()
        if not s or s.startswith(("//", "*", "/*")):
            continue
        if "System.out.print" in s:
            f.append((i, "low", "Debug print left in code", "Use a logger (e.g. SLF4J) instead of System.out.", 'logger.info("...");'))
        if re.search(r"catch\s*\(\s*(Exception|Throwable)\b", s):
            f.append((i, "medium", "Catching generic Exception", "Catch the specific exceptions you expect.", "catch (IOException e) {"))
        if "printStackTrace" in s:
            f.append((i, "medium", "printStackTrace used", "Log the exception with a logger.", 'logger.error("msg", e);'))
        if re.search(r'==\s*"|"\s*==', s):
            f.append((i, "high", "String compared with ==", "== compares references, not text.", 'name.equals("x")'))
        if re.search(r"catch\s*\(.*\)\s*\{\s*\}", s):
            f.append((i, "high", "Empty catch block", "Never swallow exceptions silently.", ""))
        if "TODO" in s:
            f.append((i, "low", "Unresolved TODO", "Link the TODO to a ticket or resolve it.", ""))
        if len(l) > 100:
            f.append((i, "low", "Line too long", "Keep lines under 100 characters.", ""))
        if s.endswith(")") and not re.match(r"(if|for|while|switch|catch|else|try|do|public|private|protected|static|class|interface|@|\}|import|package)\b", s):
            f.append((i, "high", "Missing semicolon", "This statement is missing a semicolon.", s + ";"))
    return [dict(line=a, severity=b, title=c, message=d, fix=e) for a, b, c, d, e in f]

def build_prompts(code, rules, suppressed, priority):
    numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(code.split("\n"), 1))
    system = ("You are a Java code review agent for a specific team. Reply with ONLY JSON: "
              '{"findings":[{"line":int,"severity":"high|medium|low","title":str,"message":str,"fix":str}]}. '
              "Max 6 findings, including syntax errors. Use short, generic, stable titles (e.g. 'Missing semicolon') so team feedback can be matched. "
              "Follow the team rules, never raise suppressed topics, and prioritise the priority topics.")
    user = ("Team rules:\n" + "\n".join(f"- {r}" for r in rules) +
            "\nSuppressed topics (team dismissed these):\n" + "\n".join(f"- {s}" for s in suppressed) +
            "\nPriority topics (team accepted these before):\n" + "\n".join(f"- {p}" for p in priority) +
            f"\n\nJava code:\n{numbered}")
    return system, user

def parse(text):
    return json.loads(re.sub(r"```(json)?", "", text).strip())["findings"]

def http(url, body=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={**UA, "Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def try_models(models, call):
    """Run call(model) over models; retry once when busy (429/5xx), then move on."""
    last = "no model"
    for m in models[:3]:
        for attempt in range(2):
            try:
                return call(m)
            except urllib.error.HTTPError as e:
                last = f"{m} HTTP {e.code}: {e.read().decode()[:120]}"
                if e.code in (429, 500, 502, 503):
                    time.sleep(1.5 * (attempt + 1)); continue
                break
            except Exception as e:
                last = f"{m}: {e}"; break
    raise RuntimeError(last)

def openai_compat(p, system, user):  # Groq and Cerebras
    auth = {"Authorization": "Bearer " + KEYS[p]}
    try:
        found = [m["id"] for m in http(BASES[p] + "/models", headers=auth)["data"] if not any(x in m["id"] for x in SKIP) and (p != "openrouter" or m["id"].endswith(":free"))]
    except Exception:
        found = []
    pref = [m for m in PREFER[p] if not found or m in found]
    env = os.getenv(p.upper() + "_MODEL")
    models = list(dict.fromkeys(([env] if env else []) + pref + found))
    def call(m):
        body = {"model": m, "temperature": 0.2,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if p != "openrouter":
            body["response_format"] = {"type": "json_object"}
        d = http(BASES[p] + "/chat/completions", body, auth)
        return parse(d["choices"][0]["message"]["content"])
    return try_models(models, call)

def gemini(system, user):
    h = {"x-goog-api-key": KEYS["gemini"]}
    base = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        found = sorted([m["name"].split("/")[-1] for m in http(base + "?pageSize=200", headers=h).get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in m["name"]
                        and not any(x in m["name"] for x in SKIP)], reverse=True)
        found.sort(key=lambda n: "lite" in n)
    except Exception:
        found = []
    models = list(dict.fromkeys(([os.environ["GEMINI_MODEL"]] if "GEMINI_MODEL" in os.environ else []) + found))
    def call(m):
        d = http(f"{base}/{m}:generateContent", {"systemInstruction": {"parts": [{"text": system}]},
                 "contents": [{"role": "user", "parts": [{"text": user}]}],
                 "generationConfig": {"responseMimeType": "application/json"}}, h)
        return parse(d["candidates"][0]["content"]["parts"][0]["text"])
    return try_models(models or ["gemini-3.8-flash"], call)

def claude(system, user):
    import anthropic
    msg = anthropic.Anthropic(api_key=KEYS["claude"]).messages.create(
        model=os.getenv("MODEL", "claude-sonnet-5"), max_tokens=1500, system=system,
        messages=[{"role": "user", "content": user}])
    return parse(msg.content[0].text)

def ai_review(system, user, providers):
    """Cache first (instant, outage-proof), then each provider in order."""
    ck = hashlib.sha256((system + user).encode()).hexdigest()
    try:
        cache = json.load(open("cache.json"))
    except Exception:
        cache = {}
    if ck in cache:
        return cache[ck]["f"], cache[ck]["p"]
    errors = []
    for p in providers:
        try:
            f = openai_compat(p, system, user) if p in BASES else gemini(system, user) if p == "gemini" else claude(system, user)
            cache[ck] = {"f": f, "p": p}
            json.dump(cache, open("cache.json", "w"))
            return f, p
        except Exception as e:
            print(f"[{p}] failed: {e}")
            errors.append(f"{p}: {e}")
    raise RuntimeError(" | ".join(errors))

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/review")
def review(r: Review):
    c = db(); st = stats(c)
    rules = [x["text"] for x in c.execute("SELECT text FROM rules")]
    suppressed = [t for t, v in st.items() if v["dismiss"] > v["accept"]]
    priority = [t for t, v in st.items() if v["accept"] > v["dismiss"]]
    providers = [p for p in ("groq", "cerebras", "gemini", "openrouter", "claude") if KEYS[p]]
    if not providers:
        findings, mode = demo_review(r.code), "demo (no API key)"
    else:
        try:
            findings, mode = ai_review(*build_prompts(r.code, rules, suppressed, priority), providers)
        except Exception as e:
            print("ALL PROVIDERS FAILED:", e)
            findings, mode = demo_review(r.code), "demo (AI busy, using offline checks)"
    kept = [f for f in findings if f["title"] not in suppressed]
    for f in kept:
        f["remembered"] = f["title"] in priority
        f["seen"] = sum(st.get(f["title"], {}).values())
    return {"findings": kept, "hidden": len(findings) - len(kept), "mode": mode}

@app.post("/feedback")
def feedback(f: Feedback):
    c = db()
    c.execute("INSERT INTO feedback(title, verdict) VALUES(?,?)", (f.title, f.verdict))
    if stats(c)[f.title][f.verdict] >= 2:
        text = f"{'Stop flagging' if f.verdict == 'dismiss' else 'Always flag'}: {f.title}"
        c.execute("INSERT OR IGNORE INTO rules(text, kind) VALUES(?, 'learned')", (text,))
    c.commit()
    return {"ok": True}

@app.get("/memory")
def memory():
    c = db()
    return {"rules": [dict(r) for r in c.execute("SELECT id, text, kind FROM rules ORDER BY id")], "stats": stats(c)}

@app.post("/rules")
def add_rule(r: Rule):
    c = db(); c.execute("INSERT OR IGNORE INTO rules(text, kind) VALUES(?, 'team')", (r.text,)); c.commit()
    return {"ok": True}

@app.post("/reset")
def reset():
    c = db(); c.executescript("DELETE FROM rules; DELETE FROM feedback;")
    return {"ok": True}
