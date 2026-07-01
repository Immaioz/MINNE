"""
Webapp per lanciare la pipeline e visualizzare risultati in markdown.
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import flask
import markdown

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

app = flask.Flask(__name__)
app.secret_key = "research-pipeline-webapp-secret"

running_jobs: dict[str, dict] = {}
LOG_DIR = Path("/tmp/opencode/pipeline-logs")


# ---------------------------------------------------------------------------
# template filters
# ---------------------------------------------------------------------------

@app.template_filter("datetimeformat")
def _datetimeformat(timestamp: int) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _topic_slug(topic: str) -> str:
    slug = topic.lower().strip()
    slug = __import__("re").sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")[:60]


def list_results() -> list[dict]:
    if not RESULTS_DIR.is_dir():
        return []
    entries = []
    for d in sorted(RESULTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        md_file = d / "results.md"
        if md_file.is_file():
            topic = ""
            for line in md_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("**Topic:**"):
                    topic = line.replace("**Topic:**", "").strip()
                    break
            entries.append({
                "slug": d.name,
                "topic": topic or d.name,
                "mtime": d.stat().st_mtime,
                "files": [f.name for f in d.iterdir() if f.suffix in (".md", ".pdf")],
            })
    return entries


def md_to_html(text: str) -> str:
    return markdown.markdown(text, extensions=["fenced_code", "tables", "codehilite"])


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    results = list_results()
    return flask.render_template("index.html", results=results)


@app.route("/run", methods=["POST"])
def run():
    topic = flask.request.form.get("topic", "").strip()
    if not topic:
        flask.flash("Inserisci un topic.")
        return flask.redirect("/")

    try:
        subtopics = int(flask.request.form.get("subtopics", 3))
    except ValueError:
        subtopics = 3
    try:
        articles = int(flask.request.form.get("articles", 3))
    except ValueError:
        articles = 3

    slug = _topic_slug(topic)

    # already exists
    if (RESULTS_DIR / slug / "results.md").is_file():
        return flask.redirect(f"/result/{slug}")

    # already running
    if slug in running_jobs:
        return flask.render_template("loading.html", slug=slug, topic=topic)

    # launch pipeline in background with streaming log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{slug}.log"
    status_file = LOG_DIR / f"{slug}.status"

    status_file.write_text("running", encoding="utf-8")

    def _run():
        running_jobs[slug] = {"status": "running", "topic": topic}
        cmd = [
            sys.executable, "-u", str(BASE_DIR / "pipeline.py"), topic,
            "--subtopics", str(subtopics), "--articles", str(articles),
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=BASE_DIR,
            )
            with log_file.open("w", encoding="utf-8") as f:
                for line in proc.stdout:
                    f.write(line)
                    f.flush()
            proc.wait(timeout=1800)
            final = "done" if proc.returncode == 0 else "error"
            status_file.write_text(final, encoding="utf-8")
            running_jobs[slug] = {"status": final, "topic": topic}
        except subprocess.TimeoutExpired:
            proc.kill()
            with log_file.open("a", encoding="utf-8") as f:
                f.write("\n[TIMEOUT] Pipeline interrotta dopo 1800s\n")
            status_file.write_text("error", encoding="utf-8")
            running_jobs[slug] = {"status": "error", "topic": topic}
        except Exception as e:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"\n[ERROR] {e}\n")
            status_file.write_text("error", encoding="utf-8")
            running_jobs[slug] = {"status": "error", "topic": topic}

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return flask.render_template("loading.html", slug=slug, topic=topic)


@app.route("/check/<slug>")
def check(slug):
    job = running_jobs.get(slug)
    if job is not None:
        return {"status": job["status"]}

    # fallback: file-based status (persiste tra restart del server)
    status_file = LOG_DIR / f"{slug}.status"
    if status_file.is_file():
        st = status_file.read_text(encoding="utf-8").strip()
        return {"status": st}

    if (RESULTS_DIR / slug / "results.md").is_file():
        return {"status": "done"}
    return {"status": "not_found"}


@app.route("/output/<slug>")
def output(slug):
    log_file = LOG_DIR / f"{slug}.log"
    if not log_file.is_file():
        return flask.Response("", mimetype="text/plain")

    try:
        # returns last 80 lines
        lines = log_file.read_text(encoding="utf-8").splitlines()
        tail = lines[-80:] if len(lines) > 80 else lines
        return flask.Response("\n".join(tail), mimetype="text/plain")
    except Exception:
        return flask.Response("", mimetype="text/plain")


@app.route("/result/<slug>")
def result(slug):
    md_file = RESULTS_DIR / slug / "results.md"
    if not md_file.is_file():
        flask.flash(f"Nessun risultato trovato per «{slug}».")
        return flask.redirect("/")

    md_text = md_file.read_text(encoding="utf-8")
    html_body = md_to_html(md_text)

    topic = ""
    for line in md_text.splitlines():
        if line.startswith("**Topic:**"):
            topic = line.replace("**Topic:**", "").strip()
            break

    files = [p.name for p in (RESULTS_DIR / slug).iterdir() if p.suffix in (".md", ".pdf")]

    return flask.render_template(
        "result.html", slug=slug, topic=topic or slug,
        html_body=html_body, files=files,
    )


@app.route("/raw/<slug>/<filename>")
def raw_file(slug, filename):
    return flask.send_from_directory(str(RESULTS_DIR / slug), filename)


@app.route("/api/results")
def api_results():
    return flask.jsonify(list_results())


# ---------------------------------------------------------------------------
# Chat — agente conversazionale basato su knowledge_base.md
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """\
You are a research assistant helping a scientist explore a collection of
scientific articles and novelty proposals.

You have access to a knowledge base (attached file) containing all the
articles, summaries, related work, novelty proposals, and bibliography
for a specific research topic.

Guidelines:
- Answer questions based ONLY on the information in the knowledge base.
- If the answer is not in the knowledge base, say so politely.
- When discussing an article, mention its authors and year.
- When discussing a novelty, refer to its difficulty and rationale.
- Keep answers concise but informative.
- Use markdown formatting for readability when appropriate.
"""


def _load_chat_history(slug: str) -> list[dict]:
    """Load the current chat history for a topic."""
    history_file = RESULTS_DIR / slug / "chats" / "current.json"
    if history_file.is_file():
        try:
            return json.loads(history_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_chat_history(slug: str, history: list[dict]) -> None:
    """Save the current chat history."""
    chat_dir = RESULTS_DIR / slug / "chats"
    chat_dir.mkdir(parents=True, exist_ok=True)
    (chat_dir / "current.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def _save_chat_report(slug: str, history: list[dict]) -> str:
    """Save a chat report and return the filename."""
    chat_dir = RESULTS_DIR / slug / "chats"
    chat_dir.mkdir(parents=True, exist_ok=True)

    # Build markdown report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [f"# Chat Report — {slug}\n"]
    parts.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    parts.append("---\n\n")
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"**You:**\n\n{content}\n\n")
        elif role == "assistant":
            parts.append(f"**Assistant:**\n\n{content}\n\n")
    report_name = f"chat_{timestamp}.md"
    report_path = chat_dir / report_name
    report_path.write_text("".join(parts), encoding="utf-8")
    return report_name


def _get_knowledge_base_path(slug: str) -> Path | None:
    """Return the path to the knowledge base file if it exists."""
    kb = RESULTS_DIR / slug / "knowledge_base.md"
    return kb if kb.is_file() else None


@app.route("/chat/<slug>")
def chat_page(slug):
    kb_path = _get_knowledge_base_path(slug)
    if not kb_path:
        flask.flash(f"Nessuna knowledge base trovata per «{slug}». Esegui prima la pipeline.")
        return flask.redirect(f"/result/{slug}")

    # Load topic from results
    md_file = RESULTS_DIR / slug / "results.md"
    topic = slug
    if md_file.is_file():
        for line in md_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("**Topic:**"):
                topic = line.replace("**Topic:**", "").strip()
                break

    # Load history
    history = _load_chat_history(slug)

    return flask.render_template(
        "chat.html", slug=slug, topic=topic,
        history=history,
        chat_files=sorted(
            p.name for p in (RESULTS_DIR / slug / "chats").iterdir()
            if p.suffix == ".md"
        ) if (RESULTS_DIR / slug / "chats").is_dir() else [],
    )


@app.route("/chat/<slug>/ask", methods=["POST"])
def chat_ask(slug):
    kb_path = _get_knowledge_base_path(slug)
    if not kb_path:
        return {"error": "Knowledge base not found"}, 404

    data = flask.request.get_json(silent=True)
    if not data or "message" not in data:
        return {"error": "Missing message"}, 400

    user_message = data["message"].strip()
    if not user_message:
        return {"error": "Empty message"}, 400

    # Load history
    history = _load_chat_history(slug)

    # Append user message
    history.append({"role": "user", "content": user_message, "timestamp": time.time()})

    # Build context from history (keep last N messages)
    context_messages = history[-20:]  # last 20 messages for context

    # Build the prompt for opencode
    conversation = ""
    for msg in context_messages[:-1]:  # exclude the last user message (it's the current one)
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"\n\n**{role}:** {msg['content']}"

    prompt = (
        f"{CHAT_SYSTEM_PROMPT}\n\n"
        f"---\n\n"
        f"Conversation so far:{conversation}\n\n"
        f"**User:** {user_message}\n\n"
        f"**Assistant:**"
    )

    # Call opencode with the knowledge base as context
    try:
        cmd = [
            "opencode", "run", "--agent", "researcher",
            "--file", str(kb_path),
        ]
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=300,
            cwd=BASE_DIR,
        )
        if result.returncode != 0:
            reply = f"[Error] opencode exit {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        else:
            reply = result.stdout.strip()
    except FileNotFoundError:
        reply = "[Error] opencode not found. Install it first."
    except subprocess.TimeoutExpired:
        reply = "[Error] opencode timed out (300s)."
    except Exception as e:
        reply = f"[Error] {e}"

    # Append assistant reply
    history.append({"role": "assistant", "content": reply, "timestamp": time.time()})
    _save_chat_history(slug, history)

    return {"reply": reply, "history_length": len(history)}


def _append_chat_to_knowledge_base(kb_path: Path, slug: str, history: list[dict]) -> None:
    """Append a chat session's Q&A to the knowledge base so future sessions
    can benefit from the enriched context."""
    parts = ["\n\n"]
    parts.append("---\n\n")
    parts.append(f"## Chat Knowledge — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"### User Question\n\n{content}\n\n")
        elif role == "assistant":
            parts.append(f"### Assistant Answer\n\n{content}\n\n")
    parts.append("\n")
    with open(kb_path, "a", encoding="utf-8") as f:
        f.write("".join(parts))


@app.route("/chat/<slug>/save", methods=["POST"])
def chat_save(slug):
    kb_path = _get_knowledge_base_path(slug)
    if not kb_path:
        return {"error": "Knowledge base not found"}, 404

    history = _load_chat_history(slug)
    if not history:
        return {"error": "No chat history to save"}, 400

    report_name = _save_chat_report(slug, history)

    # Enrich knowledge base with this chat session
    _append_chat_to_knowledge_base(kb_path, slug, history)

    # Clear current history
    _save_chat_history(slug, [])

    return {"report": report_name, "message": "Chat saved and knowledge base enriched."}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"\n  🌐  Research Pipeline Webapp — http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
