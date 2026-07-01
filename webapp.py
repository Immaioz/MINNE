"""
Webapp per lanciare la pipeline e visualizzare risultati in markdown.
"""

import json
import subprocess
import sys
import threading
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
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"\n  🌐  Research Pipeline Webapp — http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
