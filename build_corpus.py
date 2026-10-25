"""Build the Streamlit corpus: docs + closed issues + release notes.

Usage:
    export GITHUB_TOKEN=ghp_xxx        # free token, no scopes needed
    python build_corpus.py --max-issues 600

Output:
    corpus/docs/*.md        one file per docs page (front matter kept as metadata)
    corpus/issues/*.json    one file per closed issue, comments included, scrubbed
    corpus/releases/*.json  one file per release
    corpus/manifest.json    counts, so you can report corpus size in design.md
"""
import argparse, json, os, re, subprocess, shutil, time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()  # reads GITHUB_TOKEN from a local .env file if there is one

API = "https://api.github.com"
OUT = Path("corpus")
HEADERS = {"Accept": "application/vnd.github+json"}
if os.getenv("GITHUB_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

# ---------- scrubbing (personal data) ----------
SCRUBS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"(?<!\w)@[A-Za-z0-9][A-Za-z0-9-]{0,38}"), "[USER]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[TOKEN]"),
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "[TOKEN]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[TOKEN]"),
    (re.compile(r"/(?:Users|home)/[^/\s]+"), "/[HOME]"),
    (re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+"), r"C:\\Users\\[HOME]"),
]


def scrub(text: str) -> str:
    for pat, repl in SCRUBS:
        text = pat.sub(repl, text or "")
    return text



def safe_rmtree(path):
    """shutil.rmtree that also works on Windows, where git's files are read-only."""
    import stat

    def on_error(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass  # leftover temp folder is harmless; never crash the build for it

    if Path(path).exists():
        shutil.rmtree(path, onexc=on_error)

# ---------- docs ----------
def collect_docs():
    tmp = Path("_docs_clone")
    safe_rmtree(tmp)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/streamlit/docs.git", str(tmp)], check=True)
    dest = OUT / "docs"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for md in (tmp / "content").rglob("*.md"):
        rel = md.relative_to(tmp / "content")
        target = dest / ("__".join(rel.parts))      # flat, unique file names
        target.write_text(md.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        n += 1
    safe_rmtree(tmp)
    if n == 0:
        raise SystemExit("No .md files found under content/. The docs repo layout may have changed.")
    print(f"docs: copied {n} markdown pages")
    return n


# ---------- GitHub helpers ----------
def get(url, params=None):
    for _ in range(5):
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code in (403, 429) and r.headers.get("x-ratelimit-remaining") == "0":
            wait = max(int(r.headers.get("x-ratelimit-reset", 0)) - time.time(), 5)
            print(f"rate limited, sleeping {int(wait)}s")
            time.sleep(wait + 1)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("too many retries")


def paged(url, params, limit):
    out, page = [], 1
    while len(out) < limit:
        batch = get(url, {**params, "per_page": 100, "page": page}).json()
        if not batch:
            break
        out += batch
        page += 1
    return out[:limit]


# ---------- issues ----------
def iter_closed_issues(max_pages=100):
    """Yield closed items lazily, newest first, so we can stop as soon as we have enough."""
    for page in range(1, max_pages + 1):
        batch = get(f"{API}/repos/streamlit/streamlit/issues",
                    {"state": "closed", "sort": "updated", "direction": "desc",
                     "per_page": 100, "page": page}).json()
        if not batch:
            return
        yield from batch


def collect_issues(max_issues):
    dest = OUT / "issues"
    dest.mkdir(parents=True, exist_ok=True)
    n = seen = 0
    for it in iter_closed_issues():
        seen += 1
        if "pull_request" in it:          # the issues endpoint also returns PRs
            continue
        if it["comments"] == 0:           # no discussion = nothing checkable
            continue
        path = dest / f"issue-{it['number']}.json"
        if path.exists():                 # resumable: don't re-download on a rerun
            n += 1
        else:
            comments = paged(it["comments_url"], {}, 50)
            doc = {
                "id": f"issue-{it['number']}",
                "type": "issue",
                "number": it["number"],
                "url": it["html_url"],
                "title": scrub(it["title"]),
                "state": it["state"],
                "state_reason": it.get("state_reason"),
                "labels": [l["name"] for l in it["labels"]],
                "created_at": it["created_at"],
                "closed_at": it["closed_at"],
                "body": scrub(it["body"]),
                # author logins are deliberately NOT stored (personal data)
                "comments": [{"role": c.get("author_association", "NONE"),
                              "created_at": c["created_at"],
                              "body": scrub(c["body"])} for c in comments],
            }
            path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            n += 1
        if n % 50 == 0:
            print(f"issues: kept {n} (scanned {seen} items)")
        if n >= max_issues:
            break
    return n


# ---------- releases ----------
def collect_releases(limit=60):
    dest = OUT / "releases"
    dest.mkdir(parents=True, exist_ok=True)
    rels = paged(f"{API}/repos/streamlit/streamlit/releases", {}, limit)
    for r in rels:
        doc = {"id": f"release-{r['tag_name']}", "type": "release", "url": r["html_url"],
               "title": r["name"] or r["tag_name"], "version": r["tag_name"],
               "date": r["published_at"], "body": scrub(r["body"])}
        (dest / f"{doc['id']}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return len(rels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-issues", type=int, default=600)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    manifest = {"docs": collect_docs(),
                "issues": collect_issues(a.max_issues),
                "releases": collect_releases()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("done:", manifest)
