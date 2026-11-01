"""Clean and chunk documents, then write them to the Qdrant vector store (see store.py).

Usage:
    python index_corpus.py --recreate                 # (re)build the whole collection from corpus/
    python index_corpus.py                            # upsert corpus/ into the existing collection
    python index_corpus.py --collection librarian_c150 --chunk-size 150 --overlap 25 --recreate   # experiment
    python index_corpus.py --add notes.md --type doc --title "My page" --url https://...  # add one document
    python index_corpus.py --query "how do I keep state between reruns?"   # compare keyword / vector / hybrid

Adding content later goes through add_document(): it chunks the text with the same rules, embeds it and replaces
any earlier version of that document. The web service exposes it as POST /api/documents.
"""
import argparse, html, json, re, time
from pathlib import Path

import store

DOCS_BASE = "https://docs.streamlit.io"
ADMONITIONS = "Note|Important|Tip|Warning|Caution"
CHUNK_SIZE, OVERLAP, MIN_WORDS = 250, 40, 40           # chosen settings (see design.md)


# ---------------- chunking ----------------
def words(t):
    return len(t.split())


def split_blocks(text):
    """Split into paragraphs, keeping fenced code blocks whole. Returns [(text, is_code)]."""
    blocks, cur, in_fence = [], [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if not in_fence and cur:                 # close paragraph before the fence
                blocks.append(("\n".join(cur), False)); cur = []
            cur.append(line)
            if in_fence:                              # fence closed
                blocks.append(("\n".join(cur), True)); cur = []
            in_fence = not in_fence
        elif in_fence:
            cur.append(line)
        elif line.strip() == "":
            if cur:
                blocks.append(("\n".join(cur), False)); cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(("\n".join(cur), in_fence))
    return blocks


def pack(blocks, size, overlap):
    """Greedily pack blocks into chunks of ~size words, carrying ~overlap words forward."""
    chunks, cur, n = [], [], 0

    def flush():
        if cur:
            chunks.append("\n\n".join(b for b, _ in cur))

    for text, is_code in blocks:
        w = words(text)
        if w > size and not is_code:                  # oversized prose: hard split by words
            toks = text.split()
            step = max(size - overlap, 1)
            for i in range(0, len(toks), step):
                chunks.append(" ".join(toks[i:i + size]))
            continue
        if n + w > size and cur:
            flush()
            keep, kn = [], 0                          # overlap: reuse trailing blocks
            for b in reversed(cur):
                if kn + words(b[0]) > overlap:
                    break
                keep.insert(0, b); kn += words(b[0])
            cur, n = keep, kn
        cur.append((text, is_code)); n += w
    flush()
    return [c for c in chunks if c.strip()]


def split_sections(text):
    """Split markdown by headings (ignoring '#' inside code fences). Returns [(heading, body)]."""
    secs, head, buf, in_fence = [], "", [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and re.match(r"^#{1,4}\s+\S", line):
            if buf:
                secs.append((head, "\n".join(buf)))
            head, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if buf:
        secs.append((head, "\n".join(buf)))
    return [(h, b) for h, b in secs if b.strip()]


def front_matter(text):
    meta = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip("\"'")
            text = parts[2]
    return meta, text



# ---------------- cleaning (MDX/JSX noise) ----------------
def _attr(attrs, name):
    m = re.search(name + r'="([^"]*)"', attrs)
    return m.group(1) if m else ""


def _clean_prose(t):
    codes = []                                           # protect inline `code`
    def keep(m):
        codes.append(m.group(0)); return f"\x00{len(codes) - 1}\x00"
    t = re.sub(r"`[^`\n]+`", keep, t)

    t = re.sub(r"(?m)^[ \t]+(?=</?[A-Za-z])", "", t)                # dedent JSX/HTML lines
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)                       # markdown images
    t = re.sub(r"<Image\b.*?/>", "", t, flags=re.S)                  # <Image ... />
    t = re.sub(r"<InlineCallout\b(.*?)>(.*?)</InlineCallout>",       # keep the callout text
               lambda m: f"{_attr(m.group(1), 'bold')} {m.group(2).strip()}\n".lstrip(), t, flags=re.S)
    t = re.sub(r"<Tile\b(.*?)/>",
               lambda m: f"{_attr(m.group(1), 'title')}: {_attr(m.group(1), 'text')}\n", t, flags=re.S)
    t = re.sub(r"<i\b[^>]*>.*?</i>", "", t, flags=re.S)              # icon tags
    t = re.sub(r"</t[dh]>\s*", " | ", t)                             # tables -> "a | b |"
    t = re.sub(r"</tr>", "\n", t)
    t = re.sub(rf"<({ADMONITIONS})\b[^>]*>\s*", lambda m: f"{m.group(1)}: ", t)
    t = re.sub(rf"\s*</({ADMONITIONS})>", "\n", t)
    t = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", t, flags=re.S)         # links/anchors -> text
    t = re.sub(r"<code>(.*?)</code>", r"`\1`", t, flags=re.S)
    t = re.sub(r"\{/\*.*?\*/\}", "", t, flags=re.S)                  # JSX comments
    t = re.sub(r"</?(?:div|span|Flex|table|tbody|thead|tr|td|th|sup|sub|br|p|img|center|details|"
               r"summary|iframe|YouTube|Video|TileContainer|InlineCalloutContainer|Tile|Image|"
               r"InlineCallout)\b[^>]*/?>", "", t, flags=re.S)
    t = html.unescape(t)                                             # &mdash; &dagger; ...
    t = re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], t)
    t = re.sub(r"(?m)^[ \t]*\|[ \t]*\n?", "", t)                  # lines that are just "|"
    t = re.sub(r"(?m)[ \t]+$", "", t)                              # trailing spaces
    return re.sub(r"\n{3,}", "\n\n", t)


def clean_mdx(text):
    """Strip JSX/HTML noise but never touch fenced code blocks."""
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = _clean_prose(parts[i])
    return re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip()


def combine(a, b):
    extra = b["text"] if a["section"] == b["section"] else f"{b['section']}\n{b['text']}"
    return {**a, "text": a["text"] + "\n\n" + extra}


def merge_small(chunks, min_words, max_words):
    """Merge chunks shorter than min_words into a neighbour of the same document."""
    out, carry = [], None
    for c in chunks:
        if carry is not None:
            if carry["doc_id"] == c["doc_id"]:
                c, carry = combine(carry, c), None
            else:
                out.append(carry); carry = None
        if words(c["text"]) < min_words:
            if out and out[-1]["doc_id"] == c["doc_id"] and words(out[-1]["text"]) + words(c["text"]) <= max_words:
                out[-1] = combine(out[-1], c)
            else:
                carry = c
        else:
            out.append(c)
    if carry is not None:
        out.append(carry)
    return out


# ---------------- loading + chunking per source ----------------
def finish(chunks, min_words, cap, merge=True):
    """Merge tiny chunks into a neighbour, drop empty leftovers, number the chunks inside their document."""
    chunks = merge_small(chunks, min_words, cap) if merge else chunks
    chunks = [c for c in chunks if words(c["text"]) >= 3]
    for n, c in enumerate(chunks):
        c["chunk"] = n
    return chunks


def text_chunks(doc_id, type, title, text, url="", date=None, size=CHUNK_SIZE, overlap=OVERLAP,
                min_words=MIN_WORDS, clean=True):
    """Chunk one document's text: split at headings, pack paragraphs up to `size` words, never split code."""
    body = clean_mdx(text) if clean and type != "issue" else text
    chunks = [dict(doc_id=doc_id, type=type, title=title, section=head, url=url, date=date, text=c)
              for head, sec in (split_sections(body) or [("", body)]) for c in pack(split_blocks(sec), size, overlap)]
    return finish(chunks, min_words, size)


def doc_chunks(path, size, overlap, min_words, clean=True):
    meta, body = front_matter(path.read_text(encoding="utf-8", errors="ignore"))
    slug = meta.get("slug")
    return text_chunks(path.name, "doc", meta.get("title") or path.stem, body, DOCS_BASE + slug if slug else DOCS_BASE,
                       None, size, overlap, min_words, clean)


def issue_chunks(path, size, overlap):
    d = json.loads(path.read_text(encoding="utf-8"))
    head = f"Issue #{d['number']}: {d['title']}\nLabels: {', '.join(d['labels'])}\n\n{d['body'] or ''}"
    comments = [(f"[{c['role']}] {c['body']}", False) for c in d["comments"] if c["body"]]
    chunks = [dict(doc_id=d["id"], type="issue", title=d["title"], section=sec, url=d["url"], date=d["closed_at"],
                   text=c)
              for sec, blocks in [("Description", split_blocks(head)), ("Discussion", comments)]
              for c in pack(blocks, size, overlap)]
    return finish(chunks, 0, size, merge=False)          # issue chunks are not merged


def release_chunks(path, size, overlap, min_words, clean=True):
    d = json.loads(path.read_text(encoding="utf-8"))
    return text_chunks(d["id"], "release", d["title"], "## Notes\n" + (d["body"] or ""), d["url"], d["date"],
                       size, overlap, min_words, clean)


def build_chunks(corpus, size=CHUNK_SIZE, overlap=OVERLAP, min_words=MIN_WORDS, clean=True):
    chunks = []
    for p in sorted((corpus / "docs").glob("*.md")):
        chunks += doc_chunks(p, size, overlap, min_words, clean)
    for p in sorted((corpus / "issues").glob("*.json")):
        chunks += issue_chunks(p, size, overlap)
    for p in sorted((corpus / "releases").glob("*.json")):
        chunks += release_chunks(p, size, overlap, min_words, clean)
    return chunks


# ---------------- writing to the vector store ----------------
def add_document(doc_id, type, title, text, url="", date=None, collection=store.COLLECTION,
                 size=CHUNK_SIZE, overlap=OVERLAP, min_words=MIN_WORDS):
    """Add one document to the RAG system, or replace it if `doc_id` already exists: chunk it with the corpus rules,
    embed it and upsert it. Returns the number of chunks written."""
    chunks = text_chunks(doc_id, type, title, text, url, date, size, overlap, min_words)
    store.ensure_collection(collection)
    store.delete_document(doc_id, collection)          # drop every chunk of an earlier (maybe longer) version
    store.upsert_chunks(chunks, collection)
    return len(chunks)


def index_corpus(corpus, collection=store.COLLECTION, recreate=False, size=CHUNK_SIZE, overlap=OVERLAP,
                 min_words=MIN_WORDS, clean=True):
    t0 = time.time()
    chunks = build_chunks(Path(corpus), size, overlap, min_words, clean)
    print(f"{len(chunks)} chunks from {len({c['doc_id'] for c in chunks})} documents; embedding and uploading...")
    store.ensure_collection(collection, recreate=recreate)
    store.upsert_chunks(chunks, collection, progress=True)
    print(dict(collection=collection, points=store.count(collection), chunk_size=size, overlap=overlap,
               min_words=min_words, clean_mdx=clean, seconds=round(time.time() - t0, 1),
               avg_words=round(sum(words(c["text"]) for c in chunks) / len(chunks), 1)))


def compare(query, collection=store.COLLECTION, k=5):
    """Quick look at what each search mode returns for one question."""
    for mode in ("vector", "bm25", "hybrid"):
        print(f"\n== {mode.upper()} ==")
        for c in store.search(query, k=k, mode=mode, collection=collection)[0]:
            print(f"- [{c['type']}] {c['title']} > {c['section']}  {c['url']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--collection", default=store.COLLECTION)
    ap.add_argument("--recreate", action="store_true", help="delete and rebuild the collection")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="target words per chunk")
    ap.add_argument("--overlap", type=int, default=OVERLAP, help="words carried to the next chunk")
    ap.add_argument("--min-words", type=int, default=MIN_WORDS, help="merge doc/release chunks shorter than this")
    ap.add_argument("--no-clean", action="store_true", help="skip MDX/JSX cleaning (for ablations)")
    ap.add_argument("--query", help="don't build; compare the search modes for this question")
    ap.add_argument("--add", help="add or replace one document from this text/Markdown file")
    ap.add_argument("--type", default="doc", choices=["doc", "issue", "release"])
    ap.add_argument("--title")
    ap.add_argument("--url", default="")
    ap.add_argument("--doc-id", help="defaults to the file name")
    ap.add_argument("--date")
    a = ap.parse_args()
    if a.query:
        compare(a.query, a.collection)
    elif a.add:
        path = Path(a.add)
        n = add_document(a.doc_id or path.name, a.type, a.title or path.stem, path.read_text(encoding="utf-8"),
                         a.url, a.date, a.collection, a.chunk_size, a.overlap, a.min_words)
        print(f"{a.doc_id or path.name}: {n} chunks written to {a.collection}")
    else:
        index_corpus(a.corpus, a.collection, a.recreate, a.chunk_size, a.overlap, a.min_words, not a.no_clean)
