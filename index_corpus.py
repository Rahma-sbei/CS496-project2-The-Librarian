"""Chunk the corpus and build a flat-file hybrid index (vectors + BM25).

Usage:
    python index_corpus.py                       # build with defaults
    python index_corpus.py --chunk-size 150 --overlap 30 --out index_small
    python index_corpus.py --out index --query "how do I keep state between reruns?"

Output (in --out, default ./index):
    chunks.jsonl      one chunk per line: text + metadata (title, url, section, ...)
    embeddings.npy    float32 matrix, one normalised row per chunk
    bm25.pkl          keyword index
    config.json       chunk size, overlap, model: so every result is reproducible
"""
import argparse, json, pickle, re, time
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DOCS_BASE = "https://docs.streamlit.io"


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


# ---------------- loading + chunking per source ----------------
def doc_chunks(path, size, overlap):
    meta, body = front_matter(path.read_text(encoding="utf-8", errors="ignore"))
    title = meta.get("title") or path.stem
    slug = meta.get("slug")
    url = DOCS_BASE + slug if slug else DOCS_BASE
    for head, sec in split_sections(body):
        for c in pack(split_blocks(sec), size, overlap):
            yield dict(doc_id=path.name, type="doc", title=title, section=head,
                       url=url, date=None, text=c)


def issue_chunks(path, size, overlap):
    d = json.loads(path.read_text(encoding="utf-8"))
    head = f"Issue #{d['number']}: {d['title']}\nLabels: {', '.join(d['labels'])}\n\n{d['body'] or ''}"
    parts = [("Description", split_blocks(head))]
    comments = [(f"[{c['role']}] {c['body']}", False) for c in d["comments"] if c["body"]]
    parts.append(("Discussion", comments))
    for sec, blocks in parts:
        for c in pack(blocks, size, overlap):
            yield dict(doc_id=d["id"], type="issue", title=d["title"], section=sec,
                       url=d["url"], date=d["closed_at"], text=c)


def release_chunks(path, size, overlap):
    d = json.loads(path.read_text(encoding="utf-8"))
    for head, sec in split_sections("## Notes\n" + (d["body"] or "")):
        for c in pack(split_blocks(sec), size, overlap):
            yield dict(doc_id=d["id"], type="release", title=d["title"], section=head,
                       url=d["url"], date=d["date"], text=c)


def build_chunks(corpus, size, overlap):
    chunks = []
    for p in sorted((corpus / "docs").glob("*.md")):
        chunks += doc_chunks(p, size, overlap)
    for p in sorted((corpus / "issues").glob("*.json")):
        chunks += issue_chunks(p, size, overlap)
    for p in sorted((corpus / "releases").glob("*.json")):
        chunks += release_chunks(p, size, overlap)
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


# ---------------- indexing ----------------
def tokenize(t):
    return re.findall(r"\w+", t.lower())


def embed_text(c):
    return f"{c['title']} > {c['section']}\n{c['text']}"


def build_index(chunks, out, model_name, cfg):
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = SentenceTransformer(model_name)
    emb = model.encode([embed_text(c) for c in chunks], batch_size=64,
                       normalize_embeddings=True, show_progress_bar=True).astype("float32")
    np.save(out / "embeddings.npy", emb)
    bm25 = BM25Okapi([tokenize(embed_text(c)) for c in chunks])
    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    cfg.update(model=model_name, n_chunks=len(chunks), build_seconds=round(time.time() - t0, 1),
               avg_words=round(sum(words(c["text"]) for c in chunks) / len(chunks), 1))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(cfg)


# ---------------- quick search demo (also the seed of your ablation) ----------------
def search(out, query, k=5):
    from sentence_transformers import SentenceTransformer
    chunks = [json.loads(l) for l in open(out / "chunks.jsonl", encoding="utf-8")]
    emb = np.load(out / "embeddings.npy")
    bm25 = pickle.load(open(out / "bm25.pkl", "rb"))
    cfg = json.loads((out / "config.json").read_text())
    q = SentenceTransformer(cfg["model"]).encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    vec_rank = np.argsort(-(emb @ q))[:50]
    kw_rank = np.argsort(-bm25.get_scores(tokenize(query)))[:50]
    rrf = {}                                          # reciprocal rank fusion
    for rank_list in (vec_rank, kw_rank):
        for r, i in enumerate(rank_list):
            rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + r)
    for name, ids in [("VECTOR", vec_rank[:k]), ("BM25", kw_rank[:k]),
                      ("HYBRID", sorted(rrf, key=rrf.get, reverse=True)[:k])]:
        print(f"\n== {name} ==")
        for i in ids:
            c = chunks[int(i)]
            print(f"- [{c['type']}] {c['title']} > {c['section']}  {c['url']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default="index")
    ap.add_argument("--chunk-size", type=int, default=250, help="target words per chunk")
    ap.add_argument("--overlap", type=int, default=40, help="words carried to next chunk")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--query", help="skip building; search an existing index")
    a = ap.parse_args()
    out = Path(a.out)
    if a.query:
        search(out, a.query)
    else:
        chunks = build_chunks(Path(a.corpus), a.chunk_size, a.overlap)
        print(f"{len(chunks)} chunks")
        build_index(chunks, out, a.model, {"chunk_size": a.chunk_size, "overlap": a.overlap})
