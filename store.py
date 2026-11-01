"""Vector store for The Librarian: one Qdrant collection holds every chunk with two vectors.

    dense   384 numbers for the chunk's meaning (BAAI/bge-small-en-v1.5, cosine similarity)
    bm25    sparse keyword vector (Qdrant/bm25); Qdrant applies the word-rarity (IDF) part on the server,
            so it stays correct as documents are added or removed: there is nothing to rebuild

Each point's payload is the chunk itself: doc_id, chunk (its number inside the document), type, title, section, url,
date, text. Point ids are derived from (doc_id, chunk), so adding a document again replaces it instead of
duplicating it. `type` and `doc_id` are indexed, so filtering and deleting stay fast as the collection grows.

Connection, from .env: QDRANT_URL + QDRANT_API_KEY (Qdrant Cloud). Without QDRANT_URL it uses an embedded local
Qdrant in .qdrant/ (no server needed), so a fresh clone can still build and query its own copy.
QDRANT_COLLECTION picks the collection (default "librarian").
"""
import os, time, uuid
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

load_dotenv()
COLLECTION = os.getenv("QDRANT_COLLECTION", "librarian")
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # how bge expects questions
DENSE, SPARSE = "dense", "bm25"                                               # vector names in the collection
ID_NAMESPACE = uuid.UUID("5a1c3e4e-7b2d-4c1a-9f0e-6d8b2a4c1e37")            # fixed, so ids are reproducible


@lru_cache
def client():
    if os.getenv("QDRANT_URL"):
        return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.getenv("QDRANT_API_KEY"), timeout=60)
    return QdrantClient(path=os.getenv("QDRANT_PATH", ".qdrant"))


def retry(call, attempts=3):
    """Run a Qdrant request, retrying network failures (DNS hiccups, timeouts) with a short backoff."""
    for i in range(attempts):
        try:
            return call()
        except ResponseHandlingException:
            if i == attempts - 1:
                raise
            time.sleep(1 + 2 * i)


@lru_cache
def dense_model():
    from fastembed import TextEmbedding                 # ONNX runtime: no PyTorch, small enough for a 512 MB host
    return TextEmbedding(DENSE_MODEL)


@lru_cache
def sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(SPARSE_MODEL)


def point_id(doc_id, n):
    return str(uuid.uuid5(ID_NAMESPACE, f"{doc_id}#{n}"))


def embed_text(c):
    """What gets embedded: the heading context travels with the chunk."""
    return f"{c['title']} > {c['section']}\n{c['text']}"


def _sparse(e):
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def _type_filter(types):
    if not types:
        return None
    return models.Filter(must=[models.FieldCondition(key="type", match=models.MatchAny(any=list(types)))])


# ---------------- writing ----------------
def ensure_collection(collection=COLLECTION, recreate=False):
    c = client()
    if recreate and c.collection_exists(collection):
        c.delete_collection(collection)
    if not c.collection_exists(collection):
        c.create_collection(
            collection,
            vectors_config={DENSE: models.VectorParams(size=384, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)})
        for field in ("type", "doc_id"):
            c.create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD)


def upsert_chunks(chunks, collection=COLLECTION, batch=32, progress=False):
    """Embed chunks (dicts with doc_id, chunk, type, title, section, url, date, text) and write them."""
    ensure_collection(collection)
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        texts = [embed_text(c) for c in part]
        dense = dense_model().embed(texts, batch_size=8)    # small batches: long chunks need a lot of memory
        sparse = sparse_model().embed(texts)
        points = [models.PointStruct(id=point_id(c["doc_id"], c["chunk"]), payload=c,
                                     vector={DENSE: d.tolist(), SPARSE: _sparse(s)})
                  for c, d, s in zip(part, dense, sparse)]
        retry(lambda: client().upsert(collection, wait=True, points=points))
        if progress:
            print(f"  {min(i + batch, len(chunks))}/{len(chunks)} chunks", flush=True)


def delete_document(doc_id, collection=COLLECTION):
    selector = models.FilterSelector(filter=models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]))
    retry(lambda: client().delete(collection, wait=True, points_selector=selector))


# ---------------- reading ----------------
def search(query, k=5, pool=50, mode="hybrid", types=None, collection=COLLECTION):
    """mode: "hybrid", or "vector" / "bm25" alone (the ablation). types: only these source types.
    Both searches go to Qdrant in one request; their top `pool` lists are merged with reciprocal rank fusion.
    Returns (chunks with their dense similarity as `score`, best dense similarity)."""
    flt, requests = _type_filter(types), []
    if mode != "bm25":
        q = next(iter(dense_model().embed([QUERY_PREFIX + query]))).tolist()
        requests.append(models.QueryRequest(query=q, using=DENSE, limit=pool, filter=flt, with_payload=True))
    if mode != "vector":
        q = _sparse(next(iter(sparse_model().query_embed([query]))))
        requests.append(models.QueryRequest(query=q, using=SPARSE, limit=pool, filter=flt, with_payload=True))
    lists = [r.points for r in retry(lambda: client().query_batch_points(collection, requests))]

    payload, rrf = {}, {}
    for points in lists:                                   # one list keeps its own order
        for rank, p in enumerate(points):
            payload[p.id] = p.payload
            rrf[p.id] = rrf.get(p.id, 0) + 1 / (60 + rank)
    sims = {p.id: p.score for p in lists[0]} if mode != "bm25" else {}
    best = lists[0][0].score if mode != "bm25" and lists[0] else 0.0
    top = sorted(rrf, key=rrf.get, reverse=True)[:k]
    return [dict(payload[i], id=str(i), score=sims.get(i)) for i in top], best


def all_chunks(collection=COLLECTION):
    """Every chunk's payload, for the evaluation's gold-passage matching (scrolls the whole collection)."""
    out, offset = [], None
    while True:
        points, offset = retry(lambda: client().scroll(collection, limit=256, offset=offset, with_payload=True,
                                                       with_vectors=False))
        out += [dict(p.payload, id=str(p.id)) for p in points]
        if offset is None:
            return out


def count(collection=COLLECTION):
    return retry(lambda: client().count(collection)).count
