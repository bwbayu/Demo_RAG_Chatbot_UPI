from pinecone.grpc import PineconeGRPC as Pinecone
from main import get_embeddings
import json
import os
from dotenv import load_dotenv

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# load env
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HOST_PINECONE_DENSE = os.getenv('HOST_PINECONE_DENSE')
HOST_PINECONE_SPARSE = os.getenv('HOST_PINECONE_SPARSE')
SILICONFLOW_URL = os.getenv('SILICONFLOW_URL')
SILICONFLOW_API_KEY = os.getenv('SILICONFLOW_API_KEY')
NAMESPACE = os.getenv('NAMESPACE')
EMBED_DIM = os.getenv('EMBED_DIM')
TOP_K = 3
OUTFILE = "vector_search_results_new.json"
print("load key done")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_dense = pc.Index(host=HOST_PINECONE_DENSE)
index_sparse = pc.Index(host=HOST_PINECONE_SPARSE)

def search_one(text: str, namespace: str = NAMESPACE, top_k: int = TOP_K):
    res = index_dense.query(
        namespace=namespace,
        vector=get_embeddings(text, EMBED_DIM),
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    matches = res.get("matches", []) or []
    out = []
    for rank, item in enumerate(matches, start=1):
        md = item.get("metadata") or {}
        out.append({
            "rank": rank,
            "title": md.get("title", ""),
            "similarity": item.get("score", 0.0)
        })
    return out

def run_and_save(queries, outfile: str = OUTFILE, namespace: str = NAMESPACE, top_k: int = TOP_K):
    """
    Executes all queries, aggregates results, and writes to `outfile` in JSON:
    [
      {"query": "...", "results": [{"rank":1,"title":"...","similarity":0.XXX}, ...]},
      ...
    ]
    """
    payload = []
    for q in queries:
        results = search_one(q, namespace=namespace, top_k=top_k)
        payload.append({"query": q, "results": results})

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(payload)} query result sets to {outfile}")
    return outfile

