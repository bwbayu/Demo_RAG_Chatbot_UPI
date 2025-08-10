from pinecone.grpc import PineconeGRPC as Pinecone
from main import get_dense_embeddings, get_sparse_embeddings, create_corpus_train_bm25_model
from pinecone_text.sparse import BM25Encoder
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
EMBED_DIM = int(os.getenv('EMBED_DIM')) if os.getenv('EMBED_DIM') else None
TOP_K = 5
print("load key done")

# config
pc = Pinecone(api_key=PINECONE_API_KEY)
index_dense = pc.Index(host=HOST_PINECONE_DENSE)
index_sparse = pc.Index(host=HOST_PINECONE_SPARSE)
print("config done")

# create corpus and train bm25 model
bm25 = BM25Encoder(stem=False)
create_corpus_train_bm25_model(bm25)
print("load bm25 done")

def search_dense_index(text: str):
    dense_response = index_dense.query(
        namespace=NAMESPACE,
        vector=get_dense_embeddings(text, EMBED_DIM),
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )
    # matches = dense_response.get("matches", []) or []
    print(dense_response)

def search_sparse_index(text: str):
    sparse_response = index_sparse.query(
        namespace=NAMESPACE,
        sparse_vector=get_sparse_embeddings(text, bm25),
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )
    # matches = sparse_response.get("matches", []) or []
    print(sparse_response)

if __name__ == "__main__": 
    query = "describe the history of computer science department"
    search_dense_index(query)
    print("===============================")
    search_sparse_index(query)