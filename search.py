from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from main import get_dense_embeddings, get_sparse_embeddings, create_corpus_train_bm25_model
import os
from dotenv import load_dotenv
from collections import defaultdict
import json
import requests

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# load env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HOST_PINECONE_DENSE = os.getenv('HOST_PINECONE_DENSE')
HOST_PINECONE_SPARSE = os.getenv('HOST_PINECONE_SPARSE')
SILICONFLOW_URL_EMBEDDING = os.getenv('SILICONFLOW_URL_EMBEDDING')
SILICONFLOW_URL_RERANK = os.getenv('SILICONFLOW_URL_RERANK')
SILICONFLOW_API_KEY = os.getenv('SILICONFLOW_API_KEY')
NAMESPACE = os.getenv('NAMESPACE')
EMBED_DIM = int(os.getenv('EMBED_DIM')) if os.getenv('EMBED_DIM') else None
TOP_K = 20 # 20 -> type=facility | 5 -> history, person

# config
pc = Pinecone(api_key=PINECONE_API_KEY)
index_dense = pc.Index(host=HOST_PINECONE_DENSE)
index_sparse = pc.Index(host=HOST_PINECONE_SPARSE)

# create corpus and train bm25 model
bm25 = BM25Encoder(stem=False)
create_corpus_train_bm25_model(bm25)

def search_dense_index(text: str):
    dense_response = index_dense.query(
        namespace=NAMESPACE,
        vector=get_dense_embeddings(text, EMBED_DIM),
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )
    matches = dense_response.get("matches", []) or []
    dense_results = []
    for item in matches:
        md = item.get("metadata") or {}
        md = {key: value for key, value in md.items() if key not in {'lang', 'type'}}
        dense_results.append({
            "id": item.get("id"),
            "similarity": item.get('score', 0.0),
            "metadata": md
        })

    return dense_results

def search_sparse_index(text: str):
    sparse_response = index_sparse.query(
        namespace=NAMESPACE,
        sparse_vector=get_sparse_embeddings(text=text, bm25_model=bm25, query_type='search'),
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )
    matches = sparse_response.get("matches", []) or []
    sparse_results = []
    for item in matches:
        md = item.get("metadata") or {}
        md = {key: value for key, value in md.items() if key not in {'lang', 'type'}}
        sparse_results.append({
            "id": item.get("id"),
            "similarity": item.get('score', 0.0),
            "metadata": md
        })

    return sparse_results

"""
RRF score(d) = Σ 1/(k+rank(d)) where k is between 1-60 where d is document
"""
def rrf_fusion(dense_results, sparse_results, k=60, top_n=TOP_K):
    scores = defaultdict(float)

    # add rrf score from dense result
    for rank, res in enumerate(dense_results, 1):
        doc_id = res['id']
        scores[doc_id] = 1/(k + rank)

    # add rrf score from dense result
    for rank, res in enumerate(sparse_results, 1):
        doc_id = res['id']
        scores[doc_id] = 1/(k + rank)

    # sort by rrf score desc
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    all_results = {r['id']: r for r in dense_results + sparse_results}
    fused_results = [all_results[doc_id] for doc_id, _ in fused[:top_n]]

    return fused_results

def reranking_results(query, docs, fused_results):
    payload = {
        "model": "Qwen/Qwen3-Reranker-8B",
        "query": query,
        "documents": docs,
        "return_documents": False
    }

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(SILICONFLOW_URL_RERANK, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        reranked_data = response.json()
        reranked_results = reranked_data.get('results', [])
        
        # map original data after reranking
        final_results = []
        for res in reranked_results:
            index = res['index']
            relevance_score = res['relevance_score']
            original_result = fused_results[index]
            final_results.append({
                "id": original_result['id'],
                "similarity": relevance_score,
                "metadata": original_result['metadata']
            })
        
        return final_results
    else:
        print(f"Error in reranking: {response.status_code} - {response.text}")
        return 0

def context_generation(query, contexts, chat_history):
    context = "\n\n".join([data['metadata'].get("text", "") for data in contexts])
    template = """You are an AI assistant answering questions based strictly on the provided context and, if present, the chat history.
    Only use information that is clearly relevant to the question. Ignore unrelated or ambiguous context. 
    If the answer cannot be determined from the information provided, respond with: "I don't know."

    Context:
    {context}

    Chat history:
    {chat_history}

    Question:
    {query}
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="o4-mini")
    chain = prompt | model
    response = chain.invoke(
        {
            "context": context, 
            "query": query,
            "chat_history": chat_history
        }
    )
    # print("Question : ", query)
    # print("Response : ")
    # print(response.content)
    return response.content

def RAG_pipeline(query, chat_history):
    # query = "explain the internship program in computer science department"
    # search top k result
    dense_results = search_dense_index(query)
    sparse_results = search_sparse_index(query)
    # fused dense and sparse result using RRF
    fused_results = rrf_fusion(dense_results, sparse_results)
    # extract text data for reranking
    docs = [result['metadata'].get("text", '') for result in fused_results]
    contexts = reranking_results(query, docs, fused_results)
    # print("Reranked Results:")
    # print(json.dumps(contexts, indent=4))
    response = context_generation(query, contexts, chat_history)
    
    return response