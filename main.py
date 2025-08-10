import requests
import json
import os
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from pinecone_text.sparse import BM25Encoder

# load env
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HOST_PINECONE_DENSE = os.getenv('HOST_PINECONE_DENSE')
HOST_PINECONE_SPARSE = os.getenv('HOST_PINECONE_SPARSE')
SILICONFLOW_URL = os.getenv('SILICONFLOW_URL')
SILICONFLOW_API_KEY = os.getenv('SILICONFLOW_API_KEY')
NAMESPACE = os.getenv('NAMESPACE')
EMBED_DIM = os.getenv('EMBED_DIM')
print("load key done")

# config
pc = Pinecone(api_key=PINECONE_API_KEY)
index_dense = pc.Index(host=HOST_PINECONE_DENSE)
index_sparse = pc.Index(host=HOST_PINECONE_SPARSE)
headers = {
    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
    "Content-Type": "application/json"
}
print("config done")

def get_embeddings(text, dim_size = 1024):
    payload = {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": text,
        "encoding_format": "float",
        "dimensions": dim_size
    }

    response = requests.post(SILICONFLOW_URL, json=payload, headers=headers)

    return response.json()['data'][0]['embedding']

def generate_embedding(path_files):
    try:
        with open(path_files, 'r') as file:
            data = json.load(file)
            new_data = []
            for item in data:
                new_item = {
                    "id": item['_id'], 
                    "values": get_embeddings(item['text'], EMBED_DIM), 
                    "metadata": {key: value for key, value in item.items() if key not in {'_id'}}
                }
                new_data.append(new_item)
            
            return new_data
    except FileNotFoundError:
        print(f"Error: {path_files} not found. Please ensure the file exists in the correct directory.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path_files}. The file might be malformed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

index_dense.upsert(
  vectors=generate_embedding("data/full_text.json"),
  namespace=NAMESPACE
)
