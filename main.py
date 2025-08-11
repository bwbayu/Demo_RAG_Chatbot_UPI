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
SILICONFLOW_URL_EMBEDDING = os.getenv('SILICONFLOW_URL_EMBEDDING')
SILICONFLOW_API_KEY = os.getenv('SILICONFLOW_API_KEY')
NAMESPACE = os.getenv('NAMESPACE')
EMBED_DIM = int(os.getenv('EMBED_DIM')) if os.getenv('EMBED_DIM') else None
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

def create_corpus(corpus, folder_path):
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if len(filename.split('.')) == 2 and filename.split('.')[1] == 'json':
                file_path=folder_path+'/'+filename
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        for item in data:
                            corpus.append(item['text'])
                        
                except FileNotFoundError:
                    print(f"Error: {file_path} not found. Please ensure the file exists in the correct directory.")
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {file_path}. The file might be malformed.")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
    
    print("corpus created successfully")

def get_dense_embeddings(text, dim_size = 1024):
    payload = {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": text,
        "encoding_format": "float",
        "dimensions": dim_size
    }

    response = requests.post(SILICONFLOW_URL_EMBEDDING, json=payload, headers=headers)

    return response.json()['data'][0]['embedding']

def get_sparse_embeddings(text, bm25_model, query_type):
    if query_type == 'upsert':
        return bm25_model.encode_documents(text)
    else:
        return bm25_model.encode_queries(text)

def generate_embedding(path_files, bm25_model):
    try:
        with open(path_files, 'r') as file:
            data = json.load(file)
            dense_vectors = []
            sparse_vectors = []
            for item in data:
                # get dense embedding
                dense_item = {
                    "id": item['_id'], 
                    "values": get_dense_embeddings(item['text'], EMBED_DIM), 
                    "metadata": {key: value for key, value in item.items() if key not in {'_id'}}
                }
                dense_vectors.append(dense_item)
                # get sparse embedding
                sparse_item = {
                    "id": item['_id'], 
                    "sparse_values": get_sparse_embeddings(item['text'], bm25_model, 'upsert'), 
                    "metadata": {key: value for key, value in item.items() if key not in {'_id'}}
                }
                sparse_vectors.append(sparse_item)
            
            return dense_vectors, sparse_vectors
    except FileNotFoundError:
        print(f"Error: {path_files} not found. Please ensure the file exists in the correct directory.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path_files}. The file might be malformed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# read data
folder_path = 'data/clean'

# define bm25 model
def create_corpus_train_bm25_model(bm25):
    bm25_pralatih = 'model/bm25_params.json'
    if(os.path.exists(bm25_pralatih)):
        # load BM25 params from json
        bm25.load(bm25_pralatih)
    else:
        # create bm25 corpus for sparse vector
        bm25_corpus = []
        create_corpus(bm25_corpus, folder_path)

        # fit corpus to bm25 model
        bm25.fit(bm25_corpus)

        # store BM25 params as json
        bm25.dump("model/bm25_params.json")
    print("bm25 model successfully loaded")

if __name__ == "__main__": 
    # create corpus and train bm25 model
    bm25 = BM25Encoder(stem=False)
    create_corpus_train_bm25_model(bm25)
    print("load bm25 model done")

    # generate dense and sparse vector
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if len(filename.split('.')) == 2 and filename.split('.')[1] == 'json':
                file_path=folder_path+'/'+filename
                # insert data dense
                dense_vectors, sparse_vectors = generate_embedding(file_path, bm25_model=bm25)
                print("generate dense and sparse vector done for: ", filename)
                index_dense.upsert(
                    vectors=dense_vectors,
                    namespace=NAMESPACE
                )
                print("upsert dense vector successfully for: ", filename)
                # insert data sparse
                index_sparse.upsert(
                    namespace=NAMESPACE,
                    vectors=sparse_vectors
                )
                print("upsert sparse vector successfully for: ", filename)
