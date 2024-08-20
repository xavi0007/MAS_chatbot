from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


reader = SimpleDirectoryReader(
    input_files=["/home/xavier002/Private_LLM/data/paul_graham_essay.txt"]
)
docs = reader.load_data()
print(f"Loaded {len(docs)} docs")
# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    docs,
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)