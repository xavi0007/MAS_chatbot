from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer

"simple RAG, for fine-tuning refer ro fine_tune_rag"
class RAG:
    def __init__(self) -> None:
        self.docs = None
        self.index = None
    def read_docs(self, file_paths:list[str]):
        #read document files
        reader = SimpleDirectoryReader(
                input_files=file_paths
            )
        self.docs = reader.load_data()
        print(f"Loaded {len(self.docs)} docs")

    def initiate_models(self):
        # bge-base embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

        #Load model using Ollama
        Settings.llm = Ollama(model="llama3.1", request_timeout=360.0)

        self.index = VectorStoreIndex.from_documents(
            self.docs,
        )

    def rag_query(self, prompt:str):
        query_engine = self.index.as_query_engine()
        response = query_engine.query(prompt)
        # print(response)
        return response

