from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode,TextNode
from llama_index.finetuning import generate_qa_embedding_pairs,SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
# from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever
from copy import deepcopy
from IPython.display import display, HTML
from tqdm.notebook import tqdm
import pandas as pd
import os
from typing import List

class RAG_LLM:
    def __init__(self) -> None:
        # bge-base embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        # ollama
        Settings.llm = Ollama(model="llama3", request_timeout=360.0)

        Settings.chunk_overlap = 0
        Settings.chunk_size = 128

        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

        self.TRAIN_FILES = ["~/Private_LLM/data/10k/lyft_2021.pdf"]
        self.VAL_FILES = ["~/Private_LLM/data/10k/uber_2021.pdf"]
        self.INPUT_FILE = ["~/Private_LLM/data/paul_graham_essay.txt"]
    
        self.TRAIN_CORPUS_FPATH = "~/Private_LLM/data/10k/train_corpus.json"
        self.VAL_CORPUS_FPATH = "~/Private_LLM/data/10k/val_corpus.json"
        
        #preload
        self.RAG_index = self.read_file(self.TRAIN_FILES)
        

    def load_corpus(self, files, verbose=False):
        if verbose:
            print(f"Loading files {files}")

        reader = SimpleDirectoryReader(input_files=files)
        docs = reader.load_data()
        if verbose:
            print(f"Loaded {len(docs)} docs")

        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

        if verbose:
            print(f"Parsed {len(nodes)} nodes")

        return nodes

    def read_file(self, files: List[str]):
        reader = SimpleDirectoryReader(input_files=files)
        docs = reader.load_data()
        print(f"Loaded {len(docs)} docs")
        index = VectorStoreIndex.from_documents(
        docs,)
        return index


    def ask_RAG(self, query:str):
        query_engine = self.RAG_index.as_query_engine()
        response = query_engine.query(query)
        print(response)

    def get_retrieved_nodes(self, query_str, vector_top_k=10, reranker_top_n=3, with_reranker=False
        ):
        query_bundle = QueryBundle(query_str)
        # configure retriever
        retriever = VectorIndexRetriever(
            index=self.RAG_index,
            similarity_top_k=vector_top_k,
        )
        retrieved_nodes = retriever.retrieve(query_bundle)

        if with_reranker:
            # configure reranker
            reranker = LLMRerank(
                choice_batch_size=5,
                top_n=reranker_top_n,
            )
            retrieved_nodes = reranker.postprocess_nodes(
                retrieved_nodes, query_bundle
            )

        return retrieved_nodes


    def pretty_print(self, df):
        return display(HTML(df.to_html().replace("\\n", "<br>")))

    def visualize_retrieved_nodes(self, nodes) -> None:
        result_dicts = []
        for node in nodes:
            node = deepcopy(node)
            node.node.metadata = None
            node_text = node.node.get_text()
            node_text = node_text.replace("\n", " ")

            result_dict = {"Score": node.score, "Text": node_text}
            result_dicts.append(result_dict)

        self.pretty_print(pd.DataFrame(result_dicts))


    def generate_dataset(self):
        train_nodes = self.load_corpus(self.TRAIN_FILES, verbose=True)
        val_nodes = self.load_corpus(self.VAL_FILES, verbose=True)

        # train_dataset = generate_qa_embedding_pairs(
        #                 llm=Ollama(model="llama3", request_timeout=360.0),
        #                 nodes=train_nodes,
        #                 output_path="train_dataset.json",
        #             )
        train_dataset = None
        val_dataset = generate_qa_embedding_pairs(
                        # llm=Gemini(model_name="models/gemini-1.5-flash", api_key=self.GOOGLE_API_KEY),
                        llm = Ollama(model="llama3", request_timeout=360.0),
                        nodes=val_nodes,
                        output_path="val_dataset.json",
                    )
        return train_dataset, val_dataset
    
    def train_RAG(self, train_dataset, val_dataset) -> object:
        # train_dataset, val_dataset = self.generate_dataset()

        finetune_engine = SentenceTransformersFinetuneEngine(
            train_dataset,
            model_id="BAAI/bge-small-en",
            model_output_path="/home/xavier002/Private_LLM/models",
            val_dataset=val_dataset,
        )

        finetune_engine.finetune()
        embed_model = finetune_engine.get_finetuned_model()
        return embed_model

    
    def evaluate_RAG(self, dataset, embed_model='local:BAAI/bge-small-en', top_k=5,verbose=False):
        '''a simple hit rate metric for evaluation, retrieve top-k documents with the query,
          and it's a hit if the results contain the relevant doc.'''
        corpus = dataset.corpus
        queries = dataset.queries
        relevant_docs = dataset.relevant_docs

        nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
        index = VectorStoreIndex(
            nodes, embed_model=embed_model, show_progress=True
        )
        retriever = index.as_retriever(similarity_top_k=top_k)

        eval_results = []
        for query_id, query in queries.items():
            retrieved_nodes = retriever.retrieve(query)
            retrieved_ids = [node.node.node_id for node in retrieved_nodes]
            expected_id = relevant_docs[query_id][0]
            is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

            eval_result = {
                "is_hit": is_hit,
                "retrieved": retrieved_ids,
                "expected": expected_id,
                "query": query_id,
            }
            eval_results.append(eval_result)
        
        df_results = pd.DataFrame(eval_results)
        hit_rate = df_results["is_hit"].mean()
        print(hit_rate)
        return eval_results, hit_rate


#Unit tests functions, cannot write another py script due to file referencing errors

rag = RAG_LLM()
text_prompt="summarise this file"
rag.ask_RAG(text_prompt)

#Run to get corpus data
# train, val = rag.generate_dataset()

val_dataset = EmbeddingQAFinetuneDataset.from_json("/home/xavier002/Private_LLM/data/10k/val_dataset.json")
train_dataset = EmbeddingQAFinetuneDataset.from_json("/home/xavier002/Private_LLM/data/10k/train_dataset.json")

#before, and after
# _, hitrate_before = rag.evaluate_RAG(val_dataset)
# ft_emodel = rag.train_RAG(train_dataset, val_dataset)
# _, hitrate_after = rag.evaluate_RAG(val_dataset, ft_emodel)

#reranking
new_nodes_before = rag.get_retrieved_nodes(
    "What is Lyft's response to COVID-19?", vector_top_k=5, with_reranker=False
)
print(new_nodes_before)
# rag.visualize_retrieved_nodes(new_nodes_before)

new_nodes_after = rag.get_retrieved_nodes(
    "What is Lyft's response to COVID-19?",
    vector_top_k=20,
    reranker_top_n=5,
    with_reranker=True,
)
print(new_nodes_after)
# rag.visualize_retrieved_nodes(new_nodes_after)
