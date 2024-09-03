from ..PrivateLLM.ragLLM import RAG

rag_test = RAG()
rag_test.read_docs(['/home/xavier002/Private_LLM/data/activelearning.pdf'])
rag_test.initiate_models()
rag_test.rag_query('what is the Active Inquiry Module (AIM)')