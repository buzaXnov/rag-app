from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List

class Retriever:
    def __init__(self, vector_store: FAISS, k_val: int = 2):
        self.retriever= vector_store.as_retriever(search_kwargs={"k": k_val})

    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Function that takes a user query and returns the most similar documents based on similarity search.
        """
        retrieved_docs = self.retriever.invoke(query)
        return retrieved_docs
