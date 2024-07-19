from langchain_community.embeddings import LlamafileEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader


class Embedder:
    def __init__(self, embedding_model_url: str):
        self.embeddings_model = LlamafileEmbeddings(base_url=embedding_model_url)

    def create_vector_store(
        self, txt_path: str, chunk_size: int = 10, chunk_overlap: int = 5
    ) -> FAISS:
        """
            Function that creates a faiss index vector store based on 
            the given path where txt files are located in a directory. 
        """

        text_loader_kwargs = {"encoding": "windows-1252"}  # autodetect_encoding': True,
        # NOTE: autoencoding could not find the encoding but a random stack-overflow search did; long live koPytok
        # Source: https://stackoverflow.com/questions/48067514/utf-8-codec-cant-decode-byte-0xa0-in-position-4276-invalid-start-byte

        loader = DirectoryLoader(
            txt_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs,
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(docs)

        self.vector_store = FAISS.from_documents(
            documents=splits, embedding=self.embeddings_model
        )  # vectorstore

        return self.vector_store

    def save_vector_store(self, save_path: str) -> None:
        """
            Saves created vector store locally. 
        """
        self.vector_store.save_local(save_path)

    def load_vector_store(self, load_path: str = "faiss_index"):
        """
            Loads an existing vector store w.r.t the used embeddeing model. 
        """
        self.vector_store = FAISS.load_local(
            load_path, self.embeddings_model, allow_dangerous_deserialization=True
        )
