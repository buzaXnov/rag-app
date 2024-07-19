import os

from flask import Flask, jsonify, request
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough

from config import Config
from services.embedder import Embedder
from services.generator import Generator
from services.retriever import Retriever
from utils import *

app = Flask(__name__)

embedder = Embedder(Config.EMBEDDING_MODEL_URL)
generator = Generator(Config.GENERATION_MODEL_URL)

# Build or load FAISS index
if os.path.exists(Config.VECTOR_STORE_PATH):
    embedder.load_vector_store(Config.VECTOR_STORE_PATH)
else:
    embedder.create_vector_store(
        txt_path=Config.DATA_DIR,
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    embedder.save_vector_store(Config.VECTOR_STORE_PATH)

retriever = Retriever(embedder.vector_store, k_val=2)


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data["query"]

    prompt = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=[
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["context", "question"],
                    template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:",
                )
            )
        ],
    )

    chain = (
        {
            "context": retriever.retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | generator.generator
        | StrOutputParser()
    )

    docs = retriever.retrieve_documents(user_query)
    generated_text = chain.invoke(user_query)

    return jsonify(
        {
            "generated_text": generated_text,
            "top_k_documents": [doc_to_dict(doc) for doc in docs],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
