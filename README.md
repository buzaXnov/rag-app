# rag-app
A simple RAG app with a simple web UI answering questions about LLMs, Deep Learning in general and the methods scraped from an engineer's website working at OpenAI. 

https://lilianweng.github.io/


# Prerequisites

Make sure you have installed:
- Docker

that's it.

# Building the images

Navigate to the part of the repository that has the docker-compose.yaml file and run the following command:

`docker compose up --build -d`

`--build` flag build the images by using the Dockerfiles for each service in the dockerfiles folder.
`-d` or `--detach` is the flag that makes the services run in the background

### Built images:
1. Generation model: a local tiny llama model was used to ease debugging and to prevent rising (insane) costs of using third party APIs for this demo. 
2. Embedding model: an embedding model used for building a faiss index vector store, which was done during development as the embedding models that I can run locally and use take more time than the commercial ones (a lot more time, but they are free to use money-wise). 
3. UI: a streamlit container for a nice view of the generated text and relevant retrieved documents alongside the links to the scraped blog/paper.
4. App: an application container which is actually a vector database (named app as flask was used and is the norm in naming) that receives the inserted query, does document search and performs a text generation using Langchain. 


After the images have been built, check if they are running using:

`docker compose ps`

Once you have convinced yourself they are indeed running, enter the ui to start interacting with the model:

`http://0.0.0.0:8501`

# Relevant Questions to ask:

1. What Causes Hallucinations?
2. Approaches for applying Quantization? 
3. What is knowledge distillation? 
4. What are Vision Language Models? 
5. What is few-shot learning? 

And many more, this is simply for this simple RAG system which I plan to extend on in the future. 
