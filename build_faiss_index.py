import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamafileEmbeddings

# Load, chunk and index the contents of the blog.
web_paths = (
    "https://lilianweng.github.io/posts/2020-10-29-odqa/",
    "https://lilianweng.github.io/posts/2020-08-06-nas/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/",
    "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/",
    "https://lilianweng.github.io/posts/2022-09-08-ntk/",
    "https://lilianweng.github.io/posts/2022-06-09-vlm/",
    "https://lilianweng.github.io/posts/2022-04-15-data-gen/",
    "https://lilianweng.github.io/posts/2022-02-20-active-learning/",
    "https://lilianweng.github.io/posts/2021-12-05-semi-supervised/",
    "https://lilianweng.github.io/posts/2021-09-25-train-large/",
    "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/",
    "https://lilianweng.github.io/posts/2021-05-31-contrastive/",
    "https://lilianweng.github.io/posts/2021-03-21-lm-toxicity/",
    "https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
)
loader = WebBaseLoader(
    web_paths=web_paths,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings_model = LlamafileEmbeddings(base_url="http://localhost:8080")
vector_store = FAISS.from_documents(
    documents=splits, embedding=embeddings_model
)
vector_store.save_local("faiss_index")