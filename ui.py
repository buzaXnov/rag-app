import streamlit as st
import requests

# Streamlit UI
st.title("A (Meta) Retrieval-Augmented Generation (RAG) System about LLMs and Deep Learning")

query = st.text_input("Enter your query:")
submit_button = st.button("Submit")

if submit_button:
    if query:
        # Call the RAG application to get the response and top-k documents
        response = requests.post("http://app:5000/query", json={"query": query})

        # response = requests.post("http://localhost:5000/query", json={"query": query})
        result = response.json()

        st.subheader("Generated Response")
        st.write(result["generated_text"])

        st.subheader("Top-K Retrieved Documents")
        for doc in result["top_k_documents"]:
            st.write(doc)
    else:
        st.write("Please enter a query.")


links = [
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
]

st.sidebar.title("Used papers/blogs")
for link in links:
    st.sidebar.markdown(f"[{link}]({link})")


# streamlit run ui.py --server.port=8501 --server.address=0.0.0.0