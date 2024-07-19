from langchain_core.documents import Document

def doc_to_dict(doc: Document):
    """
    Objects of type Document are non serializable so this utils function was made
    that turns a Document object into a dictionary.
    """
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def format_docs(docs: Document):
    """
    Funciton used in the chain during invocation that takes the output of the
    retriever and formats it accordingly to insert into the final prompt. 
    """
    return "\n\n".join(doc.page_content for doc in docs)