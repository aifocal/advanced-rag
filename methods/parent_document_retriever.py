from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from web_scraping_utils import retrieve_docs_from_urls
from langchain.prompts import PromptTemplate
from web_crawling_utils import get_website_links_dynamic
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")


def parent_document_retriever(docs: list[Document], child_chunk_size: int=500):
    """
    Current Functionality: Creates a Child splitter defined on a particular child document
    chunk size. Creates a Chroma Vectorstore for adding the child documents. An in-memory
    store is used to store parent documents. A ParentDocumentRetriever is used to connect
    both the databases for enhancing retrieval by increasing context due to child documents.
    >>> parent_document_retriever(*args)
    retriever, store, vectorstore (objects)
    """
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)
    vectorstore = Chroma(
        collection_name="parent_documents", embedding_function=OpenAIEmbeddings()
    ) 
    store = InMemoryStore() 
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, # For child documents
        docstore=store, # For parent documents
        child_splitter=child_splitter,
    )
    retriever.add_documents(docs, ids=None)
    print("Created the parent-child docstore succesfully.")
    return retriever, store, vectorstore


def run_test_chatbot(url: str, chunk_size: int=2000, chunk_overlap: int=200, child_chunk_size: int=500) -> None:
    """
    Enhanced workflow: In addition to retrieving documents, this function also uses a RetrievalQAWithSourcesChain
    with a prompt template to maintain and utilize conversation history for better context in responses.
    """
    urls = get_website_links_dynamic(url)
    docs = retrieve_docs_from_urls(urls=urls, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever, _, vectorstore = parent_document_retriever(docs=docs, child_chunk_size=child_chunk_size)

    template = """
    Optional Context:
    {context}

    As a helpful assistant chatbot, You are here to provide detailed and accurate information based on our extensive resources.

    User Prompt:
    {question}

    If the question is beyond the scope of our current knowledge base, just apologise and say that you aren't aware of an answer.

    Remember, the goal is to be as helpful and informative as possible, while also being transparent about the limits of our current knowledge and resources.
    """

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever, 
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=template,
                input_variables=["context", "question"],
            ),
        },
    )

    conversation_history = []

    query = ""
    while query != "exit":
        query = input("Query: ")
        sub_res = vectorstore.similarity_search(query)
        conversation_history.append(query)
        # context = " ".join(conversation_history)
        prompt = template.format(context=sub_res[0].page_content, question=query)
        response = qa_chain.run(prompt)
        conversation_history.append(response)

        print(response)


if __name__ == "__main__":
    url = "https://www.crowe.com/sg"
    run_test_chatbot(url=url, chunk_size=1000, chunk_overlap=100, child_chunk_size=300)
    pass