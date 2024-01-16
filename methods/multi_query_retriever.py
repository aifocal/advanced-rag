from langchain_community.vectorstores import Chroma
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain.retrievers.multi_query import MultiQueryRetriever
from web_crawling_utils import get_website_links_dynamic
from web_scraping_utils import retrieve_docs_from_urls, text_splitter
from pdf_scraping_utils import extract_text_from_pdf
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
# Max tokens is the number of tokens that the LLM input can take
MAX_TOKENS = 1000
# The chunk size and overlap created for each of the vector dbs
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# The number of URLs to crawl within home page
NUM_URLS = 4
K = 2
# Set the creativity of LLMs
RETRIEVER_LLM_TEMPERATURE = 0
QNA_LLM_TEMPERATURE = 0.1


class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


class MultiQueryChatbot:
    def __init__(self, url, pdf):
        urls = get_website_links_dynamic(url, NUM_URLS)

        web_docs = retrieve_docs_from_urls(urls=urls, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        pdf_docs = text_splitter([extract_text_from_pdf(pdf)], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        embedding = OpenAIEmbeddings()
        web_vectordb = Chroma.from_documents(web_docs, embedding=embedding)
        pdf_vectordb = Chroma.from_documents(pdf_docs, embedding=embedding)

        self.llm = ChatOpenAI(temperature=RETRIEVER_LLM_TEMPERATURE, max_tokens=MAX_TOKENS)
        self.qa_llm = ChatOpenAI(temperature=QNA_LLM_TEMPERATURE, max_tokens=MAX_TOKENS)

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate three 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )

        output_parser = LineListOutputParser()
        self.llm_chain = LLMChain(llm=self.llm, prompt=query_prompt, output_parser=output_parser)

        self.web_retriever = MultiQueryRetriever(
            retriever=web_vectordb.as_retriever(
                search_kwargs={"k": K}
            ), llm_chain=self.llm_chain, parser_key="lines"
        )
        self.pdf_retriever = MultiQueryRetriever(
            retriever=pdf_vectordb.as_retriever(
                search_kwargs={"k": K}
            ), llm_chain=self.llm_chain, parser_key="lines"
        )
        

    def get_relevant_documents(self, query):
        web_docs = self.web_retriever.get_relevant_documents(query=query)
        pdf_docs = self.pdf_retriever.get_relevant_documents(query=query)
        return web_docs, pdf_docs
    
    def generate_answer(self, question, web_context, pdf_context):
        prompt = f"""
        Read the context given here carefully: {web_context}
        Read the additional context given here as well: {pdf_context}
        Remember to keep the answer concise while also providing all the relevant information. 
        """
        input_ = [
            SystemMessage(
                content=prompt
            ),
            HumanMessage(
                content=question
            )
        ]
        answer = self.qa_llm.invoke(input_).content
        return answer
    
    def trim_context(self, context, max_tokens=MAX_TOKENS):
        tokens = context.split()
        if len(tokens) > max_tokens:
            return ' '.join(tokens[:max_tokens])
        return context

    def is_context_too_large(self, context):
        return len(context.split()) > MAX_TOKENS

    def chat(self):
        while True:
            user_input = input("Query: ")
            if user_input.lower() == 'exit':
                break

            web_docs, pdf_docs = self.get_relevant_documents(user_input)

            web_context = []
            for doc in web_docs:
                web_context.extend(doc.page_content)
            web_context = "".join(web_context)
            web_context = self.trim_context(web_context)

            pdf_context = []
            for doc in pdf_docs:
                pdf_context.extend(doc.page_content)
            pdf_context = "".join(pdf_context)
            pdf_context = self.trim_context(pdf_context)

            response = self.generate_answer(user_input, web_context, pdf_context)
            print("AI: " + response)


if __name__ == "__main__":
    url = "https://www.pizzapizza.ca/store/1/delivery"
    pdf = "data/pizzapizza.pdf"
    chatbot = MultiQueryChatbot(url, pdf)
    chatbot.chat()