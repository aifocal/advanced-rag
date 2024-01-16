from llama_index import download_loader
from web_crawling_utils import get_website_links_dynamic
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def content_from_urls(urls: list[str]) -> str:
    """
    Initialize bs4 web-scraping client and obtain webpage content. Loads the web content
    from the list of urls provided and returns a list of those documents.
    >>> documents_from_urls([url1, url2, ...])
    "Website content aggregated from all the URLs"
    """
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    loader = BeautifulSoupWebReader()
    pages = loader.load_data(urls=urls)
    content = []
    for page in pages:
        content.extend(page.text)
    content = "".join(content)
    return content


def content_from_urls_dynamic(urls: list[str]) -> str:
    """
    Initialize selenium web-scraping client and obtain webpage content. Loads the web content
    from the list of urls provided and returns a list of those documents.
    >>> documents_from_urls([url1, url2, ...])
    "Website content aggregated from all the URLs"
    """
    options = Options()
    options.add_argument("--headless=new")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    total_content = ""
    for url in urls:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
        body_text = driver.find_element(By.TAG_NAME, 'body').text
        total_content += body_text + "\n"
    driver.quit()
    return total_content


def text_splitter(doc: str, chunk_size: int=1000, chunk_overlap: int=200) -> list[Document]:
    """
    Recursively remove escape sequence and whitespace.
    >>> text_splitter(["Website HTML page content ..."])
    [Doc1, Doc2, ...]
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )
    doc[0] = doc[0].replace("\n\n", "")
    doc[0] = doc[0].replace("\n", " ")
    docs = text_splitter.create_documents(doc)
    return docs


def retrieve_docs_from_urls(urls: list[str], chunk_size: int=1000, chunk_overlap: int=200) -> list[Document]:
    """
    Retrieves the final documents from the URLs. First the URLs give individual documents
    that are collated. Then that is cleaned and split into parent documents. That is then
    returned as a collection of documents.
    """
    documents = text_splitter([content_from_urls_dynamic(urls)], chunk_size, chunk_overlap)
    print(f"Parsed the content and generated {len(documents)} documents of chunk size: {chunk_size} and chunk overlap: {chunk_overlap}")
    return documents


if __name__ == "__main__":
    urls = [
        "https://www.pizzahut.ca/",
        "https://www.crowe.com/sg"
    ]
    # content = content_from_urls_dynamic(urls=urls)
    docs = retrieve_docs_from_urls(urls=urls)
    print(docs)
    pass