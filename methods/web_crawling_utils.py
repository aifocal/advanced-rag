import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_website_links(base_url: str) -> list[str]:
    """
    Returns a collection of urls given a url as input. Performs better if a
    higher directory or homepage level URL is input into it.
    >>> get_website_links("https://www.google.com")
    [url1, url2, ...]
    """
    response = requests.get(base_url)
    urls = set()
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        anchor_tags = soup.find_all('a', href=True)
        for tag in anchor_tags:
            if tag:
                url = urljoin(base_url, tag['href'])
                if url.startswith(base_url):
                    urls.add(url)
    else:
        print(f"Error: {response.status_code}")
    print(f"{len(urls)} urls found within the provided url.")
    return urls


def get_website_links_dynamic(url, max_depth=20):
    """
    Returns a collection of urls given a url as input. Performs better if a
    higher directory or homepage level URL is input into it. Works well with
    JavaScript rendered webpages.
    >>> get_website_links_dynamic("https://www.google.com")
    [url1, url2, ...]
    """
    domain = url.replace("https://", "")
    domain = domain[:domain.index("/")]
    options = Options()
    options.add_argument("--headless=new")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    wait = WebDriverWait(driver, 3)
    elements = wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, 'a')))
    urls = []
    for element in elements:
        if len(urls) > max_depth:
            break
        href = element.get_attribute('href')
        if href and domain in href:
            urls.append(href)
    print(f"{len(urls)} urls found within the provided url.")
    return urls


if __name__ == "__main__":
    url = "https://www.pizzapizza.ca/store/1/delivery"
    # url = "https://www.pizzahut.ca/"
    # url = "https://www.crowe.com/sg"
    # urls = get_website_links(url)
    urls = get_website_links_dynamic(url)
    print(urls)
    pass