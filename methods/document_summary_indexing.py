import os
import openai
import logging
from pathlib import Path
import requests
from llama_index import SimpleDirectoryReader, ServiceContext, get_response_synthesizer
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI
import sys

openai.api_key = os.environ["OPENAI_API_KEY"]
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))