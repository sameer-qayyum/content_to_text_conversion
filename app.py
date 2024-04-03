import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import logging
import pdfplumber
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import torch
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import multiprocessing
from tempfile import NamedTemporaryFile
torch.set_num_threads(1)
nltk.download('punkt')
load_dotenv()

def convert_pdf_to_images(file_path, scale=300/72):

    pdf_file = pdfium.PdfDocument(file_path)

    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    final_images = []

    for i, image in zip(page_indices, renderer):

        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))

    return final_images

def extract_text_from_img(list_dict_final_images):

    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):

        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)

def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )
    output = summary_chain.run(input_documents=docs, objective=objective)
    return output

def extract_from_pdf(pdf_url):
    #1. Download the file using the API request URL
    response = requests.get(pdf_url)
    file = response.content
    # Save the PDF file to the working directory of the API
    with open('downloaded_file.pdf', 'wb') as f:
        f.write(file)
    # Clear the contents of the file first
    with open("pdf_output.txt", "w") as f:
        pass  # Opening in 'w' mode and closing it will clear the file
    #2. Convert PDF To Images
    url='downloaded_file.pdf
    images_list = convert_pdf_to_images(url)
    #3. Extract Text from Images
    text_with_pytesseract = extract_text_from_img(images_list)
    #4. Summarize text if too large
    objective='Our goal is to create a step by step, checklist type guide from the available text. All the tiny nuances of the text should be kept'
    if len(text_with_pytesseract) > 10000:
            output = summary(objective, text_with_pytesseract)
            return output
        else:
            return text_with_pytesseract



# Set this as an API endpoint via FastAPI
app = FastAPI()
@app.post("/convert") 
def convert():
    input_url = request.values.get('input_url')
    unique_id = request.values.get('unique_id')
    file_contents = None  # Initialize to handle cases where URL doesn't match any condition
    if input_url.endswith('.pdf'):
        file_contents = extract_from_pdf(input_url)
    else:
        content, error = extract_text_from_url(input_url)
        if error:
            return jsonify(error=error), 500
        file_contents = content
    if file_contents:
        file_contents_str = str(file_contents) if not isinstance(file_contents, str) else file_contents
        predicted_title = generate_title(file_contents_str)
        #predicted_title='test title'
        return jsonify(text=file_contents, unique_id=unique_id, data_source=input_url, title=predicted_title)
    else:
        return jsonify(error="Unsupported URL or no content extracted"), 400
