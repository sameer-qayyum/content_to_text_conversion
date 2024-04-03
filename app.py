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
from fastapi import FastAPI, HTTPException, Request
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import torch
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
from tempfile import NamedTemporaryFile
import tempfile
torch.set_num_threads(1)
nltk.download('punkt')
load_dotenv()

# Initialize tokenizer and model for title generation
tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-small-medium-title-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-small-medium-title-generation")
max_input_length = 2048

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
    # Download the file using the API request URL
    response = requests.get(pdf_url)
    response.raise_for_status()  # This will ensure to raise an error for bad responses

    # Use a temporary file to save the PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
        tmp_pdf.write(response.content)
        tmp_pdf_path = tmp_pdf.name  # Store the path to the temporary file

    try:
        # 2. Convert PDF To Images
        images_list = convert_pdf_to_images(tmp_pdf_path)

        # 3. Extract Text from Images
        text_with_pytesseract = extract_text_from_img(images_list)

        # 4. Summarize text if too large
        objective = 'Our goal is to create a step by step, checklist type guide from the available text. All the tiny nuances of the text should be kept'
        if len(text_with_pytesseract) > 10000:
            output = summary(objective, text_with_pytesseract)
            return output
        else:
            return text_with_pytesseract
    finally:
        # Clean up the temporary file
        os.unlink(tmp_pdf_path)

def generate_title(text):
    inputs = ["summarize: " + text]
    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=6, max_length=24)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
    return predicted_title

def extract_text_from_url(url):
    try:
        if "youtube.com/watch?v=" in url or "youtu.be/" in url:
            video_id = url.split("v=")[1].split("&")[0] if "youtube.com" in url else url.split("youtu.be/")[1]
            return extract_youtube_transcript(video_id), None
        else:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return " ".join(soup.stripped_strings), None
    except Exception as e:
        return None, str(e)

def extract_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Extract the 'text' from each entry in the transcript and join them
        formatted_transcript = " ".join(entry['text'] for entry in transcript)
        objective='Our goal is to create a step by step, checklist type guide from the available text. All the tiny nuances of the text should be kept'
        if len(formatted_transcript)>10000:
            formatted_transcript = summary(objective, formatted_transcript)
        return formatted_transcript, None
    except Exception as e:
        return None, str(e)

# Set this as an API endpoint via FastAPI
app = FastAPI()
@app.post("/convert") 
async def convert(request: Request):
    form_data = await request.form()
    input_url = form_data.get('input_url')
    unique_id = form_data.get('unique_id')
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
        return {"text": file_contents, "unique_id": unique_id, "data_source": input_url, "title": predicted_title}
    else:
        raise HTTPException(status_code=400, detail="Unsupported URL or no content extracted")
