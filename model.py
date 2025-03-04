import os
import spacy
from PyPDF2 import PdfReader
from io import BytesIO
from docx import Document

# Load the custom SpaCy model
model_path = os.path.abspath('output/model-best')
nlp = spacy.load(model_path)

def extract_text_from_pdf(pdf_data: bytes) -> str:
    pdf_reader = PdfReader(BytesIO(pdf_data))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # print(text)
    return text

def extract_text_from_docx(docx_data: bytes) -> str:
    doc = Document(BytesIO(docx_data))
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

def extract_text_from_txt(txt_data: bytes) -> str:
    return txt_data.decode('utf-8')

def process_text_with_spacy(text: str):
    doc = nlp(text)
    named_entities = []
    for ent in doc.ents:
        named_entities.append({
            "text": ent.text,
            "label": ent.label_
        })
    # print(named_entities)
    return named_entities
