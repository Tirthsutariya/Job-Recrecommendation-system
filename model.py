import os
import spacy
from huggingface_hub import hf_hub_download  # Import for Hugging Face Hub
from PyPDF2 import PdfReader
from io import BytesIO
from docx import Document

# Hugging Face details
HUGGINGFACE_REPO = "Tirth2102/Job-Recommendation-System"  # Change to your repo

# Download model from Hugging Face Hub
def download_model_from_hub():
    model_dir = "./models/model-last"
    if not os.path.exists(model_dir):
        model_path = hf_hub_download(
            repo_id=HUGGINGFACE_REPO,
            filename="model-last",  # Folder name on Hugging Face
            cache_dir="./models"   # Local cache directory
        )
    else:
        model_path = model_dir

    # Load the spaCy model
    return spacy.load(model_path)

# Load the SpaCy model from Hugging Face
nlp = download_model_from_hub()

def extract_text_from_pdf(pdf_data: bytes) -> str:
    pdf_reader = PdfReader(BytesIO(pdf_data))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
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
    return named_entities
