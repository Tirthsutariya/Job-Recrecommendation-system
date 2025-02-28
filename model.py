import os
import spacy
from huggingface_hub import snapshot_download
from PyPDF2 import PdfReader
from io import BytesIO
from docx import Document

# Hugging Face details
HUGGINGFACE_REPO = "Tirth2102/Job-Recommendation-System"  # Change to your repo

# Download model from Hugging Face Hub
def download_model_from_hub():
    model_dir = "./models/model-last"
    if not os.path.exists(model_dir):
        print("⬇️ Downloading model from Hugging Face Hub...")
        cache_dir = snapshot_download(
            repo_id=HUGGINGFACE_REPO,
            cache_dir="./models"  # Local cache directory
        )
        # Set the model path to the specific directory
        model_path = os.path.join(cache_dir, "model-last")
        print("✅ Model downloaded from Hugging Face Hub.")
    else:
        model_path = model_dir
        print("✅ Model loaded from local cache.")
    
    # Check if the model directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")
    
    # Load the spaCy model
    try:
        nlp_model = spacy.load(model_path)
        print("✅ SpaCy model loaded successfully.")
        return nlp_model
    except Exception as e:
        print(f"❌ Failed to load spaCy model: {e}")
        raise

# Load the SpaCy model from Hugging Face
nlp = download_model_from_hub()

# Extract text from PDF
def extract_text_from_pdf(pdf_data: bytes) -> str:
    pdf_reader = PdfReader(BytesIO(pdf_data))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle pages with no text
    return text

# Extract text from DOCX
def extract_text_from_docx(docx_data: bytes) -> str:
    doc = Document(BytesIO(docx_data))
    text = [para.text for para in doc.paragraphs if para.text]  # Avoid empty lines
    return "\n".join(text)

# Extract text from TXT
def extract_text_from_txt(txt_data: bytes) -> str:
    return txt_data.decode('utf-8')

# Process text with SpaCy model
def process_text_with_spacy(text: str):
    doc = nlp(text)
    named_entities = []
    for ent in doc.ents:
        named_entities.append({
            "text": ent.text,
            "label": ent.label_
        })
    return named_entities
