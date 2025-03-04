from fastapi import FastAPI, HTTPException, Query,BackgroundTasks
from bson import ObjectId
from dbconnection import get_db
from model import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, process_text_with_spacy
from recomandation import query_with_metadata
import requests
from io import BytesIO

app = FastAPI()

# Fetch document from MongoDB
def fetch_document(user_id: str):
    db = get_db()
    doc_record = db["resumes"].find_one({"user_id": ObjectId(user_id)})
    if not doc_record:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc_record

# Extract skills from resume
def extract_skills_from_resume(file_url: str, filename: str):
    response = requests.get(file_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="File not found on server")

    file_extension = filename.split('.')[-1].lower()
    extracted_text = ""

    # Read the file data in-memory
    file_data = BytesIO(response.content).read()

    # Extract text based on file type
    if file_extension == "pdf":
        extracted_text = extract_text_from_pdf(file_data)
    elif file_extension == "docx":
        extracted_text = extract_text_from_docx(file_data)
    elif file_extension == "txt":
        extracted_text = extract_text_from_txt(file_data)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    named_entities = process_text_with_spacy(extracted_text)
    extracted_skills = [ent['text'] for ent in named_entities if ent['label'] == 'SKILLS' or ent['label'] == 'LOCATION']

    print(extracted_skills)
    return extracted_skills

# Get job recommendations with pagination
from pydantic import BaseModel


class DataSchema(BaseModel):
    user_id: str
    resume_id:str
    # page: int
    # limit: int
    
@app.post("/recommendations")
def get_recommendations(
    # doc_id: str, 
    # page: int = Query(1, gt=0, description="Page number (starting from 1)"),
    # limit: int = Query(10, gt=0, le=50, description="Number of jobs per page (default 10, max 50)")
    resume_id:str,
    user_id:str,
    # page:int,
    # limit:int,
    background_tasks: BackgroundTasks
    

):
    # user_id=request.user_id
    # resume_id=request.resume_id
    # page=request.page
    # limit=request.limit
    """
    Get job recommendations for a resume.
    - **doc_id**: Document ID in MongoDB
    - **page**: Page number (starting from 1)
    - **limit**: Number of jobs per page (default 10, max 50)
    """
    # Fetch document from DB
    doc_record = fetch_document(user_id)
    file_url = doc_record.get('file_url')

    # Validate file_url
    if not file_url:
        raise HTTPException(status_code=400, detail="File URL is missing in the document record")

    # Extract filename from URL
    filename = file_url.split("/")[-1]  # Get the last part of the URL
    if '.' not in filename:
        raise HTTPException(status_code=400, detail="Filename could not be determined from the URL")

    # Extract skills from the resume
    extracted_skills = extract_skills_from_resume(file_url, filename)
    
    # Get job recommendations with pagination
    recommendations = query_with_metadata(extracted_skills, page=1, limit=10)
    background_tasks.add_task(save_matches_in_background, user_id, resume_id, recommendations)
    return {"status": 200,"resume_id":resume_id,"user_id":user_id, "recommendations": recommendations}

# Save match data in MongoDB
def save_match(user_id: str, resume_id: str, job_id: str, score: float):
    db = get_db()
    match_data = {
        "user_id": ObjectId(user_id),
        "resume_id": resume_id,
        "job_id": job_id,
        "score": score
    }
    db["matches"].insert_one(match_data)

def save_matches_in_background(user_id, resume_id, recommendations):
    """
    Save matches to the database in the background.
    """
    for recommendation in recommendations:
        job_id = recommendation.get('job_id')
        score = recommendation.get('score', 0.0)  # Default score to 0 if not present
        save_match(user_id, resume_id, job_id, score)


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
