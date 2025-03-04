import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import torch
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Load environment variables
MONGO_URI = os.getenv('MONGO_URI')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = 'job-posting-embeddings2'

# Load the SentenceTransformer model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# MongoDB Connection
client = MongoClient(MONGO_URI)
db = client['jobPortalDB']
job_collection = db['jobposts']

# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists; if not, create it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'),
    )

# Access the index
index = pc.Index(INDEX_NAME)

# Fetch only new job descriptions from MongoDB
def fetch_new_job_descriptions():
    cursor = job_collection.find({"processed": {"$ne": True}},
                                 {'_id': 1, 'job_title': 1, 'skills': 1, 'description': 1, 'location': 1})
    return [
        {
            'id': str(job['_id']),
            'title': job.get('job_title', ''),
            'skills': job.get('skills', []),
            'description': job.get('description', ''),
            'location': job.get('location', '')
        }
        for job in cursor
    ]

# Create embeddings for new jobs
def create_new_embeddings(batch_size=32):
    job_descriptions = fetch_new_job_descriptions()
    if not job_descriptions:
        print("No new job postings found. Skipping embedding creation.")
        return

    job_texts, job_ids, metadata_list = [], [], []

    for job in job_descriptions:
        job_text = f"{job['title']} {' '.join(job['skills'])} {job['description']} {job['location']}"
        metadata = {
            "job_id": job['id'],
            "title": job['title'],
            "skills": job['skills'],
            "description": job['description'],
            "location": job['location']
        }
        job_ids.append(job['id'])
        job_texts.append(job_text)
        metadata_list.append(metadata)

    # Process in batches
    for i in tqdm(range(0, len(job_texts), batch_size), desc="Creating New Embeddings"):
        batch_ids = job_ids[i:i+batch_size]
        batch_texts = job_texts[i:i+batch_size]
        batch_metadata = metadata_list[i:i+batch_size]

        # Encode the batch
        with torch.no_grad():
            embeddings = model.encode(batch_texts, batch_size=batch_size, convert_to_tensor=True).cpu().numpy().tolist()

        # Batch upsert into Pinecone
        vectors = [(batch_ids[j], embeddings[j], batch_metadata[j]) for j in range(len(batch_ids))]
        index.upsert(vectors)

    # Mark jobs as processed in MongoDB
    job_collection.update_many({"_id": {"$in": [job['_id'] for job in job_descriptions]}}, {"$set": {"processed": True}})
    print("Embedding creation complete.")

# Query Pinecone with metadata and pagination
def query_with_metadata(extracted_skills, page=1, limit=10):
    skills_text = ' '.join(extracted_skills)
    offset = (page - 1) * limit

    # Encode skills text
    with torch.no_grad():
        skills_embedding = model.encode(skills_text, convert_to_tensor=True).cpu().numpy().tolist()

    # Query Pinecone
    query_result = index.query(vector=skills_embedding, top_k=limit + offset, include_metadata=True)

    # Apply pagination
    matches = query_result['matches'][offset:offset + limit]
    response = [{"job_id": match['id'], "score": round(((match['score'] + 1) / 2) * 100, 2)} for match in matches]
    
    return response