import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
import torch
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm





# Load environment variables for sensitive information
MONGO_URI = os.getenv('MONGO_URI')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

INDEX_NAME = 'job-posting-embeddings2'

# Load the SentenceTransformer model and use GPU if available
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

# Fetch job descriptions from MongoDB
def fetch_job_descriptions():
    cursor = job_collection.find(
        {},
        {'_id': 1, 'job_title': 1, 'skills': 1, 'description': 1, 'location': 1}
    )
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

# Create embeddings in batches
def get_or_create_embeddings(job_descriptions, batch_size=32):
    job_texts = []
    job_ids = []
    metadata_list = []

    for job in job_descriptions:
        job_id = job['id']
        job_text = (
            f"{job['title']} {' '.join(job['skills'])} {job['description']} {job['location']}"
        )
        metadata = {
            "job_id": job_id,
            "title": job['title'],
            "skills": job['skills'],
            "description": job['description'],
            "location": job['location']
        }
        job_ids.append(job_id)
        job_texts.append(job_text)
        metadata_list.append(metadata)

    # Process in batches for better performance
    for i in tqdm(range(0, len(job_texts), batch_size), desc="Processing Embeddings"):
        batch_ids = job_ids[i:i+batch_size]
        batch_texts = job_texts[i:i+batch_size]
        batch_metadata = metadata_list[i:i+batch_size]

        # Encode the batch
        with torch.no_grad():
            embeddings = model.encode(
                batch_texts, 
                batch_size=batch_size, 
                convert_to_tensor=True
            ).cpu().numpy().tolist()

        # Batch upsert into Pinecone
        vectors = [
            (batch_ids[j], embeddings[j], batch_metadata[j]) 
            for j in range(len(batch_ids))
        ]
        index.upsert(vectors)

# Ensure all job embeddings are created
def ensure_all_embeddings():
    job_descriptions = fetch_job_descriptions()
    get_or_create_embeddings(job_descriptions)




# Query with metadata and pagination
# Query with metadata and pagination
def query_with_metadata(extracted_skills, page=1, limit=10):
    skills_text = ' '.join(extracted_skills)
    
    # Calculate offset for pagination
    offset = (page - 1) * limit
    
    # Encode skills text
    with torch.no_grad():
        skills_embedding = model.encode(skills_text, convert_to_tensor=True).cpu().numpy().tolist()
    
    # Query Pinecone
    query_result = index.query(
        vector=skills_embedding,
        top_k=limit + offset,  # Get more results than needed for pagination
        include_metadata=True
    )
    
    # Apply pagination on the result
    matches = query_result['matches'][offset:offset + limit]  # Apply offset and limit
    response = []
    
    for match in matches:
        job_data = {
            "job_id": match['id'],
            "score": round(((match['score'] + 1) / 2) * 100, 2),
            # "metadata": match['metadata']
            # "user_id": user_id
        }
        response.append(job_data)
    
    # print(json.dump(response, indent=4))
    return response

