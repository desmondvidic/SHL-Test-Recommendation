import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
import json
import numpy as np
from typing import List, Optional
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on query text",
    version="1.0.0"
)

# Load SHL assessment data
def load_assessment_data(filename="transformed2.json"):
    """Lazy loader that reads data only when requested."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Function to extract specific fields when needed
def extract_assessment_fields(data):
    """Extract the necessary fields from the data."""
    shl_embeddings = [np.array(assessment["embedding"]) for assessment in data]
    shl_titles = [assessment["title"] for assessment in data]
    shl_urls = [assessment["redirect_url"] for assessment in data]
    shl_remotes = [assessment.get("remote_testing", False) for assessment in data]
    shl_adaptive_irts = [assessment.get("adaptive_IRT", False) for assessment in data]
    shl_test_types = [assessment.get("test_type", []) for assessment in data]
    shl_durations = [assessment.get("assessment length", "") for assessment in data]
    shl_descriptions = [assessment.get("description", "") for assessment in data]
    
    return shl_embeddings, shl_titles, shl_urls, shl_remotes, shl_adaptive_irts,shl_test_types, shl_durations, shl_descriptions


model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- Models ---------------- #

class Assessment(BaseModel):
    assessment: str
    link: str
    remoteTesting: str
    adaptiveIRT: str
    testTypes: Optional[List[str]]
    assessmentLength: str
    description: str

class RecommendationResponse(BaseModel):
    status: str
    message: str
    data: dict

class RecommendRequest(BaseModel):
    query: str
# ---------------- Utility Functions ---------------- #

def extract_keywords(query: str) -> set:
    return set(re.findall(r'\b\w+\b', query.lower()))

def normalize_keywords(keywords: set) -> set:
    return {word.lstrip('.') for word in keywords}

def match_title_with_keywords(keywords, title, description):
    title = title.lower()
    description = description.lower()
    for kw in keywords:
        if kw in title.split() or kw in description.split():
            return True
    return False

def get_embedding(text: str) -> np.ndarray:
    return model.encode(text, normalize_embeddings=False)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def get_significant_words(query: str) -> List[str]:
    stop_words = {'test', 'for', 'basics', 'and', 'the', 'of', 'in', 'on', 'with', 'a', 'an'}
    words = re.findall(r'\b\w+\b', query.lower())
    significant_words = [word for word in words if word not in stop_words]
    return significant_words

def recommend_assessments(query: str, k: int = 10) -> tuple:
    
    data = load_assessment_data()
    # Extract necessary fields
    shl_embeddings, shl_titles, shl_urls, shl_remotes, shl_adaptive_irts,shl_test_types, shl_durations, shl_descriptions = extract_assessment_fields(data)

    query_embedding = get_embedding(query)
    query_keywords = normalize_keywords(extract_keywords(query))
    significant_words = get_significant_words(query)

    similarities = [cosine_similarity(query_embedding, emb) for emb in shl_embeddings]
    top_indices = np.argsort(similarities)[::-1]
    recommendations = []

    for idx in top_indices:
        title = shl_titles[idx]
        description = shl_descriptions[idx]

        # Skip if no significant words match
        if not any(word in (title + " " + description).lower() for word in significant_words):
            continue

        if match_title_with_keywords(query_keywords, title, description):
            assessment_length = shl_durations[idx]
            if isinstance(assessment_length, str):
                assessment_length = assessment_length.split("=")[-1].strip()
                formatted_length = assessment_length if assessment_length else "Duration not specified"

            recommendations.append({
                "url": shl_urls[idx],
                "adaptive_support": shl_adaptive_irts[idx],
                "description": description,
                "duration": formatted_length,
                "remote_support": shl_remotes[idx],
                "test_type": shl_test_types[idx]
            })

            if len(recommendations) == k:
                break

    return recommendations, significant_words


# ---------------- API Endpoints ---------------- #

@app.get("/", tags=["Root"])
async def root():
    return JSONResponse(content={
        "status": "success",
        "message": "Welcome to the SHL Assessment Recommender API",
        "data": {}
    }, status_code=200)

@app.get("/api/v1/recommend", response_model=RecommendationResponse, tags=["Recommend"])
async def get_recommendations(
    query: str = Query(..., description="Search query for recommending assessments"),
):
    if not query.strip():
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "message": "Validation failed",
            "errors": {"query": "Query string cannot be empty"}
        })

    try:
        recommendations, sig_word = recommend_assessments(query)
        return {
            "status": "success",
            "message": "Recommendations fetched successfully",
            "data": {
                "recommendations": recommendations,
                "query": query,
                "significantWord": sig_word
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()  # Add this
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"Internal server error: {str(e)}",  # Include real error message
            "data": {}
    })



@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "healthy"}


@app.post("/recommend")
def recommend(payload: RecommendRequest):
    query = payload.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    # Your existing recommendation logic
    results = recommend_assessments(query)  # <-- Replace with your function that returns the top 1-10 assessments

    # Make sure results is a list of dicts like the example response
    return {
        "recommended_assessments": results
    }


# ---------------- Script Usage ---------------- #

def print_recommendations(query: str, k: int = 10):
    recommendations, _ = recommend_assessments(query, k)
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if os.environ.get("USE_API", "false").lower() == "true":
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        query = "Test for Java basics"
        print_recommendations(query)
