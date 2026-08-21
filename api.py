from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import load_db, generate_answer, answer_general_query
import os
from dotenv import load_dotenv
import traceback
import asyncio

# -------------------------
# LOAD ENV VARIABLES
# -------------------------
load_dotenv()

# -------------------------
# INITIALIZE APP
# -------------------------
app = FastAPI(title="AyurSutra RAG API")


# -------------------------
# CONSTANTS
# -------------------------
DB_PATH = "vector_db/classical_db"

# -------------------------
# GLOBAL DB VARIABLE
# -------------------------
db = None

async def load_db_background():
    global db
    loop = asyncio.get_running_loop()
    try:
        print("⏳ Loading Vector database and AI models in background...")
        # Run synchronous load_db in a thread pool to avoid blocking the event loop
        db = await loop.run_in_executor(None, load_db, DB_PATH)
        print("✅ Vector database loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load database: {e}")
        traceback.print_exc()

# -------------------------
# STARTUP EVENT
# -------------------------
@app.on_event("startup")
async def startup_event():
    if not os.path.exists(DB_PATH):
        print(f"⚠️ Vector DB not found at {DB_PATH}. Knowledge base will be unavailable.")
    else:
        print(f"📂 Found vector database at {DB_PATH}")
        # Offload heavy model loading so Uvicorn can bind to $PORT immediately
        asyncio.create_task(load_db_background())


# -------------------------
# ROOT ROUTE
# -------------------------
@app.get("/")
def root():
    return {"message": "AyurSutra API running 🚀"}


# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_loaded": db is not None,
        "env_check": os.getenv("GROQ_API_KEY") is not None
    }


# -------------------------
# REQUEST MODELS
# -------------------------
class CaseQuery(BaseModel):
    age: int
    gender: str
    prakriti: str
    symptoms: str
    history: str = ""


class GeneralQuery(BaseModel):
    query: str


# -------------------------
# ANALYZE PATIENT CASE
# -------------------------
@app.post("/analyze")
async def analyze_case(data: CaseQuery):
    if db is None:
        raise HTTPException(status_code=503, detail="Model and database are still loading. Please try again in a few seconds.")

    query_text = (
        f"Age: {data.age}\n"
        f"Gender: {data.gender}\n"
        f"Prakriti: {data.prakriti}\n"
        f"Symptoms: {data.symptoms}\n"
        f"History: {data.history}"
    )

    try:
        answer, docs = generate_answer(query_text, db)

        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "preview": doc.page_content[:200]
            }
            for doc in docs
        ]

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


# -------------------------
# GENERAL QUERY ENDPOINT
# -------------------------
@app.post("/query")
async def general_query(data: GeneralQuery):
    if db is None:
        raise HTTPException(status_code=503, detail="Model and database are still loading. Please try again in a few seconds.")

    try:
        answer, docs = answer_general_query(data.query, db)

        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "preview": doc.page_content[:200]
            }
            for doc in docs
        ]

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable provided by Render, default to 8000 for local
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
