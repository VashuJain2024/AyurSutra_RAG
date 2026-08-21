from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


# -------------------------
# CONFIG
# -------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"


# -------------------------
# CHECK API KEY (Moved to usage to prevent import-time crashes)
# -------------------------
def validate_config():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("❌ GROQ_API_KEY not found in environment variables")


# -------------------------
# GET EMBEDDINGS
# -------------------------
def get_embeddings():
    validate_config()
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# -------------------------
# LOAD VECTOR DATABASE
# -------------------------
def load_db(path):

    embeddings = get_embeddings()

    db = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


# -------------------------
# GENERATE ANSWER
# -------------------------
def generate_answer(query, db):

    retriever = db.as_retriever(search_kwargs={"k": 8})

    docs = retriever.invoke(query)

    # -------- Handle no results --------
    if not docs:
        return "No relevant information found in knowledge base.", []

    context = "\n\n".join(d.page_content for d in docs)

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.2
    )

    prompt = f"""
You are an expert Ayurveda physician specializing in Panchakarma.

Base your answer on the provided context. The context comes from classical
Ayurveda texts and may use Sanskrit terminology for symptoms and conditions
(e.g. Jwara = fever, Kasa = cough, Atisara = diarrhea) — connect the patient's
plain-language symptoms to these terms where relevant.
Do NOT invent treatments that are not supported by the context.

Only say "Insufficient evidence from sources." if the context has nothing
reasonably relevant to the patient's case, even after considering related or
partially-matching passages. Prefer giving a best-effort answer with a lower
confidence level over refusing outright.

Context:
{context}

Patient Case:
{query}

Provide your answer in clear sections:

1. Recommended Panchakarma therapy
2. Clinical rationale
3. Contraindications
4. Supporting evidence from context
5. Confidence level (High / Medium / Low)
"""

    response = llm.invoke(prompt)
    return response.content, docs

# -------------------------
# ANSWER GENERAL QUERY
# -------------------------
def answer_general_query(query, db):
    # Step 1: Retrieve top-k relevant documents
    docs = db.similarity_search(query, k=8)

    if not docs:
        return "I don't know based on the available Ayurvedic texts.", []

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1
    )

    # Step 2: Grounded prompt with Sanskrit-terminology awareness
    prompt = f"""
You are an expert Ayurveda assistant specialized in Panchakarma.

Answer the user's question using the provided context. The context comes
from classical Ayurveda texts and may use Sanskrit terminology for symptoms
and conditions (e.g. Jwara = fever, Kasa = cough, Atisara = diarrhea) —
connect the question to these terms where relevant.
Do NOT use external knowledge beyond connecting such terminology.
Do NOT guess or hallucinate treatments not supported by the context.

Only say "I don't know based on the available Ayurvedic texts." if the
context has nothing reasonably relevant, even after considering related or
partially-matching passages. Prefer a best-effort answer over refusing.

Context:
{context}

Question:
{query}

Instructions:
- Give a clear, practical answer.
- If applicable, mention precautions.
- Keep it concise and medically responsible.
- Do NOT fabricate details.

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs