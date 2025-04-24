import os
import io
import time
import uuid
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber
import spacy
import torch
import sqlite3
import uvicorn
import moviepy.editor as mp
from threading import Thread
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status, Header
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    pipeline, 
    TrainingArguments, 
    Trainer
)
from sentence_transformers import SentenceTransformer
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from dotenv import load_dotenv
# Import get_db_connection from auth
from auth import (
    User, UserCreate, Token, get_current_active_user, authenticate_user,
    create_access_token, hash_password, register_user, check_subscription_access,
    SUBSCRIPTION_TIERS, JWT_EXPIRATION_DELTA, get_db_connection, update_auth_db_schema

)
from auth import get_subscription_plans
# Add this import near the top with your other imports
from paypal_integration import (
    create_user_subscription, verify_subscription_payment, 
    update_user_subscription, handle_subscription_webhook, initialize_database
)
from fastapi import Request  # Add this if not already imported

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

# Initialize the database
# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Analysis API",
    description="API for analyzing legal documents, videos, and audio",
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://testing-hdq7qxb3k-hardikkandpals-projects.vercel.app/", "http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
initialize_database()
try:
    update_auth_db_schema()
    logger.info("Database schema updated successfully")
except Exception as e:
    logger.error(f"Database schema update error: {e}")


# Set device for model inference
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize chat history
chat_history = []

# Document context storage
document_contexts = {}

def store_document_context(task_id, text):
    """Store document text for later retrieval."""
    document_contexts[task_id] = text

def load_document_context(task_id):
    """Load document text for a given task ID."""
    return document_contexts.get(task_id, "")


load_dotenv()
DB_PATH = os.getenv("DB_PATH", "data/user_data.db")
os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)

def fine_tune_qa_model():
    """Fine-tunes a QA model on the CUAD dataset."""
    print("Loading base model for fine-tuning...")
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    
    # Load and preprocess CUAD dataset
    print("Loading CUAD dataset...")
    from datasets import load_dataset
    
    try:
        dataset = load_dataset("cuad")
    except Exception as e:
        print(f"Error loading CUAD dataset: {str(e)}")
        print("Downloading CUAD dataset from alternative source...")
        # Implement alternative dataset loading here
        return tokenizer, model
    
    print(f"Dataset loaded with {len(dataset['train'])} training examples")
    
    # Preprocess the dataset
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]
        
        inputs = tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            
            start_char = answer["answer_start"][0] if len(answer["answer_start"]) > 0 else 0
            end_char = start_char + len(answer["text"][0]) if len(answer["text"]) > 0 else 0
            
            sequence_ids = inputs.sequence_ids(i)
            
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            
            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    print("Preprocessing dataset...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    print("Splitting dataset...")
    train_dataset = processed_dataset["train"]
    val_dataset = processed_dataset["validation"]
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])

    training_args = TrainingArguments(
        output_dir="./fine_tuned_legal_qa",
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=100,
        load_best_model_at_end=True,
        report_to=[]
    )

    print("✅ Starting fine tuning on CUAD QA dataset...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("✅ Fine tuning completed. Saving model...")

    model.save_pretrained("./fine_tuned_legal_qa")
    tokenizer.save_pretrained("./fine_tuned_legal_qa")

    return tokenizer, model

#############################
#    Load NLP Models       #
#############################

# Initialize model variables
nlp = None
embedding_model = None
ner_model = None
speech_to_text = None
cuad_model = None
cuad_tokenizer = None
qa_model = None

summarizer = None

try:
    print("Loading spaCy model for NER...")
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading spaCy model: {str(e)}")
    nlp = None

# Replace this block:
# try:
#     print("Loading small summarizer model...")
#     summarizer = pipeline(
#         "summarization",
#         model="sshleifer/distilbart-cnn-12-6",
#         tokenizer="sshleifer/distilbart-cnn-12-6",
#         device= -1
#     )
#     print("✅ Summarizer loaded successfully")
# except Exception as e:
#     import traceback
#     print(f"⚠️ Error loading summarizer: {str(e)}")
#     print(traceback.format_exc())
#     summarizer = None

# With this DRY call:
summarizer = load_pipeline_model(
    task="summarization",
    model_path="sshleifer/distilbart-cnn-12-6",
    tokenizer_path="sshleifer/distilbart-cnn-12-6",
    device=-1
)

# Add similar DRY loading for other pipeline models as needed:
ner_model = load_pipeline_model(
    task="ner",
    model_path="dslim/bert-base-NER",
    tokenizer_path="dslim/bert-base-NER",
    device=-1
)

qa_model = load_pipeline_model(
    task="question-answering",
    model_path="deepset/roberta-base-squad2",
    tokenizer_path="deepset/roberta-base-squad2",
    device=-1
)

speech_to_text = load_pipeline_model(
    task="automatic-speech-recognition",
    model_path="openai/whisper-base",
    tokenizer_path="openai/whisper-base",
    device=-1
)
cuad_qa_model = load_pipeline_model(
    task="question-answering",
    model_path="nlpaueb/legal-bert-base-uncased-qa-cuad",
    tokenizer_path="nlpaueb/legal-bert-base-uncased-qa-cuad",
    device=0 if torch.cuda.is_available() else -1
)

try:
    print("Loading CUAD raw model and tokenizer for clause analysis...")
    cuad_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased-qa-cuad")
    cuad_model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased-qa-cuad").to(device)
    print("✅ CUAD raw model and tokenizer loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading CUAD raw model/tokenizer: {str(e)}")
    cuad_tokenizer = None
    cuad_model = None

# --- OPTIONAL: LOAD EMBEDDING MODEL IF NEEDED ---
try:
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading embedding model: {str(e)}")
    embedding_model = None


@app.get("/health/summarizer")
def summarizer_health():
    return {"summarizer_loaded": summarizer is not None}

def legal_chatbot(user_input, context):
    """Uses a real NLP model for legal Q&A."""
    global chat_history
    chat_history.append({"role": "user", "content": user_input})
    response = qa_model(question=user_input, context=context)["answer"]
    chat_history.append({"role": "assistant", "content": response})
    return response

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file using pdfplumber."""
    try:
        # Suppress pdfplumber warnings about CropBox
        import logging
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        
        with pdfplumber.open(pdf_file) as pdf:
            print(f"Processing PDF with {len(pdf.pages)} pages")
            text = ""
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                if (i + 1) % 10 == 0:  # Log progress every 10 pages
                    print(f"Processed {i + 1} pages...")
            
            print(f"✅ PDF text extraction complete: {len(text)} characters extracted")
        return text.strip() if text else None
    except Exception as e:
        print(f"❌ PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def process_video_to_text(video_file_path):
    """Extract audio from video and convert to text."""
    try:
        print(f"Processing video file at {video_file_path}")
        temp_audio_path = os.path.join("temp", "extracted_audio.wav")
        video = mp.VideoFileClip(video_file_path)
        video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
        print(f"Audio extracted to {temp_audio_path}")
        result = speech_to_text(temp_audio_path)
        transcript = result["text"]
        print(f"Transcription completed: {len(transcript)} characters")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return transcript
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Video processing failed: {str(e)}")

def process_audio_to_text(audio_file_path):
    """Process audio file and convert to text."""
    try:
        print(f"Processing audio file at {audio_file_path}")
        result = speech_to_text(audio_file_path)
        transcript = result["text"]
        print(f"Transcription completed: {len(transcript)} characters")
        return transcript
    except Exception as e:
        print(f"Error in audio processing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

def extract_named_entities(text):
    """Extracts named entities from legal text."""
    max_length = 10000
    entities = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i+max_length]
        doc = nlp(chunk)
        entities.extend([{"entity": ent.text, "label": ent.label_} for ent in doc.ents])
    return entities

def analyze_risk(text):
    """Analyzes legal risk in the document using keyword-based analysis."""
    risk_keywords = {
        "Liability": ["liability", "responsible", "responsibility", "legal obligation"],
        "Termination": ["termination", "breach", "contract end", "default"],
        "Indemnification": ["indemnification", "indemnify", "hold harmless", "compensate", "compensation"],
        "Payment Risk": ["payment", "terms", "reimbursement", "fee", "schedule", "invoice", "money"],
        "Insurance": ["insurance", "coverage", "policy", "claims"],
    }
    risk_scores = {category: 0 for category in risk_keywords}
    lower_text = text.lower()
    for category, keywords in risk_keywords.items():
        for keyword in keywords:
            risk_scores[category] += lower_text.count(keyword.lower())
    return risk_scores

def extract_context_for_risk_terms(text, risk_keywords, window=1):
    """
    Extracts and summarizes the context around risk terms.
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    risk_contexts = {category: [] for category in risk_keywords}
    for i, sent in enumerate(sentences):
        sent_text_lower = sent.text.lower()
        for category, details in risk_keywords.items():
            for keyword in details["keywords"]:
                if keyword.lower() in sent_text_lower:
                    start_idx = max(0, i - window)
                    end_idx = min(len(sentences), i + window + 1)
                    context_chunk = " ".join([s.text for s in sentences[start_idx:end_idx]])
                    risk_contexts[category].append(context_chunk)
    summarized_contexts = {}
    for category, contexts in risk_contexts.items():
        if contexts:
            combined_context = " ".join(contexts)
            try:
                summary_result = summarizer(combined_context, max_length=100, min_length=30, do_sample=False)
                summary = summary_result[0]['summary_text']
            except Exception as e:
                summary = "Context summarization failed."
            summarized_contexts[category] = summary
        else:
            summarized_contexts[category] = "No contextual details found."
    return summarized_contexts

def get_detailed_risk_info(text):
    """
    Returns detailed risk information by merging risk scores with descriptive details
    and contextual summaries from the document.
    """
    risk_details = {
        "Liability": {
            "description": "Liability refers to the legal responsibility for losses or damages.",
            "common_concerns": "Broad liability clauses may expose parties to unforeseen risks.",
            "recommendations": "Review and negotiate clear limits on liability.",
            "example": "E.g., 'The party shall be liable for direct damages due to negligence.'"
        },
        "Termination": {
            "description": "Termination involves conditions under which a contract can be ended.",
            "common_concerns": "Unilateral termination rights or ambiguous conditions can be risky.",
            "recommendations": "Ensure termination clauses are balanced and include notice periods.",
            "example": "E.g., 'Either party may terminate the agreement with 30 days notice.'"
        },
        "Indemnification": {
            "description": "Indemnification requires one party to compensate for losses incurred by the other.",
            "common_concerns": "Overly broad indemnification can shift significant risk.",
            "recommendations": "Negotiate clear limits and carve-outs where necessary.",
            "example": "E.g., 'The seller shall indemnify the buyer against claims from product defects.'"
        },
        "Payment Risk": {
            "description": "Payment risk pertains to terms regarding fees, schedules, and reimbursements.",
            "common_concerns": "Vague payment terms or hidden charges increase risk.",
            "recommendations": "Clarify payment conditions and include penalties for delays.",
            "example": "E.g., 'Payments must be made within 30 days, with a 2% late fee thereafter.'"
        },
        "Insurance": {
            "description": "Insurance risk covers the adequacy and scope of required coverage.",
            "common_concerns": "Insufficient insurance can leave parties exposed in unexpected events.",
            "recommendations": "Review insurance requirements to ensure they meet the risk profile.",
            "example": "E.g., 'The contractor must maintain liability insurance with at least $1M coverage.'"
        }
    }
    risk_scores = analyze_risk(text)
    risk_keywords_context = {
        "Liability": {"keywords": ["liability", "responsible", "responsibility", "legal obligation"]},
        "Termination": {"keywords": ["termination", "breach", "contract end", "default"]},
        "Indemnification": {"keywords": ["indemnification", "indemnify", "hold harmless", "compensate", "compensation"]},
        "Payment Risk": {"keywords": ["payment", "terms", "reimbursement", "fee", "schedule", "invoice", "money"]},
        "Insurance": {"keywords": ["insurance", "coverage", "policy", "claims"]}
    }
    risk_contexts = extract_context_for_risk_terms(text, risk_keywords_context, window=1)
    detailed_info = {}
    for risk_term, score in risk_scores.items():
        if score > 0:
            info = risk_details.get(risk_term, {"description": "No details available."})
            detailed_info[risk_term] = {
                "score": score,
                "description": info.get("description", ""),
                "common_concerns": info.get("common_concerns", ""),
                "recommendations": info.get("recommendations", ""),
                "example": info.get("example", ""),
                "context_summary": risk_contexts.get(risk_term, "No context available.")
            }
    return detailed_info

def analyze_contract_clauses(text):
    """Analyzes contract clauses using the fine-tuned CUAD QA model."""
    max_length = 512
    step = 256
    clauses_detected = []
    try:
        clause_types = list(cuad_model.config.id2label.values())
    except Exception as e:
        clause_types = [
            "Obligations of Seller", "Governing Law", "Termination", "Indemnification",
            "Confidentiality", "Insurance", "Non-Compete", "Change of Control",
            "Assignment", "Warranty", "Limitation of Liability", "Arbitration",
            "IP Rights", "Force Majeure", "Revenue/Profit Sharing", "Audit Rights"
        ]
    chunks = [text[i:i+max_length] for i in range(0, len(text), step) if i+step < len(text)]
    for chunk in chunks:
        inputs = cuad_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = cuad_model(**inputs)
        predictions = torch.sigmoid(outputs.start_logits).cpu().numpy()[0]
        for idx, confidence in enumerate(predictions):
            if confidence > 0.5 and idx < len(clause_types):
                clauses_detected.append({"type": clause_types[idx], "confidence": float(confidence)})
    aggregated_clauses = {}
    for clause in clauses_detected:
        clause_type = clause["type"]
        if clause_type not in aggregated_clauses or clause["confidence"] > aggregated_clauses[clause_type]["confidence"]:
            aggregated_clauses[clause_type] = clause
    return list(aggregated_clauses.values())

def summarize_text(text):
    """Summarizes legal text using the summarizer model."""
    try:
        print(f"Summarizer at call time: {summarizer}")
        if summarizer is None:
            return "Basic analysis (NLP models not available)"
        
        # Split text into chunks if it's too long
        max_chunk_size = 1024
        if len(text) > max_chunk_size:
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return " ".join(summaries)
        else:
            summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        return "Summarization failed. Please try again later."

@app.post("/analyze_legal_document")
async def analyze_legal_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Analyzes a legal document (PDF) and returns insights based on subscription tier."""
    try:
        # Calculate file size in MB
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        # Check subscription access for document analysis
        check_subscription_access(current_user, "basic_document_analysis", file_size_mb)
        
        print(f"Processing file: {file.filename}")
        
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        # Extract text from PDF
        text = extract_text_from_pdf(tmp_path)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Generate a task ID
        task_id = str(uuid.uuid4())
        
        # Store document context for later retrieval
        store_document_context(task_id, text)
        
        # Basic analysis available to all tiers
        summary = summarize_text(text)
        entities = extract_named_entities(text)
        risk_scores = analyze_risk(text)
        
        # Prepare response based on subscription tier
        response = {
            "task_id": task_id,
            "summary": summary,
            "entities": entities,
            "risk_assessment": risk_scores,
            "subscription_tier": current_user.subscription_tier
        }
        
        # Add premium features if user has access
        if current_user.subscription_tier == "premium_tier":  
            # Add detailed risk assessment
            if "detailed_risk_assessment" in SUBSCRIPTION_TIERS[current_user.subscription_tier]["features"]:
                detailed_risk = get_detailed_risk_info(text)
                response["detailed_risk_assessment"] = detailed_risk
            
            # Add contract clause analysis
            if "contract_clause_analysis" in SUBSCRIPTION_TIERS[current_user.subscription_tier]["features"]:
                clauses = analyze_contract_clauses(text)
                response["contract_clauses"] = clauses
        
        return response

    except HTTPException as e:
        # Let FastAPI handle HTTPException (like 403)
        raise
    except Exception as e:
        print(f"Error analyzing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")

@app.post("/analyze_legal_video")
async def analyze_legal_video(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Analyzes legal video by transcribing and analyzing the transcript."""
    try:
        # Calculate file size in MB
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        # Check subscription access for video analysis
        check_subscription_access(current_user, "video_analysis", file_size_mb)
        
        print(f"Processing video file: {file.filename}")
        
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        # Process video to extract transcript
        transcript = process_video_to_text(tmp_path)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        if not transcript:
            raise HTTPException(status_code=400, detail="Could not extract transcript from video")
        
        # Generate a task ID
        task_id = str(uuid.uuid4())
        
        # Store document context for later retrieval
        store_document_context(task_id, transcript)
        
        # Basic analysis
        summary = summarize_text(transcript)
        entities = extract_named_entities(transcript)
        risk_scores = analyze_risk(transcript)
        
        # Prepare response
        response = {
            "task_id": task_id,
            "transcript": transcript,
            "summary": summary,
            "entities": entities,
            "risk_assessment": risk_scores,
            "subscription_tier": current_user.subscription_tier
        }
        
        # Add premium features if user has access
        if current_user.subscription_tier == "premium_tier":
            # Add detailed risk assessment
            if "detailed_risk_assessment" in SUBSCRIPTION_TIERS[current_user.subscription_tier]["features"]:
                detailed_risk = get_detailed_risk_info(transcript)
                response["detailed_risk_assessment"] = detailed_risk
        
        return response
    
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")


@app.post("/legal_chatbot/{task_id}")
async def chat_with_document(
    task_id: str,
    question: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """Chat with a document using the legal chatbot."""
    try:
        # Check if user has access to chatbot feature
        if "chatbot" not in SUBSCRIPTION_TIERS[current_user.subscription_tier]["features"]:
            raise HTTPException(
                status_code=403,
                detail=f"The chatbot feature is not available in your {current_user.subscription_tier} subscription. Please upgrade to access this feature."
            )
        
        # Check if document context exists
        context = load_document_context(task_id)
        if not context:
            raise HTTPException(status_code=404, detail="Document context not found. Please analyze a document first.")
        
        # Use the chatbot to answer the question
        answer = legal_chatbot(question, context)
        
        return {"answer": answer, "chat_history": chat_history}
    
    except Exception as e:
        print(f"Error in chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in chatbot: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Legal Document Analysis API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #2c3e50;
                }
                .endpoint {
                    background-color: #f8f9fa;
                    padding: 15px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                }
                .method {
                    font-weight: bold;
                    color: #e74c3c;
                }
            </style>
        </head>
        <body>
            <h1>Legal Document Analysis API</h1>
            <p>Welcome to the Legal Document Analysis API. This API provides tools for analyzing legal documents, videos, and audio.</p>
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <p><span class="method">POST</span> /analyze_legal_document - Analyze a legal document (PDF)</p>
            </div>
            <div class="endpoint">
                <p><span class="method">POST</span> /analyze_legal_video - Analyze a legal video</p>
            </div>
            <div class="endpoint">
                <p><span class="method">POST</span> /analyze_legal_audio - Analyze legal audio</p>
            </div>
            <div class="endpoint">
                <p><span class="method">POST</span> /legal_chatbot/{task_id} - Chat with a document</p>
            </div>
            <div class="endpoint">
                <p><span class="method">POST</span> /register - Register a new user</p>
            </div>
            <div class="endpoint">
                <p><span class="method">POST</span> /token - Login to get an access token</p>
            </div>
            <div class="endpoint">
                <p><span class="method">GET</span> /users/me - Get current user information</p>
            </div>
            <div class="endpoint">
                <p><span class="method">POST</span> /subscribe/{tier} - Subscribe to a plan</p>
            </div>
            <p>For more details, visit the <a href="/docs">API documentation</a>.</p>
        </body>
    </html>
    """)

@app.post("/register", response_model=Token)
async def register_new_user(user_data: UserCreate):
    """Register a new user with a free subscription"""
    try:
        success, result = register_user(user_data.email, user_data.password)
        
        if not success:
            raise HTTPException(status_code=400, detail=result)
            
        return {"access_token": result["access_token"], "token_type": "bearer"}
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Endpoint for OAuth2 token generation"""
    try:
        # Add debug logging
        logger.info(f"Token request for username: {form_data.username}")
        
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            logger.warning(f"Authentication failed for: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token = create_access_token(user.id)
        if not access_token:
            logger.error(f"Failed to create access token for user: {user.id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token",
            )
        
        logger.info(f"Login successful for: {form_data.username}")
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Token endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login error: {str(e)}",
        )


@app.get("/debug/token")
async def debug_token(authorization: str = Header(None)):
    """Debug endpoint to check token validity"""
    try:
        if not authorization:
            return {"valid": False, "error": "No authorization header provided"}
        
        # Extract token from Authorization header
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            return {"valid": False, "error": "Not a bearer token"}
        
        # Log the token for debugging
        logger.info(f"Debugging token: {token[:10]}...")
        
        # Try to validate the token
        try:
            user = await get_current_active_user(token)
            return {"valid": True, "user_id": user.id, "email": user.email}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    except Exception as e:
        return {"valid": False, "error": f"Token debug error: {str(e)}"}


@app.post("/login")
async def api_login(email: str, password: str):
    success, result = login_user(email, password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result
        )
    return result

@app.get("/health")
def health_check():
    """Simple health check endpoint to verify the API is running"""
    return {"status": "ok", "message": "API is running"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.post("/analyze_legal_audio")
async def analyze_legal_audio(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Analyzes legal audio by transcribing and analyzing the transcript."""
    try:
        # Calculate file size in MB
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        # Check subscription access for audio analysis
        check_subscription_access(current_user, "audio_analysis", file_size_mb)
        
        print(f"Processing audio file: {file.filename}")
        
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        # Process audio to extract transcript
        transcript = process_audio_to_text(tmp_path)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        if not transcript:
            raise HTTPException(status_code=400, detail="Could not extract transcript from audio")
        
        # Generate a task ID
        task_id = str(uuid.uuid4())
        
        # Store document context for later retrieval
        store_document_context(task_id, transcript)
        
        # Basic analysis
        summary = summarize_text(transcript)
        entities = extract_named_entities(transcript)
        risk_scores = analyze_risk(transcript)
        
        # Prepare response
        response = {
            "task_id": task_id,
            "transcript": transcript,
            "summary": summary,
            "entities": entities,
            "risk_assessment": risk_scores,
            "subscription_tier": current_user.subscription_tier
        }
        
        # Add premium features if user has access
        if current_user.subscription_tier == "premium_tier":  # Change from premium_tier to premium
            # Add detailed risk assessment
            if "detailed_risk_assessment" in SUBSCRIPTION_TIERS[current_user.subscription_tier]["features"]:
                detailed_risk = get_detailed_risk_info(transcript)
                response["detailed_risk_assessment"] = detailed_risk
        
        return response
    
    except Exception as e:
        print(f"Error analyzing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing audio: {str(e)}")



# Add these new endpoints before the if __name__ == "__main__" line
@app.get("/users/me/subscription")
async def get_user_subscription(current_user: User = Depends(get_current_active_user)):
    """Get the current user's subscription details"""
    try:
        # Get subscription details from database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the most recent active subscription
        try:
            cursor.execute(
                "SELECT id, tier, status, created_at, expires_at, paypal_subscription_id FROM subscriptions "
                "WHERE user_id = ? AND status = 'active' ORDER BY created_at DESC LIMIT 1",
                (current_user.id,)
            )
            subscription = cursor.fetchone()
        except sqlite3.OperationalError as e:
            # Handle missing tier column
            if "no such column: tier" in str(e):
                logger.warning("Subscriptions table missing 'tier' column. Returning default subscription.")
                subscription = None
            else:
                raise
        
        # Get subscription tiers with pricing directly from SUBSCRIPTION_TIERS
        subscription_tiers = {
            "free_tier": {
                "price": SUBSCRIPTION_TIERS["free_tier"]["price"],
                "currency": SUBSCRIPTION_TIERS["free_tier"]["currency"],
                "features": SUBSCRIPTION_TIERS["free_tier"]["features"]
            },
            "standard_tier": {
                "price": SUBSCRIPTION_TIERS["standard_tier"]["price"],
                "currency": SUBSCRIPTION_TIERS["standard_tier"]["currency"],
                "features": SUBSCRIPTION_TIERS["standard_tier"]["features"]
            },
            "premium_tier": {
                "price": SUBSCRIPTION_TIERS["premium_tier"]["price"],
                "currency": SUBSCRIPTION_TIERS["premium_tier"]["currency"],
                "features": SUBSCRIPTION_TIERS["premium_tier"]["features"]
            }
        }
        
        if subscription:
            sub_id, tier, status, created_at, expires_at, paypal_id = subscription
            result = {
                "id": sub_id,
                "tier": tier,
                "status": status,
                "created_at": created_at,
                "expires_at": expires_at,
                "paypal_subscription_id": paypal_id,
                "current_tier": current_user.subscription_tier,
                "subscription_tiers": subscription_tiers
            }
        else:
            result = {
                "tier": "free_tier",
                "status": "active",
                "current_tier": current_user.subscription_tier,
                "subscription_tiers": subscription_tiers
            }
        
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error getting subscription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting subscription: {str(e)}")
# Add this model definition before your endpoints
class SubscriptionCreate(BaseModel):
    tier: str

@app.post("/create_subscription")
async def create_subscription(
    subscription: SubscriptionCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a subscription for the current user"""
    try:
        # Log the request for debugging
        logger.info(f"Creating subscription for user {current_user.email} with tier {subscription.tier}")
        logger.info(f"Available tiers: {list(SUBSCRIPTION_TIERS.keys())}")
        
        # Validate tier
        valid_tiers = ["standard_tier", "premium_tier"]
        if subscription.tier not in valid_tiers:
            logger.warning(f"Invalid tier requested: {subscription.tier}")
            raise HTTPException(status_code=400, detail=f"Invalid tier: {subscription.tier}. Must be one of {valid_tiers}")
        
        # Create subscription
        logger.info(f"Calling create_user_subscription with email: {current_user.email}, tier: {subscription.tier}")
        success, result = create_user_subscription(current_user.email, subscription.tier)
        
        if not success:
            logger.error(f"Failed to create subscription: {result}")
            raise HTTPException(status_code=400, detail=result)
        
        logger.info(f"Subscription created successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error creating subscription: {str(e)}")
        # Include the full traceback for better debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error creating subscription: {str(e)}")

@app.post("/subscribe/{tier}")
async def subscribe_to_tier(
    tier: str,
    current_user: User = Depends(get_current_active_user)
):
    """Subscribe to a specific tier"""
    try:
        # Validate tier
        valid_tiers = ["standard_tier", "premium_tier"]
        if tier not in valid_tiers:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {tier}. Must be one of {valid_tiers}")
        
        # Create subscription
        success, result = create_user_subscription(current_user.email, tier)
        
        if not success:
            raise HTTPException(status_code=400, detail=result)
        
        return result
    except Exception as e:
        logger.error(f"Error creating subscription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating subscription: {str(e)}")

@app.post("/subscription/create")
async def create_subscription(request: Request, current_user: User = Depends(get_current_active_user)):
    """Create a subscription for the current user"""
    try:
        data = await request.json()
        tier = data.get("tier")
        
        if not tier:
            return JSONResponse(
                status_code=400,
                content={"detail": "Tier is required"}
            )
        
        # Log the request for debugging
        logger.info(f"Creating subscription for user {current_user.email} with tier {tier}")
        
        # Create the subscription using the imported function directly
        success, result = create_user_subscription(current_user.email, tier)
        
        if success:
            # Make sure we're returning the approval_url in the response
            logger.info(f"Subscription created successfully: {result}")
            logger.info(f"Approval URL: {result.get('approval_url')}")
            
            return {
                "success": True,
                "data": {
                    "approval_url": result["approval_url"],
                    "subscription_id": result["subscription_id"],
                    "tier": result["tier"]
                }
            }
        else:
            logger.error(f"Failed to create subscription: {result}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "detail": result}
            )
    except Exception as e:
        logger.error(f"Error creating subscription: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Error creating subscription: {str(e)}"}
        )

@app.post("/admin/initialize-paypal-plans")
async def initialize_paypal_plans(request: Request):
    """Initialize PayPal subscription plans"""
    try:
        # This should be protected with admin authentication in production
        plans = initialize_subscription_plans()
        
        if plans:
            return JSONResponse(
                status_code=200,
                content={"success": True, "plans": plans}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "detail": "Failed to initialize plans"}
            )
    except Exception as e:
        logger.error(f"Error initializing PayPal plans: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Error initializing plans: {str(e)}"}
        )


@app.post("/subscription/verify")
async def verify_subscription(request: Request, current_user: User = Depends(get_current_active_user)):
    """Verify a subscription after payment"""
    try:
        data = await request.json()
        subscription_id = data.get("subscription_id")
        
        if not subscription_id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "detail": "Subscription ID is required"}
            )
        
        logger.info(f"Verifying subscription: {subscription_id}")
        
        # Verify the subscription with PayPal
        success, result = verify_paypal_subscription(subscription_id)
        
        if not success:
            logger.error(f"Subscription verification failed: {result}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "detail": str(result)}
            )
        
        # Update the user's subscription in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the subscription details
        cursor.execute(
            "SELECT tier FROM subscriptions WHERE paypal_subscription_id = ?", 
            (subscription_id,)
        )
        subscription = cursor.fetchone()
        
        if not subscription:
            # This is a new subscription, get the tier from the PayPal response
            tier = "standard_tier"  # Default to standard tier
            # You could extract the tier from the PayPal plan ID if needed
            
            # Create a new subscription record
            sub_id = str(uuid.uuid4())
            start_date = datetime.now()
            expires_at = start_date + timedelta(days=30)
            
            cursor.execute(
                "INSERT INTO subscriptions (id, user_id, tier, status, created_at, expires_at, paypal_subscription_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sub_id, current_user.id, tier, "active", start_date, expires_at, subscription_id)
            )
        else:
            # Update existing subscription
            tier = subscription[0]
            cursor.execute(
                "UPDATE subscriptions SET status = 'active' WHERE paypal_subscription_id = ?",
                (subscription_id,)
            )
        
        # Update user's subscription tier
        cursor.execute(
            "UPDATE users SET subscription_tier = ? WHERE id = ?",
            (tier, current_user.id)
        )
        
        conn.commit()
        conn.close()
        
        return JSONResponse(
            status_code=200,
            content={"success": True, "detail": "Subscription verified successfully"}
        )
        
    except Exception as e:
        logger.error(f"Error verifying subscription: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Error verifying subscription: {str(e)}"}
        )

@app.post("/subscription/webhook")
async def subscription_webhook(request: Request):
    """Handle PayPal subscription webhooks"""
    try:
        payload = await request.json()
        success, result = handle_subscription_webhook(payload)
        
        if not success:
            logger.error(f"Webhook processing failed: {result}")
            return {"status": "error", "message": result}
        
        return {"status": "success", "message": result}
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return {"status": "error", "message": f"Error processing webhook: {str(e)}"}

@app.get("/subscription/verify/{subscription_id}")
async def verify_subscription(
    subscription_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Verify a subscription payment and update user tier"""
    try:
        # Verify the subscription
        success, result = verify_subscription_payment(subscription_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Subscription verification failed: {result}")
        
        # Get the plan ID from the subscription to determine tier
        plan_id = result.get("plan_id", "")
        
        # Connect to DB to get the tier for this plan
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT tier FROM paypal_plans WHERE plan_id = ?", (plan_id,))
        tier_result = cursor.fetchone()
        conn.close()
        
        if not tier_result:
            raise HTTPException(status_code=400, detail="Could not determine subscription tier")
        
        tier = tier_result[0]
        
        # Update the user's subscription
        success, update_result = update_user_subscription(current_user.email, subscription_id, tier)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to update subscription: {update_result}")
        
        return {
            "message": f"Successfully subscribed to {tier} tier",
            "subscription_id": subscription_id,
            "status": result.get("status", ""),
            "next_billing_time": result.get("billing_info", {}).get("next_billing_time", "")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Subscription verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Subscription verification failed: {str(e)}")

@app.post("/webhook/paypal")
async def paypal_webhook(request: Request):
    """Handle PayPal subscription webhooks"""
    try:
        payload = await request.json()
        logger.info(f"Received PayPal webhook: {payload.get('event_type', 'unknown event')}")
        
        # Process the webhook
        result = handle_subscription_webhook(payload)
        
        return {"status": "success", "message": "Webhook processed"}
    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}")
        # Return 200 even on error to acknowledge receipt to PayPal
        return {"status": "error", "message": str(e)}

# Add this to your startup code
@app.on_event("startup")
async def startup_event():
    """Initialize subscription plans on startup"""
    try:
        # Initialize PayPal subscription plans if needed
        # If you have an initialize_subscription_plans function in your paypal_integration.py,
        # you can call it here
        print("Application started successfully")
    except Exception as e:
        print(f"Error during startup: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8500)

