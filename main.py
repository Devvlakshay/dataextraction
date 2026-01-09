"""
Main FastAPI Application for Document Extraction
Integrates Aadhaar and PAN card extraction with Qwen fallback support
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
import aiohttp
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    BASE_DIR = Path(__file__).parent
    
    # Model paths for Aadhaar
    MODEL1_PATH = BASE_DIR / os.environ.get("MODEL1_PATH", "models/model1.pt")
    MODEL2_PATH = BASE_DIR / os.environ.get("MODEL2_PATH", "models/model2.pt")
    
    # Directories
    DOWNLOAD_DIR = BASE_DIR / os.environ.get("DOWNLOAD_DIR", "workspace/downloads")
    DATA_DIR = BASE_DIR / os.environ.get("DATA_DIR", "workspace/data")
    
    # Aadhaar data storage
    AADHAR_EXCEL_PATH = DATA_DIR / "aadhar_details.xlsx"
    AADHAR_PKL_PATH = DATA_DIR / "aadhar_numbers.pkl"
    
    # PAN card data storage
    PANCARD_EXCEL_PATH = DATA_DIR / "pancard_details.xlsx"
    PANCARD_PKL_PATH = DATA_DIR / "pancard_numbers.pkl"
    
    # Default confidence threshold
    DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.15))
    
    # Fallback settings
    FALLBACK_THRESHOLD_AADHAAR = int(os.environ.get("FALLBACK_THRESHOLD_AADHAAR", 2))
    FALLBACK_THRESHOLD_PANCARD = int(os.environ.get("FALLBACK_THRESHOLD_PANCARD", 1))


config = Config()

# Ensure directories exist
config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Import pipeline modules ---
# These are imported conditionally to avoid startup errors if models aren't present

pipeline = None
qwen_fallback = None

def initialize_aadhaar_pipeline():
    """Initialize the Aadhaar processing pipeline"""
    global pipeline
    try:
        from aadhar import ComprehensiveAadhaarPipeline
        pipeline = ComprehensiveAadhaarPipeline(
            model1_path=str(config.MODEL1_PATH),
            model2_path=str(config.MODEL2_PATH)
        )
        logger.info("Aadhaar pipeline initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Aadhaar pipeline: {e}")
        return False

def initialize_qwen_fallback():
    """Initialize the Qwen fallback module"""
    global qwen_fallback
    try:
        from qwen_fallback import QwenFallback
        qwen_fallback = QwenFallback()
        if qwen_fallback.check_ollama_status():
            logger.info("Qwen fallback initialized and ready")
            return True
        else:
            logger.warning("Qwen fallback initialized but Ollama not ready")
            return True  # Still return True, will work when Ollama is started
    except Exception as e:
        logger.error(f"Failed to initialize Qwen fallback: {e}")
        return False

# --- FastAPI App ---
app = FastAPI(
    title="Document Extraction API",
    description="API for extracting data from Aadhaar and PAN cards with Qwen fallback support",
    version="1.0.0"
)

# --- Request Models ---
class AadhaarProcessRequest(BaseModel):
    user_id: str = "default_user"
    front_url: HttpUrl
    back_url: HttpUrl
    confidence_threshold: float = Field(config.DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    use_fallback: bool = True  # Enable/disable Qwen fallback

class PanCardProcessRequest(BaseModel):
    user_id: str = "default_user"
    image_url: HttpUrl

class Base64ImageRequest(BaseModel):
    user_id: str = "default_user"
    image_base64: str
    doc_type: str = "pancard"  # "aadhaar" or "pancard"
    use_fallback: bool = True

# --- Utility Functions ---
async def download_image(session: aiohttp.ClientSession, url: str, filepath: Path) -> bool:
    """Download image from URL"""
    try:
        async with session.get(str(url), timeout=30) as response:
            response.raise_for_status()
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(await response.read())
            return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False

def save_to_excel(data: Dict[str, Any], excel_path: Path):
    """Save/append data to Excel file"""
    try:
        if excel_path.exists():
            df_existing = pd.read_excel(excel_path)
            df_new = pd.DataFrame([data])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.DataFrame([data])
        
        df_combined.to_excel(excel_path, index=False)
        logger.info(f"Data saved to {excel_path}")
    except Exception as e:
        logger.error(f"Failed to save to Excel: {e}")

def save_to_pkl(number: str, pkl_path: Path, key: str):
    """Save number to PKL file"""
    try:
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = []
        
        # Check for duplicates
        existing_numbers = [entry.get(key, "") for entry in data if isinstance(entry, dict)]
        if number not in existing_numbers:
            data.append({key: number, "timestamp": datetime.now().isoformat()})
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Number saved to {pkl_path}")
            return True
        else:
            logger.info(f"Number already exists in {pkl_path}")
            return False
    except Exception as e:
        logger.error(f"Failed to save to PKL: {e}")
        return False

def check_duplicate_number(number: str, pkl_path: Path, key: str) -> Optional[Dict]:
    """Check if number already exists in PKL"""
    try:
        if not pkl_path.exists():
            return None
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        for entry in data:
            if isinstance(entry, dict) and entry.get(key, "").replace(" ", "") == number.replace(" ", ""):
                return entry
        return None
    except Exception as e:
        logger.error(f"Error checking duplicates: {e}")
        return None

# --- Import fallback helper functions ---
def check_missing_fields_aadhaar(data: Dict[str, Any]) -> list:
    """Check which Aadhaar fields are missing or invalid"""
    import re
    required_fields = ["aadharnumber", "name", "dob", "gender", "address", "pincode"]
    missing = []
    
    for field in required_fields:
        value = data.get(field, "")
        
        # Check if empty or Invalid Format
        if not value or value == "" or value == "Invalid Format":
            missing.append(field)
            continue
        
        # Validate Aadhaar number format (should be exactly 12 digits)
        if field == "aadharnumber":
            clean_aadhar = re.sub(r'\D', '', str(value))
            if len(clean_aadhar) != 12 or not clean_aadhar.isdigit():
                missing.append(field)
                continue
        
        # Detect gibberish text (too many special chars, random patterns)
        if field in ["name", "address"]:
            str_value = str(value)
            # Count special characters and non-alphabetic characters
            special_chars = len(re.findall(r'[^a-zA-Z0-9\s,.-]', str_value))
            total_chars = len(str_value.replace(' ', ''))
            
            # If more than 30% special chars, consider it gibberish
            if total_chars > 0 and (special_chars / total_chars) > 0.3:
                missing.append(field)
                continue
            
            # Check for too many single characters (OCR noise)
            words = str_value.split()
            single_chars = sum(1 for w in words if len(w) == 1)
            if len(words) > 0 and (single_chars / len(words)) > 0.5:
                missing.append(field)
                continue
    
    return missing

def check_missing_fields_pancard(data: Dict[str, Any]) -> list:
    """Check which PAN card fields are missing"""
    required_fields = ["pan_number", "name", "father_name", "dob"]
    missing = []
    for field in required_fields:
        value = data.get(field, "")
        if not value or value == "":
            missing.append(field)
    return missing

def merge_results(primary: Dict, fallback: Dict, fields: list) -> Dict:
    """Merge primary and fallback results"""
    merged = primary.copy()
    for field in fields:
        primary_val = primary.get(field, "")
        if not primary_val or primary_val == "" or primary_val == "Invalid Format":
            fallback_val = fallback.get(field, "")
            if fallback_val:
                merged[field] = fallback_val
    return merged

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize pipelines on startup"""
    logger.info("Starting Document Extraction API...")
    
    # Initialize Aadhaar pipeline
    initialize_aadhaar_pipeline()
    
    # Initialize Qwen fallback
    initialize_qwen_fallback()
    
    logger.info("API startup complete")

# --- Aadhaar Endpoints ---
@app.post("/verify_aadhar", tags=["Aadhaar Processing"])
async def verify_aadhaar(request: AadhaarProcessRequest):
    """
    Process Aadhaar card images and extract data.
    Uses Qwen fallback if primary pipeline fails to extract all fields.
    """
    global pipeline, qwen_fallback
    
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "data": {"user_id": request.user_id, "details": "Aadhaar pipeline not initialized"},
                "message": "service_unavailable"
            }
        )
    
    task_id = hashlib.md5(f"{request.user_id}_{datetime.now().timestamp()}".encode()).hexdigest()
    user_download_dir = config.DOWNLOAD_DIR / request.user_id / task_id
    user_download_dir.mkdir(parents=True, exist_ok=True)
    
    front_path = user_download_dir / "front.jpg"
    back_path = user_download_dir / "back.jpg"
    
    logger.info(f"[user_id={request.user_id}] Processing Aadhaar - task_id={task_id}")
    
    # Download images
    async with aiohttp.ClientSession() as session:
        downloads = await asyncio.gather(
            download_image(session, str(request.front_url), front_path),
            download_image(session, str(request.back_url), back_path)
        )
    
    front_downloaded, back_downloaded = downloads
    
    if not front_downloaded or not back_downloaded:
        shutil.rmtree(user_download_dir, ignore_errors=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {"user_id": request.user_id, "details": "Failed to download images"},
                "message": "download_failed"
            }
        )
    
    # Process with primary pipeline
    try:
        from aadhar import extract_main_fields
        
        result = pipeline.process_images(
            [str(front_path), str(back_path)],
            request.user_id,
            task_id,
            request.confidence_threshold
        )
        
        if 'error' in result:
            # Check if it's a security error
            if result.get("error") == "print_aadhar_detected":
                shutil.rmtree(user_download_dir, ignore_errors=True)
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "data": {"user_id": request.user_id, "details": "Print Aadhaar detected"},
                        "message": "print_aadhar_detected"
                    }
                )
            
            # For other errors, try fallback if enabled
            if request.use_fallback and qwen_fallback:
                logger.info(f"[user_id={request.user_id}] Primary pipeline failed, using Qwen fallback")
                main_data = qwen_fallback.extract_aadhaar_data(str(front_path), str(back_path))
                fallback_used = True
            else:
                shutil.rmtree(user_download_dir, ignore_errors=True)
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "data": {"user_id": request.user_id, "details": result.get("error")},
                        "message": "processing_error"
                    }
                )
        else:
            organized = result.get('organized_results', {})
            main_data = extract_main_fields(organized)
            fallback_used = False
            
            # Check if we should use fallback for missing fields
            if request.use_fallback and qwen_fallback:
                missing_fields = check_missing_fields_aadhaar(main_data)
                if len(missing_fields) >= config.FALLBACK_THRESHOLD_AADHAAR:
                    logger.info(f"[user_id={request.user_id}] Missing fields {missing_fields}, using Qwen fallback")
                    fallback_data = qwen_fallback.extract_aadhaar_data(
                        str(front_path), str(back_path), missing_fields
                    )
                    main_data = merge_results(
                        main_data, fallback_data,
                        ["aadharnumber", "name", "dob", "gender", "address", "pincode"]
                    )
                    fallback_used = True
        
        # Handle masked Aadhaar
        if main_data == "masked_aadhar" or main_data.get("aadharnumber") == "masked_aadhar":
            shutil.rmtree(user_download_dir, ignore_errors=True)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "data": {"user_id": request.user_id, "details": "Masked Aadhaar detected"},
                    "message": "masked_aadhar_detected"
                }
            )
        
        # Add metadata
        main_data['user_id'] = request.user_id
        main_data['timestamp'] = datetime.now().isoformat()
        main_data['fallback_used'] = fallback_used
        
        # Check for duplicates
        aadhar_number = main_data.get('aadharnumber', '').replace(' ', '')
        if aadhar_number:
            duplicate = check_duplicate_number(aadhar_number, config.AADHAR_PKL_PATH, "aadharnumber")
            if duplicate:
                shutil.rmtree(user_download_dir, ignore_errors=True)
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "data": {"aadharnumber": aadhar_number, "duplicate": True},
                        "message": "duplicate_aadhar_found"
                    }
                )
        
        # Save to Excel (all details)
        save_to_excel(main_data, config.AADHAR_EXCEL_PATH)
        
        # Save to PKL (only Aadhaar number)
        if aadhar_number:
            save_to_pkl(aadhar_number, config.AADHAR_PKL_PATH, "aadharnumber")
        
        # Cleanup
        shutil.rmtree(user_download_dir, ignore_errors=True)
        
        # Prepare response
        response_data = {
            "user_id": request.user_id,
            "aadharnumber": main_data.get("aadharnumber", ""),
            "name": main_data.get("name", ""),
            "dob": main_data.get("dob", ""),
            "gender": main_data.get("gender", ""),
            "address": main_data.get("address", ""),
            "pincode": main_data.get("pincode", ""),
            "fallback_used": fallback_used
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": response_data,
                "message": "aadhar_data_saved"
            }
        )
        
    except Exception as e:
        logger.error(f"[user_id={request.user_id}] Error: {e}")
        shutil.rmtree(user_download_dir, ignore_errors=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "data": {"user_id": request.user_id, "details": str(e)},
                "message": "internal_error"
            }
        )

# --- PAN Card Endpoints ---
@app.post("/verify_pancard", tags=["PAN Card Processing"])
async def verify_pancard(request: PanCardProcessRequest):
    """
    Process PAN card image and extract data.
    Uses Qwen fallback if primary detection fails.
    """
    global qwen_fallback
    
    task_id = hashlib.md5(f"{request.user_id}_{datetime.now().timestamp()}".encode()).hexdigest()
    user_download_dir = config.DOWNLOAD_DIR / request.user_id / task_id
    user_download_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = user_download_dir / "pancard.jpg"
    
    logger.info(f"[user_id={request.user_id}] Processing PAN card - task_id={task_id}")
    
    # Download image
    async with aiohttp.ClientSession() as session:
        downloaded = await download_image(session, str(request.image_url), image_path)
    
    if not downloaded:
        shutil.rmtree(user_download_dir, ignore_errors=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {"user_id": request.user_id, "details": "Failed to download image"},
                "message": "download_failed"
            }
        )
    
    try:
        # For PAN card, we use Qwen fallback directly as the primary method
        # (since the original pancard.py uses Celery and external dependencies)
        
        main_data = {}
        fallback_used = False
        
        # Try to import and use pancard processing if available
        try:
            # Note: Original pancard.py uses Celery, so we'll use Qwen for direct processing
            raise ImportError("Using Qwen for PAN processing")
        except ImportError:
            # Use Qwen fallback for PAN card extraction
            if qwen_fallback:
                logger.info(f"[user_id={request.user_id}] Using Qwen for PAN card extraction")
                main_data = qwen_fallback.extract_pancard_data(str(image_path))
                fallback_used = True
            else:
                shutil.rmtree(user_download_dir, ignore_errors=True)
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "data": {"user_id": request.user_id, "details": "Qwen fallback not available"},
                        "message": "service_unavailable"
                    }
                )
        
        if "error" in main_data:
            shutil.rmtree(user_download_dir, ignore_errors=True)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "data": {"user_id": request.user_id, "details": main_data.get("error")},
                    "message": "extraction_failed"
                }
            )
        
        # Add metadata
        main_data['user_id'] = request.user_id
        main_data['timestamp'] = datetime.now().isoformat()
        main_data['fallback_used'] = fallback_used
        
        # Check for duplicates
        pan_number = main_data.get('pan_number', '').replace(' ', '')
        if pan_number:
            duplicate = check_duplicate_number(pan_number, config.PANCARD_PKL_PATH, "pan_number")
            if duplicate:
                shutil.rmtree(user_download_dir, ignore_errors=True)
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "data": {"pan_number": pan_number, "duplicate": True},
                        "message": "duplicate_pancard_found"
                    }
                )
        
        # Save to Excel (all details)
        save_to_excel(main_data, config.PANCARD_EXCEL_PATH)
        
        # Save to PKL (only PAN number)
        if pan_number:
            save_to_pkl(pan_number, config.PANCARD_PKL_PATH, "pan_number")
        
        # Cleanup
        shutil.rmtree(user_download_dir, ignore_errors=True)
        
        # Prepare response
        response_data = {
            "user_id": request.user_id,
            "pan_number": main_data.get("pan_number", ""),
            "name": main_data.get("name", ""),
            "father_name": main_data.get("father_name", ""),
            "dob": main_data.get("dob", ""),
            "fallback_used": fallback_used
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": response_data,
                "message": "pancard_data_saved"
            }
        )
        
    except Exception as e:
        logger.error(f"[user_id={request.user_id}] Error: {e}")
        shutil.rmtree(user_download_dir, ignore_errors=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "data": {"user_id": request.user_id, "details": str(e)},
                "message": "internal_error"
            }
        )

@app.post("/process_pan_upload", tags=["PAN Card Processing"])
async def process_pan_upload(
    file: UploadFile = File(...),
    user_id: str = Form("default_user"),
    email_id: str = Form(None),
    mob_no: str = Form(None)
):
    """Process PAN card from uploaded file"""
    global qwen_fallback
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Invalid file format"}
        )
    
    task_id = hashlib.md5(f"{user_id}_{datetime.now().timestamp()}".encode()).hexdigest()
    user_download_dir = config.DOWNLOAD_DIR / user_id / task_id
    user_download_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = user_download_dir / f"pancard_{file.filename}"
    
    # Save uploaded file
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        if qwen_fallback:
            main_data = qwen_fallback.extract_pancard_data(str(image_path))
        else:
            shutil.rmtree(user_download_dir, ignore_errors=True)
            return JSONResponse(
                status_code=503,
                content={"success": False, "message": "Qwen fallback not available"}
            )
        
        if "error" in main_data:
            shutil.rmtree(user_download_dir, ignore_errors=True)
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": main_data.get("error")}
            )
        
        # Add metadata and save
        main_data['user_id'] = user_id
        main_data['mob_no'] = mob_no
        main_data['email_id'] = email_id
        main_data['timestamp'] = datetime.now().isoformat()
        
        pan_number = main_data.get('pan_number', '')
        if pan_number:
            duplicate = check_duplicate_number(pan_number, config.PANCARD_PKL_PATH, "pan_number")
            if duplicate:
                shutil.rmtree(user_download_dir, ignore_errors=True)
                return JSONResponse(
                    status_code=200,
                    content={"success": True, "data": {"pan_number": pan_number}, "message": "duplicate_pancard_found"}
                )
        
        save_to_excel(main_data, config.PANCARD_EXCEL_PATH)
        if pan_number:
            save_to_pkl(pan_number, config.PANCARD_PKL_PATH, "pan_number")
        
        shutil.rmtree(user_download_dir, ignore_errors=True)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "pan_number": main_data.get("pan_number", ""),
                    "name": main_data.get("name", ""),
                    "father_name": main_data.get("father_name", ""),
                    "dob": main_data.get("dob", "")
                },
                "message": "pancard_data_saved"
            }
        )
        
    except Exception as e:
        shutil.rmtree(user_download_dir, ignore_errors=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# --- Data Retrieval Endpoints ---
@app.get("/user/{user_id}/aadhar", tags=["Data Retrieval"])
def get_user_aadhar(user_id: str):
    """Get Aadhaar records for a user"""
    try:
        if config.AADHAR_EXCEL_PATH.exists():
            df = pd.read_excel(config.AADHAR_EXCEL_PATH)
            df['user_id'] = df['user_id'].astype(str)
            user_data = df[df['user_id'] == user_id].to_dict('records')
            
            if user_data:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "data": {"user_id": user_id, "records": user_data},
                        "message": "records_found"
                    }
                )
        
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "data": {"user_id": user_id},
                "message": "no_records_found"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

@app.get("/user/{user_id}/pancard", tags=["Data Retrieval"])
def get_user_pancard(user_id: str):
    """Get PAN card records for a user"""
    try:
        if config.PANCARD_EXCEL_PATH.exists():
            df = pd.read_excel(config.PANCARD_EXCEL_PATH)
            df['user_id'] = df['user_id'].astype(str)
            user_data = df[df['user_id'] == user_id].to_dict('records')
            
            if user_data:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "data": {"user_id": user_id, "records": user_data},
                        "message": "records_found"
                    }
                )
        
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "data": {"user_id": user_id},
                "message": "no_records_found"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# --- Health Check Endpoints ---
@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Check API and service health"""
    global pipeline, qwen_fallback
    
    health_status = {
        "api": "healthy",
        "aadhaar_pipeline": "initialized" if pipeline else "not_initialized",
        "qwen_fallback": "initialized" if qwen_fallback else "not_initialized"
    }
    
    # Check Ollama if fallback is initialized
    if qwen_fallback:
        try:
            ollama_ready = qwen_fallback.check_ollama_status()
            health_status["ollama"] = "ready" if ollama_ready else "not_ready"
        except:
            health_status["ollama"] = "error"
    
    all_healthy = all([
        health_status.get("aadhaar_pipeline") == "initialized" or health_status.get("qwen_fallback") == "initialized"
    ])
    
    return JSONResponse(
        status_code=200 if all_healthy else 503,
        content={
            "success": all_healthy,
            "data": health_status,
            "message": "service_healthy" if all_healthy else "service_degraded"
        }
    )

@app.get("/fallback/status", tags=["Monitoring"])
async def fallback_status():
    """Check Qwen fallback status"""
    global qwen_fallback
    
    if not qwen_fallback:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "data": {"status": "not_initialized"},
                "message": "qwen_fallback_not_available"
            }
        )
    
    ollama_ready = qwen_fallback.check_ollama_status()
    
    return JSONResponse(
        status_code=200 if ollama_ready else 503,
        content={
            "success": ollama_ready,
            "data": {
                "status": "ready" if ollama_ready else "ollama_not_ready",
                "model": qwen_fallback.model
            },
            "message": "fallback_ready" if ollama_ready else "start_ollama"
        }
    )

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8103,
        reload=True
    )
