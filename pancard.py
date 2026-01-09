import os
import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image
import json
import re
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import uvicorn
import pickle
import pandas as pd

# Import Qwen fallback for when primary detection fails
try:
    from qwen_fallback import QwenFallback, check_missing_fields_pancard
    QWEN_FALLBACK_AVAILABLE = True
except ImportError:
    QWEN_FALLBACK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Qwen fallback module not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base64ImageRequest(BaseModel):
    image_base64: str
    user_id: str = "default_user"

class ImagePathRequest(BaseModel):
    image_path: str
    user_id: str = "default_user"

class ProcessPANRequest(BaseModel):
    user_id: str = "default_user"

class CleanPANProcessor:
    def __init__(self, model_path='model/best_pancard.pt', data_dir='data'):
        """Initialize the PAN processor with clean and simple extraction"""
        
        # Dynamic device selection
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"✅ CUDA is available. Model will run on GPU ({torch.cuda.get_device_name(0)}).")
        else:
            self.device = "cpu"
            logger.info("⚠️ CUDA not available. Model will fall back to CPU.")
        
        # Load YOLO model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = YOLO(model_path)
        logger.info(f"✅ YOLO model loaded successfully from {model_path}")
        
        # Define confidence threshold for poor detections
        self.MIN_CONFIDENCE_THRESHOLD = 0.35  # Below this, consider as "pan_notfound"
        
        # Setup data directory for storing summaries
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.pkl_path = self.data_dir / "summary.pkl"
        self.csv_path = self.data_dir / "summary.csv"
        
        # Initialize Qwen fallback
        self.qwen_fallback = None
        if QWEN_FALLBACK_AVAILABLE:
            try:
                self.qwen_fallback = QwenFallback()
                if self.qwen_fallback.check_ollama_status():
                    logger.info("✅ Qwen fallback initialized and ready")
                else:
                    logger.warning("⚠️ Qwen fallback initialized but Ollama not ready")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize Qwen fallback: {e}")
        
        self._check_tesseract()
    
    def _check_tesseract(self):
        """Check if Tesseract is properly installed"""
        try:
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR is available")
        except pytesseract.TesseractNotFoundError:
            logger.critical("❌ Tesseract not found. Please install Tesseract OCR.")
            raise RuntimeError("Tesseract not found")
    
    def _try_detection_with_rotation(self, image: np.ndarray) -> Tuple[Any, np.ndarray, int, float]:
        """
        Try YOLO detection with automatic rotation correction
        Returns: (results, corrected_image, rotation_used, avg_confidence)
        """
        # First try without rotation
        results = self.model.predict(
            source=image,
            conf=0.25,
            iou=0.45,
            save=False,
            device=self.device,
            verbose=False
        )
        
        # Check detection quality
        detection_count = 0
        avg_confidence = 0
        
        if results and len(results) > 0:
            for box in results[0].boxes:
                class_name = self.model.names[int(box.cls)]
                if class_name in ['pan', 'name', 'father', 'dob']:
                    detection_count += 1
                    avg_confidence += float(box.conf)
            
            if detection_count > 0:
                avg_confidence /= detection_count
        
        # If good detections, use them
        if detection_count >= 3 and avg_confidence > 0.6:
            logger.info(f"✅ Good detections without rotation: {detection_count} fields, avg conf: {avg_confidence:.3f}")
            return results, image, 0, avg_confidence
        
        logger.info(f"⚠️ Initial detections: {detection_count} fields, trying rotations...")
        
        # Try different rotations
        best_results = results
        best_image = image
        best_rotation = 0
        best_score = detection_count * avg_confidence
        best_avg_conf = avg_confidence
        
        for angle in [90, 180, 270]:
            if angle == 90:
                rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            test_results = self.model.predict(
                source=rotated,
                conf=0.25,
                iou=0.45,
                save=False,
                device=self.device,
                verbose=False
            )
            
            test_count = 0
            test_conf = 0
            
            if test_results and len(test_results) > 0:
                for box in test_results[0].boxes:
                    class_name = self.model.names[int(box.cls)]
                    if class_name in ['pan', 'name', 'father', 'dob']:
                        test_count += 1
                        test_conf += float(box.conf)
                
                if test_count > 0:
                    test_conf /= test_count
            
            test_score = test_count * test_conf
            
            if test_score > best_score:
                best_score = test_score
                best_results = test_results
                best_image = rotated
                best_rotation = angle
                best_avg_conf = test_conf
        
        if best_rotation != 0:
            logger.info(f"✅ Using {best_rotation}° rotation")
        
        return best_results, best_image, best_rotation, best_avg_conf
    
    def _simple_preprocess(self, image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """Simple and effective preprocessing"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Scale up if too small
        h, w = gray.shape[:2]
        if h < 50:
            scale = 100 / h
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Enhance contrast if needed
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        
        return gray
    
    def _extract_pan(self, image: np.ndarray) -> str:
        """Extract PAN number with strict validation"""
        # Try different preprocessing
        images_to_try = [
            self._simple_preprocess(image, enhance_contrast=False),
            self._simple_preprocess(image, enhance_contrast=True),
        ]
        
        for img in images_to_try:
            # Try different PSM modes
            for psm in [8, 7, 13]:
                try:
                    # With whitelist
                    config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    text = pytesseract.image_to_string(img, config=config).strip()
                    
                    # Extract valid PAN
                    clean = text.upper().replace(' ', '').replace('\n', '')
                    pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', clean)
                    if pan_match:
                        return pan_match.group()
                    
                    # Without whitelist
                    text = pytesseract.image_to_string(img, config=f'--psm {psm}').strip()
                    clean = text.upper().replace(' ', '').replace('\n', '')
                    pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', clean)
                    if pan_match:
                        return pan_match.group()
                        
                except:
                    continue
        
        return ""
    
    def _extract_name(self, image: np.ndarray) -> str:
        """Extract name with better configuration"""
        best_name = ""
        
        # Preprocess
        gray = self._simple_preprocess(image, enhance_contrast=True)
        
        # Try different PSM modes for names
        for psm in [7, 6, 8, 11]:
            try:
                # OCR without restrictive whitelist
                config = f'--psm {psm} -l eng'
                text = pytesseract.image_to_string(gray, config=config).strip()
                
                # Clean the text
                # Remove non-alphabetic characters except spaces
                cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
                
                # Remove single letters and very short words (likely noise)
                words = []
                for word in cleaned.split():
                    if len(word) > 2:  # Keep words with more than 2 characters
                        words.append(word)
                    elif word.upper() in ['JR', 'SR', 'II', 'III']:  # Keep valid short words
                        words.append(word)
                
                if words:
                    name = ' '.join(words)
                    # Check if this looks like a valid name
                    if len(words) <= 5 and len(name) > len(best_name):  # Names usually have 1-5 words
                        best_name = name
                        
            except:
                continue
        
        # Format the name properly with title case and normal spaces
        if best_name:
            return ' '.join(word.title() for word in best_name.split())
        
        return ""
    
    def _extract_dob(self, image: np.ndarray) -> str:
        """Extract date of birth in YYYY-MM-DD format"""
        # Preprocess
        gray = self._simple_preprocess(image, enhance_contrast=True)
        
        for psm in [7, 8, 6]:
            try:
                # Try with whitelist
                config = f'--psm {psm} -c tessedit_char_whitelist=0123456789/-'
                text = pytesseract.image_to_string(gray, config=config).strip()
                
                # Look for date pattern
                date_match = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', text)
                if date_match:
                    day, month, year = date_match.groups()
                    day, month, year = int(day), int(month), int(year)
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2010:
                        # Return in YYYY-MM-DD format
                        return f"{year}-{month:02d}-{day:02d}"
                
                # Try without whitelist
                text = pytesseract.image_to_string(gray, config=f'--psm {psm}').strip()
                date_match = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', text)
                if date_match:
                    day, month, year = date_match.groups()
                    day, month, year = int(day), int(month), int(year)
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2010:
                        return f"{year}-{month:02d}-{day:02d}"
                        
            except:
                continue
        
        return ""
    
    def check_duplicate(self, pan_number: str) -> Dict[str, Any]:
        """Check if PAN number already exists in database"""
        if not pan_number or not self.pkl_path.exists():
            return None
        
        try:
            with open(self.pkl_path, 'rb') as f:
                all_data = pickle.load(f)
            
            for entry in all_data:
                if entry.get('pan', '').upper() == pan_number.upper():
                    return {
                        'matched_user_id': entry.get('user_id', ''),
                        'name': entry.get('name', '')
                    }
        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")
        
        return None
    
    def save_to_database(self, data: Dict[str, Any]):
        """Save data to PKL and CSV files"""
        try:
            # Save to PKL
            all_data = []
            if self.pkl_path.exists():
                with open(self.pkl_path, 'rb') as f:
                    all_data = pickle.load(f)
            
            all_data.append(data)
            
            with open(self.pkl_path, 'wb') as f:
                pickle.dump(all_data, f)
            
            # Save to CSV
            df = pd.DataFrame([data])
            if self.csv_path.exists():
                df.to_csv(self.csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.csv_path, mode='w', header=True, index=False)
            
            logger.info(f"✅ Data saved for user: {data.get('user_id')}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def process_pan_card(self, image_path: str, user_id: str = "default_user") -> Dict[str, Any]:
        """Main processing pipeline for PAN card"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            logger.info(f"Processing PAN card for user: {user_id}")
            
            # Try detection with automatic rotation
            results, corrected_image, rotation_used, avg_confidence = self._try_detection_with_rotation(image)
            
            if rotation_used != 0:
                logger.info(f"🔄 Image rotated by {rotation_used}°")
            
            # Check if detections are too poor (confidence below threshold)
            if avg_confidence < self.MIN_CONFIDENCE_THRESHOLD:
                logger.warning(f"❌ Poor detection confidence: {avg_confidence:.3f}")
                
                # Try Qwen fallback
                if self.qwen_fallback:
                    logger.info("🔄 Using Qwen fallback for PAN extraction...")
                    fallback_data = self.qwen_fallback.extract_pancard_data(image_path)
                    
                    if "error" not in fallback_data:
                        # Map fallback fields to our format
                        data = {
                            "pan": fallback_data.get("pan_number", ""),
                            "name": fallback_data.get("name", ""),
                            "father": fallback_data.get("father_name", ""),
                            "dob": fallback_data.get("dob", "")
                        }
                        
                        # Check for duplicates
                        if data['pan']:
                            duplicate = self.check_duplicate(data['pan'])
                            if duplicate:
                                logger.warning(f"⚠️ Duplicate PAN found: {data['pan']}")
                                return {
                                    "status": False,
                                    "matched_user_id": duplicate['matched_user_id'],
                                    "name": duplicate['name'],
                                    "message": "duplicate_found"
                                }
                        
                        has_empty = any(v == "" for v in data.values())
                        
                        if not has_empty:
                            # Save to database
                            save_data = {
                                "user_id": user_id,
                                "pan": data["pan"],
                                "name": data["name"],
                                "father": data["father"],
                                "dob": data["dob"],
                                "timestamp": datetime.now().isoformat(),
                                "fallback_used": True
                            }
                            self.save_to_database(save_data)
                            
                            return {
                                "status": True,
                                "user_id": user_id,
                                "data": data,
                                "message": "pan_found",
                                "fallback_used": True
                            }
                        else:
                            return {
                                "status": False,
                                "user_id": user_id,
                                "data": data,
                                "message": "details_not_found",
                                "fallback_used": True
                            }
                
                return {
                    "status": False,
                    "user_id": user_id,
                    "data": {
                        "pan": "",
                        "name": "",
                        "father": "",
                        "dob": ""
                    },
                    "message": "pan_notfound"
                }
            
            # Initialize data dictionary
            data = {
                "pan": "",
                "name": "",
                "father": "",
                "dob": ""
            }
            
            # Process detections
            best_detections = {}
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = self.model.names[class_id]
                    
                    if class_name not in ['pan', 'name', 'father', 'dob']:
                        continue
                    
                    if class_name not in best_detections or confidence > best_detections[class_name]['confidence']:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Add padding
                        padding = 10
                        y1 = max(0, y1 - padding)
                        y2 = min(corrected_image.shape[0], y2 + padding)
                        x1 = max(0, x1 - padding)
                        x2 = min(corrected_image.shape[1], x2 + padding)
                        
                        best_detections[class_name] = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence
                        }
            
            # Check if no valid detections found
            if not best_detections:
                # Try Qwen fallback
                if self.qwen_fallback:
                    logger.info("🔄 No detections found, using Qwen fallback...")
                    fallback_data = self.qwen_fallback.extract_pancard_data(image_path)
                    
                    if "error" not in fallback_data:
                        data = {
                            "pan": fallback_data.get("pan_number", ""),
                            "name": fallback_data.get("name", ""),
                            "father": fallback_data.get("father_name", ""),
                            "dob": fallback_data.get("dob", "")
                        }
                        
                        if data['pan']:
                            duplicate = self.check_duplicate(data['pan'])
                            if duplicate:
                                return {
                                    "status": False,
                                    "matched_user_id": duplicate['matched_user_id'],
                                    "name": duplicate['name'],
                                    "message": "duplicate_found"
                                }
                        
                        has_empty = any(v == "" for v in data.values())
                        if not has_empty:
                            save_data = {
                                "user_id": user_id,
                                "pan": data["pan"],
                                "name": data["name"],
                                "father": data["father"],
                                "dob": data["dob"],
                                "timestamp": datetime.now().isoformat(),
                                "fallback_used": True
                            }
                            self.save_to_database(save_data)
                            return {
                                "status": True,
                                "user_id": user_id,
                                "data": data,
                                "message": "pan_found",
                                "fallback_used": True
                            }
                        else:
                            return {
                                "status": False,
                                "user_id": user_id,
                                "data": data,
                                "message": "details_not_found",
                                "fallback_used": True
                            }
                
                return {
                    "status": False,
                    "user_id": user_id,
                    "data": data,
                    "message": "pan_notfound"
                }
            
            # Extract text from each field
            for field_name, detection in best_detections.items():
                x1, y1, x2, y2 = detection['bbox']
                cropped = corrected_image[y1:y2, x1:x2].copy()
                
                # Use appropriate extraction method
                if field_name == 'pan':
                    extracted_text = self._extract_pan(cropped)
                elif field_name in ['name', 'father']:
                    extracted_text = self._extract_name(cropped)
                elif field_name == 'dob':
                    extracted_text = self._extract_dob(cropped)
                else:
                    extracted_text = ""
                
                data[field_name] = extracted_text
                logger.info(f"✅ {field_name}: {extracted_text if extracted_text else 'Not found'} (conf: {detection['confidence']:.3f})")
            
            # Check if any field is empty
            has_empty_fields = any(value == "" for value in data.values())
            
            # Check for duplicates if PAN found
            if data['pan']:
                duplicate = self.check_duplicate(data['pan'])
                if duplicate:
                    logger.warning(f"⚠️ Duplicate PAN found: {data['pan']}")
                    return {
                        "status": False,
                        "matched_user_id": duplicate['matched_user_id'],
                        "name": duplicate['name'],
                        "message": "duplicate_found"
                    }
            
            # Determine status and message
            fallback_used = False
            
            if has_empty_fields:
                # Try Qwen fallback to fill missing fields
                if self.qwen_fallback:
                    missing_fields = [k for k, v in data.items() if v == ""]
                    logger.info(f"🔄 Missing fields {missing_fields}, trying Qwen fallback...")
                    
                    # Map our field names to fallback field names
                    fallback_field_map = {
                        "pan": "pan_number",
                        "name": "name",
                        "father": "father_name",
                        "dob": "dob"
                    }
                    fallback_missing = [fallback_field_map[f] for f in missing_fields]
                    
                    fallback_data = self.qwen_fallback.extract_pancard_data(image_path, fallback_missing)
                    
                    if "error" not in fallback_data:
                        # Merge results - primary takes precedence
                        for field in missing_fields:
                            fallback_key = fallback_field_map[field]
                            if fallback_data.get(fallback_key):
                                data[field] = fallback_data[fallback_key]
                                logger.info(f"✅ Fallback filled {field}: {data[field]}")
                        
                        fallback_used = True
                        has_empty_fields = any(v == "" for v in data.values())
            
            if has_empty_fields:
                status = False
                message = "details_not_found"
            else:
                status = True
                message = "pan_found"
                
                # Save to database if all fields are present
                save_data = {
                    "user_id": user_id,
                    "pan": data["pan"],
                    "name": data["name"],
                    "father": data["father"],
                    "dob": data["dob"],
                    "timestamp": datetime.now().isoformat(),
                    "fallback_used": fallback_used
                }
                self.save_to_database(save_data)
            
            result = {
                "status": status,
                "user_id": user_id,
                "data": data,
                "message": message
            }
            
            if fallback_used:
                result["fallback_used"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PAN card: {str(e)}")
            return {
                "status": False,
                "user_id": user_id,
                "data": {
                    "pan": "",
                    "name": "",
                    "father": "",
                    "dob": ""
                },
                "message": "pan_notfound"
            }

# FastAPI app setup
app = FastAPI(title="PAN Card Processing API with User Management", version="7.0.0")
processor = None

@app.on_event("startup")
async def startup_event():
    global processor
    try:
        processor = CleanPANProcessor()
        logger.info("✅ PAN processor initialized successfully")
    except Exception as e:
        logger.critical(f"❌ Failed to initialize processor: {e}")
        raise

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": processor.device if processor else "not_initialized",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/process_pan")
async def process_pan_file(file: UploadFile = File(...), user_id: str = "default_user"):
    """Process PAN card from file upload with user_id"""
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        result = processor.process_pan_card(temp_path, user_id)
        
        os.remove(temp_path)
        
        return result
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process_pan_base64")
async def process_pan_base64(request: Base64ImageRequest):
    """Process PAN card from base64 encoded image"""
    try:
        image_data = base64.b64decode(request.image_base64)
        
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        result = processor.process_pan_card(temp_path, request.user_id)
        
        os.remove(temp_path)
        
        return result
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process_pan_path")
async def process_pan_path(request: ImagePathRequest):
    """Process PAN card from file path or URL"""
    try:
        import requests
        if request.image_path.startswith("http://") or request.image_path.startswith("https://"):
            response = requests.get(request.image_path, timeout=10)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
            
            temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            result = processor.process_pan_card(temp_path, request.user_id)
            os.remove(temp_path)
        else:
            result = processor.process_pan_card(request.image_path, request.user_id)

        return result
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/user/{user_id}")
def get_user_data(user_id: str):
    """Get user data by user_id"""
    try:
        if not processor.pkl_path.exists():
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "user_not_found"}
            )
        
        with open(processor.pkl_path, 'rb') as f:
            all_data = pickle.load(f)
        
        user_data = [entry for entry in all_data if entry.get('user_id') == user_id]
        
        if not user_data:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "user_not_found"}
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "user_id": user_id,
                "data": user_data,
                "message": "user_found"
            }
        )
        
    except Exception as e:
        logger.error(f"Error fetching user data: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": "error_fetching_data"}
        )

@app.delete("/user/{user_id}")
def delete_user_data(user_id: str):
    """Delete user data by user_id"""
    try:
        user_found = False
        
        # Delete from PKL
        if processor.pkl_path.exists():
            with open(processor.pkl_path, 'rb') as f:
                all_data = pickle.load(f)
            
            original_length = len(all_data)
            filtered_data = [entry for entry in all_data if entry.get('user_id') != user_id]
            
            if len(filtered_data) < original_length:
                user_found = True
                with open(processor.pkl_path, 'wb') as f:
                    pickle.dump(filtered_data, f)
        
        # Delete from CSV
        if processor.csv_path.exists():
            df = pd.read_csv(processor.csv_path)
            original_length = len(df)
            df_filtered = df[df['user_id'] != user_id]
            
            if len(df_filtered) < original_length:
                user_found = True
                df_filtered.to_csv(processor.csv_path, index=False)
        
        if not user_found:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "user_not_found"}
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "user_id": user_id,
                "message": "user_deleted"
            }
        )
        
    except Exception as e:
        logger.error(f"Error deleting user data: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": "error_deleting_data"}
        )

if __name__ == '__main__':
    if not os.path.exists('model/best_pancard.pt'):
        print("❌ Warning: Model file 'model/best_pancard.pt' not found!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)