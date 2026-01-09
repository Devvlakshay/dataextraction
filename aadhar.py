import asyncio
import hashlib
import json
import logging
import math # Added for rotation calculation
import os
import pickle
import shutil
import re
import sys
import tempfile
import time 
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiofiles
import aiohttp
import cv2
import numpy as np
import pandas as pd  # Added for CSV writing
import pytesseract
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, HttpUrl
from ultralytics import YOLO

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Core Pipeline Logic (Fully In-Memory) ---

class ComprehensiveAadhaarPipeline:
    # MODIFIED: Removed confidence_threshold from __init__ as it's now stateless regarding the threshold.
    def __init__(self, model1_path, model2_path, other_lang_code='hin+tel+ben'):
        ### NEW: Dynamic device selection (CUDA/CPU) ###
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"CUDA is available. Models will run on GPU ({torch.cuda.get_device_name(0)}).")
        else:
            self.device = "cpu"
            logger.info("CUDA not available. Models will fall back to CPU.")
        ### END NEW ###

        self.model1_path = model1_path
        self.model2_path = model2_path
        
        logger.info("Checking for YOLO models on filesystem...")
        if not Path(self.model1_path).exists():
            logger.critical(f"Model1 not found at {self.model1_path}. Aborting startup.")
            raise FileNotFoundError(f"Model1 not found at {self.model1_path}")
        if not Path(self.model2_path).exists():
            logger.critical(f"Model2 not found at {self.model2_path}. Aborting startup.")
            raise FileNotFoundError(f"Model2 not found at {self.model2_path}")

        logger.info("Loading models directly from filesystem...")
        self.model1 = YOLO(self.model1_path)
        self.model2 = YOLO(self.model2_path)
        
        self.other_lang_code = other_lang_code

        self._check_tesseract()

        logger.info("YOLOv8 models loaded successfully from filesystem.")
        logger.info(f"Model1 classes: {self.model1.names}")
        logger.info(f"Model2 classes: {self.model2.names}")

        self.card_classes = {i: name for i, name in self.model1.names.items()}
        self.entity_classes = {
            0: 'aadharnumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
            4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
            8: 'name', 9: 'name_otherlang', 10: 'pincode'
        }
        logger.info(f"Pipeline initialized to use '{self.other_lang_code}' for other language fields.")

    def _check_tesseract(self):
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            logger.critical("Tesseract executable not found. Please install Tesseract OCR and ensure it's in your PATH.")
            raise RuntimeError("Tesseract not found")
        except Exception as e:
             logger.critical(f"An error occurred while checking Tesseract: {e}")
             raise RuntimeError(f"Error checking Tesseract: {e}")

    def detect_and_crop_cards(self, image_paths: List[str], confidence_threshold: float) -> Dict[str, List[Dict[str, Any]]]:
        """Step 1: Detect Aadhaar front/back cards and pass cropped image data (np.array) forward."""
        logger.info(f"\nStep 1: Detecting Aadhaar cards in {len(image_paths)} images (Threshold: {confidence_threshold})")
        cropped_cards = {'front': [], 'back': []}
        for image_path in image_paths:
            logger.info(f"  Processing: {Path(image_path).name}")
            results = self.model1(str(image_path), device=self.device)
            img = cv2.imread(str(image_path))
            input_filename = Path(image_path).stem
            detected = False
            for box in results[0].boxes:
                if float(box.conf[0]) < confidence_threshold: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.card_classes.get(int(box.cls[0]), "unknown")
                if class_name == 'print_aadhar':
                    raise ValueError("print_aadhar_detected")
                if class_name not in ['aadhar_front', 'aadhar_back']: continue
                detected = True
                crop = img[y1:y2, x1:x2]
                card_type = class_name.replace('aadhar_', '')
                card_data = {
                    "image": crop,
                    "name": f"{input_filename}_{class_name}_conf{int(float(box.conf[0])*100)}",
                    "type": card_type
                }
                cropped_cards[card_type].append(card_data)
                logger.info(f" Detected {class_name}")
            if not detected:
                logger.warning(f" No Aadhaar card detected in {Path(image_path).name}.")
        return cropped_cards

    def detect_entities_in_cards(self, cropped_cards: Dict[str, List[Dict[str, Any]]], confidence_threshold: float):
        """Step 2: Detect entities using in-memory card image data."""
        all_card_data = cropped_cards.get('front', []) + cropped_cards.get('back', [])
        logger.info(f"\nStep 2: Detecting entities in {len(all_card_data)} cards (Threshold: {confidence_threshold})")
        all_detections = {}

        for card_info in all_card_data:
            card_name = card_info['name']
            img = card_info['image']
            card_type = card_info['type']
            logger.info(f"  Processing: {card_name}")
            
            results = self.model2(img, device=self.device)
            card_detections = []
            
            for box in results[0].boxes:
                if float(box.conf[0]) < confidence_threshold: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.entity_classes.get(int(box.cls[0]), "unknown")
                card_detections.append({'bbox': (x1, y1, x2, y2), 'class_name': class_name, 'confidence': float(box.conf[0])})
                
            logger.info(f"  Detected {len(card_detections)} entities in {card_name}")
            all_detections[card_name] = {
                "card_image": img,
                "card_type": card_type,
                "detections": card_detections
            }
        return all_detections

    def crop_entities(self, all_detections: Dict[str, Dict[str, Any]]):
        """Step 3: Crop individual entities and add their image data to the detection dictionary."""
        logger.info(f"\nStep 3: Cropping individual entities")
        for card_name, card_data in all_detections.items():
            img = card_data['card_image']
            for i, detection in enumerate(card_data['detections']):
                x1, y1, x2, y2 = detection['bbox']
                crop = img[y1:y2, x1:x2]
                detection['cropped_image'] = crop
                entity_key = f"{card_name}_{detection['class_name']}_{i}"
                detection['entity_key'] = entity_key
                logger.info(f"   Cropped entity: {entity_key}")
        return all_detections
    
    def _correct_entity_orientation_and_preprocess(self, entity_image: np.ndarray, entity_key: str, osd_confidence_threshold: float = 0.5) -> Optional[Image.Image]:
        """
        Takes a numpy array for a single cropped entity, attempts to correct its orientation, 
        and returns a preprocessed PIL Image without saving any files.
        """
        try:
            img = entity_image
            if img is None or img.size == 0:
                logger.warning(f"  Entity image data for {entity_key} is empty, skipping.")
                return None
            
            h, w = img.shape[:2]
            if h < 100:
                scale_factor = 100 / h
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                img_for_analysis = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                img_for_analysis = img

            best_rotation = self._detect_orientation_by_letters(img_for_analysis, entity_key)
            
            if best_rotation is None:
                try:
                    osd = pytesseract.image_to_osd(img_for_analysis, output_type=pytesseract.Output.DICT)
                    if osd['orientation_conf'] > osd_confidence_threshold:
                        best_rotation = osd['rotate']
                        logger.info(f" Using Tesseract OSD for {entity_key}: {best_rotation}° (conf: {osd['orientation_conf']:.2f})")
                    else:
                        best_rotation = 0
                except pytesseract.TesseractError as e:
                    logger.warning(f" Both letter-based and OSD failed for {entity_key}. Assuming 0° rotation. Details: {e}")
                    best_rotation = 0
            
            corrected_img = img
            if best_rotation != 0:
                logger.info(f"   Correcting entity {entity_key} orientation by {best_rotation}°")
                if best_rotation == 90: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif best_rotation == 180: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_180)
                elif best_rotation == 270: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            h_corr, w_corr = corrected_img.shape[:2]
            if h_corr > w_corr and 'address' not in entity_key:
                logger.info(f"   Rotating vertical entity {entity_key} to horizontal format")
                corrected_img = cv2.rotate(corrected_img, cv2.ROTATE_90_CLOCKWISE)

            pil_img = Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))
            return pil_img
            
        except Exception as e:
            logger.error(f"   Unhandled error during entity orientation/preprocessing for {entity_key}: {e}")
            return None

    def _detect_orientation_by_letters(self, img: np.ndarray, entity_key: str) -> Optional[int]:
        """
        Detect the correct orientation by analyzing letter shapes and OCR confidence
        at different rotation angles.
        """
        try:
            rotations = [0, 90, 180, 270]
            rotation_scores = {}
            
            for rotation in rotations:
                if rotation == 0:
                    rotated_img = img
                elif rotation == 90:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_180)
                elif rotation == 270:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                
                score = self._calculate_orientation_score(rotated_img, rotation)
                rotation_scores[rotation] = score
                logger.debug(f"      Rotation {rotation}°: score = {score:.3f}")
            
            best_rotation = max(rotation_scores.keys(), key=lambda k: rotation_scores[k])
            best_score = rotation_scores[best_rotation]
            
            if best_score > 0.1:
                logger.info(f" Letter-based analysis for {entity_key}: {best_rotation}° (score: {best_score:.3f})")
                return best_rotation
            else:
                logger.warning(f"   Letter-based analysis inconclusive for {entity_key} (best score: {best_score:.3f})")
                return None
                
        except Exception as e:
            logger.warning(f"  Error in letter-based orientation detection for {entity_key}: {e}")
            return None

    def _calculate_orientation_score(self, img: np.ndarray, rotation: int) -> float:
        """
        Calculate a score for how likely this orientation is correct.
        """
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            ocr_score = self._get_ocr_confidence_score(gray)
            shape_score = self._analyze_letter_shapes(gray)
            line_score = self._analyze_text_lines(gray)
            
            total_score = (ocr_score * 0.5 + shape_score * 0.3 + line_score * 0.2)
            return total_score
            
        except Exception as e:
            logger.debug(f"      Error calculating orientation score: {e}")
            return 0.0

    def _get_ocr_confidence_score(self, gray_img: np.ndarray) -> float:
        """Get OCR confidence and text quality score"""
        try:
            psm_modes = [6, 7, 8, 13]
            best_confidence = 0.0
            best_text_length = 0
            
            for psm in psm_modes:
                try:
                    data = pytesseract.image_to_data(gray_img, config=f'--psm {psm}', output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        text_length = sum(len(text.strip()) for text in data['text'] if text.strip())
                        
                        if avg_confidence > best_confidence or (avg_confidence == best_confidence and text_length > best_text_length):
                            best_confidence = avg_confidence
                            best_text_length = text_length
                            
                except pytesseract.TesseractError:
                    continue
            
            confidence_factor = best_confidence / 100.0
            length_factor = min(best_text_length / 10.0, 1.0)
            return confidence_factor * 0.7 + length_factor * 0.3
            
        except Exception:
            return 0.0

    def _analyze_letter_shapes(self, gray_img: np.ndarray) -> float:
        """Analyze the shapes of detected contours to determine if they look like upright letters"""
        try:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return 0.0
            
            upright_score = 0.0
            valid_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20: continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 5 or h < 5 or w > gray_img.shape[1] * 0.8 or h > gray_img.shape[0] * 0.8: continue
                
                aspect_ratio = h / w
                if 0.3 <= aspect_ratio <= 4.0:
                    valid_contours += 1
                    if 1.0 <= aspect_ratio <= 2.5: upright_score += 1.0
                    elif 0.5 <= aspect_ratio <= 3.5: upright_score += 0.7
                    else: upright_score += 0.3
            
            if valid_contours == 0: return 0.0
            return min(upright_score / valid_contours, 1.0)
            
        except Exception:
            return 0.0

    def _analyze_text_lines(self, gray_img: np.ndarray) -> float:
        """Analyze text line orientation using morphological operations"""
        try:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            horizontal_pixels = cv2.countNonZero(horizontal_lines)
            vertical_pixels = cv2.countNonZero(vertical_lines)
            
            total_pixels = horizontal_pixels + vertical_pixels
            if total_pixels == 0: return 0.5
            
            horizontal_ratio = horizontal_pixels / total_pixels
            return horizontal_ratio
            
        except Exception:
            return 0.5

    def perform_multi_language_ocr(self, all_detections: Dict[str, Dict[str, Any]]):
        """Step 4: Correcting orientation and perform OCR on in-memory entity images."""
        logger.info(f"\nStep 4: Correcting Entity Orientation & Performing Multi-Language OCR")
        ocr_results = {}
        for card_name, card_data in all_detections.items():
            for detection in card_data['detections']:
                cropped_image = detection.get('cropped_image')
                entity_key = detection.get('entity_key')
                class_name = detection.get('class_name')

                if cropped_image is None or entity_key is None: continue

                logger.info(f"  Processing entity: {entity_key} (Class: {class_name})")
                lang_to_use = self.other_lang_code if class_name and class_name.endswith('_other_lang') else 'eng'
                
                processed_pil_img = self._correct_entity_orientation_and_preprocess(cropped_image, entity_key)

                if processed_pil_img:
                    try:
                        text = pytesseract.image_to_string(processed_pil_img, lang=lang_to_use, config='--psm 6')
                        extracted_text = ' '.join(text.split()).strip()
                        ocr_results[entity_key] = extracted_text
                    except Exception as e:
                        logger.error(f" OCR failed for {entity_key}: {e}")
                        ocr_results[entity_key] = None
        return ocr_results

    def organize_results_by_card_type(self, all_detections: Dict[str, Dict[str, Any]], ocr_results: Dict[str, str], confidence_threshold: float):
        """Step 5: Organizing final results from in-memory data structures."""
        logger.info("\nStep 5: Organizing final results")
        organized_results = {
            'front': {}, 'back': {},
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'confidence_threshold_used': confidence_threshold
            }
        }
        for card_name, card_data in all_detections.items():
            card_type = card_data['card_type']
            
            if card_name not in organized_results[card_type]:
                organized_results[card_type][card_name] = {'entities': {}}
            
            for detection in card_data['detections']:
                 entity_name = detection['class_name']
                 entity_key = detection.get('entity_key')
                 extracted_text = ocr_results.get(entity_key)

                 if entity_name not in organized_results[card_type][card_name]['entities']:
                      organized_results[card_type][card_name]['entities'][entity_name] = []
                 
                 organized_results[card_type][card_name]['entities'][entity_name].append({
                     'confidence': detection['confidence'], 
                     'bbox': detection['bbox'],
                     'extracted_text': extracted_text
                 })
        return organized_results
            
    def process_images(self, image_paths, user_id: str, task_id: str, confidence_threshold: float, verbose=True):
        """Main pipeline function to process images in-memory without saving any intermediate or final files."""
        try:
            if verbose:
                logger.info(f"[user_id={user_id}] Starting In-Memory Pipeline for task {task_id}")

            cropped_cards = self.detect_and_crop_cards(image_paths, confidence_threshold)
            if not cropped_cards.get('front', []) and not cropped_cards.get('back', []):
                logger.warning(f"[user_id={user_id}] No Aadhaar card detected in any provided images.")
                return {'error': 'no_aadhar_detected'}

            all_detections = self.detect_entities_in_cards(cropped_cards, confidence_threshold)

            self.crop_entities(all_detections)

            ocr_results = self.perform_multi_language_ocr(all_detections)

            organized_results = self.organize_results_by_card_type(all_detections, ocr_results, confidence_threshold)

            if verbose:
                logger.info(f"[user_id={user_id}] Pipeline processing completed successfully in memory.")
            return {'organized_results': organized_results}

        except ValueError as ve:
            logger.error(f"[user_id={user_id}] SECURITY ERROR in pipeline: {ve}")
            return {'error': str(ve), 'security_flagged': True, 'step': 'card_detection'}
        except Exception as e:
            logger.error(f"[user_id={user_id}] Unhandled error in pipeline: {e}\n{traceback.format_exc()}")
            return {'error': str(e), 'traceback': traceback.format_exc(), 'step': 'unknown'}


# --- FastAPI Application Setup ---

app = FastAPI(
    title="Aadhaar Processing API",
    description="API for synchronously processing Aadhaar cards using YOLO and multi-language OCR.",
    version="2.4.0-central-config"
)

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL1_PATH = BASE_DIR / os.environ.get("MODEL1_PATH")
    MODEL2_PATH = BASE_DIR / os.environ.get("MODEL2_PATH")
    DOWNLOAD_DIR = BASE_DIR / Path(os.environ.get("DOWNLOAD_DIR","workspace/AadharAuth/download"))
    
    IMAGE_NOT_SCAN_DIR = Path(os.environ.get("IMAGE_NOT_SCAN_DIR","workspace/AadharAuth/ImageNotScan"))
    SUMMARY_DATA_DIR = Path(os.environ.get("SUMMARY_DATA_DIR", "workspace/AadharAuth/data"))

    # --- SINGLE SOURCE OF TRUTH FOR CONFIDENCE THRESHOLD ---
    DEFAULT_CONFIDENCE_THRESHOLD = float(0.15)

config = Config()
pipeline: Optional[ComprehensiveAadhaarPipeline] = None

class AadhaarProcessRequest(BaseModel):
    user_id: str = "default_user"
    # --- Added new fields here ---
    name: Optional[str] = None
    mob_no: Optional[str] = None
    email_id: Optional[str] = None
    # ---------------------------
    front_url: HttpUrl
    back_url: HttpUrl
    # MODIFIED: The default value for the API request is now taken from the central config.
    confidence_threshold: float = Field(config.DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        config.IMAGE_NOT_SCAN_DIR.mkdir(parents=True, exist_ok=True)
        config.SUMMARY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # MODIFIED: Pipeline is now stateless regarding threshold, no need to pass it here.
        pipeline = ComprehensiveAadhaarPipeline(
            model1_path=str(config.MODEL1_PATH),
            model2_path=str(config.MODEL2_PATH),
        )
        logger.info("Pipeline models loaded successfully.")
    except Exception as e:
        logger.critical(f"Pipeline initialization failed: {e}", exc_info=True)
        sys.exit(1)

async def download_image(session: aiohttp.ClientSession, url: str, filepath: Path) -> bool:
    try:
        async with session.get(str(url), timeout=30) as response:
            response.raise_for_status()
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(await response.read())
            return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False

def extract_main_fields(organized_results: Dict[str, Any]) -> Dict[str, Any]:
    fields = ['aadharnumber', 'dob', 'gender', 'name', 'address', 'pincode']
    data = {key: "" for key in fields}
    for side in ['front', 'back']:
        for card in organized_results.get(side, {}).values():
            for field in fields:
                if field in card['entities'] and card['entities'][field]:
                    all_texts = [item.get('extracted_text', '') for item in card['entities'][field]]
                    first_valid_text = next((text for text in all_texts if text), '')
                    if first_valid_text:
                        data[field] = first_valid_text
                        
    if data.get('aadharnumber'):
        aad = data['aadharnumber']

        # detect masked aadhaar
        if re.search(r'X{4}', aad, re.IGNORECASE):
            return "masked_aadhar"
    
        # clean spaces for real aadhaar
        data['aadharnumber'] = aad.replace(" ", "")

    
    if data.get('dob'):
        # Extract all digit groups from dob string
        digit_groups = re.findall(r'\d+', data['dob'])
        digits_only = ''.join(digit_groups)
        # If 8 digits, try to parse as ddmmyyyy
        if len(digits_only) == 8:
            try:
                parsed_date = datetime.strptime(digits_only, '%d%m%Y')
                data['dob'] = parsed_date.strftime('%d-%m-%Y')
            except ValueError:
                data['dob'] = 'Invalid Format'
        else:
            # If dob contains a year (e.g., 'Year of Birth : 1991'), use the first 4-digit group as year
            year = next((g for g in digit_groups if len(g) == 4), None)
            if year:
                data['dob'] = year
            else:
                data['dob'] = 'Invalid Format'
                
    # --- Extract pincode from address if not valid ---
    if data['pincode']:
        digits_only = re.sub(r'\D', '', data['pincode'])
        if len(digits_only) == 6:
            data['pincode'] = digits_only
        else:
            # Try to extract from address
            match = re.search(r'\b\d{6}\b', data.get('address', ''))
            if match:
                data['pincode'] = match.group(0)
            else:
                data['pincode'] = 'Invalid Format'
    else:
        # If pincode is empty, try to extract from address
        match = re.search(r'\b\d{6}\b', data.get('address', ''))
        if match:
            data['pincode'] = match.group(0)
        else:
            data['pincode'] = 'Invalid Format'
    
    # --- Gender normalization (improved logic) ---
    if data['gender']:
        gender = data['gender'].strip().lower()
        if gender == 'male':
            data['gender'] = 'Male'
        elif gender == 'female':
            data['gender'] = 'Female'
        else:
            data['gender'] = 'Other'
        
    return data
############################
def load_data():
    data = {}
    if os.path.exists(csv_path):
        data['csv'] = pd.read_csv(csv_path)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data['json'] = pd.DataFrame(json.load(f))
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data['pkl'] = pickle.load(f)
    return data

def save_data(data):
    data['csv'].to_csv(csv_path, index=False)
    data['json'].to_json(json_path, orient='records', indent=2)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data['pkl'], f)
        


####################


@app.post("/verify_aadhar", response_class=JSONResponse, tags=["Aadhaar Processing"])
async def verify_aadhaar_sync(request: AadhaarProcessRequest):
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "data": {
                    "user_id": request.user_id,
                    "details": "Pipeline not initialized"
                },
                "message": "service_unavailable"
            }
        )

    task_id = hashlib.md5(f"{request.user_id}_{datetime.now().timestamp()}".encode()).hexdigest()
    user_download_dir = config.DOWNLOAD_DIR / request.user_id / task_id
    user_download_dir.mkdir(parents=True, exist_ok=True)
    front_path = user_download_dir / "front.jpg"
    back_path = user_download_dir / "back.jpg"
    logger.info(f"[user_id={request.user_id}] Started processing for task_id={task_id}")

    async with aiohttp.ClientSession() as session:
        downloads = await asyncio.gather(
            download_image(session, str(request.front_url), front_path),
            download_image(session, str(request.back_url), back_path)
        )

        # Check which images failed to download
        front_downloaded, back_downloaded = downloads

        if not front_downloaded and not back_downloaded:
            logger.error(f"[user_id={request.user_id}] Failed to download both front and back images.")
            shutil.rmtree(user_download_dir, ignore_errors=True)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "data": {
                        "user_id": request.user_id,
                        "details": "Unable to download both front and back images from provided URLs"
                    },
                    "message": "failed_to_both_images"
                }
            )
        elif not front_downloaded:
            logger.error(f"[user_id={request.user_id}] Failed to download front image.")
            shutil.rmtree(user_download_dir, ignore_errors=True)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "data": {
                        "user_id": request.user_id,
                        "details": "Unable to download front image from provided URL"
                    },
                    "message": "failed_to_front_images"
                }
            )
        elif not back_downloaded:
            logger.error(f"[user_id={request.user_id}] Failed to download back image.")
            shutil.rmtree(user_download_dir, ignore_errors=True)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "data": {
                        "user_id": request.user_id,
                        "details": "Unable to download back image from provided URL"
                    },
                    "message": "failed_to_back_images"
                }
            )

    # The threshold from the request (which defaults to the central config) is passed to the pipeline.
    result = pipeline.process_images(
        [str(front_path), str(back_path)],
        request.user_id,
        task_id,
        request.confidence_threshold
    )

    if 'error' in result:
        logger.error(f"[user_id={request.user_id}] Error during processing: {result.get('error')} (step: {result.get('step', 'unknown')})")
        shutil.rmtree(user_download_dir, ignore_errors=True)

        # Handle security violation
        if result.get("error") == "print_aadhar_detected":
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "data": {
                        "user_id": request.user_id,
                        "details": "Print Aadhaar card detected - security violation"
                    },
                    "message": "print_aadhar_detected"
                }
            )

        # Handle no detection error
        if result.get("error") == "no_aadhar_detected":
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "data": {
                        "user_id": request.user_id,
                        "details": "Neither front nor back card detected"
                    },
                    "message": "no_aadhar_detected"
                }
            )

        # Handle other processing errors
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "data": {
                    "user_id": request.user_id,
                    "details": "Error during image processing pipeline",
                    "error_step": result.get("step", "unknown")
                },
                "message": "image_processing_error"
            }
        )

    organized = result.get('organized_results', {})
    
    # --- Card presence validation ---
    detected_front = bool(organized.get('front')) and any(organized['front'])
    detected_back = bool(organized.get('back')) and any(organized['back'])
    
    if not detected_front and not detected_back:
        logger.warning(f"[user_id={request.user_id}] Neither front nor back card detected after processing.")
        shutil.rmtree(user_download_dir, ignore_errors=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {
                    "user_id": request.user_id,
                    "details": "Neither front nor back card detected"
                },
                "message": "no_aadhar_detected"
            }
        )
    elif not detected_front:
        logger.warning(f"[user_id={request.user_id}] Front card not detected after processing.")
        shutil.rmtree(user_download_dir, ignore_errors=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {
                    "user_id": request.user_id,
                    "details": "Front card not detected"
                },
                "message": "missing_front_card"
            }
        )
    elif not detected_back:
        logger.warning(f"[user_id={request.user_id}] Back card not detected after processing.")
        shutil.rmtree(user_download_dir, ignore_errors=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {
                    "user_id": request.user_id,
                    "details": "Back card not detected"
                },
                "message": "missing_back_card"
            }
        )

    main_data = extract_main_fields(organized)
    main_data['user_id'] = request.user_id
    
    # --- ADDED: Insert input fields into data to be saved ---
    # main_data['input_name'] = request.name
    main_data['mob_no'] = request.mob_no
    main_data['email_id'] = request.email_id
    # -------------------------------------------------------

    pkl_path = config.SUMMARY_DATA_DIR / "summary.pkl"
    csv_path = config.SUMMARY_DATA_DIR / "summary.csv"
    json_path = config.SUMMARY_DATA_DIR / "summary.json"
    
    aadhar_number = main_data.get('aadharnumber', '').replace(' ', '')
    if pkl_path.exists() and aadhar_number:
        try:
            with open(pkl_path, 'rb') as pf:
                all_data = pickle.load(pf)
            for entry in all_data:
                if entry.get('aadharnumber', '').replace(' ', '') == aadhar_number:
                    shutil.rmtree(user_download_dir, ignore_errors=True)
                    duplicate_data = {
                        "matched_id": entry.get("user_id", ""),
                        "name": entry.get("name", ""),
                        "aadharnumber": entry.get("aadharnumber", "")
                    }
                    logger.warning(f"Duplicate Aadhaar {aadhar_number} found for user {entry.get('user_id')}.")
                    return JSONResponse(
                        status_code=200,
                        content={
                            "success": True,
                            # "user_id": request.user_id,
                            "data": duplicate_data,
                            "message": "duplicate_aadhar_found"
                        }
                    )
        except Exception as e:
            logger.error(f"Failed to check PKL for duplicates: {e}")

    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(main_data, jf, indent=4, ensure_ascii=False)

    try:
        all_data = []
        if pkl_path.exists():
            with open(pkl_path, 'rb') as pf:
                all_data = pickle.load(pf)
        all_data.append(main_data)
        with open(pkl_path, 'wb') as pf:
            pickle.dump(all_data, pf)
    except Exception as e:
        logger.error(f"Failed to update PKL file: {e}")

    try:
        df = pd.DataFrame([main_data])
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
    except Exception as e:
        logger.error(f"Failed to update CSV file: {e}")
    
    logger.info(f"[user_id={request.user_id}] Processing complete. Cleaning up user folder.")
    shutil.rmtree(user_download_dir, ignore_errors=True)

    # Prepare response data
    response_data = {
        "user_id": request.user_id,
        # "input_name": request.name,      # Added to output
        # "mob_no": request.mob_no,  # Added to output
        # "email_id": request.email_id, # Added to output
        "aadharnumber": main_data.get("aadharnumber", ""),
        "dob": main_data.get("dob", ""),
        "gender": main_data.get("gender", ""),
        "name": main_data.get("name", ""),
        "address": main_data.get("address", ""),
        "pincode": main_data.get("pincode", "")
    }

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            # "user_id": request.user_id,
            "data": response_data,
            "message": "aadhar_data_saved"
        }
    )


@app.get("/user/{user_id}", tags=["Aadhaar Processing"])
def get_user_data(user_id: str):
    pkl_path = config.SUMMARY_DATA_DIR / "summary.pkl"

    user_data = []

    if pkl_path.exists():
        try:
            with open(pkl_path, 'rb') as pf:
                pkl_data = pickle.load(pf)
            user_data = [entry for entry in pkl_data if entry.get("user_id") == user_id]
        except Exception as e:
            logger.error(f"Error reading PKL: {e}")
            user_data = []

    if not user_data:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                # "user_id": user_id,
                "data": {
                    "user_id": request.user_id if 'request' in locals() else user_id,
                    "details": "No records found for this user"
                },
                "message": "user_not_found"
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            # "user_id": user_id,
            "data": {
                "user_id": user_id,
                "records_found": len(user_data),
                "user_data": user_data
            },
            "message": "user_data_found"
        }
    )


@app.delete("/user/{user_id}", tags=["Aadhaar Processing"])
def delete_user_data(user_id: str):
    pkl_path = config.SUMMARY_DATA_DIR / "summary.pkl"
    csv_path = config.SUMMARY_DATA_DIR / "summary.csv"
    json_path = config.SUMMARY_DATA_DIR / "summary.json"

    records_deleted = 0
    storage_formats_updated = []

    # --- PKL ---
    if pkl_path.exists():
        try:
            with open(pkl_path, 'rb') as pf:
                pkl_data = pickle.load(pf)
            original_len = len(pkl_data)
            new_pkl_data = [entry for entry in pkl_data if entry.get("user_id") != user_id]
            deleted_count = original_len - len(new_pkl_data)
            if deleted_count > 0:
                records_deleted += deleted_count
                storage_formats_updated.append("pkl")
                with open(pkl_path, 'wb') as pf:
                    pickle.dump(new_pkl_data, pf)
        except Exception as e:
            logger.error(f"Error updating PKL: {e}")

    # --- CSV ---
    if csv_path.exists():
        try:
            csv_df = pd.read_csv(csv_path)
            csv_df["user_id"] = csv_df["user_id"].astype(str)
            original_len = len(csv_df)
            new_csv_df = csv_df[csv_df["user_id"] != str(user_id)]
            deleted_count = original_len - len(new_csv_df)
            if deleted_count > 0:
                storage_formats_updated.append("csv")
                new_csv_df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.error(f"Error updating CSV: {e}")
            
    # --- JSON ---
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)
            if json_data.get("user_id") == user_id:
                storage_formats_updated.append("json")
                os.remove(json_path)
        except Exception as e:
            logger.error(f"Error deleting JSON: {e}")

    if records_deleted == 0:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "data": {
                    "user_id": user_id,
                    "details": "No records found for this user to delete"
                },
                "message": "user_not_found"
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "data": {
                "user_id": user_id,
                "records_deleted": records_deleted,
                "storage_formats_updated": storage_formats_updated
            },
            "message": "user_data_deleted"
        }
    )


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Checks the health of the service, including pipeline initialization and device availability.
    """
    if pipeline and hasattr(pipeline, 'device'):
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "pipeline_status": "initialized",
                    "inference_device": pipeline.device,
                    "torch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                },
                "message": "service_healthy"
            }
        )
    else:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "data": {
                    "pipeline_status": "not_initialized",
                    "details": "The main processing pipeline is not available"
                },
                "message": "service_unhealthy"
            }
        )
if __name__ == "__main__":
    uvicorn.run("gpupipelineupdated:app", host="0.0.0.0", port=8111, reload=True)