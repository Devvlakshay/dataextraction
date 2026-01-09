"""
Qwen Fallback Module for Aadhaar and PAN Card Data Extraction
Uses Qwen3-VL:8b from Ollama for vision-based extraction when primary pipeline fails
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, List, Literal
from PIL import Image
import io

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"
QWEN_MODEL = "qwen3-vl:8b"  # Using 8B model - 2B is too weak


class QwenFallback:
    """
    Fallback processor using Qwen3-VL:8b from Ollama for document data extraction
    when the primary YOLO-based pipeline fails to detect entities.
    """

    def __init__(self, ollama_url: str = OLLAMA_API_URL, model: str = QWEN_MODEL):
        self.ollama_url = ollama_url
        self.model = model
        logger.info(f"QwenFallback initialized with model: {self.model}")

    def _encode_image_to_base64(self, image_path: str, max_size: int = 800) -> str:
        """Convert image file to base64 string, resizing if needed to reduce payload"""
        try:
            with Image.open(image_path) as img:
                # Resize if too large
                width, height = img.size
                if width > max_size or height > max_size:
                    ratio = min(max_size / width, max_size / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"[QwenFallback] Resized image from {width}x{height} to {new_size[0]}x{new_size[1]}")
                
                # Convert to RGB if needed (for PNG with alpha)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"[QwenFallback] Error encoding image: {e}")
            # Fallback to direct read
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")

    def _call_ollama(self, prompt: str, images: List[str]) -> Optional[str]:
        """P
        Call Ollama API with Qwen3-VL model for vision tasks
        
        Args:
            prompt: Text prompt for the model
            images: List of base64 encoded images
            
        Returns:
            Model response text or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": images,
                "stream": False,
                "think": False,  # Disable thinking mode for direct response
                "options": {
                    "temperature": 0.1,  # Low temperature for more deterministic output
                    "num_predict": 2048  # Increased token limit
                }
            }
            
            logger.info(f"[QwenFallback] Calling Ollama with model: {self.model}")
            logger.info(f"[QwenFallback] Number of images: {len(images)}")
            logger.info(f"[QwenFallback] Waiting for response (timeout: 120s)...")
            response = requests.post(self.ollama_url, json=payload, timeout=120)  # Increased timeout for 8B
            response.raise_for_status()
            logger.info(f"[QwenFallback] Response received from Ollama")
            
            result = response.json()
            
            # Check for errors in the response
            if "error" in result:
                logger.error(f"[QwenFallback] Ollama returned error: {result.get('error')}")
                return None
            
            # Try 'response' field first, then 'thinking' field (some models use 'thinking')
            response_text = result.get("response", "") or result.get("thinking", "")
            logger.info(f"[QwenFallback] Response length: {len(response_text)} chars")
            
            # Log the full result for debugging
            logger.debug(f"[QwenFallback] Full Ollama result: {result}")
            
            if response_text:
                logger.info(f"[QwenFallback] Raw response (first 200 chars): {response_text[:200]}")
            else:
                logger.warning(f"[QwenFallback] Empty response received from Ollama")
                logger.warning(f"[QwenFallback] Full result keys: {result.keys()}")
            
            # Log ALL fields from result to debug where the actual JSON is
            logger.info(f"[QwenFallback] DEBUG - Full result dump:")
            for key, value in result.items():
                if key not in ['context', 'total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration']:
                    # Skip timing/context fields, log everything else
                    logger.info(f"[QwenFallback]   {key}: {str(value)[:500]}")
            
            return response_text
            
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama. Ensure Ollama is running with `ollama serve`")
            return None
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after 120 seconds")
            logger.error(f"The model might be too slow or stuck processing the images")
            logger.error(f"Consider: 1) Using a smaller model, 2) Reducing image size, 3) Checking Ollama logs")
            return None
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return None

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from model response, handling various formats"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                logger.info(f"[QwenFallback] Parsed JSON: {parsed}")
                return parsed
            logger.warning(f"[QwenFallback] No JSON found in response")
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {}

    def extract_aadhaar_data(
        self,
        front_image_path: str,
        back_image_path: str,
        missing_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract Aadhaar card data using Qwen vision model
        
        Args:
            front_image_path: Path to front side image
            back_image_path: Path to back side image
            missing_fields: List of specific fields that need extraction (if None, extract all)
            
        Returns:
            Dictionary with extracted Aadhaar data
        """
        logger.info(f"[QwenFallback] Processing Aadhaar card images")
        
        # Encode images with resizing
        images = []
        if Path(front_image_path).exists():
            logger.info(f"[QwenFallback] Front image found: {front_image_path}")
            front_b64 = self._encode_image_to_base64(front_image_path)
            logger.info(f"[QwenFallback] Front image base64 size: {len(front_b64) / 1024:.1f} KB")
            images.append(front_b64)
        else:
            logger.warning(f"[QwenFallback] Front image NOT found: {front_image_path}")
            
        if Path(back_image_path).exists():
            logger.info(f"[QwenFallback] Back image found: {back_image_path}")
            back_b64 = self._encode_image_to_base64(back_image_path)
            logger.info(f"[QwenFallback] Back image base64 size: {len(back_b64) / 1024:.1f} KB")
            images.append(back_b64)
        else:
            logger.warning(f"[QwenFallback] Back image NOT found: {back_image_path}")
        
        if not images:
            logger.error("[QwenFallback] No valid images found for Aadhaar extraction")
            return {"error": "no_images_found"}

        # Build prompt based on missing fields
        if missing_fields:
            fields_str = ", ".join(missing_fields)
            field_instruction = f"Focus on extracting these specific fields: {fields_str}"
        else:
            field_instruction = "Extract all available fields"

        prompt = f"""Analyze the Aadhaar card images and extract data in JSON format.

Return ONLY this JSON structure:
{{
    "aadharnumber": "12-digit number",
    "name": "person name",
    "dob": "DD-MM-YYYY",
    "gender": "Male/Female/Other",
    "address": "full address",
    "pincode": "6-digit code"
}}

Extract the requested fields: {fields_str if missing_fields else "all fields"}"""

        response = self._call_ollama(prompt, images)
        
        if response is None:
            logger.error("[QwenFallback] Failed to connect to Ollama for Aadhaar extraction")
            return {"error": "ollama_connection_failed"}
        
        if not response or response.strip() == "":
            logger.error("[QwenFallback] Ollama returned empty response for Aadhaar extraction")
            return {"error": "ollama_empty_response"}

        extracted_data = self._parse_json_response(response)
        
        # Post-process the data
        extracted_data = self._postprocess_aadhaar_data(extracted_data)
        
        logger.info(f"[QwenFallback] Aadhaar extraction complete: {list(extracted_data.keys())}")
        return extracted_data

    def _postprocess_aadhaar_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate extracted Aadhaar data"""
        result = {
            "aadharnumber": "",
            "name": "",
            "dob": "",
            "gender": "",
            "address": "",
            "pincode": ""
        }
        
        # Aadhaar number
        if data.get("aadharnumber"):
            aadhar = str(data["aadharnumber"]).replace(" ", "").replace("-", "")
            if "XXXX" in aadhar.upper() or "xxxx" in aadhar:
                result["aadharnumber"] = "masked_aadhar"
            elif len(re.sub(r'\D', '', aadhar)) == 12:
                result["aadharnumber"] = re.sub(r'\D', '', aadhar)
            else:
                result["aadharnumber"] = aadhar

        # Name
        if data.get("name"):
            result["name"] = str(data["name"]).strip()

        # DOB
        if data.get("dob"):
            dob = str(data["dob"]).strip()
            # Try to normalize date format
            digits = re.sub(r'\D', '', dob)
            if len(digits) == 8:
                result["dob"] = f"{digits[:2]}-{digits[2:4]}-{digits[4:]}"
            elif len(digits) == 4:  # Year only
                result["dob"] = digits
            else:
                result["dob"] = dob

        # Gender
        if data.get("gender"):
            gender = str(data["gender"]).strip().lower()
            if gender in ["male", "m"]:
                result["gender"] = "Male"
            elif gender in ["female", "f"]:
                result["gender"] = "Female"
            else:
                result["gender"] = "Other"

        # Address
        if data.get("address"):
            result["address"] = str(data["address"]).strip()

        # Pincode
        if data.get("pincode"):
            pincode = re.sub(r'\D', '', str(data["pincode"]))
            if len(pincode) == 6:
                result["pincode"] = pincode
            else:
                # Try to extract from address
                match = re.search(r'\b\d{6}\b', result.get("address", ""))
                if match:
                    result["pincode"] = match.group(0)
        else:
            # Try to extract from address
            match = re.search(r'\b\d{6}\b', result.get("address", ""))
            if match:
                result["pincode"] = match.group(0)

        return result

    def extract_pancard_data(self, image_path: str, missing_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract PAN card data using Qwen vision model
        
        Args:
            image_path: Path to PAN card image
            missing_fields: List of specific fields that need extraction (if None, extract all)
            
        Returns:
            Dictionary with extracted PAN card data
        """
        logger.info(f"[QwenFallback] Processing PAN card image: {image_path}")
        
        if not Path(image_path).exists():
            logger.error(f"[QwenFallback] PAN card image NOT found: {image_path}")
            return {"error": "image_not_found"}
        
        logger.info(f"[QwenFallback] PAN card image found: {image_path}")

        # Encode image with resizing
        image_b64 = self._encode_image_to_base64(image_path)
        logger.info(f"[QwenFallback] PAN image base64 size: {len(image_b64) / 1024:.1f} KB")

        # Build prompt based on missing fields
        if missing_fields:
            fields_str = ", ".join(missing_fields)
            field_instruction = f"Focus on extracting these specific fields: {fields_str}"
        else:
            field_instruction = "Extract all available fields"

        prompt = f"""You are an expert document analyzer. Analyze this PAN (Permanent Account Number) card image and extract the information.
{field_instruction}

IMPORTANT: Return ONLY a valid JSON object with these exact keys (use empty string "" if not found):
{{
    "pan_number": "10-character PAN number (format: 5 letters, 4 digits, 1 letter like ABCDE1234F)",
    "name": "Full name as shown on card",
    "father_name": "Father's name as shown on card",
    "dob": "Date of birth in DD-MM-YYYY format"
}}

Rules:
1. PAN number format: 5 uppercase letters + 4 digits + 1 uppercase letter (e.g., ABCDE1234F)
2. Name should be exactly as printed on the card
3. Father's name is usually shown as "Father's Name" on the card
4. Date of birth should be in DD-MM-YYYY format
5. Return ONLY the JSON, no explanations

Analyze the image now:"""

        response = self._call_ollama(prompt, [image_b64])
        
        if not response:
            logger.error("[QwenFallback] No response from Ollama for PAN extraction")
            return {"error": "ollama_no_response"}

        extracted_data = self._parse_json_response(response)
        
        # Post-process the data
        extracted_data = self._postprocess_pancard_data(extracted_data)
        
        logger.info(f"[QwenFallback] PAN extraction complete: {list(extracted_data.keys())}")
        return extracted_data

    def _postprocess_pancard_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate extracted PAN card data"""
        result = {
            "pan_number": "",
            "name": "",
            "father_name": "",
            "dob": ""
        }
        
        # PAN number
        if data.get("pan_number"):
            pan = str(data["pan_number"]).upper().replace(" ", "").replace("-", "")
            # Validate PAN format: 5 letters + 4 digits + 1 letter
            pan_pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]$'
            if re.match(pan_pattern, pan):
                result["pan_number"] = pan
            else:
                # Try to extract valid PAN from string
                match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', pan)
                if match:
                    result["pan_number"] = match.group()
                else:
                    result["pan_number"] = pan  # Keep as-is for review

        # Name
        if data.get("name"):
            result["name"] = str(data["name"]).strip()

        # Father's name
        if data.get("father_name"):
            result["father_name"] = str(data["father_name"]).strip()

        # DOB
        if data.get("dob"):
            dob = str(data["dob"]).strip()
            # Try to normalize date format
            digits = re.sub(r'\D', '', dob)
            if len(digits) == 8:
                result["dob"] = f"{digits[:2]}-{digits[2:4]}-{digits[4:]}"
            else:
                result["dob"] = dob

        return result

    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and the model is available"""
        try:
            # Check Ollama server
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error("Ollama server not responding")
                return False
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if not any(self.model in name for name in model_names):
                logger.warning(f"Model {self.model} not found. Available: {model_names}")
                logger.info(f"Please run: ollama pull {self.model}")
                return False
            
            logger.info(f"Ollama ready with model: {self.model}")
            return True
            
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Start it with: ollama serve")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return False


def check_missing_fields_aadhaar(data: Dict[str, Any]) -> List[str]:
    """
    Check which Aadhaar fields are missing or empty
    
    Args:
        data: Dictionary with extracted Aadhaar data
        
    Returns:
        List of missing field names
    """
    required_fields = ["aadharnumber", "name", "dob", "gender", "address", "pincode"]
    missing = []
    
    for field in required_fields:
        value = data.get(field, "")
        if not value or value == "" or value == "Invalid Format":
            missing.append(field)
    
    return missing


def check_missing_fields_pancard(data: Dict[str, Any]) -> List[str]:
    """
    Check which PAN card fields are missing or empty
    
    Args:
        data: Dictionary with extracted PAN card data
        
    Returns:
        List of missing field names
    """
    required_fields = ["pan_number", "name", "father_name", "dob"]
    missing = []
    
    for field in required_fields:
        value = data.get(field, "")
        if not value or value == "":
            missing.append(field)
    
    return missing


def should_use_fallback_aadhaar(data: Dict[str, Any], threshold: int = 2) -> bool:
    """
    Determine if Qwen fallback should be used for Aadhaar
    
    Args:
        data: Extracted data from primary pipeline
        threshold: Minimum number of missing fields to trigger fallback
        
    Returns:
        True if fallback should be used
    """
    missing = check_missing_fields_aadhaar(data)
    return len(missing) >= threshold


def should_use_fallback_pancard(data: Dict[str, Any], threshold: int = 1) -> bool:
    """
    Determine if Qwen fallback should be used for PAN card
    
    Args:
        data: Extracted data from primary pipeline
        threshold: Minimum number of missing fields to trigger fallback
        
    Returns:
        True if fallback should be used
    """
    missing = check_missing_fields_pancard(data)
    return len(missing) >= threshold


def merge_extraction_results(
    primary_data: Dict[str, Any],
    fallback_data: Dict[str, Any],
    doc_type: Literal["aadhaar", "pancard"]
) -> Dict[str, Any]:
    """
    Merge primary pipeline results with fallback results
    Priority: Primary data takes precedence unless empty
    
    Args:
        primary_data: Data from primary YOLO pipeline
        fallback_data: Data from Qwen fallback
        doc_type: Type of document ("aadhaar" or "pancard")
        
    Returns:
        Merged data dictionary
    """
    if doc_type == "aadhaar":
        fields = ["aadharnumber", "name", "dob", "gender", "address", "pincode"]
    else:  # pancard
        fields = ["pan_number", "name", "father_name", "dob"]
    
    merged = {}
    for field in fields:
        primary_value = primary_data.get(field, "")
        fallback_value = fallback_data.get(field, "")
        
        # Use primary if it has valid value, otherwise use fallback
        if primary_value and primary_value != "" and primary_value != "Invalid Format":
            merged[field] = primary_value
        else:
            merged[field] = fallback_value
    
    # Keep any additional fields from primary data
    for key, value in primary_data.items():
        if key not in merged:
            merged[key] = value
    
    return merged


# Singleton instance for easy import
qwen_fallback = QwenFallback()


if __name__ == "__main__":
    # Test the fallback module
    fallback = QwenFallback()
    
    # Check Ollama status
    if fallback.check_ollama_status():
        print("Qwen Fallback module is ready!")
    else:
        print("Please ensure Ollama is running with the Qwen model")
        print(f"Run: ollama pull {QWEN_MODEL}")
