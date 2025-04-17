import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import shutil
import os
import uuid
import logging
import re
import io # Needed for reading file bytes

# Image Processing imports (Optional for Google Vision but can keep)
# from PIL import Image # No longer strictly needed if not using Tesseract's PIL conversion
import cv2 # Still useful if you want to add preprocessing back later
import numpy as np

# spaCy import
import spacy

# Google Cloud Vision Import
from google.cloud import vision

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tesseract configuration commented out (using Google Vision)
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Load spaCy Model ---
# Load the model once when the application starts for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    logger.error(
        "spaCy model 'en_core_web_sm' not found. "
        "Please run 'python -m spacy download en_core_web_sm' "
        "in your virtual environment."
    )
    nlp = None # Extraction function will check for this

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Business Card Scanner API",
    description="Upload a business card image to extract information using Google Cloud Vision OCR + spaCy NER.",
    version="1.3.0", # Updated version number
)

# --- Pydantic Model for Response (Includes job_title, excludes raw_text) ---
class CardInfo(BaseModel):
    company_name: str | None = None
    company_type: str | None = None
    person_name: str | None = None
    job_title: str | None = None # Added job title field
    phone_number: str | None = None
    email: str | None = None
    address: str | None = None
    # raw_text field is removed

# --- File Handling Setup ---
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Image Preprocessing Function (Just reads bytes for Google Vision) ---
def preprocess_image(image_path: str) -> bytes:
    """Loads image and returns its byte content."""
    try:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        logger.info(f"Read image bytes for Google Vision: {image_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading image file {image_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read image file: {e}")

# --- Google Cloud Vision OCR Function ---
def perform_ocr_google(image_path: str) -> str:
    """Performs OCR on an image file using Google Cloud Vision API."""
    try:
        logger.info(f"Performing Google Cloud Vision OCR on: {image_path}")
        client = vision.ImageAnnotatorClient()

        content = preprocess_image(image_path)
        image = vision.Image(content=content)

        response = client.document_text_detection(image=image)

        if response.error.message:
            logger.error(f"Google Cloud Vision API error: {response.error.message}")
            raise HTTPException(
                status_code=500,
                detail=f"Google Cloud Vision API error: {response.error.message}"
            )

        if response.full_text_annotation:
            extracted_text = response.full_text_annotation.text
            logger.info("Google Cloud Vision OCR completed successfully.")
            return extracted_text
        else:
            logger.warning("Google Cloud Vision did not detect any text.")
            return ""
    except HTTPException as http_exc: # Pass through HTTP exceptions from preprocessing
        raise http_exc
    except Exception as e:
        logger.error(f"Error during Google Cloud OCR processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")

# --- Information Extraction Function (spaCy + Regex + Job Title) ---
def extract_information_spacy(text: str) -> dict:
    """
    Extracts information using spaCy NER + Regex, including Job Title.
    (Corrected version ensuring break statement is within its loop)
    """
    logger.info("Starting information extraction with Job Title detection.")

    if nlp is None:
        logger.error("spaCy model 'nlp' object is None. Cannot perform NER.")
        return {
            "company_name": None, "company_type": None, "person_name": None,
            "job_title": None, "phone_number": None, "email": None, "address": None,
            "_error": "spaCy model not loaded"
        }

    data = {
        "company_name": None, "company_type": None, "person_name": None,
        "job_title": None, "phone_number": None, "email": None, "address": None,
    }

    # --- Text Cleaning ---
    text = re.sub(r'\s{2,}', ' ', text.strip())
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    lines_lower = [line.lower() for line in lines]

    # --- Regex Definitions ---
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_regex = r'(?:(?:Tel|Phone|Mobile|Mob|Fax|F)[:\s]*)?(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,}[-.\s]?\d{3,}\b'
    phone_prefix_regex = r'^[MFTWECP][\s:+]+'

    # --- Initial Regex Extraction ---
    emails_found = re.findall(email_regex, text)
    phones_found = re.findall(phone_regex, text)

    if emails_found:
        data['email'] = emails_found[0]
        logger.info(f"Regex assigned email: {data['email']}")

    if phones_found: # Score and select best phone
        scored_phones = []
        for phone in phones_found:
            cleaned_phone = re.sub(r'[^\d+() -]', '', phone).strip()
            digit_count = sum(c.isdigit() for c in cleaned_phone)
            if digit_count >= 7:
                score = digit_count
                for line_idx, line in enumerate(lines):
                    if cleaned_phone in line:
                         if re.match(phone_prefix_regex, line.strip(), re.IGNORECASE): score += 10
                         if line_idx > len(lines) / 2: score += 2
                         break # Correct break: from inner line search loop
                scored_phones.append((score, cleaned_phone))
        if scored_phones:
            scored_phones.sort(key=lambda x: x[0], reverse=True)
            data['phone_number'] = scored_phones[0][1]
            logger.info(f"Regex assigned phone (score {scored_phones[0][0]}): {data['phone_number']}")

    # --- spaCy NER Processing ---
    doc = nlp(text)
    persons = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent for ent in doc.ents if ent.label_ == "ORG"]
    locations = [ent for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]]
    logger.info(f"spaCy found {len(persons)} PERSON, {len(orgs)} ORG, {len(locations)} location entities.")

    # --- Assign Person Name ---
    person_name_line_index = -1
    person_name_assigned = None
    if persons:
        scored_persons = []
        for i, person_ent in enumerate(persons):
            name = person_ent.text.strip()
            if (data['email'] and name in data['email']) or \
               (data['phone_number'] and name in data['phone_number']) or \
               sum(c.isdigit() for c in name) > 0 or \
               len(name.split()) > 4 or len(name.split()) < 2: continue
            score = len(name)
            if re.fullmatch(r'^[A-Z][a-z]+(?:\s+([A-Z][a-z.]+|[A-Z]\.)){0,2}$', name): score += 10
            line_idx = -1
            try: start_char = person_ent.start_char; line_idx = text[:start_char].count('\n');
            except Exception: pass
            if line_idx != -1 and line_idx < len(lines) / 3: score += 5
            scored_persons.append((score, name, line_idx))
        if scored_persons:
             scored_persons.sort(key=lambda x: x[0], reverse=True)
             data['person_name'] = scored_persons[0][1]
             person_name_line_index = scored_persons[0][2]
             person_name_assigned = data['person_name']
             logger.info(f"spaCy assigned PERSON (score {scored_persons[0][0]}): {data['person_name']} around line {person_name_line_index}")

    # --- Assign Job Title ---
    job_title_keywords = ['Officer', 'Manager', 'Director', 'Assistant', 'Engineer', 'Sales', 'Marketing', 'CEO', 'CTO', 'President', 'Specialist', 'Consultant', 'Analyst', 'Services', 'Executive', 'Representative', 'Procurement']
    job_title_assigned = None
    if person_name_assigned and person_name_line_index != -1:
        start_search_index = max(0, person_name_line_index)
        end_search_index = min(len(lines), start_search_index + 3)
        for i in range(start_search_index, end_search_index): # Loop for job title search
            line = lines[i]; line_lower = lines_lower[i]
            if any(keyword.lower() in line_lower for keyword in job_title_keywords):
                 if line != person_name_assigned and \
                    not (data['email'] and data['email'] in line) and \
                    not (data['phone_number'] and data['phone_number'] in line) and \
                    len(line.split()) < 6:
                      data['job_title'] = line
                      job_title_assigned = data['job_title']
                      logger.info(f"Assigned Job Title: {data['job_title']}")
                      break # Correct break: Stop searching for title once found
        # No 'else' needed here, if loop finishes without break, no title assigned

    # --- Assign Company Name & Type ---
    company_name_assigned = None
    if orgs:
        scored_orgs = []
        company_suffixes = ['Limited', 'Ltd', 'Inc', 'Corp', 'Solutions', 'Group', 'PLC']; company_keywords = ['Bank', 'Consulting', 'Media', 'Tech', 'Logistics']
        for org_ent in orgs: # Loop through potential orgs
            org_name = org_ent.text.strip(); type_found_in_scoring = None
            if (person_name_assigned and org_name == person_name_assigned) or len(org_name) < 4: continue
            score = len(org_name)
            for suffix in company_suffixes: # Inner loop to check suffixes for scoring
                 if re.search(r'\b' + re.escape(suffix) + r'\b', org_name, re.IGNORECASE):
                     score += 20; type_found_in_scoring = suffix;
                     break # Correct break: from suffix checking loop for scoring
            if not type_found_in_scoring:
                for keyword in company_keywords: # Inner loop to check keywords for scoring
                    if re.search(r'\b' + re.escape(keyword) + r'\b', org_name, re.IGNORECASE):
                        score += 10;
                        break # Correct break: from keyword checking loop for scoring
            scored_orgs.append((score, org_name, type_found_in_scoring)) # Note: type_found_in_scoring is just a hint

        if scored_orgs:
             scored_orgs.sort(key=lambda x: x[0], reverse=True)
             best_org_name = scored_orgs[0][1]
             best_org_type_guess = scored_orgs[0][2] # This was the suffix found during scoring (if any)
             final_type = None; cleaned_org_name = best_org_name

             # Re-check the chosen best_org_name accurately for the suffix
             if best_org_type_guess: # Start check with the guess from scoring
                 match = re.search(r'\b' + re.escape(best_org_type_guess) + r'\b', best_org_name, re.IGNORECASE);
                 if match: final_type = match.group(0)

             if not final_type: # If scoring guess wasn't right or none found, check all suffixes again
                  for suffix in company_suffixes: # <<< This is the relevant loop for the Pylance error
                      match = re.search(r'\b' + re.escape(suffix) + r'\b', best_org_name, re.IGNORECASE);
                      if match:
                          final_type = match.group(0)
                          # --- THIS IS THE BREAK STATEMENT IN QUESTION ---
                          # It is correctly inside the 'for suffix in company_suffixes:' loop.
                          break
             # --- END OF SECTION FOR THE BREAK ---

             if final_type:
                 data['company_type'] = final_type;
                 cleaned_org_name = re.sub(r'\b' + re.escape(final_type) + r'\b', '', cleaned_org_name, flags=re.IGNORECASE).strip(' ,')
             data['company_name'] = cleaned_org_name.strip();
             company_name_assigned = data['company_name'];
             logger.info(f"spaCy assigned ORG (score {scored_orgs[0][0]}): {data['company_name']} (Type: {data['company_type']})")

    # --- Address Assembly ---
    # (Keep Address logic from previous version)
    address_keywords = ['Plot', 'Avenue', 'Street', 'Road', 'Drive', 'P.O. Box', 'Box', 'Floor', 'Suite', 'Ste', 'Building', 'St', 'Rd', 'Ave', 'Ln']
    address_parts = []; location_texts = {loc.text.strip().lower() for loc in locations if len(loc.text.strip()) > 2}
    for line in lines:
        is_assigned_field = (person_name_assigned and line == person_name_assigned) or \
                           (job_title_assigned and line == job_title_assigned) or \
                           (company_name_assigned and data['company_type'] is None and line == company_name_assigned) or \
                           (company_name_assigned and data['company_type'] and data['company_type'] in line and company_name_assigned in line) or \
                           (data['email'] and line == data['email']) or \
                           (data['phone_number'] and data['phone_number'] in line)
        if is_assigned_field: continue
        line_cleaned_prefixes = re.sub(phone_prefix_regex, '', line).strip()
        if re.fullmatch(phone_regex, line_cleaned_prefixes) or re.fullmatch(email_regex, line): continue
        if any(keyword.lower() in line.lower() for keyword in job_title_keywords) and \
           not any(addr_kw.lower() in line.lower() for addr_kw in address_keywords): continue
        line_lower = line.lower()
        if any(keyword.lower() in line_lower for keyword in address_keywords) or \
           re.search(r'\b\d{4,}\b', line) or \
           any(loc_text in line_lower for loc_text in location_texts):
             address_parts.append(line.strip(':., '))
    unique_address_parts = []; seen = set()
    for part in address_parts: part_lower = part.lower();
    if part_lower not in seen: unique_address_parts.append(part); seen.add(part_lower)
    if unique_address_parts:
        address_string = ", ".join(unique_address_parts);
        if company_name_assigned and company_name_assigned in address_string: address_string = address_string.replace(company_name_assigned, '').strip(' ,')
        if data['company_type'] and data['company_type'] in address_string: address_string = address_string.replace(data['company_type'], '').strip(' ,')
        address_string = re.sub(r'\s{2,}', ' ', address_string).strip(' ,')
        data['address'] = address_string
        logger.info(f"Assembled address (filtered): {data['address']}")

    logger.info(f"Information extraction finished. Data: {data}")
    return data


# --- API Endpoint (Corrected try/finally structure) ---
@app.post("/scan-card/", response_model=CardInfo)
async def scan_business_card(file: UploadFile = File(..., description="Business card image file (JPEG, PNG recommended).")):
    """
    Receives a business card image, performs OCR (Google Vision), extracts info, returns JSON.
    """
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file.content_type}'. Please upload an image.")

    temp_file_path = None # Initialize to ensure cleanup check works
    try:
        # --- File Saving and Processing ---
        file_extension = os.path.splitext(file.filename)[1] or '.png' # Ensure extension
        temp_file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_extension}")
        logger.info(f"Received file: {file.filename}. Saving temporarily to: {temp_file_path}")

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Temporary file saved successfully.")

        # --- OCR ---
        raw_text_from_ocr = perform_ocr_google(temp_file_path)

        # --- Information Extraction ---
        extracted_data = extract_information_spacy(raw_text_from_ocr)

        # Check for internal extraction errors (like spaCy model missing)
        if isinstance(extracted_data, dict) and "_error" in extracted_data:
             error_msg = extracted_data.get("_error", "Unknown extraction error")
             logger.error(f"Information extraction failed: {error_msg}")
             raise HTTPException(status_code=500, detail=f"Information extraction failed: {error_msg}")

        # --- Return Response ---
        # Pydantic model validation handles the exclusion of raw_text and inclusion of job_title
        return CardInfo(**extracted_data)

    except HTTPException as http_exc:
        # Re-raise specific HTTP errors (from OCR, preprocessing, etc.)
        logger.warning(f"HTTP Exception during processing: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        # --- Cleanup: Ensure the temporary file is always deleted ---
        # This block now correctly corresponds to the outer try
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Temporary file deleted: {temp_file_path}")
            except OSError as e:
                logger.error(f"Error deleting temporary file {temp_file_path}: {e}")
        # Ensure the uploaded file stream is closed
        if file and hasattr(file, 'file') and not file.file.closed:
            try:
                 await file.close()
                 logger.info("Closed uploaded file stream.")
            except Exception as e:
                 logger.error(f"Error closing uploaded file stream: {e}")


# --- Root endpoint ---
@app.get("/", include_in_schema=False)
def read_root():
    """Provides a simple welcome message for the API root."""
    return {"message": "Business Card Scanner API v1.3 (Google Vision + Job Title) is running. Go to /docs for API documentation."}

# --- Main execution block ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Business Card Scanner API server (v1.3 Google Vision + Job Title)...")
    # Check environment variable
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
         logger.warning("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Google Cloud Vision API calls will likely fail.")
    else:
         logger.info("GOOGLE_APPLICATION_CREDENTIALS environment variable is set.")
    # Check spaCy model
    if nlp is None:
        logger.warning("spaCy model failed to load during startup. NER features will be unavailable.")

    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)