# --- Imports ---
import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, ValidationError # Added ValidationError
import shutil
import os
import uuid
import logging
import re
import io # Needed for reading file bytes
import json # Needed for loading credentials from JSON string
from dotenv import load_dotenv # Import dotenv
from typing import Annotated, List, Tuple, Dict, Any # For type hinting

# spaCy import
import spacy

# Google Cloud Vision Import
from google.cloud import vision
from google.oauth2 import service_account # For loading creds from JSON content

# --- Load Environment Variables ---
# Load .env file BEFORE accessing os.environ below
load_dotenv()
logger_init = logging.getLogger(__name__) # Temp logger for startup
logger_init.info("Attempted to load environment variables from .env file.")

# --- Configuration ---
# Use basicConfig AFTER load_dotenv potentially sets logging config vars
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Main logger

# --- Load spaCy Model ---
NLP_MODEL = None # Initialize global variable for the model
try:
    NLP_MODEL = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    logger.error(
        "spaCy model 'en_core_web_sm' not found. "
        "Please run 'python -m spacy download en_core_web_sm' "
        "in your virtual environment."
    )
    # Extraction function will check if NLP_MODEL is None

# --- Constants ---
# Regex Patterns (Validated)
EMAIL_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_REGEX = r'(?:(?:Tel|Phone|Mobile|Mob|Fax|F)[:\s]*)?(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,}[-.\s]?\d{3,}\b' # Corrected last part
PHONE_PREFIX_REGEX = r'^[MFTWECP][\s:+]+'
WEBSITE_REGEX = r'\b(?<![@\w])(?:https?://|www\.)?\s?([a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9](?:\.[a-zA-Z0-9.-]+))\b'
NAME_STRUCTURE_REGEX = r'^[A-Z][a-z]+(?:\s+([A-Z][a-z.]+|[A-Z]\.)){1,2}$'

# Keywords
ADDRESS_KEYWORDS = ['Plot', 'Avenue', 'Street', 'Road', 'Drive', 'P.O. Box', 'Box', 'Floor', 'Suite', 'Ste', 'Building', 'St', 'Rd', 'Ave', 'Ln']
COMPANY_SUFFIXES = ['Limited', 'Ltd', 'Inc', 'Corp', 'Solutions', 'Group', 'PLC', 'GmbH', 'LLC']
COMPANY_KEYWORDS = ['Bank', 'Consulting', 'Media', 'Tech', 'Logistics', 'Holdings', 'Industries', 'Enterprises']
JOB_TITLE_KEYWORDS = ['Officer', 'Manager', 'Director', 'Assistant', 'Engineer', 'Sales', 'Marketing', 'CEO', 'CTO', 'President', 'Specialist', 'Consultant', 'Analyst', 'Services', 'Executive', 'Representative', 'Procurement', 'Head']

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Business Card Scanner API",
    description="Upload a business card image to extract information using Google Cloud Vision OCR + spaCy NER.",
    version="1.3.4", # Updated version number reflecting fixes
)

# --- Pydantic Model for Response ---
class CardInfo(BaseModel):
    company_name: str | None = None
    company_type: str | None = None
    person_name: str | None = None
    job_title: str | None = None
    phone_number: str | None = None # Primary phone number
    email: str | None = None
    website: str | None = None # Added website
    address: str | None = None

# --- File Handling Setup ---
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Image Preprocessing Function ---
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

# --- Google Cloud Vision OCR Function (with Correct Credential Handling) ---
def perform_ocr_google(image_path: str) -> str:
    """Performs OCR on an image file using Google Cloud Vision API."""
    try:
        logger.info(f"Performing Google Cloud Vision OCR on: {image_path}")
        # --- Start Credential Handling ---
        client = None
        creds_path_or_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path_or_json:
             logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
             raise ValueError("Google Credentials not configured.")

        if os.path.isfile(creds_path_or_json): # Local dev: Path provided
             logger.info("Using Google credentials from file path specified in ENV.")
             client = vision.ImageAnnotatorClient() # SDK handles path automatically
        else: # Deployed: Assume JSON content in ENV var
             logger.info("Attempting to load Google credentials from ENV variable content.")
             try:
                 credentials_dict = json.loads(creds_path_or_json)
                 credentials = service_account.Credentials.from_service_account_info(credentials_dict)
                 client = vision.ImageAnnotatorClient(credentials=credentials)
                 logger.info("Successfully loaded Google credentials from ENV content.")
             except Exception as cred_err:
                  logger.error(f"Failed to parse/load Google credentials from ENV variable content: {cred_err}")
                  raise ValueError(f"Invalid Google Credentials JSON in environment variable: {cred_err}")
        # --- End Credential Handling ---

        if client is None: # Safeguard
             raise ValueError("Failed to initialize Google Vision client.")

        content = preprocess_image(image_path)
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)

        if response.error.message:
            logger.error(f"Google Cloud Vision API error: {response.error.message}")
            raise HTTPException(status_code=500, detail=f"Google Cloud Vision API error: {response.error.message}")

        if response.full_text_annotation:
            extracted_text = response.full_text_annotation.text
            logger.info("Google Cloud Vision OCR completed successfully.")
            return extracted_text
        else:
            logger.warning("Google Cloud Vision did not detect any text.")
            return ""
    except HTTPException as http_exc:
        raise http_exc
    except ValueError as val_err: # Catch credential config errors
         logger.error(f"Configuration error during OCR: {val_err}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Server Configuration Error: {val_err}")
    except Exception as e:
        logger.error(f"Error during Google Cloud OCR processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")


# --- Helper Functions for Information Extraction ---

def _clean_text(text: str) -> tuple[str, list[str], list[str]]:
    """Basic text cleaning and line splitting."""
    text_cleaned = re.sub(r'\s{2,}', ' ', text.strip())
    lines = [line.strip() for line in text_cleaned.split('\n') if line.strip()]
    lines_lower = [line.lower() for line in lines]
    return text_cleaned, lines, lines_lower

def _extract_emails(text: str) -> str | None:
    """Extracts the most likely email address."""
    emails_found = re.findall(EMAIL_REGEX, text)
    if emails_found:
        logger.info(f"Regex found email(s): {emails_found}")
        return emails_found[0]
    return None

def _extract_phones(text: str, lines: list[str]) -> tuple[str | None, list[dict]]:
    """
    Extracts phone numbers from text, scores them based on context,
    and returns the most relevant (primary) phone number and a list of all found numbers with details.
    """
    phones_found = re.findall(PHONE_REGEX, text)
    scored_phones = []; all_phones_details = []
    if phones_found:
        for phone in phones_found:
            cleaned_phone = re.sub(r'[^\d+() -]', '', phone).strip()
            digit_count = sum(c.isdigit() for c in cleaned_phone)
            if digit_count >= 7:
                score = digit_count; context = "Other"
                for line_idx, line in enumerate(lines):
                    if cleaned_phone in line:
                        line_strip = line.strip()
                        prefix_match = re.match(PHONE_PREFIX_REGEX, line_strip, re.IGNORECASE)
                        if prefix_match:
                            score += 10
                            # --- FIXED PREFIX EXTRACTION ---
                            prefix = prefix_match.group(0)[0].upper() # Get first char of the whole match
                            # --- END FIX ---
                            if prefix == 'M': context = "Mobile"
                            elif prefix == 'F': context = "Fax"
                            elif prefix == 'T': context = "Tel"
                            elif prefix == 'P': context = "Phone"
                        if line_idx > len(lines) / 2: score += 2
                        break # Break from inner line search loop
                scored_phones.append((score, cleaned_phone))
                all_phones_details.append({"number": cleaned_phone, "type": context, "score": score})
    primary_phone = None
    if scored_phones:
        scored_phones.sort(key=lambda x: x[0], reverse=True)
        primary_phone = scored_phones[0][1]
        logger.info(f"Regex assigned primary phone (score {scored_phones[0][0]}): {primary_phone}")
    all_phones_details.sort(key=lambda x: x['score'], reverse=True)
    return primary_phone, all_phones_details

def _extract_websites(text: str) -> str | None:
    """Extracts the first likely website using the revised regex."""
    matches = list(re.finditer(WEBSITE_REGEX, text)) # Uses the updated constant
    if not matches: return None
    first_match = matches[0]; domain_match = first_match.group(1); full_match_text = first_match.group(0).strip()
    full_url = domain_match
    if full_match_text.lower().startswith("www."): full_url = full_match_text
    elif full_match_text.lower().startswith("https://"): full_url = full_match_text
    elif full_match_text.lower().startswith("http://"): full_url = full_match_text
    elif re.search(r'^[Ww][\s:.]+\s*' + re.escape(domain_match), text, re.MULTILINE): full_url = f"www.{domain_match}"
    full_url = full_url.replace("www. ", "www.")
    logger.info(f"Regex found potential website: {full_url}")
    return full_url

def _find_person(doc: spacy.tokens.Doc, extracted_data: dict) -> tuple[str | None, int]:
    """Finds the most likely person name using spaCy PERSON entities."""
    persons = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    person_name_assigned = None; person_name_line_index = -1
    if persons:
        scored_persons = []
        for i, person_ent in enumerate(persons):
            name = person_ent.text.strip()
            if (extracted_data.get('email') and name in extracted_data['email']) or \
               (extracted_data.get('phone_number') and name in extracted_data['phone_number']) or \
               sum(c.isdigit() for c in name) > 0 or len(name.split()) > 4 or len(name.split()) < 2: continue
            score = len(name)
            if re.fullmatch(NAME_STRUCTURE_REGEX, name): score += 10
            line_idx = -1;
            try: start_char = person_ent.start_char; line_idx = doc.text[:start_char].count('\n');
            except Exception: pass
            if line_idx != -1 and line_idx < len(doc.text.split('\n')) / 3: score += 5
            scored_persons.append((score, name, line_idx))
        if scored_persons:
             scored_persons.sort(key=lambda x: x[0], reverse=True)
             person_name_assigned = scored_persons[0][1]
             person_name_line_index = scored_persons[0][2]
             logger.info(f"spaCy assigned PERSON (score {scored_persons[0][0]}): {person_name_assigned} around line {person_name_line_index}")
    return person_name_assigned, person_name_line_index

def _find_job_title(lines: list[str], lines_lower: list[str], person_name: str | None, person_name_line_index: int, extracted_data: dict) -> str | None:
    """Finds the job title, often near the person's name."""
    job_title_assigned = None
    if person_name and person_name_line_index != -1:
        start_search_index = max(0, person_name_line_index)
        end_search_index = min(len(lines), start_search_index + 3)
        for i in range(start_search_index, end_search_index): # Loop to find title
            line = lines[i]; line_lower = lines_lower[i]
            if any(keyword.lower() in line_lower for keyword in JOB_TITLE_KEYWORDS):
                 if line != person_name and \
                    not (extracted_data.get('email') and extracted_data['email'] in line) and \
                    not (extracted_data.get('phone_number') and extracted_data['phone_number'] in line) and \
                    not (extracted_data.get('company_name') and extracted_data['company_name'] in line) and \
                    len(line.split()) < 7:
                      job_title_assigned = line
                      logger.info(f"Assigned Job Title: {job_title_assigned}")
                      break # Break from title search loop
    return job_title_assigned

def _find_org(doc: spacy.tokens.Doc, person_name: str | None) -> tuple[str | None, str | None]:
    """Finds the most likely organization name and type using spaCy ORG entities."""
    orgs = [ent for ent in doc.ents if ent.label_ == "ORG"]
    company_name_assigned = None; company_type_assigned = None
    if orgs:
        scored_orgs = []
        for org_ent in orgs: # Loop through potential orgs
            org_name = org_ent.text.strip(); type_found_in_scoring = None
            if (person_name and org_name == person_name) or len(org_name) < 3: continue
            score = len(org_name)
            for suffix in COMPANY_SUFFIXES: # Inner loop 1
                 if re.search(r'\b' + re.escape(suffix) + r'\b', org_name, re.IGNORECASE):
                     score += 50; type_found_in_scoring = suffix; # Increased suffix boost
                     break # Correct break
            if not type_found_in_scoring:
                for keyword in COMPANY_KEYWORDS: # Inner loop 2
                    if re.search(r'\b' + re.escape(keyword) + r'\b', org_name, re.IGNORECASE):
                        score += 10;
                        break # Correct break
            if org_name.isupper() and len(org_name) > 4: score += 5 # ALL CAPS boost
            scored_orgs.append((score, org_name, type_found_in_scoring))
        if scored_orgs:
             scored_orgs.sort(key=lambda x: x[0], reverse=True)
             best_org_name = scored_orgs[0][1]; best_org_type_guess = scored_orgs[0][2]
             final_type = None; cleaned_org_name = best_org_name
             if best_org_type_guess:
                 match = re.search(r'\b' + re.escape(best_org_type_guess) + r'\b', best_org_name, re.IGNORECASE);
                 if match: final_type = match.group(0)
             if not final_type:
                  for suffix in COMPANY_SUFFIXES: # Inner loop 3
                      match = re.search(r'\b' + re.escape(suffix) + r'\b', best_org_name, re.IGNORECASE);
                      if match: final_type = match.group(0); break # Correct break
             if final_type:
                 company_type_assigned = final_type;
                 cleaned_org_name = re.sub(r'\b' + re.escape(final_type) + r'\b', '', cleaned_org_name, flags=re.IGNORECASE).strip(' ,')
             cleaned_org_name = cleaned_org_name.replace('\n', ' ').strip() # Clean newlines
             company_name_assigned = cleaned_org_name;
             logger.info(f"spaCy assigned ORG (score {scored_orgs[0][0]}): {company_name_assigned} (Type: {company_type_assigned})")
    return company_name_assigned, company_type_assigned

def _assemble_address(lines: list[str], lines_lower: list[str], locations: list[spacy.tokens.Token], data: dict) -> str | None:
    """Assembles the address from relevant lines, filtering out other data more strictly."""
    address_parts = []; location_texts = {loc.text.strip().lower() for loc in locations if len(loc.text.strip()) > 2}
    person_name_assigned = data.get('person_name'); job_title_assigned = data.get('job_title'); company_name_assigned = data.get('company_name'); company_type_assigned = data.get('company_type'); email_assigned = data.get('email'); phone_assigned = data.get('phone_number'); website_assigned = data.get('website')
    for i, line in enumerate(lines):
        line_lower = lines_lower[i]; line_strip = line.strip(':., ')
        # Filter assigned fields
        is_assigned_field = (person_name_assigned and line_strip == person_name_assigned) or \
                           (job_title_assigned and line_strip == job_title_assigned) or \
                           (company_name_assigned and company_type_assigned is None and line_strip == company_name_assigned) or \
                           (company_name_assigned and company_type_assigned and company_type_assigned in line_strip and company_name_assigned in line_strip) or \
                           (email_assigned and line_strip == email_assigned) or \
                           (phone_assigned and phone_assigned in line_strip) or \
                           (website_assigned and website_assigned in line_strip)
        if is_assigned_field: continue
        # Filter lines that look like only phone/email/titles/websites
        line_cleaned_prefixes = re.sub(PHONE_PREFIX_REGEX, '', line_strip).strip()
        if re.fullmatch(PHONE_REGEX, line_cleaned_prefixes): continue
        if re.fullmatch(EMAIL_REGEX, line_strip): continue
        website_match = re.search(WEBSITE_REGEX, line_strip)
        if website_match and len(website_match.group(0)) > len(line_strip) * 0.7: continue
        if any(keyword.lower() in line_lower for keyword in JOB_TITLE_KEYWORDS) and \
           not any(addr_kw.lower() in line_lower for addr_kw in ADDRESS_KEYWORDS): continue
        # Check for Address Clues
        has_keyword = any(keyword.lower() in line_lower for keyword in ADDRESS_KEYWORDS)
        has_zip_like = re.search(r'\b\d{4,7}\b', line_strip) or re.search(r'\b\d{3}\s?\d{3}\b', line_strip)
        has_location = any(loc_text in line_lower for loc_text in location_texts)
        if has_keyword or has_location or has_zip_like:
             if re.fullmatch(r'[\d\s/()-]+', line_strip) and not has_location and not has_zip_like: continue # Filter lines with only numbers/symbols unless geo
             address_parts.append(line_strip)
    # Deduplicate and Join
    unique_address_parts = []; seen = set()
    for part in address_parts: part_lower = part.lower();
    if part_lower not in seen: unique_address_parts.append(part); seen.add(part_lower)
    if unique_address_parts:
        address_string = ", ".join(unique_address_parts);
        # Final cleaning
        if company_name_assigned and company_name_assigned in address_string: address_string = address_string.replace(company_name_assigned, '').strip(' ,')
        if company_type_assigned and company_type_assigned in address_string: address_string = address_string.replace(company_type_assigned, '').strip(' ,')
        if job_title_assigned and job_title_assigned in address_string: address_string = address_string.replace(job_title_assigned, '').strip(' ,')
        address_string = re.sub(r'\s{2,}', ' ', address_string).strip(' ,')
        if address_string: logger.info(f"Assembled address (filtered): {address_string}"); return address_string
    logger.info("No address assembled.")
    return None

# --- Main Extraction Function (Refactored) ---
def extract_information_spacy(text: str) -> dict:
    """
    Extracts information using spaCy NER + Regex, refactored with helper functions.
    """
    logger.info("Starting information extraction (Refactored v1.3.4 - Final Fixes).")
    if NLP_MODEL is None:
        logger.error("spaCy model 'NLP_MODEL' object is None. Cannot perform NER.")
        return {"_error": "spaCy model not loaded"}

    data = { # Initialize with all keys expected by CardInfo
        "company_name": None, "company_type": None, "person_name": None,
        "job_title": None, "phone_number": None, "email": None,
        "website": None, "address": None,
    }
    text_cleaned, lines, lines_lower = _clean_text(text)
    if not text_cleaned: return data

    doc = NLP_MODEL(text_cleaned)
    persons = [ent for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent for ent in doc.ents if ent.label_ == "ORG"]
    locations = [ent for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]]
    logger.info(f"spaCy found {len(persons)} PERSON, {len(orgs)} ORG, {len(locations)} location entities.")

    # Extract simple patterns first
    data['email'] = _extract_emails(text_cleaned)
    data['phone_number'], _ = _extract_phones(text_cleaned, lines) # Call updated function
    extracted_website = _extract_websites(text_cleaned)

    # --- REVISED Website Filtering ---
    data['website'] = extracted_website # Assume valid initially
    if extracted_website and data['email']:
        email_parts = data['email'].split('@')
        email_local_part = email_parts[0]
        email_domain_part = email_parts[1] if len(email_parts) > 1 else None
        website_compare = re.sub(r'^(https?://)?(www\.)?', '', extracted_website, flags=re.IGNORECASE)
        if website_compare == email_local_part:
            logger.warning(f"Ignoring website '{extracted_website}' as it matches email local part '{email_local_part}'.")
            data['website'] = None
        elif email_domain_part and website_compare == email_domain_part:
             logger.warning(f"Ignoring website '{extracted_website}' as it matches email domain part '{email_domain_part}'.")
             data['website'] = None
    if data['website']: logger.info(f"Assigned website: {data['website']}")
    elif extracted_website and data['website'] is None: logger.info(f"Filtered out potential website: {extracted_website}")
    # --- End REVISED Website Filtering ---

    person_name_assigned, person_name_line_index = _find_person(doc, data)
    data['person_name'] = person_name_assigned

    # Use the updated _find_org function
    company_name_assigned, company_type_assigned = _find_org(doc, person_name_assigned)
    data['company_name'] = company_name_assigned
    data['company_type'] = company_type_assigned

    # Update data dict before calling title/address functions that use it for filtering
    data['job_title'] = _find_job_title(lines, lines_lower, person_name_assigned, person_name_line_index, data)

    # Use the updated _assemble_address function
    data['address'] = _assemble_address(lines, lines_lower, locations, data)

    logger.info(f"Information extraction finished. Data: {data}")
    data.pop("_error", None) # Remove internal error key if present
    return data


# --- API Endpoint (No Database) ---
@app.post("/scan-card/", response_model=CardInfo)
async def scan_business_card(
    file: UploadFile = File(..., description="Business card image file (JPEG, PNG recommended).")
):
    """
    Receives card image, performs OCR (Google Vision), extracts info, returns JSON.
    (No database interaction)
    """
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file.content_type}'. Please upload an image.")

    temp_file_path = None
    original_filename = file.filename

    try:
        file_extension = os.path.splitext(original_filename)[1] or '.png'
        temp_file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_extension}")
        logger.info(f"Received file: {original_filename}. Saving temporarily to: {temp_file_path}")

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Temporary file saved successfully.")

        # --- OCR ---
        raw_text_from_ocr = perform_ocr_google(temp_file_path) # Using Google Vision

        # --- Information Extraction ---
        extracted_data = extract_information_spacy(raw_text_from_ocr)

        # Check for internal extraction errors
        if isinstance(extracted_data, dict) and "_error" in extracted_data:
             error_msg = extracted_data.get("_error", "Unknown extraction error")
             logger.error(f"Information extraction failed: {error_msg}")
             raise HTTPException(status_code=500, detail=f"Information extraction failed: {error_msg}")

        # --- Prepare and Return Response ---
        try:
             card_info_response = CardInfo(**extracted_data)
             return card_info_response
        except ValidationError as pydantic_err:
             logger.error(f"Pydantic validation failed for extracted data: {extracted_data}. Error: {pydantic_err}", exc_info=True)
             raise HTTPException(status_code=422, detail=f"Validation failed for extracted data: {pydantic_err.errors()}")

    except HTTPException as http_exc:
        logger.warning(f"HTTP Exception during processing: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred processing file {original_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        # --- Cleanup ---
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path); logger.info(f"Temporary file deleted: {temp_file_path}")
            except OSError as e: logger.error(f"Error deleting temporary file {temp_file_path}: {e}")
        if file and hasattr(file, 'file') and not file.file.closed:
            try: await file.close(); logger.info("Closed uploaded file stream.")
            except Exception as e: logger.error(f"Error closing uploaded file stream: {e}")

# --- Root endpoint ---
@app.get("/", include_in_schema=False)
def read_root():
    """Provides a simple welcome message for the API root."""
    return {"message": "Business Card Scanner API v1.3.4 (No DB) is running. Go to /docs for API documentation."}

# --- Main execution block ---
if __name__ == "__main__":
    # --- No Database Table Creation Call Here ---

    import uvicorn
    logger.info("Starting Business Card Scanner API server (v1.3.4 No DB)...")
    # Environment variable checks
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
         logger.warning("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Google Cloud Vision API calls will likely fail.")
    else:
         logger.info("GOOGLE_APPLICATION_CREDENTIALS environment variable is set.")
    if NLP_MODEL is None: # Check global variable used by extraction function
        logger.warning("spaCy model failed to load during startup. NER features will be unavailable.")

    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Keep reload=True for local dev
