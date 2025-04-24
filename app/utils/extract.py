import fitz
import re
from pdf2image import convert_from_path
import pytesseract
import logging
from llama_index.core import Document, VectorStoreIndex
import json
from typing import Dict
from datetime import datetime

import PyPDF2
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from app.core.config import settings
import os
from llama_index.core import Settings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database URIs from settings
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
GEMINI_MODEL = settings.GEMINI_MODEL
GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_EMBEDDING_MODEL = settings.GEMINI_EMBEDDING_MODEL

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# LlamaIndex settings


def configure_settings():
    """Configure LlamaIndex settings"""

    # Reset any existing settings
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    Settings.llm = Gemini()
    Settings.embed_model = GeminiEmbedding(model_name=GEMINI_EMBEDDING_MODEL)


DEBUG = False  # Set to True to enable debug logging
DISABLE_LOGGING = False  # Set to True to completely disable logging

# Configure root logger
if not DISABLE_LOGGING:
    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# Configure module logger
logger = logging.getLogger(__name__)
if DISABLE_LOGGING:
    logger.disabled = True
else:
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)


class PDFTableOfContents:
    def __init__(self, pdf_path: str):
        """Initialize with the path to the PDF file"""
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.logger = logging.getLogger(__name__)
        if DISABLE_LOGGING:
            self.logger.disabled = True
        else:
            self.logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        configure_settings()

    def extract_toc(self) -> Dict:
        """Extract table of contents from the PDF"""
        toc = self.doc.get_toc()

        # Initialize the structure
        toc_data = {
            "modules": [],
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "source_file": self.pdf_path.split("/")[-1],
            },
            # These fields will be set by the API endpoint
            "course_id": None,
            "course_title": None,
        }

        current_module = None
        module_count = 0
        seen_titles = set()  # Track seen module titles to avoid duplicates

        for level, title, page in toc:
            # Skip entries without a title
            if not title.strip():
                continue

            # Level 1 entries are modules
            if level == 1:
                if current_module:
                    # Only add the module if its title hasn't been seen before
                    if current_module["title"] not in seen_titles:
                        toc_data["modules"].append(current_module)
                        seen_titles.add(current_module["title"])
                    else:
                        # If it's a duplicate, merge its units with the existing module
                        for existing_module in toc_data["modules"]:
                            if existing_module["title"] == current_module["title"]:
                                # Add units from the duplicate module to the existing one
                                for unit in current_module["units"]:
                                    if unit not in existing_module["units"]:
                                        existing_module["units"].append(unit)
                                break

                module_count += 1
                current_module = {
                    "id": None,  # Will be set when creating the module in the database
                    "title": title.strip(),
                    "description": "",
                    "order": module_count,
                    "units": [],
                }
            # Level 2 entries are units
            elif level == 2 and current_module:
                unit = {
                    "id": None,  # Will be set when creating the unit in the database
                    "title": title.strip(),
                    "description": "",
                    "order": len(current_module["units"]) + 1,
                    "content_generated": False,
                }
                current_module["units"].append(unit)

        # Add the last module if exists
        if current_module:
            # Only add the module if its title hasn't been seen before
            if current_module["title"] not in seen_titles:
                toc_data["modules"].append(current_module)
                seen_titles.add(current_module["title"])
            else:
                # If it's a duplicate, merge its units with the existing module
                for existing_module in toc_data["modules"]:
                    if existing_module["title"] == current_module["title"]:
                        # Add units from the duplicate module to the existing one
                        for unit in current_module["units"]:
                            if unit not in existing_module["units"]:
                                existing_module["units"].append(unit)
                        break

        # If no modules were found, create a default module with the entire PDF as a unit
        if not toc_data["modules"]:
            # Create a default module
            default_module = {
                "id": None,
                "title": "Course Content",
                "description": "",
                "order": 1,
                "units": [
                    {
                        "id": None,
                        "title": "Full Course",
                        "description": "",
                        "order": 1,
                        "content_generated": False,
                    }
                ],
            }
            toc_data["modules"].append(default_module)

        return toc_data

    def extract_text_from_pages(self, start_page: int, end_page: int) -> str:
        """Extract text from a range of pages"""
        text = ""

        # Adjust page numbers to 0-based index
        start_page = max(0, start_page - 1)
        end_page = min(len(self.doc) - 1, end_page - 1)

        for page_num in range(start_page, end_page + 1):
            page = self.doc[page_num]
            text += page.get_text()

        return text.strip()

    def __del__(self):
        """Close the PDF document when the object is destroyed"""
        if hasattr(self, "doc"):
            self.doc.close()

    def find_toc_page_range(self, pdf_path, max_toc_pages=10):
        """
        Finds the range of pages that the Table of Contents (TOC) spans in a PDF.
        """
        try:
            self.logger.info(f"Finding TOC page range for PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            self.logger.info(f"Total pages in PDF: {total_pages}")

            max_toc_pages = min(max_toc_pages, total_pages)
            toc_start = None
            toc_end = None

            # Try to extract bookmarks first
            toc = doc.get_toc()
            if toc:
                self.logger.info(f"Found bookmarks: {len(toc)} entries")
                toc_start = toc[0][2] + 1
                toc_end = toc[-1][2] + 1
                self.logger.info(
                    f"Found TOC in bookmarks, spans pages {toc_start} to {toc_end}."
                )
                return toc_start, toc_end

            # Fallback to content analysis
            self.logger.info(
                f"Falling back to content analysis for first {max_toc_pages} pages"
            )
            for page_num in range(max_toc_pages):
                self.logger.debug(f"Processing page {page_num + 1}...")
                page = doc.load_page(page_num)
                text = page.get_text()

                # Log a preview of the text
                if page_num < 3:  # Only log first 3 pages to avoid too much output
                    self.logger.debug(
                        f"Page {page_num + 1} text preview: {text[:200]}..."
                    )

                if re.search(r"table\s+of\s+contents|contents", text, re.I):
                    if toc_start is None:
                        toc_start = page_num + 1
                    toc_end = page_num + 1
                    self.logger.info(f"TOC found on page {page_num + 1}.")

            if toc_start and toc_end:
                self.logger.info(f"TOC spans pages {toc_start} to {toc_end}.")
                return toc_start, toc_end
            else:
                self.logger.warning("No TOC found in first {max_toc_pages} pages.")
                # If no TOC found, assume it's on the first page
                self.logger.info("Assuming TOC is on the first page.")
                return 1, 1
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            # If there's an error, assume TOC is on the first page
            self.logger.info("Error occurred, assuming TOC is on the first page.")
            return 1, 1

    def ocr_page(self, pdf_path, page_num):
        """Converts a PDF page to image and performs OCR."""
        try:
            images = convert_from_path(
                pdf_path, first_page=page_num + 1, last_page=page_num + 1
            )
            return pytesseract.image_to_string(images[0]) if images else ""
        except Exception as e:
            self.logger.error(f"OCR failed for page {page_num + 1}: {str(e)}")
            return ""

    def load_pdf_page_range(self, pdf_path, start_page, end_page):
        """Load text content from specific page range of a PDF.

        Args:
            pdf_path (str): Path to the PDF file
            start_page (int): Starting page number (inclusive)
            end_page (int): Ending page number (inclusive)

        Returns:
            dict: Dictionary with page numbers as keys and page text as values
        """
        page_texts = {}
        self.logger.info(
            f"Loading PDF pages {start_page} to {end_page} from {pdf_path}"
        )

        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                self.logger.info(f"PDF has {total_pages} pages")

                # Adjust page range to be within bounds
                start_page = max(0, start_page)
                end_page = min(len(pdf_reader.pages) - 1, end_page)
                self.logger.info(f"Adjusted page range: {start_page} to {end_page}")

                for page_num in range(start_page - 1, end_page):
                    try:
                        text = pdf_reader.pages[page_num].extract_text()
                        page_texts[page_num] = text
                        self.logger.debug(
                            f"Successfully extracted text from page {page_num + 1} (length: {len(text)})"
                        )
                        # Log a preview of the text for the first few pages
                        if page_num < start_page + 2:
                            self.logger.debug(
                                f"Page {page_num + 1} text preview: {text[:200]}..."
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Error extracting text from page {page_num + 1}: {str(e)}"
                        )
                        page_texts[page_num] = ""

                self.logger.info(f"Extracted text from {len(page_texts)} pages")
                return page_texts

        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
            return None

    def get_toc_prompt(self, toc_text: str) -> str:
        """Generate prompt for TOC extraction."""
        prompt = """
        Extract the table of contents from the text below. Identify modules and their units, even if the text uses different naming conventions (e.g., 'Unit', numeric sections, etc.). 
        Output ONLY a raw JSON object (without any markdown formatting or code block markers) where each module has a title, description, and a list of units, following this structure:

        {{
        "modules": [
            {{
            "title": "Module Title Here",
            "description": "It is found in page X - Y",
            "order": 1,
            "units": [
                {{"title": "Unit 1", "description": "It is found in page X - Y", "order": 1}},
                {{"title": "Unit 2", "description": "It is found in page X - Y", "order": 2}},
                ...
            ]
            }},
            ...
        ]
        }}

        Important: 
        1. Return ONLY the JSON object without any markdown formatting or code block markers (no ```json or ``` tags).
        2. For each module and unit, include a description that indicates the page range where the content is found (e.g., "It is found in page 1 - 50").
        3. If you can't determine the exact page range, use a reasonable estimate based on the TOC structure.

        TOC Text:
        \"\"\"
        {}
        \"\"\"
        """.format(toc_text)
        return prompt

    def extract_toc_text(self, pdf_path: str) -> str:
        """Extract TOC text from PDF."""
        # Find TOC page range
        toc_range = self.find_toc_page_range(pdf_path)
        if not toc_range:
            self.logger.warning("No Table of Contents found.")
            return ""

        self.logger.info(
            f"Table of Contents spans pages: {toc_range[0]} to {toc_range[1]}"
        )

        # Load text from the TOC pages
        toc_text = self.load_pdf_page_range(pdf_path, toc_range[0], toc_range[1])
        if not toc_text:
            self.logger.warning("Failed to extract TOC text")
            return ""

        # Concatenate all page texts
        full_toc_text = "\n".join(toc_text.values())
        self.logger.info("Successfully extracted TOC text")
        return full_toc_text

    def process_text_with_llama(self, toc_text: str, prompt: str):
        """Process text using LlamaIndex and return the query response."""
        try:
            if not toc_text or toc_text.strip() == "":
                self.logger.error("Empty TOC text provided")
                return "Error: Empty TOC text"

            self.logger.info(f"Processing TOC text of length: {len(toc_text)}")
            self.logger.debug(f"TOC text: {toc_text[:200]}...")

            # Configure LlamaIndex settings
            configure_settings()
            self.logger.info("Configured LlamaIndex settings")

            doc = Document(text=toc_text)
            self.logger.info("Created Document object")

            index = VectorStoreIndex.from_documents([doc])
            self.logger.info("Built VectorStoreIndex")

            query_engine = index.as_query_engine()
            self.logger.info("Created query engine")

            self.logger.info("Sending prompt to LLM")
            response = query_engine.query(prompt)
            self.logger.info("Got response from query engine")

            # Log the actual response for debugging
            response_str = str(response)
            self.logger.debug(f"Raw response: {response_str}")

            # Try to extract JSON from the response if it's wrapped in markdown or other text
            json_str = response_str
            # Look for JSON-like content between curly braces
            json_match = re.search(r"(\{[\s\S]*\})", response_str)
            if json_match:
                json_str = json_match.group(1)
                self.logger.debug(f"Extracted JSON string: {json_str}")

            try:
                result = json.loads(json_str)
                self.logger.info("Successfully parsed JSON response")

                # Ensure metadata is present
                if "metadata" not in result:
                    self.logger.warning(
                        "Result does not have 'metadata' key, adding it"
                    )
                    result["metadata"] = {
                        "generated_at": datetime.utcnow().isoformat(),
                        "source_file": "unknown.pdf",
                    }

                return result
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {str(e)}")
                self.logger.debug(f"Invalid JSON: {json_str}")
                # Return a default structure with empty modules
                return {
                    "modules": [],
                    "metadata": {
                        "generated_at": datetime.utcnow().isoformat(),
                        "source_file": "unknown.pdf",
                    },
                }

        except Exception as e:
            self.logger.error(
                f"Error processing text with LlamaIndex: {str(e)}", exc_info=True
            )
            # Return a default structure with empty modules
            return {
                "modules": [],
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "source_file": "unknown.pdf",
                },
            }

    def convert_toc_to_json(self, pdf_path: str):
        """Convert PDF TOC to JSON using LlamaIndex."""
        self.logger.info(f"Processing PDF: {pdf_path}")

        toc_text = self.extract_toc_text(pdf_path)
        if not toc_text:
            self.logger.error("Failed to extract TOC text")
            # Create a default structure with empty modules
            default_result = {
                "modules": [],
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "source_file": pdf_path.split("/")[-1],
                },
                "course_id": None,  # Will be set by the API
                "course_title": None,  # Will be set by the API
            }
            self.logger.info(
                f"Returning default structure: {json.dumps(default_result, indent=2)}"
            )
            return default_result

        self.logger.info(f"Extracted TOC text length: {len(toc_text)}")
        self.logger.debug(f"TOC text preview: {toc_text[:500]}...")

        prompt = self.get_toc_prompt(toc_text)
        self.logger.info("Generated prompt")
        self.logger.debug(f"Prompt: {prompt}")

        try:
            result = self.process_text_with_llama(toc_text, prompt)
            self.logger.info(f"Query Response type: {type(result)}")
            self.logger.info(f"Query Response: {json.dumps(result, indent=2)}")

            # Check if result is a string (error message)
            if isinstance(result, str):
                self.logger.error(f"Error from LLM: {result}")
                # Create a default structure with empty modules
                default_result = {
                    "modules": [],
                    "metadata": {
                        "generated_at": datetime.utcnow().isoformat(),
                        "source_file": pdf_path.split("/")[-1],
                    },
                    "course_id": None,  # Will be set by the API
                    "course_title": None,  # Will be set by the API
                }
                self.logger.info(
                    f"Returning default structure: {json.dumps(default_result, indent=2)}"
                )
                return default_result

            # Ensure the result has the expected structure
            if "modules" not in result:
                self.logger.error("Result does not have 'modules' key")
                result["modules"] = []

            # Ensure metadata is present
            if "metadata" not in result:
                self.logger.warning("Result does not have 'metadata' key, adding it")
                result["metadata"] = {
                    "generated_at": datetime.utcnow().isoformat(),
                    "source_file": pdf_path.split("/")[-1],
                }

            # Log the number of modules
            self.logger.info(f"Number of modules found: {len(result['modules'])}")

            # Remove duplicate modules based on title
            seen_titles = set()
            unique_modules = []

            for module in result.get("modules", []):
                if "units" not in module:
                    self.logger.warning(
                        f"No 'units' key in module: {module.get('title', 'Unknown')}"
                    )
                    module["units"] = []

                # Ensure description has page range information
                if "description" not in module or not module["description"]:
                    module["description"] = (
                        "It is found in page 1 - 50"  # Default page range
                    )
                elif "page" not in module["description"].lower():
                    module["description"] = (
                        f"{module['description']} (Pages: 1 - 50)"  # Add default page range
                    )

                # Check if this module title has been seen before
                if module.get("title") not in seen_titles:
                    seen_titles.add(module.get("title"))
                    unique_modules.append(module)
                else:
                    # If it's a duplicate, merge its units with the existing module
                    for existing_module in unique_modules:
                        if existing_module.get("title") == module.get("title"):
                            # Add units from the duplicate module to the existing one
                            for unit in module.get("units", []):
                                if unit not in existing_module["units"]:
                                    existing_module["units"].append(unit)
                            break

            # Replace the modules list with the deduplicated one
            result["modules"] = unique_modules

            # Ensure each unit has content_generated field and proper description
            for module in result.get("modules", []):
                for unit in module.get("units", []):
                    if "content_generated" not in unit:
                        unit["content_generated"] = False

                    # Ensure unit description has page range information
                    if "description" not in unit or not unit["description"]:
                        unit["description"] = (
                            "It is found in page 1 - 50"  # Default page range
                        )
                    elif "page" not in unit["description"].lower():
                        unit["description"] = (
                            f"{unit['description']} (Pages: 1 - 50)"  # Add default page range
                        )

            return result

        except Exception as e:
            self.logger.error(f"Error in convert_toc_to_json: {str(e)}", exc_info=True)
            # Create a default structure with empty modules
            default_result = {
                "modules": [],
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "source_file": pdf_path.split("/")[-1],
                },
                "course_id": None,  # Will be set by the API
                "course_title": None,  # Will be set by the API
            }
            self.logger.info(
                f"Returning default structure: {json.dumps(default_result, indent=2)}"
            )
            return default_result


if __name__ == "__main__":
    pdf_path = "uploads/1/20250414_190907_G10-Mathematics-STB-2023-web.pdf"
    toc_handler = PDFTableOfContents(pdf_path)
    json_result = toc_handler.convert_toc_to_json(pdf_path)
    print(json_result)

    if json_result:
        print("\nExample of accessing dictionary data:")
        first_module = json_result["modules"][0]
        print(f"First module title: {first_module['title']}")
        print(f"First module description: {first_module['description']}")

        first_unit = first_module["units"][0]
        print(f"First unit title: {first_unit['title']}")
        print(f"First unit description: {first_unit['description']}")
