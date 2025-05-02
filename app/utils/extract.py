import fitz
import re
from pdf2image import convert_from_path
import pytesseract
import logging
from llama_index.core import Document, VectorStoreIndex
import json
from typing import Dict
from datetime import datetime
import warnings
import contextlib
import time

import PyPDF2
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from app.core.config import settings
import os
from llama_index.core import Settings
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import io


# Create a filter to suppress specific pdfminer warnings
class PDFMinerWarningFilter(logging.Filter):
    def filter(self, record):
        # Filter out common pdfminer warnings about gray stroke colors
        if "Cannot set gray stroke color because" in record.getMessage():
            return False
        if "Cannot set gray non-stroke color because" in record.getMessage():
            return False
        if "invalid float value" in record.getMessage():
            return False
        return True


# Apply the filter to pdfminer
pdfminer_logger = logging.getLogger("pdfminer")
pdfminer_logger.addFilter(PDFMinerWarningFilter())


# Context manager to temporarily suppress specific warnings
@contextlib.contextmanager
def suppress_pdfminer_warnings():
    # Save original warning filter settings
    original_filters = warnings.filters.copy()
    # Ignore specific pdfminer warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pdfminer")
    try:
        yield
    finally:
        # Restore original warning filter settings
        warnings.filters = original_filters


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
        try:
            self.doc = fitz.open(pdf_path)
        except Exception as e:
            self.logger.error(f"Error opening PDF: {str(e)}")
            self.doc = None

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
        if hasattr(self, "doc") and self.doc:
            try:
                self.doc.close()
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.warning(f"Error closing PDF: {str(e)}")

        # Try to delete the temporary file if it has a temp file pattern
        if hasattr(self, "pdf_path") and self.pdf_path:
            if "temp" in self.pdf_path.lower() or "tmp" in self.pdf_path.lower():
                self.safe_delete_file(self.pdf_path)

    def safe_delete_file(self, file_path):
        """
        Safely delete a file with multiple attempts to handle Windows file locks.
        Returns True if successfully deleted, False otherwise.
        """
        if not file_path or not os.path.exists(file_path):
            return False

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                os.remove(file_path)
                self.logger.info(f"Successfully deleted file: {file_path}")
                return True
            except PermissionError:
                # File might be locked by Windows, wait a bit and retry
                if hasattr(self, "logger"):
                    self.logger.warning(
                        f"File locked, retry {attempt + 1}/{max_attempts}: {file_path}"
                    )
                time.sleep(1)  # Wait 1 second before retry
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.warning(
                        f"Could not delete file ({str(e)}): {file_path}"
                    )
                return False

        if hasattr(self, "logger"):
            self.logger.warning(
                f"Failed to delete file after {max_attempts} attempts: {file_path}"
            )
        return False

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
                        # Set TOC end to be 2 pages after the start, but don't exceed total pages
                        toc_end = min(toc_start + 2, total_pages)
                        self.logger.info(
                            f"TOC found on page {page_num + 1}, extending range to page {toc_end}."
                        )
                        break

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
        """Converts a PDF page to image and performs OCR with enhanced error handling."""
        try:
            self.logger.info(f"Performing OCR on page {page_num + 1}")
            # Use higher DPI for better OCR results
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=400,  # Increased from 300 to 400 for better quality
            )

            if not images:
                self.logger.warning(f"No images generated for page {page_num + 1}")
                return ""

            # Use more advanced OCR configuration
            ocr_config = "--psm 6 --oem 3 -l eng+ara+chi_sim+fra+spa+rus --dpi 400"

            # Try advanced configuration first
            ocr_text = pytesseract.image_to_string(images[0], config=ocr_config)

            # If we got little or no text, try with different page segmentation modes
            if len(ocr_text.strip()) < 100:
                self.logger.info(
                    "First OCR attempt returned little text, trying alternative modes"
                )

                # Try different page segmentation modes
                for psm in [1, 3, 4, 11, 12]:
                    alt_config = f"--psm {psm} --oem 3 -l eng"
                    alt_text = pytesseract.image_to_string(images[0], config=alt_config)
                    if len(alt_text.strip()) > len(ocr_text.strip()) * 1.5:
                        ocr_text = alt_text
                        self.logger.info(f"Better results with PSM {psm}")
                        break

            if ocr_text:
                self.logger.debug(f"OCR text preview: {ocr_text[:200]}...")
                return ocr_text
            else:
                self.logger.warning(f"OCR returned empty text for page {page_num + 1}")
                return ""

        except Exception as e:
            self.logger.error(f"OCR failed for page {page_num + 1}: {str(e)}")
            return ""

    def is_text_corrupted(self, text_sample):
        """
        Determine if extracted text appears to be corrupted due to font issues.

        Args:
            text_sample: A sample of text to check for corruption issues

        Returns:
            bool: True if the text appears corrupted, False otherwise
        """
        if not text_sample or len(text_sample) < 10:
            return False

        # Check for high percentage of non-ASCII/non-printable characters
        non_ascii_count = sum(
            1
            for c in text_sample
            if ord(c) > 127 or (ord(c) < 32 and c not in ["\n", "\t", "\r"])
        )
        if len(text_sample) > 0 and non_ascii_count / len(text_sample) > 0.3:
            self.logger.warning(
                "High percentage of non-ASCII characters detected, likely font corruption"
            )
            return True

        # Check for common corruption patterns
        corruption_patterns = [
            # Repeated special chars (common in font corruption)
            r"[\x00-\x1F\x7F-\xFF]{5,}",
            # Repeated single character (like )
            r"(.)\1{5,}",
            # Unusual symbol sequences common in broken fonts
            r"[]{3,}",
            # Unicode replacement characters
            r"[\ufffd\ufeff]{2,}",
        ]

        for pattern in corruption_patterns:
            if re.search(pattern, text_sample):
                self.logger.warning(f"Detected corruption pattern: {pattern}")
                return True

        # Check for readability by counting real words
        words = [w for w in text_sample.split() if len(w) > 1]
        if words:
            import string

            # Count words containing mostly normal characters
            normal_words = 0
            for word in words:
                if sum(1 for c in word if c in string.ascii_letters) / len(word) > 0.7:
                    normal_words += 1

            # If less than 30% of "words" are actually readable, text is likely corrupted
            if normal_words / len(words) < 0.3:
                self.logger.warning(
                    "Low percentage of readable words, likely font corruption"
                )
                return True

        return False

    def extract_text_from_corrupted_pdf(self, pdf_path, start_page=None, end_page=None):
        """
        Extract text from a PDF with corrupted text encoding using OCR.
        This is a specialized method for handling PDFs with font encoding issues.

        Args:
            pdf_path: Path to the PDF file
            start_page: First page to process (1-based index)
            end_page: Last page to process (1-based index)

        Returns:
            str: Extracted text
        """
        self.logger.info(f"Using OCR to extract text from corrupted PDF: {pdf_path}")

        try:
            # Determine page range
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            if start_page is None:
                start_page = 1
            if end_page is None:
                end_page = total_pages

            # Adjust to valid range
            start_page = max(1, min(start_page, total_pages))
            end_page = max(start_page, min(end_page, total_pages))

            self.logger.info(f"Processing pages {start_page} to {end_page} with OCR")

            # Extract text using OCR
            text_parts = []
            for page_num in range(start_page - 1, end_page):
                ocr_text = self.ocr_page(pdf_path, page_num)
                if ocr_text:
                    text_parts.append(ocr_text)
                    self.logger.info(f"OCR successful for page {page_num + 1}")
                else:
                    self.logger.warning(f"OCR failed for page {page_num + 1}")

            result = "\n\n".join(text_parts)
            self.logger.info(f"Extracted {len(result)} characters using OCR")
            return result

        except Exception as e:
            self.logger.error(f"Error during OCR extraction: {str(e)}")
            return ""

    def extract_toc_text(self, pdf_path: str) -> str:
        """Extract TOC text from PDF with enhanced methods for complex PDFs."""
        # Find TOC page range
        toc_range = self.find_toc_page_range(pdf_path)
        if not toc_range:
            self.logger.warning("No Table of Contents found.")
            return ""

        self.logger.info(
            f"Table of Contents spans pages: {toc_range[0]} to {toc_range[1]}"
        )

        # First, check for text corruption with a sample
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count > 0:
                # Get text samples from a few pages
                text_samples = []
                # Sample first page, a middle page, and last page
                sample_pages = [
                    0,
                    min(5, doc.page_count - 1),
                    min(doc.page_count - 1, 10),
                ]
                for page_idx in sample_pages:
                    if page_idx < doc.page_count:
                        sample = doc[page_idx].get_text()[:1000]
                        if sample:
                            text_samples.append(sample)

                doc.close()

                # Check if text samples indicate corruption
                combined_sample = "".join(text_samples)
                if combined_sample and self.is_text_corrupted(combined_sample):
                    self.logger.warning(
                        "PDF appears to have corrupted text encoding, using OCR approach"
                    )
                    # Use specialized extraction for corrupted PDFs
                    toc_text = self.extract_text_from_corrupted_pdf(
                        pdf_path, start_page=toc_range[0], end_page=toc_range[1]
                    )
                    if toc_text and len(toc_text.strip()) > 100:
                        self.logger.info(
                            "Successfully extracted TOC from corrupted PDF using OCR"
                        )
                        return toc_text
        except Exception as e:
            self.logger.error(f"Error checking for text corruption: {str(e)}")

        # For problematic PDFs, try OCR first if it appears to be a scanned document
        try:
            # Check if this is likely a scanned PDF by checking text extraction on first page
            doc = fitz.open(pdf_path)
            if doc.page_count > 0:
                first_page = doc[0]
                text = first_page.get_text()
                # If very little text is extracted, it's likely a scan
                if len(text.strip()) < 100:
                    self.logger.info("Document appears to be scanned, trying OCR first")
                    doc.close()

                    # Try OCR on TOC pages
                    toc_text = ""
                    for page_num in range(toc_range[0] - 1, toc_range[1]):
                        ocr_text = self.ocr_page(pdf_path, page_num)
                        if ocr_text:
                            toc_text += ocr_text + "\n"
                            self.logger.info(f"OCR successful for page {page_num + 1}")
                        else:
                            self.logger.warning(f"OCR failed for page {page_num + 1}")

                    if toc_text and len(toc_text.strip()) > 100:
                        self.logger.info("Successfully extracted TOC text using OCR")
                        return toc_text
                else:
                    doc.close()
        except Exception as e:
            self.logger.error(f"Error checking if document is scanned: {str(e)}")

        # First attempt: Try pdfminer.six for better font handling
        try:
            self.logger.info("Trying pdfminer.six extraction")
            toc_text = ""

            # Create custom parameters for better text extraction
            laparams = LAParams(
                char_margin=1.0,
                line_margin=0.5,
                word_margin=0.1,
                boxes_flow=0.5,
                detect_vertical=True,
            )

            # Extract full PDF text first with warning suppression
            with suppress_pdfminer_warnings():
                # Try with different encoding settings
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        full_text = extract_text(
                            pdf_path, laparams=laparams, codec=encoding
                        )
                        if full_text and len(full_text.strip()) > 100:
                            self.logger.info(
                                f"Successful extraction with {encoding} encoding"
                            )
                            break
                    except Exception as e:
                        self.logger.warning(
                            f"Failed with {encoding} encoding: {str(e)}"
                        )
                        full_text = ""

            # Split by pages (approximate)
            pages = full_text.split("\f")  # Form feed character often separates pages

            # Get text from TOC pages
            for page_num in range(toc_range[0] - 1, min(toc_range[1], len(pages))):
                if page_num < len(pages):
                    toc_text += pages[page_num] + "\n"

            # Check if we got meaningful content
            if (
                toc_text
                and len(toc_text.strip()) > 100
                and not toc_text.startswith("/gid")
            ):
                # Clean text of special characters that aren't meaningful
                toc_text = re.sub(
                    r"[^\x00-\x7F\s\.\:\-\,\;\(\)\[\]\{\}\/\\\|]", " ", toc_text
                )
                # Remove excessive whitespace
                toc_text = re.sub(r"\s+", " ", toc_text)
                self.logger.info("Successfully extracted TOC text using pdfminer.six")
                return toc_text
            else:
                self.logger.warning(
                    "pdfminer.six extraction yielded insufficient content, trying PyMuPDF"
                )
        except Exception as e:
            self.logger.error(f"Error during pdfminer.six extraction: {str(e)}")

        # Second attempt: Try PyMuPDF (fitz) with different extraction methods
        try:
            toc_text = ""
            doc = fitz.open(pdf_path)

            # Try different text extraction methods for each page
            extraction_methods = ["text", "html", "dict", "json", "rawdict", "xhtml"]

            for page_num in range(toc_range[0] - 1, toc_range[1]):
                if page_num < len(doc):
                    page = doc[page_num]
                    page_text = ""

                    # Try each extraction method
                    for method in extraction_methods:
                        try:
                            if method == "text":
                                # Simple text extraction
                                method_text = page.get_text(method)
                            elif method == "dict" or method == "rawdict":
                                # Process structured data
                                data = page.get_text(method)
                                method_text = ""
                                for block in data.get("blocks", []):
                                    if "lines" in block:
                                        for line in block["lines"]:
                                            line_text = ""
                                            for span in line.get("spans", []):
                                                if "text" in span and not span[
                                                    "text"
                                                ].startswith("/gid"):
                                                    line_text += span["text"] + " "
                                            method_text += line_text + "\n"
                            elif method in ["html", "xhtml"]:
                                # Extract text from HTML
                                html = page.get_text(method)
                                # Simple HTML tag removal (not perfect but fast)
                                method_text = re.sub(r"<[^>]+>", " ", html)
                            else:
                                # Other formats - convert to string
                                method_text = str(page.get_text(method))

                            # Check if we got meaningful text
                            if (
                                method_text
                                and len(method_text.strip()) > 50
                                and not method_text.startswith("/gid")
                            ):
                                if len(method_text) > len(page_text):
                                    page_text = method_text
                                    self.logger.debug(
                                        f"Method {method} gave better results"
                                    )
                        except Exception as e:
                            self.logger.debug(f"Method {method} failed: {str(e)}")

                    if page_text:
                        toc_text += page_text + "\n"

            doc.close()

            # Check if we got meaningful content
            if (
                toc_text
                and len(toc_text.strip()) > 100
                and not toc_text.startswith("/gid")
            ):
                self.logger.info("Successfully extracted TOC text using PyMuPDF")
                return toc_text
            else:
                self.logger.warning(
                    "PyMuPDF extraction yielded insufficient or invalid content, trying PyPDF2"
                )
        except Exception as e:
            self.logger.error(f"Error during PyMuPDF extraction: {str(e)}")

        # Third attempt: Try PyPDF2
        try:
            self.logger.info("Trying PyPDF2 extraction")
            toc_text = ""

            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                self.logger.info(f"PDF has {total_pages} pages")

                # Adjust page range to be within bounds
                start_page = max(1, toc_range[0])
                end_page = min(total_pages, toc_range[1])

                for page_num in range(start_page - 1, end_page):
                    try:
                        text = pdf_reader.pages[page_num].extract_text()
                        if text:
                            toc_text += text + "\n"
                            self.logger.debug(
                                f"Successfully extracted text from page {page_num + 1}"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Error extracting text from page {page_num + 1}: {str(e)}"
                        )

            # Check if we got meaningful content
            if (
                toc_text
                and len(toc_text.strip()) > 100
                and not toc_text.startswith("/gid")
            ):
                self.logger.info("Successfully extracted TOC text using PyPDF2")
                return toc_text
            else:
                self.logger.warning(
                    "PyPDF2 extraction yielded insufficient or invalid content, trying OCR"
                )
        except Exception as e:
            self.logger.error(f"Error during PyPDF2 extraction: {str(e)}")

        # Fourth attempt: Try OCR for complex PDFs
        try:
            self.logger.info("Falling back to OCR for TOC extraction")
            toc_text = ""

            for page_num in range(toc_range[0] - 1, toc_range[1]):
                ocr_text = self.ocr_page(pdf_path, page_num)
                if ocr_text:
                    toc_text += ocr_text + "\n"
                    self.logger.info(f"OCR successful for page {page_num + 1}")
                else:
                    self.logger.warning(f"OCR failed for page {page_num + 1}")

            if toc_text and len(toc_text.strip()) > 100:
                self.logger.info("Successfully extracted TOC text using OCR")
                return toc_text
            else:
                self.logger.error(
                    "All extraction methods failed to get meaningful TOC content"
                )
                return ""
        except Exception as e:
            self.logger.error(f"Error during OCR extraction: {str(e)}")
            return ""

    def get_toc_prompt(self, toc_text: str) -> str:
        """Generate improved prompt for TOC extraction with better page number handling."""
        prompt = """
        Extract the table of contents from the text below. Identify modules and their units, even if the text uses different naming conventions (e.g., 'Unit', 'Chapter', 'Section', numeric sections, etc.).
        
        IMPORTANT: For each module and unit, you MUST extract the EXACT page numbers mentioned in the table of contents.
        Look for patterns like:
        - "Chapter 1 ................... 15" (meaning page 15)
        - "1.2 Topic Name .......... 23-45" (meaning pages 23 to 45)
        - "Unit 3: Name [p.78-92]" (meaning pages 78 to 92)
        
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
        1. Return ONLY the JSON object without any markdown formatting.
        2. For each module and unit title, extract ONLY the actual title without page numbers.
        3. For each module and unit description, PRECISELY indicate the page range where it's found based on the TOC (e.g., "It is found in page 15 - 20").
        4. If only a single page is mentioned, use that as both start and end (e.g., "It is found in page 15 - 15").
        5. If you can't find page numbers for an item, estimate based on surrounding items.
        6. Ignore any text that appears to be metadata, headers, footers, or internal PDF identifiers.
        7. Make sure all titles are cleaned up and don't contain page numbers or dots (e.g., "Chapter 1............15" should become just "Chapter 1").

        TOC Text:
        \"\"\"
        {}
        \"\"\"
        """.format(toc_text)
        return prompt

    def extract_page_numbers(self, text):
        """
        Extract page numbers from TOC text entries.
        Handles patterns like:
        - "Title....45"
        - "Title [p.23-45]"
        - "Title (pages 12-15)"
        - "Title - page 34"
        """
        # Various regex patterns to match page number formats
        patterns = [
            r"(\d+)\s*-\s*(\d+)",  # Format: 45-67
            r"page\s+(\d+)\s*-\s*(\d+)",  # Format: page 45-67
            r"pages\s+(\d+)\s*-\s*(\d+)",  # Format: pages 45-67
            r"p\.\s*(\d+)\s*-\s*(\d+)",  # Format: p.45-67
            r"\[p\.(\d+)-(\d+)\]",  # Format: [p.45-67]
            r"\(pages?\s+(\d+)\s*-\s*(\d+)\)",  # Format: (pages 45-67)
            r"\.{3,}\s*(\d+)",  # Format: .......45 (dots followed by number)
        ]

        # First look for page ranges
        for pattern in patterns:
            matches = re.search(pattern, text)
            if matches:
                try:
                    start, end = int(matches.group(1)), int(matches.group(2))
                    return start, end
                except (IndexError, ValueError):
                    # If second group doesn't exist or conversion fails
                    try:
                        # Just one page number found
                        page = int(matches.group(1))
                        return page, page
                    except (IndexError, ValueError):
                        continue

        # If no range found, look for single page numbers
        single_patterns = [
            r"page\s+(\d+)",  # Format: page 45
            r"p\.\s*(\d+)",  # Format: p.45
            r"\[p\.(\d+)\]",  # Format: [p.45]
            r"\(page\s+(\d+)\)",  # Format: (page 45)
            r"\.{3,}\s*(\d+)",  # Format: .......45 (dots followed by number)
            r"\s(\d+)$",  # Format: title 45 (number at end)
        ]

        for pattern in single_patterns:
            matches = re.search(pattern, text)
            if matches:
                try:
                    page = int(matches.group(1))
                    return page, page
                except (IndexError, ValueError):
                    continue

        # Default if no page numbers found
        return None, None

    def clean_title(self, title):
        """
        Clean up titles by removing page numbers, dots, and other non-title elements.
        Also limits title length to prevent database errors.
        """
        # First, check if title is suspiciously long (likely captured content instead of a title)
        if len(title) > 1000:
            self.logger.warning(
                f"Title extremely long ({len(title)} chars), likely content not title"
            )
            # Try to extract a real title from the first line or first few words
            lines = title.split("\n")
            first_line = lines[0].strip() if lines else ""

            # If first line is reasonable length, use it
            if 0 < len(first_line) <= 200:
                title = first_line
            else:
                # Otherwise take first few words
                words = title.split()
                if len(words) > 5:
                    title = " ".join(words[:5]) + "..."
                else:
                    # Last resort - use generic title
                    title = "Course Content"

        # Remove trailing dots and spaces followed by numbers (common TOC format)
        title = re.sub(r"\.{2,}\s*\d+\s*$", "", title)

        # Remove page number indicators
        title = re.sub(r"\[p\.\d+(?:-\d+)?\]", "", title)
        title = re.sub(r"\(pages?\s+\d+(?:-\d+)?\)", "", title)
        title = re.sub(r"page\s+\d+(?:-\d+)?", "", title)

        # Remove typical PDF headers/footers
        title = re.sub(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", "", title, flags=re.IGNORECASE)

        # Remove content that appears to be repetitions (common in badly extracted text)
        # E.g., "Biology Biology Biology Biology Biology"
        words = title.split()
        if len(words) > 4:
            unique_words = set(w.lower() for w in words)
            # If there are many words but few unique ones, it's likely repetition
            if len(words) > 10 and len(unique_words) < len(words) / 3:
                # Take a subset with minimal repetition
                seen = set()
                filtered_words = []
                for word in words:
                    if word.lower() not in seen:
                        seen.add(word.lower())
                        filtered_words.append(word)
                    if len(filtered_words) >= 5:  # Limit to 5 unique words
                        break
                title = " ".join(filtered_words)
                if len(title) < 3:  # Ensure we have something
                    title = "Course Content"

        # Clean up any remaining dots at the end
        title = re.sub(r"\.+\s*$", "", title)

        # Trim whitespace
        title = title.strip()

        # Last check - if title is long but isn't structured like a title (e.g., entire paragraph)
        # look for common title patterns
        if len(title) > 50:
            # Look for common title patterns
            title_patterns = [
                r"^(?:chapter|module|unit|section)\s+\d+\s*:?\s*(.{3,50})",  # Chapter 1: Title
                r"^(?:\d+\.)\s+(.{3,50})",  # 1. Title
                r"^(?:[IVX]+\.)\s+(.{3,50})",  # I. Title (Roman numerals)
                r"^(?:table of contents|contents|toc)",  # TOC
                r"^(?:.*?)\btextbook\b(.*?)$",  # Textbook Title
                r"^(.*?\b(?:grade|level)\b.*?)$",  # Grade level
                r"^(.*?\bcourse\b.*?)$",  # Course Title
            ]

            for pattern in title_patterns:
                match = re.search(pattern, title, re.IGNORECASE)
                if match and match.group(1):
                    title = match.group(1).strip()
                    break

        # Limit title length to prevent database errors (PostgreSQL index limit is 8191 bytes)
        # Use a conservative limit of 200 characters for titles (most real titles are shorter)
        if len(title) > 200:
            self.logger.warning(
                f"Title too long ({len(title)} chars), truncating: {title[:50]}..."
            )
            title = title[:197] + "..."

        # Ensure title is never empty
        if not title.strip():
            title = "Course Content"

        return title

    def validate_toc_structure(self, result):
        """
        Validate and fix TOC structure to prevent database errors.
        """
        self.logger.info("Validating TOC structure to prevent database errors")

        # Ensure default metadata
        if "metadata" not in result:
            result["metadata"] = {
                "generated_at": datetime.utcnow().isoformat(),
                "source_file": os.path.basename(self.pdf_path),
            }

        # Ensure modules key exists
        if "modules" not in result:
            result["modules"] = []

        # Set maximum safe field sizes for database
        MAX_TITLE_LENGTH = 500  # Conservative limit
        MAX_DESC_LENGTH = 1000  # Conservative limit

        # Filter and validate modules
        valid_modules = []
        for i, module in enumerate(result.get("modules", [])):
            # Skip completely invalid modules
            if not isinstance(module, dict):
                self.logger.warning(
                    f"Skipping invalid module at index {i}: not a dictionary"
                )
                continue

            # Ensure module has required fields
            if "title" not in module or not module["title"]:
                self.logger.warning(f"Skipping module without title at index {i}")
                continue

            # Limit title size
            if len(module["title"]) > MAX_TITLE_LENGTH:
                module["title"] = module["title"][: MAX_TITLE_LENGTH - 3] + "..."
                self.logger.warning(f"Truncated module title at index {i}")

            # Ensure description exists and limit its size
            if "description" not in module or not module["description"]:
                module["description"] = "No description available"
            if len(module["description"]) > MAX_DESC_LENGTH:
                module["description"] = (
                    module["description"][: MAX_DESC_LENGTH - 3] + "..."
                )
                self.logger.warning(f"Truncated module description at index {i}")

            # Ensure order exists and is numeric
            if "order" not in module or not isinstance(module["order"], int):
                module["order"] = i + 1

            # Ensure units exists and is a list
            if "units" not in module or not isinstance(module["units"], list):
                module["units"] = []

            # Validate units
            valid_units = []
            for j, unit in enumerate(module.get("units", [])):
                # Skip invalid units
                if not isinstance(unit, dict):
                    self.logger.warning(
                        f"Skipping invalid unit at module {i}, unit {j}"
                    )
                    continue

                # Ensure unit has required fields
                if "title" not in unit or not unit["title"]:
                    self.logger.warning(
                        f"Skipping unit without title at module {i}, unit {j}"
                    )
                    continue

                # Limit title size
                if len(unit["title"]) > MAX_TITLE_LENGTH:
                    unit["title"] = unit["title"][: MAX_TITLE_LENGTH - 3] + "..."
                    self.logger.warning(f"Truncated unit title at module {i}, unit {j}")

                # Ensure description exists and limit its size
                if "description" not in unit or not unit["description"]:
                    unit["description"] = "No description available"
                if len(unit["description"]) > MAX_DESC_LENGTH:
                    unit["description"] = (
                        unit["description"][: MAX_DESC_LENGTH - 3] + "..."
                    )
                    self.logger.warning(
                        f"Truncated unit description at module {i}, unit {j}"
                    )

                # Ensure order exists and is numeric
                if "order" not in unit or not isinstance(unit["order"], int):
                    unit["order"] = j + 1

                # Ensure content_generated exists
                if "content_generated" not in unit:
                    unit["content_generated"] = False

                valid_units.append(unit)

            # Replace units with validated list
            module["units"] = valid_units
            valid_modules.append(module)

        # Replace modules with validated list
        result["modules"] = valid_modules

        # If we ended up with no valid modules, create a default one
        if not result["modules"]:
            self.logger.warning("No valid modules found, creating a default module")
            result["modules"] = [
                {
                    "title": "Course Content",
                    "description": "Automatically generated course content",
                    "order": 1,
                    "units": [
                        {
                            "title": "Full Course",
                            "description": "Complete course content",
                            "order": 1,
                            "content_generated": False,
                        }
                    ],
                }
            ]

        return result

    def process_toc_with_regex(self, toc_text):
        """
        Process TOC text with regex to extract modules and units with page numbers.
        This is used as a fallback when LLM-based extraction fails.
        """
        # If toc_text is extremely long (over 10000 chars), it's likely not a TOC
        # but the entire document content. Try to find the actual TOC section
        if len(toc_text) > 10000:
            self.logger.warning(
                f"TOC text extremely long ({len(toc_text)} chars), searching for actual TOC section"
            )

            # Look for common TOC markers
            toc_patterns = [
                r"(?:Table\s+of\s+Contents|Contents|TOC)[\s\S]{10,3000}?(?:\n\s*\d|\n\s*Chapter|\n\s*Section)",
                r"(?:\n\s*\d+\..*?\d+\s*\n\s*\d+\..*?\d+\s*)",  # Numbered sections with page numbers
            ]

            for pattern in toc_patterns:
                match = re.search(pattern, toc_text, re.IGNORECASE)
                if match:
                    toc_text = match.group(0)
                    self.logger.info(
                        f"Found actual TOC section ({len(toc_text)} chars)"
                    )
                    break

        lines = toc_text.split("\n")
        result = {
            "modules": [],
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "source_file": os.path.basename(self.pdf_path)
                if hasattr(self, "pdf_path")
                else "unknown.pdf",
            },
            "course_id": None,
            "course_title": None,
        }

        # Try to extract course title from the first few lines if it exists
        course_title = None
        for i in range(min(5, len(lines))):
            if lines[i].strip() and not re.match(
                r"^(?:table\s+of\s+contents|contents|toc)$",
                lines[i].strip(),
                re.IGNORECASE,
            ):
                potential_title = self.clean_title(lines[i].strip())
                if 3 < len(potential_title) < 100:  # Reasonable title length
                    course_title = potential_title
                    break

        if course_title:
            result["course_title"] = course_title

        current_module = None
        module_count = 0

        # Look for patterns indicating chapters/modules and their units
        chapter_patterns = [
            r"^(?:Chapter|Module|Unit|Part|Section)\s+\d+",  # Chapter 1, Module 2
            r"^\d+\.\s+[A-Z]",  # 1. Title
            r"^[IVXLCDM]+\.\s+",  # I. II. III. (Roman numerals)
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line appears to be a chapter/module title
            is_chapter = any(re.match(pattern, line) for pattern in chapter_patterns)
            # Units often have sub-numbering or indentation
            is_unit = re.match(r"^\s+", line) or re.match(r"^\d+\.\d+", line)

            # Extract page numbers
            start_page, end_page = self.extract_page_numbers(line)

            # Clean the title
            clean_title = self.clean_title(line)

            if is_chapter or (not is_unit and not current_module):
                # This is a new module
                module_count += 1
                current_module = {
                    "title": clean_title,
                    "description": f"It is found in page {start_page or 1} - {end_page or 100}",
                    "order": module_count,
                    "units": [],
                }
                result["modules"].append(current_module)
            elif current_module and (is_unit or len(current_module["units"]) > 0):
                # This is a unit under the current module
                unit = {
                    "title": clean_title,
                    "description": f"It is found in page {start_page or 1} - {end_page or 100}",
                    "order": len(current_module["units"]) + 1,
                    "content_generated": False,
                }
                current_module["units"].append(unit)

        # If we didn't find any chapters/modules, try a more aggressive approach
        if not result["modules"]:
            # Treat any line with page numbers as a potential module
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                start_page, end_page = self.extract_page_numbers(line)
                if start_page:
                    module_count += 1
                    module = {
                        "title": self.clean_title(line),
                        "description": f"It is found in page {start_page} - {end_page}",
                        "order": module_count,
                        "units": [],
                    }
                    result["modules"].append(module)

        # If still no modules found, create a default one
        if not result["modules"]:
            result["modules"].append(
                {
                    "title": "Course Content",
                    "description": "Automatically generated course content",
                    "order": 1,
                    "units": [
                        {
                            "title": "Full Course",
                            "description": "Complete course content",
                            "order": 1,
                            "content_generated": False,
                        }
                    ],
                }
            )

        return result

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

    def extract_text_with_pdfminer(self, pdf_path, start_page=None, end_page=None):
        """
        Extract text from PDF using pdfminer.six which has better handling of fonts and encodings.

        Args:
            pdf_path: Path to the PDF file
            start_page: Starting page number (1-based index)
            end_page: Ending page number (1-based index)

        Returns:
            Extracted text as string
        """
        try:
            self.logger.info(f"Extracting text with pdfminer.six from {pdf_path}")

            # Create custom parameters for better text extraction
            laparams = LAParams(
                char_margin=1.0,
                line_margin=0.5,
                word_margin=0.1,
                boxes_flow=0.5,
                detect_vertical=True,
            )

            # Set page range if specified
            page_numbers = None
            if start_page is not None and end_page is not None:
                page_numbers = list(
                    range(start_page - 1, end_page)
                )  # Convert to 0-based index

            # Extract text while suppressing warnings
            with suppress_pdfminer_warnings():
                text = extract_text(
                    pdf_path, laparams=laparams, page_numbers=page_numbers
                )

            # Clean problematic characters
            text = re.sub(r"[^\x00-\x7F\s\.\:\-\,\;\(\)\[\]\{\}\/\\\|]", " ", text)
            # Remove excessive whitespace
            text = re.sub(r"\s+", " ", text)

            if text:
                self.logger.info(
                    f"Successfully extracted {len(text)} characters with pdfminer.six"
                )
                self.logger.debug(f"Text preview: {text[:200]}...")
                return text
            else:
                self.logger.warning("pdfminer.six extraction returned empty text")
                return ""

        except Exception as e:
            self.logger.error(f"Error in pdfminer.six extraction: {str(e)}")
            return ""

    def convert_toc_to_json(self, pdf_path: str):
        """Convert PDF TOC to JSON using multiple extraction methods."""
        self.logger.info(f"Processing PDF: {pdf_path}")
        # Store pdf_path as instance variable for use in other methods
        self.pdf_path = pdf_path

        # Create a default structure that will be used if extraction fails
        default_result = {
            "modules": [],
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "source_file": os.path.basename(pdf_path),
            },
            "course_id": None,  # Will be set by the API
            "course_title": None,  # Will be set by the API
        }

        # Extract basic document information first
        try:
            # Try to get document title from PDF metadata
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            if metadata and "title" in metadata and metadata["title"]:
                title = metadata["title"]
                if title and len(title) > 3:
                    default_result["course_title"] = title
                    self.logger.info(f"Extracted title from PDF metadata: {title}")
            doc.close()
        except Exception as e:
            self.logger.error(f"Error extracting PDF metadata: {str(e)}")

        # For corrupted PDFs, try to extract structured info directly
        try:
            # Try direct pdfminer.six extraction of the whole document
            full_text = self.extract_text_with_pdfminer(pdf_path)

            # Look for TOC section using common markers
            toc_markers = [
                "table of contents",
                "contents",
                "toc",
                "index",
                "chapter",
                "section",
                "part",
                "module",
                "unit",
            ]

            # Find potential TOC sections
            found_toc = False
            toc_text = ""

            # Split into lines and look for TOC
            lines = full_text.split("\n")
            for i, line in enumerate(lines):
                line_lower = line.lower()

                # Check if this line indicates the start of a TOC
                if any(marker in line_lower for marker in toc_markers):
                    found_toc = True
                    toc_start = i
                    # Collect up to 100 lines after TOC marker for analysis
                    end_idx = min(i + 100, len(lines))
                    toc_text = "\n".join(lines[i:end_idx])
                    self.logger.info(f"Found potential TOC starting at line {i}")
                    break

            if found_toc and toc_text:
                self.logger.info("Processing extracted TOC text")

                # Try regex-based extraction first as it works better for corrupted text
                try:
                    regex_result = self.process_toc_with_regex(toc_text)
                    if regex_result["modules"] and len(regex_result["modules"]) > 0:
                        self.logger.info(
                            f"Successfully extracted TOC using regex from pdfminer text: {len(regex_result['modules'])} modules"
                        )
                        # Validate and fix TOC structure
                        validated_result = self.validate_toc_structure(regex_result)
                        return validated_result
                except Exception as e:
                    self.logger.error(
                        f"Error in regex-based TOC extraction from pdfminer text: {str(e)}"
                    )
        except Exception as e:
            self.logger.error(f"Error during direct pdfminer extraction: {str(e)}")

        # 1. Try to extract bookmarks directly from PDF if available
        try:
            doc = fitz.open(pdf_path)
            toc_bookmarks = doc.get_toc()

            if toc_bookmarks and len(toc_bookmarks) > 0:
                self.logger.info(f"Found {len(toc_bookmarks)} bookmarks in PDF")

                # Process bookmarks into modules and units
                modules = {}
                current_module = None
                module_count = 0

                for level, title, page in toc_bookmarks:
                    if not title.strip() or title.startswith("/gid"):
                        continue

                    # Clean the title
                    clean_title = self.clean_title(title)

                    # Level 1 entries are modules
                    if level == 1:
                        module_count += 1
                        current_module = {
                            "title": clean_title,
                            "description": f"It is found in page {page}",
                            "order": module_count,
                            "units": [],
                        }
                        modules[clean_title] = current_module
                    # Level 2+ entries are units
                    elif level >= 2 and current_module:
                        unit = {
                            "title": clean_title,
                            "description": f"It is found in page {page}",
                            "order": len(current_module["units"]) + 1,
                            "content_generated": False,
                        }
                        current_module["units"].append(unit)

                # Check if we've successfully extracted meaningful modules
                if len(modules) > 0:
                    bookmark_result = default_result.copy()
                    bookmark_result["modules"] = list(modules.values())
                    self.logger.info(
                        f"Successfully extracted TOC from bookmarks: {len(bookmark_result['modules'])} modules"
                    )

                    # Add some validation to ensure we got meaningful content
                    valid_modules = [
                        m
                        for m in bookmark_result["modules"]
                        if m["title"] and not m["title"].startswith("/gid")
                    ]

                    if len(valid_modules) > 0:
                        self.logger.info("Using bookmarks-based TOC structure")
                        doc.close()
                        # Validate and fix TOC structure
                        validated_result = self.validate_toc_structure(bookmark_result)
                        return validated_result
                    else:
                        self.logger.warning(
                            "Bookmarks extraction yielded invalid content, falling back to text extraction"
                        )
            else:
                self.logger.info(
                    "No bookmarks found in PDF, proceeding with text extraction"
                )

            doc.close()
        except Exception as e:
            self.logger.error(f"Error extracting bookmarks: {str(e)}")

        # 2. Fallback to text extraction if bookmarks approach fails
        toc_text = self.extract_toc_text(pdf_path)
        if not toc_text or len(toc_text.strip()) < 100:
            self.logger.error("Failed to extract meaningful TOC text")
            self.logger.info(f"Returning default structure with empty modules")
            # Validate and fix TOC structure
            validated_result = self.validate_toc_structure(default_result)
            return validated_result

        self.logger.info(f"Extracted TOC text length: {len(toc_text)}")
        self.logger.debug(f"TOC text preview: {toc_text[:500]}...")

        # 3. Try regex-based TOC processing first (more reliable for page numbers)
        try:
            regex_result = self.process_toc_with_regex(toc_text)
            if regex_result["modules"] and len(regex_result["modules"]) > 0:
                self.logger.info(
                    f"Successfully extracted TOC using regex: {len(regex_result['modules'])} modules"
                )
                # Validate and fix TOC structure
                validated_result = self.validate_toc_structure(regex_result)
                return validated_result
            else:
                self.logger.warning(
                    "Regex-based TOC extraction failed, trying LLM approach"
                )
        except Exception as e:
            self.logger.error(f"Error in regex-based TOC extraction: {str(e)}")

        # Check for common issues in extracted text
        if toc_text.count("/gid") > 10 or toc_text.count("obj") > 10:
            self.logger.warning(
                "Extracted text appears to contain PDF internal markers instead of content"
            )
            # Try to clean up the text
            toc_text = re.sub(r"/gid\d+", "", toc_text)
            toc_text = re.sub(r"\d+ \d+ obj", "", toc_text)

            # Check if we still have meaningful content after cleanup
            if len(toc_text.strip()) < 100:
                self.logger.error("Insufficient text after cleanup")
                # Validate and fix TOC structure
                validated_result = self.validate_toc_structure(default_result)
                return validated_result

        # 4. Try LLM-based approach as a fallback
        prompt = self.get_toc_prompt(toc_text)
        self.logger.info("Generated prompt for LLM")

        try:
            result = self.process_text_with_llama(toc_text, prompt)
            self.logger.info(f"Query Response type: {type(result)}")

            # Check if result is a string (error message)
            if isinstance(result, str):
                self.logger.error(f"Error from LLM: {result}")
                # Validate and fix TOC structure
                validated_result = self.validate_toc_structure(default_result)
                return validated_result

            # Ensure the result has the expected structure
            if not isinstance(result, dict):
                self.logger.error(f"Result is not a dictionary: {type(result)}")
                # Validate and fix TOC structure
                validated_result = self.validate_toc_structure(default_result)
                return validated_result

            if "modules" not in result:
                self.logger.error("Result does not have 'modules' key")
                result["modules"] = []

            # Ensure metadata is present
            if "metadata" not in result:
                self.logger.warning("Result does not have 'metadata' key, adding it")
                result["metadata"] = {
                    "generated_at": datetime.utcnow().isoformat(),
                    "source_file": os.path.basename(pdf_path),
                }

            # Add required fields for API
            result["course_id"] = None  # Will be set by the API
            result["course_title"] = None  # Will be set by the API

            # Log the number of modules
            self.logger.info(f"Number of modules found: {len(result['modules'])}")

            # Clean up module data
            cleaned_modules = []
            seen_titles = set()

            for module in result.get("modules", []):
                # Skip invalid modules
                if not module.get("title") or module.get("title").startswith("/gid"):
                    continue

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

                # Clean up units
                valid_units = []
                for unit in module.get("units", []):
                    # Skip invalid units
                    if not unit.get("title") or unit.get("title").startswith("/gid"):
                        continue

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

                    valid_units.append(unit)

                module["units"] = valid_units

                # Check if this module title has been seen before
                if module.get("title") not in seen_titles:
                    seen_titles.add(module.get("title"))
                    cleaned_modules.append(module)
                else:
                    # If it's a duplicate, merge its units with the existing module
                    for existing_module in cleaned_modules:
                        if existing_module.get("title") == module.get("title"):
                            # Add units from the duplicate module to the existing one
                            for unit in module.get("units", []):
                                if unit not in existing_module["units"]:
                                    existing_module["units"].append(unit)
                            break

            # Replace the modules list with the cleaned one
            result["modules"] = cleaned_modules

            # If we ended up with no valid modules, return the default structure
            if not result["modules"]:
                self.logger.warning("No valid modules found after cleanup")
                # Validate and fix TOC structure
                validated_result = self.validate_toc_structure(default_result)
                return validated_result

            # Ensure order is sequential for both modules and units
            for i, module in enumerate(result["modules"]):
                module["order"] = i + 1
                for j, unit in enumerate(module["units"]):
                    unit["order"] = j + 1

            self.logger.info(f"Final structure has {len(result['modules'])} modules")

            # Validate and fix TOC structure
            validated_result = self.validate_toc_structure(result)
            return validated_result

        except Exception as e:
            self.logger.error(f"Error in convert_toc_to_json: {str(e)}", exc_info=True)
            # Validate and fix TOC structure
            validated_result = self.validate_toc_structure(default_result)
            return validated_result

    def get_text_with_best_method(self, pdf_path, start_page=None, end_page=None):
        """
        Get text from a PDF using the best available extraction method.
        This tries multiple methods and returns the best result.

        Args:
            pdf_path: Path to the PDF file
            start_page: First page to process (1-based index)
            end_page: Last page to process (1-based index)

        Returns:
            str: Extracted text
        """
        self.logger.info(f"Extracting text from {pdf_path} using best method")

        # First, check if the PDF has corrupted text encoding
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count > 0:
                # Sample text from a few pages
                text_sample = ""
                for page_idx in [0, min(3, doc.page_count - 1)]:
                    if page_idx < doc.page_count:
                        text_sample += doc[page_idx].get_text()[:500]

                doc.close()

                # If text appears corrupted, use OCR right away
                if self.is_text_corrupted(text_sample):
                    self.logger.warning("Detected corrupted text encoding, using OCR")
                    return self.extract_text_from_corrupted_pdf(
                        pdf_path, start_page, end_page
                    )
        except Exception as e:
            self.logger.error(f"Error checking for text corruption: {str(e)}")

        # Try all methods and pick the best result
        results = []
        error_count = 0

        # Method 1: pdfminer.six with different encodings
        try:
            self.logger.info("Trying pdfminer.six extraction")
            text = self.extract_text_with_pdfminer(pdf_path, start_page, end_page)
            if text and len(text.strip()) > 50:
                results.append((text, len(text), "pdfminer"))
        except Exception as e:
            self.logger.error(f"pdfminer.six extraction failed: {str(e)}")
            error_count += 1

        # Method 2: PyMuPDF
        try:
            self.logger.info("Trying PyMuPDF extraction")
            text = ""
            doc = fitz.open(pdf_path)

            # Determine page range
            total_pages = len(doc)
            if start_page is None:
                start_page = 1
            if end_page is None:
                end_page = total_pages

            # Adjust to valid range
            start_page = max(1, min(start_page, total_pages))
            end_page = max(start_page, min(end_page, total_pages))

            # Extract text
            for page_num in range(start_page - 1, end_page):
                if page_num < total_pages:
                    page = doc[page_num]
                    page_text = page.get_text()
                    text += page_text + "\n"

            doc.close()

            if text and len(text.strip()) > 50:
                results.append((text, len(text), "pymupdf"))
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {str(e)}")
            error_count += 1

        # Method 3: PyPDF2
        try:
            self.logger.info("Trying PyPDF2 extraction")
            text = ""

            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                # Determine page range
                if start_page is None:
                    start_page = 1
                if end_page is None:
                    end_page = total_pages

                # Adjust to valid range
                start_page = max(1, min(start_page, total_pages))
                end_page = max(start_page, min(end_page, total_pages))

                # Extract text
                for page_num in range(start_page - 1, end_page):
                    if page_num < total_pages:
                        page_text = pdf_reader.pages[page_num].extract_text()
                        if page_text:
                            text += page_text + "\n"

            if text and len(text.strip()) > 50:
                results.append((text, len(text), "pypdf2"))
        except Exception as e:
            self.logger.error(f"PyPDF2 extraction failed: {str(e)}")
            error_count += 1

        # If all methods failed or returned little text, try OCR
        if (
            len(results) == 0
            or (results and max(len(r[0]) for r in results) < 100)
            or error_count == 3
        ):
            self.logger.info(
                "Text extraction methods failed or returned little text, trying OCR"
            )
            try:
                text = self.extract_text_from_corrupted_pdf(
                    pdf_path, start_page, end_page
                )
                if text and len(text.strip()) > 50:
                    results.append((text, len(text), "ocr"))
            except Exception as e:
                self.logger.error(f"OCR extraction failed: {str(e)}")

        # Return the best result (the one with the most text)
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            best_text, length, method = results[0]
            self.logger.info(
                f"Selected best extraction method: {method} with {length} characters"
            )
            return best_text
        else:
            self.logger.error("All extraction methods failed")
            return ""


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
