"""
File parsing utilities for CSV, PDF, JSON, Excel.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pymupdf  # PyMuPDF

from app.logger import setup_logger
from app.utils.exceptions import ParseError

logger = setup_logger(__name__)


class FileParser:
    """Unified parser for common data file formats."""

    @staticmethod
    def parse_csv(file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Parse CSV file with smart defaults.

        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas read_csv arguments

        Returns:
            DataFrame

        Raises:
            ParseError: If parsing fails
        """
        try:
            logger.info(f"📊 Parsing CSV: {file_path.name}")

            # Try common encodings
            encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        # Handle Indian number format (1,23,456)
                        thousands=",",
                        **kwargs,
                    )
                    logger.debug(f"✅ CSV parsed with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ParseError("Could not parse CSV with any common encoding")

            logger.info(f"✅ CSV parsed: {df.shape[0]} rows × {df.shape[1]} columns")
            return df

        except Exception as e:
            logger.error(f"❌ CSV parse error: {e}")
            raise ParseError(f"Failed to parse CSV: {e}")

    @staticmethod
    def parse_json(file_path: Path) -> Dict[str, Any]:
        """
        Parse JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON as dictionary

        Raises:
            ParseError: If parsing fails
        """
        try:
            logger.info(f"📋 Parsing JSON: {file_path.name}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"✅ JSON parsed: {len(data)} root keys")
            return data

        except Exception as e:
            logger.error(f"❌ JSON parse error: {e}")
            raise ParseError(f"Failed to parse JSON: {e}")

    @staticmethod
    def parse_excel(file_path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Parse Excel file.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name (default: first sheet)

        Returns:
            DataFrame

        Raises:
            ParseError: If parsing fails
        """
        try:
            logger.info(f"📊 Parsing Excel: {file_path.name}")

            df = pd.read_excel(file_path, sheet_name=sheet_name or 0)

            logger.info(f"✅ Excel parsed: {df.shape[0]} rows × {df.shape[1]} columns")
            return df

        except Exception as e:
            logger.error(f"❌ Excel parse error: {e}")
            raise ParseError(f"Failed to parse Excel: {e}")

    @staticmethod
    def parse_pdf_text(file_path: Path) -> str:
        """
        Extract text from PDF.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text

        Raises:
            ParseError: If parsing fails
        """
        try:
            logger.info(f"📄 Parsing PDF: {file_path.name}")

            doc = pymupdf.open(file_path)
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text_parts.append(page.get_text())

            doc.close()

            full_text = "\n\n".join(text_parts)
            logger.info(
                f"✅ PDF parsed: {len(full_text)} characters from {len(doc)} pages"
            )
            return full_text

        except Exception as e:
            logger.error(f"❌ PDF parse error: {e}")
            raise ParseError(f"Failed to parse PDF: {e}")

    @staticmethod
    def parse_pdf_tables(file_path: Path) -> List[pd.DataFrame]:
        """
        Extract tables from PDF.

        Args:
            file_path: Path to PDF file

        Returns:
            List of DataFrames (one per table)

        Raises:
            ParseError: If parsing fails
        """
        try:
            logger.info(f"📊 Extracting tables from PDF: {file_path.name}")

            doc = pymupdf.open(file_path)
            tables = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Try to find tables (basic implementation)
                # For production, consider using tabula-py or camelot
                page_tables = page.find_tables()

                for table in page_tables:
                    # Convert to DataFrame
                    df = table.to_pandas()
                    if not df.empty:
                        tables.append(df)

            doc.close()

            logger.info(f"✅ Found {len(tables)} tables in PDF")
            return tables

        except Exception as e:
            logger.error(f"❌ PDF table extraction error: {e}")
            raise ParseError(f"Failed to extract tables from PDF: {e}")

    @staticmethod
    def smart_parse(file_path: Path) -> Any:
        """
        Auto-detect file type and parse accordingly.

        Args:
            file_path: Path to file

        Returns:
            Parsed data (DataFrame, dict, or str)

        Raises:
            ParseError: If file type unsupported or parsing fails
        """
        suffix = file_path.suffix.lower()

        parsers = {
            ".csv": FileParser.parse_csv,
            ".json": FileParser.parse_json,
            ".xlsx": FileParser.parse_excel,
            ".xls": FileParser.parse_excel,
            ".pdf": FileParser.parse_pdf_text,
        }

        parser = parsers.get(suffix)
        if not parser:
            raise ParseError(f"Unsupported file type: {suffix}")

        return parser(file_path)


# Global parser instance (THIS WAS MISSING!)
file_parser = FileParser()
