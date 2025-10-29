#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF to TXT Converter
Uses pypdf and pdfminer.six as dual fallback methods
"""
from pathlib import Path
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
import logging

class PDFConverter:
    """PDF to TXT converter"""
    
    def __init__(self, max_pages=None):
        """
        Args:
            max_pages: Maximum page limit (None=all pages)
        """
        self.max_pages = max_pages
        self.logger = logging.getLogger(__name__)
    
    def convert(self, pdf_path, txt_path):
        """
        Convert a single PDF
        
        Args:
            pdf_path: PDF file path
            txt_path: Output TXT path
            
        Returns:
            bool: Returns True on success
        """
        pdf_path = Path(pdf_path)
        txt_path = Path(txt_path)
        
        # If TXT already exists and is non-empty, skip
        if txt_path.exists() and txt_path.stat().st_size > 100:
            return True
        
        try:
            # Method 1: Use pypdf
            text = self._extract_with_pypdf(pdf_path)
            
            # If failed or content too short, try Method 2
            if not text or len(text) < 100:
                text = self._extract_with_pdfminer(pdf_path)
            
            if text and len(text) > 50:
                # Clean text
                text = self._clean_text(text)
                
                # Save
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                return True
            else:
                self.logger.warning(f"No text extracted from {pdf_path.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to convert {pdf_path.name}: {e}")
            return False
    
    def _extract_with_pypdf(self, pdf_path):
        """Extract using pypdf"""
        try:
            reader = PdfReader(pdf_path, strict=False)
            text = []
            
            n_pages = len(reader.pages)
            limit = min(n_pages, self.max_pages) if self.max_pages else n_pages
            
            for i in range(limit):
                page_text = reader.pages[i].extract_text() or ""
                text.append(page_text)
            
            return "\n".join(text).strip()
        except:
            return ""
    
    def _extract_with_pdfminer(self, pdf_path):
        """Extract using pdfminer.six"""
        try:
            kwargs = {}
            if self.max_pages:
                kwargs['maxpages'] = self.max_pages
            
            return pdfminer_extract_text(str(pdf_path), **kwargs).strip()
        except:
            return ""
    
    def _clean_text(self, text):
        """Clean text"""
        import re
        
        # Remove excess whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove control characters (preserve newlines and tabs)
        text = ''.join(ch for ch in text if ch >= ' ' or ch in '\n\t')
        
        return text.strip()

if __name__ == "__main__":
    # Test code
    converter = PDFConverter(max_pages=20)
    
    test_pdf = Path("test.pdf")
    test_txt = Path("test.txt")
    
    if test_pdf.exists():
        success = converter.convert(test_pdf, test_txt)
        print(f"Conversion {'successful' if success else 'failed'}")
