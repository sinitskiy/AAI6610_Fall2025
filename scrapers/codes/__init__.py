#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scrapers Package
Contains all data scraping modules
"""
# Only import modules with class definitions
from .pdf_to_txt import PDFConverter

# Optional scrapers - only import those with classes
try:
    from .scraper_arxiv import ArxivScraper
except (ImportError, AttributeError):
    ArxivScraper = None

try:
    from .scraper_biorxiv import BioRxivScraper
except (ImportError, AttributeError):
    BioRxivScraper = None

try:
    from .scraper_openalex import OpenAlexScraper
except (ImportError, AttributeError):
    OpenAlexScraper = None

try:
    from .scraper_news import NewsScraper
except (ImportError, AttributeError):
    NewsScraper = None

# Define package's public interface
__all__ = [
    'PDFConverter',
    'ArxivScraper',
    'BioRxivScraper',
    'OpenAlexScraper',
    'NewsScraper',
]

# Package version information
__version__ = '1.0.0'
__author__ = 'AAI6610 Group 2'

# Package-level configuration
SUPPORTED_SCRAPERS = [
    'linkedin',
    'reddit',
    'arxiv',
    'biorxiv',
    'openalex',
    'news'
]

def get_scraper(scraper_type):
    """
    Factory function: returns corresponding scraper class based on type
    
    Args:
        scraper_type: scraper type string
        
    Returns:
        corresponding scraper class
    """
    scrapers = {
        'arxiv': ArxivScraper,
        'biorxiv': BioRxivScraper,
        'openalex': OpenAlexScraper,
        'news': NewsScraper,
    }
    
    scraper_class = scrapers.get(scraper_type.lower())
    
    if scraper_class is None:
        raise ValueError(f"Unknown scraper type: {scraper_type}")
    
    return scraper_class

# Initialization code when package is imported
print(f"Scrapers package v{__version__} loaded")
print(f"   Available scrapers: {', '.join(SUPPORTED_SCRAPERS)}")
