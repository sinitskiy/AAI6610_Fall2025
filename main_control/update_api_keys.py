#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API Key Update Solution for Scraper Scripts
Change hard-coded keys to load from environment variables/configuration
"""

# ============================================================================
# Solution 1: Simple Approach - Direct use of os.environ
# ============================================================================

# ========== scraper_linkedin.py Update ==========
"""
Original code:
GOOGLE_API_KEY = "AIzaSyChu1dkW1nB4ZVc0MsQIC2D2akpB772Gm8"
SEARCH_ENGINE_ID = "c78859e23e54344a3"

Updated to:
"""
import os

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")

# Add validation
if not GOOGLE_API_KEY or not SEARCH_ENGINE_ID:
    print("Warning: Google API credentials not found!")
    print("   Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID in .env file")


# ========== scraper_reddit.py Update ==========
"""
Original code:
REDDIT_CLIENT_ID="NJl1oqWfoXySI0LtxZqfLw"
REDDIT_CLIENT_SECRET="e1HUShZV-RSnNi90ncnJAidfX7d2LA"
REDDIT_USER_AGENT = "python:ml_post_scraper:v1.0 (by u/Ok-Mobile-2410)"

Updated to:
"""
import os

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "python:ml_post_scraper:v1.0")

# Add validation
if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    print("Warning: Reddit API credentials not found!")
    print("   Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file")


# ========== scraper_openalex.py Update ==========
"""
Original code:
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "xiang.siq@northeastern.edu").strip()
OPENREVIEW_MAILTO = os.environ.get("OPENREVIEW_MAILTO", "").strip()
OPENALEX_BASE = "https://api.openalex.org"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
OPENREVIEW_BASE = "https://api.openreview.net"
DEFAULT_USER_AGENT = "AAI6610-Pipeline/1.0 (mailto:xiang.siq@northeastern.edu)"

Updated to:
"""
import os

# Read from environment variables
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "xiang.siq@northeastern.edu").strip()
OPENREVIEW_MAILTO = os.environ.get("OPENREVIEW_MAILTO", UNPAYWALL_EMAIL).strip()

# API URLs
OPENALEX_BASE = os.environ.get("OPENALEX_BASE_URL", "https://api.openalex.org")
UNPAYWALL_BASE = os.environ.get("UNPAYWALL_BASE_URL", "https://api.unpaywall.org/v2")
OPENREVIEW_BASE = os.environ.get("OPENREVIEW_BASE_URL", "https://api.openreview.net")

# User Agent
DEFAULT_USER_AGENT = os.environ.get("OPENALEX_USER_AGENT", 
                                    f"AAI6610-Pipeline/1.0 (mailto:{UNPAYWALL_EMAIL})")

# Add validation
if not UNPAYWALL_EMAIL or UNPAYWALL_EMAIL == "xiang.siq@northeastern.edu":
    print("Info: Using default email for Unpaywall API")


# ========== cluster_engine.py Update ==========
"""
Original code:
openai.api_key = "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"

Updated to:
"""
import os
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# Add validation
if not openai.api_key:
    print("Error: OpenAI API key not found!")
    print("   Please set OPENAI_API_KEY in .env file")
    raise ValueError("OpenAI API key is required for clustering")


# ============================================================================
# Solution 2: Use Configuration Loader (Recommended - More Flexible)
# ============================================================================

"""
Add at the beginning of each script:
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration loader
try:
    from main_control.config_loader import get_config_loader
    
    # Load configuration
    config_loader = get_config_loader(PROJECT_ROOT)
    
    # Get API keys
    # LinkedIn
    GOOGLE_API_KEY = config_loader.get_api_key('google', 'api_key')
    SEARCH_ENGINE_ID = config_loader.get_api_key('google', 'search_engine_id')
    
    # Reddit
    REDDIT_CLIENT_ID = config_loader.get_api_key('reddit', 'client_id')
    REDDIT_CLIENT_SECRET = config_loader.get_api_key('reddit', 'client_secret')
    REDDIT_USER_AGENT = config_loader.get_api_key('reddit', 'user_agent')
    
    # Academic APIs
    UNPAYWALL_EMAIL = config_loader.get_api_key('unpaywall', 'email')
    OPENREVIEW_MAILTO = config_loader.get_api_key('openreview', 'mailto')
    
    # OpenAI
    openai.api_key = config_loader.get_api_key('openai', 'key')
    
    print("API keys loaded from configuration")

except ImportError:
    # If config loader not available, fallback to environment variables
    print("Config loader not available, using environment variables")
    
    import os
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")
    REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
    # ... etc


# ============================================================================
# Complete Example: Updated version of scraper_linkedin.py
# ============================================================================

"""
Add at the top of file, after existing imports:
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LinkedIn Crawler - Adapted path to whole_pipeline
"""

import os
import re
import pickle
import json
import shutil
# ... other imports ...

from pathlib import Path

# ========== Path Configuration ==========
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRAPER_OUTPUTS = PROJECT_ROOT / "scrapers" / "outputs"
LINKEDIN_COOKIES_FILE = Path(__file__).parent / "linkedin_cookies.json"
POST_FOLDER = SCRAPER_OUTPUTS / "linkedin_posts"

# ========== API Configuration - Load from Environment Variables ==========
# Method 1: Direct use of os.environ
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")

# Validation
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found!")
    print("   Please set it in .env file")
    exit(1)

if not SEARCH_ENGINE_ID:
    print("Error: GOOGLE_SEARCH_ENGINE_ID not found!")
    print("   Please set it in .env file")
    exit(1)

print(f"Google API configured")

# Search keywords
KEYWORDS = [
    "Applied to uncertainty prediction in ML models",
    "uncertainty estimation deep learning",
    # ... other keywords ...
]

# ... rest of code remains unchanged ...


# ============================================================================
# Complete Example: Updated version of cluster_engine.py
# ============================================================================

"""
Update at the top of file:
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering Engine - Topic clustering for filtered text
"""

import sys
import io
import os
import openai

# Windows GBK encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ========== OpenAI API Configuration ==========
# Read from environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# Validation
if not openai.api_key:
    print("Error: OPENAI_API_KEY not found!")
    print("   Please set it in .env file")
    print("   Required for generating embeddings")
    exit(1)

print(f"OpenAI API configured")

# ... other imports and configuration ...

# ============================================================================
# Path Configuration
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "..", "outputs")
# ... rest of code remains unchanged ...
