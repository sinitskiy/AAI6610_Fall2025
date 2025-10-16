
import requests
import os
import json
import time
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import feedparser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AntibodyNewsScrScraper:
    """Main class for scraping antibody engineering news from various sources"""
    
    def __init__(self, output_dir="antibody_news", delay=1):
        """
        Initialize the scraper
        
        Args:
            output_dir (str): Directory to save text files
            delay (int): Delay between requests in seconds
        """
        self.output_dir = output_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Define news sources
        self.news_sources = {
            'academic_journals': [
                {
                    'name': 'Antibody Therapeutics (Oxford Academic)',
                    'url': 'https://academic.oup.com/abt',
                    'rss': 'https://academic.oup.com/rss/site_5404/3135.xml'
                },
                {
                    'name': 'mAbs Journal (Taylor & Francis)',
                    'url': 'https://www.tandfonline.com/journals/kmab20',
                    'rss': 'https://www.tandfonline.com/feed/rss/kmab20'
                },
                {
                    'name': 'Antibodies MDPI',
                    'url': 'https://www.mdpi.com/journal/antibodies',
                    'rss': 'https://www.mdpi.com/rss/journal/antibodies'
                },
                {
                    'name': 'Frontiers in Immunology - Antibody Research',
                    'url': 'https://www.frontiersin.org/journals/immunology/sections/cancer-immunity-and-immunotherapy',
                    'search_url': 'https://www.frontiersin.org/search?query=antibody+engineering'
                },
                {
                    'name': 'Nature Biotechnology',
                    'url': 'https://www.nature.com/nbt/',
                    'search_url': 'https://www.nature.com/search?q=antibody+engineering'
                }
            ],
            'industry_news': [
                {
                    'name': 'BioPharma Dive',
                    'url': 'https://www.biopharmadive.com/',
                    'search_url': 'https://www.biopharmadive.com/search/?q=antibody+engineering'
                },
                {
                    'name': 'FierceBiotech',
                    'url': 'https://www.fiercebiotech.com/',
                    'search_url': 'https://www.fiercebiotech.com/search/site/antibody%20engineering'
                },
                {
                    'name': 'BioWorld',
                    'url': 'https://www.bioworld.com/',
                    'search_url': 'https://www.bioworld.com/search?q=antibody+engineering'
                },
                {
                    'name': 'GenomeWeb',
                    'url': 'https://www.genomeweb.com/',
                    'search_url': 'https://www.genomeweb.com/search?query=antibody+engineering'
                }
            ],
            'university_news': [
                {
                    'name': 'Cornell Chronicle',
                    'url': 'https://news.cornell.edu/',
                    'search_url': 'https://news.cornell.edu/search?keys=antibody+engineering'
                },
                {
                    'name': 'MIT News',
                    'url': 'https://news.mit.edu/',
                    'search_url': 'https://news.mit.edu/search/antibody%20engineering'
                },
                {
                    'name': 'Stanford Medicine News',
                    'url': 'https://med.stanford.edu/news/',
                    'search_url': 'https://med.stanford.edu/news/search.html?q=antibody+engineering'
                },
                {
                    'name': 'Harvard Medical School News',
                    'url': 'https://hms.harvard.edu/news',
                    'search_url': 'https://hms.harvard.edu/news?search=antibody%20engineering'
                }
            ],
            'biotech_companies': [
                {
                    'name': 'Genentech Press Releases',
                    'url': 'https://www.gene.com/media/press-releases',
                    'keywords': ['antibody', 'monoclonal', 'bispecific']
                },
                {
                    'name': 'Gilead Sciences News',
                    'url': 'https://www.gilead.com/news-and-press',
                    'keywords': ['antibody', 'immunotherapy', 'biologics']
                },
                {
                    'name': 'Amgen Newsroom',
                    'url': 'https://www.amgen.com/newsroom/',
                    'keywords': ['antibody', 'therapeutic', 'engineering']
                }
            ]
        }

    def extract_publication_date(self, soup, url):
        """
        Extract publication date from various HTML elements
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            url (str): Article URL for context
            
        Returns:
            str: Publication date or 'Unknown'
        """
        # Common date selectors
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            'meta[name="date"]',
            'time[datetime]',
            '.publish-date',
            '.publication-date',
            '.article-date',
            '.news-date',
            '.date-published'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('content') or element.get('datetime') or element.get_text().strip()
                if date_str:
                    return self.parse_date_string(date_str)
        
        # Try to extract from URL if it contains date pattern
        url_date_match = re.search(r'/(\d{4}/\d{1,2}/\d{1,2})', url)
        if url_date_match:
            return url_date_match.group(1).replace('/', '-')
            
        # Try to find date in text content
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{1,2}-\d{1,2})',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}'
        ]
        
        text_content = soup.get_text()
        for pattern in date_patterns:
            match = re.search(pattern, text_content)
            if match:
                return self.parse_date_string(match.group(1))
                
        return 'Unknown'

    def parse_date_string(self, date_str):
        """
        Parse various date string formats
        
        Args:
            date_str (str): Date string to parse
            
        Returns:
            str: Formatted date string (YYYY-MM-DD)
        """
        try:
            # Try ISO format first
            if 'T' in date_str:
                date_str = date_str.split('T')[0]
            
            # Try different formats
            formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%b %d, %Y']
            
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
                    
            return date_str
        except Exception:
            return 'Unknown'

    def scrape_url(self, url, source_name):
        """
        Scrape content from a single URL
        
        Args:
            url (str): URL to scrape
            source_name (str): Name of the source
            
        Returns:
            dict: Article data
        """
        try:
            logger.info(f"Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else 'No Title'
            
            # Extract publication date
            pub_date = self.extract_publication_date(soup, url)
            
            # Extract main content
            content_selectors = [
                'article', '.article-content', '.entry-content',
                '.post-content', '.news-content', 'main'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove script and style elements
                    for script in content_elem(["script", "style"]):
                        script.decompose()
                    content = content_elem.get_text().strip()
                    break
            
            if not content:
                # Fall back to body content
                body = soup.find('body')
                if body:
                    for script in body(["script", "style", "nav", "header", "footer"]):
                        script.decompose()
                    content = body.get_text().strip()
            
            # Clean up content
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r'\s+', ' ', content)
            
            return {
                'title': title_text,
                'url': url,
                'source': source_name,
                'publication_date': pub_date,
                'content': content,
                'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None

    def scrape_rss_feed(self, rss_url, source_name):
        """
        Scrape articles from RSS feed
        
        Args:
            rss_url (str): RSS feed URL
            source_name (str): Name of the source
            
        Returns:
            list: List of article URLs
        """
        try:
            logger.info(f"Parsing RSS feed: {rss_url}")
            feed = feedparser.parse(rss_url)
            
            articles = []
            for entry in feed.entries[:10]:  # Limit to recent 10 articles
                if hasattr(entry, 'link') and hasattr(entry, 'title'):
                    # Check if title contains antibody-related keywords
                    title = entry.title.lower()
                    if any(keyword in title for keyword in ['antibody', 'antibodies', 'monoclonal', 'therapeutic', 'immunotherapy', 'bispecific']):
                        articles.append({
                            'url': entry.link,
                            'title': entry.title,
                            'summary': getattr(entry, 'summary', ''),
                            'published': getattr(entry, 'published', 'Unknown')
                        })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {rss_url}: {str(e)}")
            return []

    def save_article(self, article_data):
        """
        Save article data to a text file
        
        Args:
            article_data (dict): Article data to save
        """
        try:
            # Create filename
            title = re.sub(r'[^\w\s-]', '', article_data['title'])
            title = re.sub(r'[-\s]+', '-', title)[:100]  # Limit length
            date_str = article_data['publication_date'].replace(':', '-')
            filename = f"{date_str}_{title}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            # Ensure unique filename
            counter = 1
            original_filepath = filepath
            while os.path.exists(filepath):
                name, ext = os.path.splitext(original_filepath)
                filepath = f"{name}_{counter}{ext}"
                counter += 1
            
            # Write article content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article_data['title']}\n")
                f.write(f"Source: {article_data['source']}\n")
                f.write(f"URL: {article_data['url']}\n")
                f.write(f"Publication Date: {article_data['publication_date']}\n")
                f.write(f"Scraped Date: {article_data['scraped_date']}\n")
                f.write("="*80 + "\n\n")
                f.write(article_data['content'])
            
            logger.info(f"Saved article: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving article: {str(e)}")

    def search_google_news(self, query="antibody engineering", num_results=20):
        """
        Search Google News for antibody engineering articles
        
        Args:
            query (str): Search query
            num_results (int): Number of results to fetch
            
        Returns:
            list: List of article URLs
        """
        try:
            # Note: This is a simplified approach. For production use,
            # consider using Google News RSS or official APIs
            search_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en&gl=US&ceid=US:en"
            
            articles = self.scrape_rss_feed(search_url, "Google News")
            return articles[:num_results]
            
        except Exception as e:
            logger.error(f"Error searching Google News: {str(e)}")
            return []

    def run_full_scrape(self):
        """
        Run complete scraping process for all sources
        """
        logger.info("Starting comprehensive antibody engineering news scrape...")
        
        all_articles = []
        
        # Scrape academic journals
        logger.info("Scraping academic journals...")
        for source in self.news_sources['academic_journals']:
            if 'rss' in source:
                articles = self.scrape_rss_feed(source['rss'], source['name'])
                for article in articles:
                    article_data = self.scrape_url(article['url'], source['name'])
                    if article_data:
                        all_articles.append(article_data)
                        self.save_article(article_data)
                        time.sleep(self.delay)
        
        # Scrape specific URLs (like the Cornell article you mentioned)
        specific_urls = [
            {
                'url': 'https://news.cornell.edu/stories/2025/08/bioengineered-bacteria-could-lead-therapeutic-antibody-drugs',
                'source': 'Cornell Chronicle'
            },
            {
                'url': 'https://news.vumc.org/2025/03/10/vumc-to-develop-ai-technology-for-therapeutic-antibody-discovery/',
                'source': 'Vanderbilt University Medical Center'
            }
        ]
        
        logger.info("Scraping specific articles...")
        for item in specific_urls:
            article_data = self.scrape_url(item['url'], item['source'])
            if article_data:
                all_articles.append(article_data)
                self.save_article(article_data)
                time.sleep(self.delay)
        
        # Search Google News for recent articles
        logger.info("Searching Google News...")
        news_articles = self.search_google_news("antibody engineering breakthrough 2025", 10)
        for article in news_articles:
            article_data = self.scrape_url(article['url'], "Google News")
            if article_data:
                all_articles.append(article_data)
                self.save_article(article_data)
                time.sleep(self.delay)
        
        # Generate summary report
        self.generate_summary_report(all_articles)
        
        logger.info(f"Scraping complete! Found {len(all_articles)} articles.")
        logger.info(f"Articles saved to: {os.path.abspath(self.output_dir)}")

    def generate_summary_report(self, articles):
        """
        Generate a summary report of all scraped articles
        
        Args:
            articles (list): List of article data dictionaries
        """
        try:
            report_path = os.path.join(self.output_dir, "summary_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("ANTIBODY ENGINEERING NEWS SCRAPING SUMMARY REPORT\n")
                f.write("="*60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total articles found: {len(articles)}\n\n")
                
                f.write("RECOMMENDED NEWS SOURCES FOR ANTIBODY ENGINEERING:\n")
                f.write("-"*50 + "\n\n")
                
                # Academic Journals
                f.write("ACADEMIC JOURNALS:\n")
                f.write("• Antibody Therapeutics (Oxford Academic) - https://academic.oup.com/abt\n")
                f.write("• mAbs Journal (Taylor & Francis) - https://www.tandfonline.com/journals/kmab20\n")
                f.write("• Antibodies (MDPI) - https://www.mdpi.com/journal/antibodies\n")
                f.write("• Frontiers in Immunology - https://www.frontiersin.org/journals/immunology\n")
                f.write("• Nature Biotechnology - https://www.nature.com/nbt/\n")
                f.write("• Science Translational Medicine - https://stm.sciencemag.org/\n\n")
                
                # Industry News
                f.write("INDUSTRY NEWS SOURCES:\n")
                f.write("• BioPharma Dive - https://www.biopharmadive.com/\n")
                f.write("• FierceBiotech - https://www.fiercebiotech.com/\n")
                f.write("• BioWorld - https://www.bioworld.com/\n")
                f.write("• GenomeWeb - https://www.genomeweb.com/\n")
                f.write("• BioCentury - https://www.biocentury.com/\n")
                f.write("• The Antibody Society - https://www.antibodysociety.org/\n\n")
                
                # University News
                f.write("UNIVERSITY & RESEARCH INSTITUTION NEWS:\n")
                f.write("• MIT News - https://news.mit.edu/\n")
                f.write("• Stanford Medicine News - https://med.stanford.edu/news/\n")
                f.write("• Harvard Medical School News - https://hms.harvard.edu/news\n")
                f.write("• Cornell Chronicle - https://news.cornell.edu/\n")
                f.write("• UCSF News - https://www.ucsf.edu/news\n\n")
                
                # Company News
                f.write("MAJOR BIOTECH/PHARMA COMPANY NEWSROOMS:\n")
                f.write("• Genentech - https://www.gene.com/media/press-releases\n")
                f.write("• Gilead Sciences - https://www.gilead.com/news-and-press\n")
                f.write("• Amgen - https://www.amgen.com/newsroom/\n")
                f.write("• Regeneron - https://newsroom.regeneron.com/\n")
                f.write("• AbbVie - https://news.abbvie.com/\n\n")
                
                # Article summaries
                f.write("ARTICLES FOUND IN THIS SCRAPE:\n")
                f.write("-"*40 + "\n\n")
                
                for i, article in enumerate(articles, 1):
                    f.write(f"{i}. {article['title']}\n")
                    f.write(f"   Source: {article['source']}\n")
                    f.write(f"   Date: {article['publication_date']}\n")
                    f.write(f"   URL: {article['url']}\n")
                    f.write(f"   Content preview: {article['content'][:200]}...\n\n")
        
            logger.info(f"Summary report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")

def main():
    """
    Main function to run the antibody engineering news scraper
    """
    print("Antibody Engineering News Scraper")
    print("="*40)
    
    # Initialize scraper
    scraper = AntibodyNewsScrScraper()
    
    # Run complete scraping
    scraper.run_full_scrape()
    
    print(f"\nScraping completed! Check the '{scraper.output_dir}' directory for results.")
    print("\nRecommended similar news channels for antibody engineering:")
    print("• The Antibody Society (https://www.antibodysociety.org/)")
    print("• BioPharma Dive (https://www.biopharmadive.com/)")
    print("• Nature Biotechnology (https://www.nature.com/nbt/)")
    print("• Antibody Therapeutics Journal (https://academic.oup.com/abt)")
    print("• MIT Technology Review - Biotech (https://www.technologyreview.com/topic/biotech/)")

if __name__ == "__main__":
    main()
