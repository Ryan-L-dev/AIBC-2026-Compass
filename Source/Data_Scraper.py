"""
Data Scraper for CPF.gov.sg
============================
Scrapes structured content from the CPF website using its sitemap.
Respects robots.txt, filters URLs by allowed paths, extracts main
content as markdown, and saves results to mirrored folder structure
with a CSV log.

Usage:
    python Data_Scraper.py

Configuration:
    Modify values in Constants.py to change target site,
    allowed paths, output directory, delay, or verbosity.

Dependencies:
    pip install requests beautifulsoup4 pandas tqdm markdownify
"""

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import xml.etree.ElementTree as ET
import os
import re
import time
import hashlib
from collections import Counter
import pandas as pd
from tqdm import tqdm

from Source.Constants import Constants


class DataScraper:
    """
    Web scraper that extracts main content from CPF.gov.sg pages as markdown.

    Workflow:
        1. Parse robots.txt to determine which URLs are allowed
        2. Read the sitemap XML to discover all available URLs
        3. Filter URLs to only those matching allowed path prefixes
        4. Scrape each allowed URL, converting main content to markdown
        5. Save scraped markdown to a mirrored folder structure
        6. Generate a CSV log with scrape results and metadata
    """

    def __init__(self):
        """Initialise scraper: parse robots.txt, load previous log, and prepare state."""
        self.rp = self._get_robots_txt()
        self.log = []  # Accumulates per-URL scrape results
        self.prev_hashes = self._load_previous_hashes()  # For change detection

    def _print(self, msg):
        """Print a message only when VERBOSE mode is enabled."""
        if Constants.VERBOSE:
            print(msg)

    def _load_previous_hashes(self):
        """
        Load content hashes from the previous scrape log.

        Used to determine if a page's content has changed since the last run,
        which indicates whether re-embedding is required.

        Returns:
            dict: Mapping of URL → previous content hash.
        """
        if not os.path.exists(Constants.LOG_FILE):
            return {}
        try:
            df = pd.read_csv(Constants.LOG_FILE)
            if Constants.COL_CONTENT_HASH not in df.columns:
                return {}
            return dict(zip(df[Constants.COL_URL], df[Constants.COL_CONTENT_HASH]))
        except Exception:
            return {}

    # ==========================================================================
    # STEP 1 — ROBOTS.TXT
    # ==========================================================================

    def _get_robots_txt(self):
        """
        Fetch and parse the target site's robots.txt.

        Returns a RobotFileParser instance that can be queried to check
        whether a given URL is allowed for crawling.
        """
        rp = RobotFileParser()
        try:
            response = requests.get(
                f"https://{Constants.BASE_DOMAIN}/robots.txt",
                headers=Constants.HEADERS,
                timeout=Constants.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            rp.parse(response.text.splitlines())
            self._print("  robots.txt read successfully")
        except requests.RequestException as e:
            self._print(f"  Warning: Could not read robots.txt — {e}")
            self._print("  Proceeding without robots.txt restrictions")
        return rp

    def _is_allowed(self, url):
        """Check if the URL is permitted by robots.txt rules."""
        if self.rp is None:
            return False  # Fail-safe: block scraping if robots.txt was unreadable
        return self.rp.can_fetch("*", url)

    # ==========================================================================
    # STEP 2 — SITEMAP PARSING
    # ==========================================================================

    def _get_sitemap_urls(self, sitemap_url):
        """
        Recursively fetch all page URLs from the sitemap.

        Handles both sitemap index files (which reference other sitemaps)
        and regular sitemaps (which list page URLs directly).

        Returns:
            list[str]: All discovered page URLs.
        """
        response = requests.get(sitemap_url, headers=Constants.HEADERS, timeout=Constants.REQUEST_TIMEOUT)
        root = ET.fromstring(response.content)
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # If this is a sitemap index, recurse into each child sitemap
        sitemaps = root.findall("sm:sitemap/sm:loc", namespace)
        if sitemaps:
            urls = []
            for sitemap in sitemaps:
                self._print(f"  Found nested sitemap: {sitemap.text}")
                urls.extend(self._get_sitemap_urls(sitemap.text))
            return urls

        # Otherwise extract page URLs directly
        return [loc.text for loc in root.findall("sm:url/sm:loc", namespace)]

    def _is_in_allowed_paths(self, url):
        """Return True if the URL's path starts with any configured allowed prefix."""
        parsed = urlparse(url)
        return any(parsed.path.startswith(path) for path in Constants.ALLOWED_PATHS)

    # ==========================================================================
    # STEP 3 — CONTENT EXTRACTION
    # ==========================================================================

    def _scrape_content(self, soup):
        """
        Extract main content from a BeautifulSoup-parsed page as markdown.

        Process:
            1. Remove noisy HTML tags (scripts, nav, footer, etc.)
            2. Remove elements with known non-content CSS classes
            3. Locate the main content container (main/article/content)
            4. Convert HTML to markdown using markdownify
            5. Clean up excessive blank lines

        Returns:
            str: Page content converted to markdown format.
        """
        # Strip noisy tags
        for tag in Constants.NOISE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        # Strip noisy CSS class elements
        for css_class in Constants.NOISE_CLASSES:
            for element in soup.find_all(class_=css_class):
                element.decompose()

        # Prefer the main content area; fall back to full page
        target = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup
        )

        # Convert custom heading divs to proper h1-h6 tags
        # CPF site uses classes like 'headline-1' through 'headline-6'
        for level in range(1, 7):
            for el in target.find_all(class_=re.compile(rf'headline-{level}\b')):
                el.name = f"h{level}"

        # Convert to markdown
        markdown = md(str(target), heading_style="ATX", strip=['img'])

        # Remove known noise lines (template boilerplate)
        lines = markdown.splitlines()
        noise_set = set(Constants.NOISE_LINES)
        lines = [l for l in lines if l.strip() not in noise_set]
        markdown = "\n".join(lines)

        # Collapse 3+ consecutive blank lines into 2
        markdown = re.sub(r'\n{3,}', '\n\n', markdown).strip()

        # Truncate at boilerplate markers (FAQs, Resources, Related Reads)
        for marker in Constants.TRUNCATE_MARKERS:
            idx = markdown.find(marker)
            if idx != -1:
                markdown = markdown[:idx].strip()

        return markdown

    def _url_to_filepath(self, url):
        """
        Convert a URL into a local folder path and filename.

        Mirrors the URL path structure under OUTPUT_DIR.
        Example:
            https://www.cpf.gov.sg/member/home-ownership/buying-a-home
            → folder: scraped_pages/member/home-ownership
            → filename: buying-a-home.md

        Returns:
            tuple[str, str]: (folder_path, filename)
        """
        parsed = urlparse(url)
        parts = [p for p in parsed.path.strip("/").split("/") if p]

        if not parts:
            folder = Constants.OUTPUT_DIR
            filename = "index.md"
        else:
            folder = os.path.join(Constants.OUTPUT_DIR, *parts[:-1]) if len(parts) > 1 else Constants.OUTPUT_DIR
            filename = f"{parts[-1]}.md"

        return folder, filename

    def _url_to_category_path(self, url):
        """Extract URL path segments as a hierarchical category path string.

        Example:
            https://www.cpf.gov.sg/member/account-services/fighting-scams/cpf-account-safeguards
            → 'account services/fighting scams/cpf account safeguards'
        """
        parsed = urlparse(url)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        # Skip the first segment (e.g. 'member') as it's the top-level route
        segments = parts[1:] if len(parts) > 1 else parts
        cleaned = [re.sub(r'[^a-zA-Z0-9 ]', ' ', seg.replace('-', ' ')).strip() for seg in segments]
        return "/".join(cleaned)

    def _build_log_entry(self, url, scraped, reason, content_hash=None, needs_embedding=False):
        """Create a standardised log entry dict."""
        return {
            Constants.COL_URL: url,
            Constants.COL_SCRAPED: scraped,
            Constants.COL_REASON: reason,
            Constants.COL_CATEGORY: self._url_to_category_path(url),
            Constants.COL_SCRAPED_DATE: time.strftime("%Y%m%d"),
            Constants.COL_CONTENT_HASH: content_hash,
            Constants.COL_NEEDS_EMBEDDING: needs_embedding,
        }

    # ==========================================================================
    # MAIN PROCESS
    # ==========================================================================

    def process(self):
        """
        Execute the full scraping pipeline.

        Steps:
            1. Read and parse the sitemap to discover URLs
            2. Filter URLs to only those in ALLOWED_PATHS
            3. Scrape each URL: fetch page, extract text, save to file
            4. Skip URLs blocked by robots.txt or that redirect elsewhere
            5. Save a CSV log summarising all scrape attempts
        """
        # Discover URLs from sitemap
        self._print("Reading sitemap...")
        all_urls = self._get_sitemap_urls(Constants.SITEMAP_URL)
        self._print(f"  Total URLs in sitemap : {len(all_urls)}")

        # Apply path filter
        filtered_urls = [url for url in all_urls if self._is_in_allowed_paths(url)]
        self._print(f"  URLs after filtering  : {len(filtered_urls)}")

        # Scrape each filtered URL
        # Note that the log will be used to store the metadata for each URL
        self.log = []
        for url in tqdm(filtered_urls, desc="  Scraping", disable=not Constants.VERBOSE):

            # Respect robots.txt
            if not self._is_allowed(url):
                #self._print(f"  SKIPPED (robots.txt): {url}")
                self.log.append(self._build_log_entry(url, Constants.SCRAPED_NO, "robots.txt"))
                continue

            try:
                # Fetch the page with redirect following
                session = requests.Session()
                session.max_redirects = Constants.MAX_REDIRECTS
                response = session.get(url, headers=Constants.HEADERS, timeout=Constants.REQUEST_TIMEOUT)
                response.raise_for_status()

                # Skip pages that redirect to a different URL
                if response.url != url:
                    #self._print(f"  SKIPPED (redirect): {url} -> {response.url}")
                    self.log.append(self._build_log_entry(url, Constants.SCRAPED_NO, f"Redirected to {response.url}"))
                    continue

                # Parse HTML and extract markdown content
                soup = BeautifulSoup(response.text, "html.parser")
                text = self._scrape_content(soup)

                # Skip pages with no meaningful content
                if not text.strip():
                    self.log.append(self._build_log_entry(url, Constants.SCRAPED_NO, "Empty content after processing"))
                    continue

                content_hash = hashlib.sha256(text.encode()).hexdigest()

                # Save to mirrored folder structure
                folder, filename = self._url_to_filepath(url)
                os.makedirs(folder, exist_ok=True)
                output_path = os.path.join(folder, filename)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"URL: {url}\n")
                    f.write(f"Hash: {content_hash}\n")
                    f.write(f"Category: {self._url_to_category_path(url)}\n")
                    f.write(f"Date Scraped: {time.strftime('%Y%m%d')}\n")
                    f.write(Constants.FILE_SEPARATOR + "\n")
                    f.write(text)

                # Determine if content has changed since last scrape
                prev_hash = self.prev_hashes.get(url)
                needs_embedding = prev_hash is None or prev_hash != content_hash

                #self._print(f"  Saved  : {output_path}")
                #if not needs_embedding:
                #    self._print(f"  Hash unchanged — skip re-embedding")

                self.log.append(self._build_log_entry(url, Constants.SCRAPED_YES, "", content_hash, needs_embedding))

                # Politeness delay between requests
                time.sleep(Constants.DELAY)

            except requests.RequestException as e:
                #self._print(f"  ERROR  : {url} — {e}")
                self.log.append(self._build_log_entry(url, Constants.SCRAPED_NO, f"Error: {e}"))

        # Write the results log
        self._save_log()

        # Detect repeated template lines across all scraped files
        self._detect_template_lines()

        # Save all discovered categories to Resources
        self._save_categories(filtered_urls)

    # ==========================================================================
    # STEP 4 — CATEGORY EXPORT
    # ==========================================================================

    def _save_categories(self, urls):
        """Write category paths (flat list including intermediate paths) and category tree to Resources."""
        all_paths = set()
        for url in urls:
            full_path = self._url_to_category_path(url)
            # Add the full path and all intermediate parent paths
            parts = full_path.split("/")
            for i in range(1, len(parts) + 1):
                all_paths.add("/".join(parts[:i]))

        os.makedirs(os.path.dirname(Constants.CATEGORIES_LIST_FILE), exist_ok=True)

        # Flat list of all unique category paths (including parents)
        with open(Constants.CATEGORIES_LIST_FILE, "w", encoding="utf-8") as f:
            for path in sorted(all_paths):
                f.write(path + "\n")

        # Build and write tree representation
        tree = self._build_category_tree(all_paths)
        tree_str = self._render_tree(tree)
        with open(Constants.CATEGORIES_TREE_FILE, "w", encoding="utf-8") as f:
            f.write(tree_str)

        self._print(f"  Categories saved: {len(all_paths)} paths, tree written")

    def _build_category_tree(self, paths):
        """Build a nested dict representing the category hierarchy."""
        tree = {}
        for path in paths:
            parts = path.split("/")
            node = tree
            for part in parts:
                node = node.setdefault(part, {})
        return tree

    def _render_tree(self, tree, prefix=""):
        """Render a nested dict as a tree string with box-drawing characters."""
        lines = []
        items = sorted(tree.keys())
        for i, key in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{key}")
            extension = "    " if is_last else "│   "
            if tree[key]:
                lines.append(self._render_tree(tree[key], prefix + extension))
        return "\n".join(lines)

    # ==========================================================================
    # STEP 5 — TEMPLATE LINE DETECTION
    # ==========================================================================

    def _detect_template_lines(self):
        """
        Post-processing detection step: identify lines repeated across a high
        percentage of scraped files (likely website template/boilerplate).

        Outputs candidates to Resources/template_lines.txt for manual review.
        Developers should review this file and add confirmed template lines
        to Constants.NOISE_LINES, which are stripped during scraping.
        """
        # Collect all .md files
        md_files = []
        for root, _, files in os.walk(Constants.OUTPUT_DIR):
            for f in files:
                if f.endswith(".md"):
                    md_files.append(os.path.join(root, f))

        if len(md_files) < 3:
            self._print("  Skipping template detection — fewer than 3 files")
            return

        # Count in how many files each non-empty line appears
        line_file_count = Counter()
        for filepath in md_files:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()[Constants.FILE_HEADER_LINES:]
            unique_lines = {
                line.strip() for line in lines
                if line.strip() and len(line.strip()) >= Constants.TEMPLATE_LINE_MIN_LENGTH
            }
            for line in unique_lines:
                line_file_count[line] += 1

        # Identify template lines exceeding the threshold
        threshold_count = len(md_files) * Constants.TEMPLATE_LINE_THRESHOLD
        template_lines = {
            line for line, count in line_file_count.items()
            if count >= threshold_count
        }

        if not template_lines:
            self._print("  No new template line candidates detected")
            return

        self._print(f"  Detected {len(template_lines)} template line candidates — see {Constants.TEMPLATE_LINES_FILE}")

        # Write candidates to Resources folder for manual review
        os.makedirs(os.path.dirname(Constants.TEMPLATE_LINES_FILE), exist_ok=True)
        with open(Constants.TEMPLATE_LINES_FILE, "w", encoding="utf-8") as f:
            for line in sorted(template_lines):
                f.write(line + "\n")

    # ==========================================================================
    # STEP 6 — LOG OUTPUT
    # ==========================================================================

    def _save_log(self):
        """
        Save scrape results to CSV. Metadata is already computed inline
        during scraping, so this method simply writes the log and prints
        a summary including embedding statistics.
        """
        df = pd.DataFrame(self.log)
        df.to_csv(Constants.LOG_FILE, index=False)

        # Print summary
        total = len(df)
        scraped = len(df[df[Constants.COL_SCRAPED] == Constants.SCRAPED_YES])
        skipped = len(df[df[Constants.COL_SCRAPED] == Constants.SCRAPED_NO])
        needs_embed = len(df[df[Constants.COL_NEEDS_EMBEDDING] == True])

        self._print(f"\n  Pages scraped       : {scraped}/{total}")
        self._print(f"  Pages skipped       : {skipped}/{total}")
        self._print(f"  Needs re-embedding  : {needs_embed}/{scraped}")
        self._print(f"  Log saved to        : {Constants.LOG_FILE}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    scraper = DataScraper()
    scraper.process()
