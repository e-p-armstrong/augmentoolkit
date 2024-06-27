import requests
from bs4 import BeautifulSoup
import time
import re

# Function to fetch and parse the content of a Wikipedia page
def fetch_wikipedia_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove references and footnotes
    for sup in soup.find_all('sup', {'class': 'reference'}):
        sup.decompose()
    
    # Extract the main content of the page
    content = soup.find('div', {'id': 'bodyContent'}).get_text(separator='\n', strip=True)
    
    # Extract the title of the page for file naming
    title = soup.find('h1', {'id': 'firstHeading'}).get_text(separator='\n', strip=True)
    return title, content

# Function to get Wikipedia links from a page
def get_wikipedia_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract links to other Wikipedia pages
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/wiki/') and not re.search(r':|#', href):
            full_url = f'https://en.wikipedia.org{href}'
            links.append(full_url)
    
    # Remove duplicates
    links = list(set(links))
    return links

# Function to save content to a .txt file
def save_to_file(title, content):
    # Sanitize the title to create a valid filename
    filename = re.sub(r'[\\/*?:"<>|]', "_", title) + '.txt'
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

# Function to scrape a Wikipedia page and its linked pages with a depth of 1
def scrape_wikipedia_page(url, delay=0.05):
    # Scrape the main page
    main_title, main_content = fetch_wikipedia_content(url)
    save_to_file(main_title, main_content)
    print(f'Scraped and saved content from: {url}')
    
    # Scrape the linked pages
    linked_pages = get_wikipedia_links(url)
    for link in linked_pages:
        time.sleep(delay)  # Delay to avoid stressing Wikipedia's servers
        try:
            title, content = fetch_wikipedia_content(link)
            save_to_file(title, content)
            print(f'Scraped and saved content from: {link}')
        except Exception as e:
            print(f'Failed to scrape {link}: {e}')

# Example usage
main_url = 'https://en.wikipedia.org/wiki/Ancient_Rome'
scrape_wikipedia_page(main_url)
