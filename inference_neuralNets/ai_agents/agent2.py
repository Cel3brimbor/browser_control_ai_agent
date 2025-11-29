import requests
import json
from bs4 import BeautifulSoup

LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
# model is gemma 4b

def analyze_webpage_subject(url):
    """
    Fetches the raw HTML content, extracts the main text, and sends a truncated version to the LM Studio model for subject analysis.
    Model tries to find clues in the URL first, then uses HTML to help with inference.
    """
    try:
        print(f"Fetching content from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # get all the visible text from the page
        text_content = soup.get_text()
        
        #clean up whitespace and extra newlines
        cleaned_text = " ".join(text_content.split())
        
        # truncate the text to a manageable size to fit within the model's context window
        MAX_LENGTH = 5000
        truncated_content = cleaned_text[:MAX_LENGTH]
        
        print(truncated_content)
        
        print(f"\n\nSending a truncated text of {len(truncated_content)} characters to the model with URL {url}")
        
        prompt = f"""
        Here is the URL of a webpage: {url}.
        Using the domain name, what is the main subject or topic of the page?
        If unable to be determined or domain name is too generic, use help from a truncated portion of the text from that same webpage for more information: \n{truncated_content}\n 
        State your response concisely. Do not provide a detailed summary or list of entities, just the primary subject."""

        lm_studio_headers = {
            "Content-Type": "application/json",
        }
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False
        }

        lm_studio_response = requests.post(LM_STUDIO_API_URL, headers=lm_studio_headers, data=json.dumps(data))
        lm_studio_response.raise_for_status()
        
        model_response = lm_studio_response.json()
        
        print("\nModel Response:")
        print(model_response['choices'][0]['message']['content'])
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        print("Please check the URL and ensure your LM Studio server is running with a model loaded.")
        
if __name__ == "__main__":
    #target_url = "https://en.wikipedia.org/wiki/International_Space_Station"
    target_url = input("Enter URL for analysis: ")

    analyze_webpage_subject(target_url)