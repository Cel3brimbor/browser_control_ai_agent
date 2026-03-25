# test prompt engineering and agent's return values

import requests
import json

LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
# model is gemma 4b

def analyze_webpage_subject(currentTabTitles, newTabTitle):
    """
    Fetches the raw HTML content, extracts the main text, and sends a truncated version to the LM Studio model for subject analysis.
    Model tries to find clues in the URL first, then uses HTML to help with inference.
    """
    try:
        # prompt = f"""
        # These are the current tab titles in the user's browser {currentTabTitles}. They are comma seperated.
        # Does the newly opened tab's title {newTabTitle} seem to align or be on task compared to the current ones?
        # Return 0 or 1 only. 0 for off-task, 1 for on-task.
        # Do not say anything other than one of those two numbers."""

        prompt = f"""
        These are the current tab titles in the user's browser: ${currentTabTitles}. They are comma-seperated.
        Does the newly opened tab's title "${newTabTitle}" seem to align or be on-task compared to the current ones?
        It is on-task if it most likely helps the user's current tabs, and off-task if it is most likely a distraction.
        Return 0 or 1 only. 0 for off-task, 1 for on-task."""

        # prompt = f"""
        # These are the current tab titles in the user's browser: ${currentTabTitles}. They are comma-seperated.
        # Does the newly opened tab's title "${newTabTitle}" seem to align or be on-task compared to the current ones?
        # It is on-task if it most likely helps the user's current tabs, and off-task if it is most likely a distraction.
        # Return whether you think it is on task or off task. Give reasoning."""

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
            "temperature": 0.0,
            "max_tokens": 10,
            "stream": False
        }

        lm_studio_response = requests.post(LM_STUDIO_API_URL, headers=lm_studio_headers, data=json.dumps(data))
        lm_studio_response.raise_for_status()
        
        model_response = lm_studio_response.json()

        print("\nModel Response:")
        print(model_response['choices'][0]['message']['content'])
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    current_tabs = input("Enter current tabs (at least 4 and comma seperated): ")
    new_tab = input("Enter new tab: ")

    analyze_webpage_subject(current_tabs, new_tab)