import requests
import json
import time

def main():
    """
    A command-line tool to determine the subject or task of a web page
    based on its content using an LM Studio server.
    """
    api_url = "http://localhost:1234/v1/chat/completions"

    system_prompt = (
        "You are an AI assistant specialized in analyzing web page content to determine the "
        "primary subject or task. Based on the provided text, identify the single, most "
        "relevant subject or task. Respond with only a concise, one-sentence description "
        "of that subject. For example, if the text is about a product page, your response "
        "should be 'The task is shopping for a new gadget.' or if it's a news article, "
        "'The subject is current events related to space exploration.' Do not include any "
        "other commentary or introductory phrases."
    )

    while True:
        try:
            user_input = input("\nEnter URL, title, or content to analyze: ")
            
            if user_input.lower() == 'exit':
                break
            
            if not user_input.strip():
                print("Input cannot be empty. Please try again.")
                continue

            print("Analyzing...")

            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7,
                "stream": False
            }
            
            headers = {
                "Content-Type": "application/json",
            }

            retries = 0
            max_retries = 5
            base_delay = 1.0  # seconds

            while retries < max_retries:
                try:
                    response = requests.post(
                        api_url,
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    if result.get("choices") and result["choices"][0].get("message"):
                        analysis_result = result["choices"][0]["message"]["content"].strip()
                        print(f"\nAnalysis Result: {analysis_result}")
                    else:
                        print("\nCould not determine the subject. Please check the LM Studio server logs.")
                    
                    break # exit

                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 429 and retries < max_retries:
                        delay = base_delay * (2 ** retries)
                        print(f"Rate limit exceeded (HTTP 429). Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        retries += 1
                    else:
                        print(f"\nHTTP error occurred: {http_err}")
                        print("Check that LM Studio is running and the port is correct.")
                        break
                except requests.exceptions.RequestException as req_err:
                    print(f"\nNetwork error occurred: {req_err}")
                    print("Check that LM Studio is running and the port is correct.")
                    break
            
            if retries == max_retries:
                print("\nMaximum retry attempts reached. Please try again later.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()