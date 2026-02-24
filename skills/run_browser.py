import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import computer_use

def run_browser(prompt: str) -> str:
    """
    Opens a browser and automates interactions based on the provided prompt using the computer_use agent.
    """
    output = ""
    client = computer_use.create_client()
    playwright, browser, page = computer_use.setup_browser()

    try:
        output = computer_use.run_agent_loop(client, page, prompt=prompt)
    except Exception as e:
        output = f"An error occurred: {str(e)}"
    finally:
        browser.close()
        playwright.stop()
    return output

if __name__ == "__main__":
    prompt = "Find data scientist jobs on indeed.com and compile me a list of 5. Make sure the job is in canada. Summarize the job description for each role in 1-2 sentences."
    result = run_browser(prompt)
    print(result)