import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Browser Summarizer file located alongside this module
BROWSER_SUMMARIZER_FILE = Path(__file__).parent / "agent_instructions/browser_summarizer.md"

import computer_use
import llm

def use_browser(prompt: str) -> str:
    """
    Runs actions based on the provided prompt using the computer_use agent.
    Returns the output of the agent's actions as a string.
    """
    output = ""
    client = computer_use.create_client()
    _playwright, _browser, page = computer_use.setup_browser(reuse_existing=True)

    try:
        output = computer_use.run_agent_loop(client, page, prompt=prompt)

        output = llm._run_model_api(output, system_instructions="", tool_use_allowed=False, force_tool=False, temperature=1.0)
    except Exception as e:
        output = f"An error occurred: {str(e)}"
    return output

if __name__ == "__main__":
    prompt = "Find data scientist jobs on indeed.com and compile me a list of 5. Make sure the job is in canada. Summarize the job description for each role in 1-2 sentences."
    result = use_browser(prompt)
    print(result)