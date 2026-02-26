from pathlib import Path
import computer_use
import llm

# Browser Summarizer file located alongside this module
BROWSER_SUMMARIZER_FILE = Path(__file__).parent.parent / "agent_instructions/browser_summarizer.md"

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

        output = llm._run_model_api(output, system_instructions=BROWSER_SUMMARIZER_FILE.read_text(encoding="utf-8"), tool_use_allowed=False, force_tool=False, temperature=1.0)
    except Exception as e:
        output = f"An error occurred: {str(e)}"
    return output