from google import genai
from google.genai import types
import os
import json
from pathlib import Path
from credentials import GEMINI_API_KEY as GEMINI_API_KEY
import tools

# Manager file located alongside this module
MANAGER_FILE = Path(__file__).parent / "agent_instructions/manager.md"

# Followup file located alongside this module
FOLLOWUP_FILE = Path(__file__).parent / "agent_instructions/followup.md"

# Reviewer file located alongside this module
REVIEWER_FILE = Path(__file__).parent / "agent_instructions/reviewer.md"

# Summarize file located alongside this module
TEXTER_FILE = Path(__file__).parent / "agent_instructions/texter.md"

# Execution order file path
EXECUTION_ORDER_FILE = Path(__file__).parent / "sub-agents/execution_order.json"

model = "gemini-3.1-pro-preview"  # Specify the model to use for generating responses

# Global flag indicating whether the LLM is currently running (True) or idle (False)
llm_running = False


def generate_response(user_message: str, verbose: bool = True) -> str:
    global llm_running

    llm_running = True
    exit_string = ""
    tools.append_history(role="user", text=user_message)

    while exit_string != "<yes>":
        # clear the 'sub-agents/execution_order.json' file at the start of each loop iteration
        if EXECUTION_ORDER_FILE.exists():
            EXECUTION_ORDER_FILE.unlink()

        # Get the manager's response based on the conversation history and the new user message
        manager_response = _run_model_api(user_message, tools.get_conversation_history() + MANAGER_FILE.read_text(encoding="utf-8"), model, verbose=verbose)

        # Check for the file 'sub-agents/execution_order.json' to determine if the manager has issued an execution order
        if EXECUTION_ORDER_FILE.exists():
            tools.append_history(role="llm", text=manager_response)

            # read the execution order from the .json file
            with EXECUTION_ORDER_FILE.open("r", encoding="utf-8") as f:
                execution_order_dict = tools.parse_execution_order(json.load(f))
            for task_name, task_info in execution_order_dict.items():
                sub_agent_response = _run_model_api(task_info["instruction"], system_instructions="You are the " + task_name + "sub-agent. ", model=model, verbose=verbose)
                tools.append_history(role="llm", text=sub_agent_response)

        else:
            exit_string = ""
            continue

        print("Revewing")
        exit_string = _run_model_api(tools.get_conversation_history(), REVIEWER_FILE.read_text(encoding="utf-8"), model, verbose=verbose)
        tools.append_history(role="llm", text=exit_string)

    text_response = _run_model_api(tools.get_conversation_history(), TEXTER_FILE.read_text(encoding="utf-8"), model, verbose=verbose)
    tools.append_history(role="llm", text=text_response)
    tools.archive_history()
    llm_running = False
    return text_response

def _run_google_search(query: str) -> str:
    """Run a Google Search using the Gemini API's Google Search tool."""
    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

    client = genai.Client()
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = client.models.generate_content(
        model=model,
        contents=query,
        config=config,
    )
    print("Used Google Search tool with query:", query)
    return response.candidates[0].content.parts[0].text or ""


def _run_model_api(text: str, system_instructions: str, model: str, verbose: bool = True) -> str:
    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

    client = genai.Client()
    agent_tools = types.Tool(function_declarations=[
        types.FunctionDeclaration.from_callable(client=client, callable=tools.run_python),
        types.FunctionDeclaration.from_callable(client=client, callable=_run_google_search)])
    config = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tools=[agent_tools]
        )

    function_output = ""

    if verbose:
        print("Running Gemini API...")

    response = client.models.generate_content(
        model=model,
        config=config,
        contents=text,
    )

    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                if part.function_call.name == "run_python":
                    function_output += tools.run_python(part.function_call.args['code'])
                if part.function_call.name == "_run_google_search":
                    function_output += _run_google_search(part.function_call.args['query'])

    output = ""

    if function_output == "":
        output = response.text or ""
    else:
        config = types.GenerateContentConfig(
            system_instruction=FOLLOWUP_FILE.read_text(encoding="utf-8")
        )

        print("Running follow-up Gemini API call...")
        followup_response = client.models.generate_content(
            model=model,
            config=config,
            contents=text + "\n\nTool output:\n\n" + function_output,
        )
    output = function_output + "\n\n" + (followup_response.text or "")
    if verbose:
        print(output)
        print("Gemini API call complete.")
    return output