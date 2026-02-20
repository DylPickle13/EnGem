from google import genai
from google.genai import types
import os
import json
from pathlib import Path
from credentials import GEMINI_API_KEY as GEMINI_API_KEY
import tools

# Intent file located alongside this module
INTENT_FILE = Path(__file__).parent / "agent_instructions/intent.md"

# Manager file located alongside this module
MANAGER_FILE = Path(__file__).parent / "agent_instructions/manager.md"

# Followup file located alongside this module
FOLLOWUP_FILE = Path(__file__).parent / "agent_instructions/followup.md"

# Sub-agent file located alongside this module
SUB_AGENT_FILE = Path(__file__).parent / "agent_instructions/sub_agent.md"

# Reviewer file located alongside this module
REVIEWER_FILE = Path(__file__).parent / "agent_instructions/reviewer.md"

# Summarize file located alongside this module
TEXTER_FILE = Path(__file__).parent / "agent_instructions/texter.md"

# Execution order file path
EXECUTION_ORDER_FILE = Path(__file__).parent / "sub-agents/execution_order.json"

model = "gemini-2.5-pro"  # Specify the model to use for generating responses

# Global flag indicating whether the LLM is currently running (True) or idle (False)
llm_running = False


def generate_response(user_message: str, verbose: bool = True) -> str:
    global llm_running

    llm_running = True
    exit_string = ""
    tools.append_history(role="user", text=user_message)
    print("User message appended to history. Starting response generation...")

    try:
        intent_response = _run_model_api(tools.get_conversation_history() + user_message, INTENT_FILE.read_text(encoding="utf-8"), model, verbose=verbose)
    except Exception as e:
        err_msg = f"Error generating response: {e}"
        print(err_msg)
        return err_msg

    if intent_response != "<complex>":
        print("Intent classified as simple. Returning direct response.")
        tools.append_history(role="IntentClassifier", text=intent_response)
        return intent_response

    while True:
        # clear the 'sub-agents/execution_order.json' file at the start of each loop iteration
        if EXECUTION_ORDER_FILE.exists():
            try:
                EXECUTION_ORDER_FILE.unlink()
                print("Cleared existing execution order file.")
            except Exception as e:
                print(f"Warning: could not clear execution order file: {e}")

        # Get the manager's response based on the conversation history and the new user message
        print("Getting manager response...")
        try:
            manager_response = _run_model_api(tools.get_conversation_history() + user_message, MANAGER_FILE.read_text(encoding="utf-8"), model, verbose=verbose)
        except Exception as e:
            err_msg = f"Error generating manager response: {e}"
            print(err_msg)
            return err_msg

        # Check for the file 'sub-agents/execution_order.json' to determine if the manager has issued an execution order
        tools.append_history(role="Manager", text=manager_response)

        # read the execution order from the .json file
        if not EXECUTION_ORDER_FILE.exists():
            print("No execution order file found. Assuming manager has completed their response.")
            continue
        else:
            try:
                with EXECUTION_ORDER_FILE.open("r", encoding="utf-8") as f:
                    execution_order_dict = json.load(f)
            except Exception as e:
                err_msg = f"Error reading execution order file: {e}"
                print(err_msg)
                return err_msg

        for agent in execution_order_dict['sub_agents']:
            try:
                print(f"Running sub-agent '{agent['task_name']}'")
                sub_agent_response = _run_model_api(tools.get_conversation_history() + agent['instruction'], system_instructions=SUB_AGENT_FILE.read_text(encoding="utf-8"), model=model, verbose=verbose)
            except Exception as e:
                err_msg = f"Error generating response for sub-agent '{agent['task_name']}': {e}"
                print(err_msg)
                sub_agent_response = err_msg
            tools.append_history(role=agent['task_name'], text=sub_agent_response)

        print("Getting reviewer response...")
        try:
            exit_string = _run_model_api(tools.get_conversation_history() + "\n\nThe user's original message was: " + user_message, REVIEWER_FILE.read_text(encoding="utf-8"), model, verbose=verbose)
        except Exception as e:
            err_msg = f"Error generating reviewer response: {e}"
            print(err_msg)
            return err_msg
        tools.append_history(role="Reviewer", text=exit_string)
        if exit_string == "<yes>":
            print("Review complete. User's request has been fulfilled.")
            break

    print("Generating final response for the user...")
    try:
        text_response = _run_model_api(tools.get_conversation_history(), TEXTER_FILE.read_text(encoding="utf-8"), model, verbose=verbose)
    except Exception as e:
        err_msg = f"Error generating final response: {e}"
        print(err_msg)
        return err_msg
    tools.append_history(role="Texter", text=text_response)
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
    follow_up_response = ""

    if function_output == "":
        output = response.text or ""
    else:
        config = types.GenerateContentConfig(
            system_instruction=FOLLOWUP_FILE.read_text(encoding="utf-8")
        )

        if verbose:
            print("Running follow-up Gemini API call...")
        follow_up_response = client.models.generate_content(
            model=model,
            config=config,
            contents=text + "\n\nTool output:\n\n" + function_output,
        )
        output = function_output + "\n\n" + (follow_up_response.text or "")
    if verbose:
        print("Gemini API call complete.")
    return output


if __name__ == "__main__":
    # open the sub-agents/execution_order.json file and print its contents (for debugging)
    if EXECUTION_ORDER_FILE.exists():
        try:
            with EXECUTION_ORDER_FILE.open("r", encoding="utf-8") as f:
                execution_order_dict = json.load(f)
        except Exception as e:
            print(f"Error reading execution order file: {e}")
        
        for agent in execution_order_dict['sub_agents']:
            print(f"Agent name: {agent['task_name']}")
            print(f"Agent instruction: {agent['instruction']}")