from google import genai
from google.genai import types
import os
import json
from pathlib import Path
from config import GEMINI_API_KEY as GEMINI_API_KEY
from config import model as model
import tools
import skills.vector_database as vector_database
import skills.run_python as run_python
import skills.run_google_search as run_google_search
import skills.git_push as git_push

# Memory Retriever file located alongside this module
MEMORY_RETRIEVER_FILE = Path(__file__).parent / "agent_instructions/memory_retriever.md"

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

# Memory Extractor file located alongside this module
MEMORY_EXTRACTOR_FILE = Path(__file__).parent / "agent_instructions/memory_extractor.md"

# Execution order file path
EXECUTION_ORDER_FILE = Path(__file__).parent / "sub-agents/execution_order.json"

# Global flag indicating whether the LLM is currently running (True) or idle (False)
llm_running = False


def generate_response(user_message: str, verbose: bool = True) -> str:
    global llm_running

    llm_running = True
    exit_string = ""
    tools.append_history(role="user", text=user_message)
    print("User message appended to history. Starting response generation...")

    relevant_memories = vector_database.get_default_store().search_memories(tools.get_conversation_history(), limit=5)
    relevant_memories_text = "\n\n".join([f"Memory: {memory.text}\nMetadata: {json.dumps(memory.metadata)}" for memory in relevant_memories])
    try:
        memory_retriever_response = _run_model_api(tools.get_conversation_history() + relevant_memories_text, MEMORY_RETRIEVER_FILE.read_text(encoding="utf-8"), model, tool_use_allowed=False, verbose=verbose)
        if memory_retriever_response.strip() != "<NO_RELEVANT_MEMORY>":
            tools.append_history(role="MemoryRetriever", text=memory_retriever_response)
            print("Memory retriever response generated and appended to history.")
    except Exception as e:
        err_msg = f"Error generating memory retriever response: {e}"
        print(err_msg)
        return err_msg

    try:
        intent_response = _run_model_api(tools.get_conversation_history() + user_message, INTENT_FILE.read_text(encoding="utf-8"), model, tool_use_allowed=True, verbose=verbose)
    except Exception as e:
        err_msg = f"Error generating response: {e}"
        print(err_msg)
        return err_msg

    if intent_response != "<complex>":
        print("Intent classified as simple. Returning direct response.")
        tools.append_history(role="IntentClassifier", text=intent_response)
        memory_extractor_response = _run_model_api(tools.get_conversation_history(), MEMORY_EXTRACTOR_FILE.read_text(encoding="utf-8"), model, tool_use_allowed=False, verbose=verbose)
        if memory_extractor_response.strip() != "<NO_MEMORY>":
            print("Extracted memory from user message. Writing to vector database.")
            try:
                vector_database.get_default_store().write_memory(memory_extractor_response)
            except Exception as e:
                print(f"Error writing memory to vector database: {e}")
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
            manager_response = _run_model_api(tools.get_conversation_history() + user_message, MANAGER_FILE.read_text(encoding="utf-8"), model, tool_use_allowed=True, verbose=verbose)
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
                sub_agent_response = _run_model_api(tools.get_conversation_history() + agent['instruction'], system_instructions=SUB_AGENT_FILE.read_text(encoding="utf-8"), model=model, tool_use_allowed=True, verbose=verbose)
            except Exception as e:
                err_msg = f"Error generating response for sub-agent '{agent['task_name']}': {e}"
                print(err_msg)
                sub_agent_response = err_msg
            tools.append_history(role=agent['task_name'], text=sub_agent_response)

        print("Getting reviewer response...")
        try:
            exit_string = _run_model_api(tools.get_conversation_history() + "\n\nThe user's original message was: " + user_message, REVIEWER_FILE.read_text(encoding="utf-8"), model, tool_use_allowed=False, verbose=verbose)
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
        text_response = _run_model_api(tools.get_conversation_history(), TEXTER_FILE.read_text(encoding="utf-8"), model, tool_use_allowed=False, verbose=verbose)
    except Exception as e:
        err_msg = f"Error generating final response: {e}"
        print(err_msg)
        return err_msg
    tools.append_history(role="Texter", text=text_response)
    memory_extractor_response = _run_model_api(tools.get_conversation_history(), MEMORY_EXTRACTOR_FILE.read_text(encoding="utf-8"), model, tool_use_allowed=False, verbose=verbose)
    if memory_extractor_response.strip() != "<NO_MEMORY>":
        print("Extracted memory from sub-agent execution. Writing to vector database.")
        try:
            vector_database.get_default_store().write_memory(memory_extractor_response)
        except Exception as e:
            print(f"Error writing memory to vector database: {e}")
    llm_running = False
    return text_response


def _run_model_api(text: str, system_instructions: str, model: str, tool_use_allowed: bool = True, verbose: bool = True) -> str:
    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

    client = genai.Client()
    agent_tools = types.Tool(function_declarations=[
        types.FunctionDeclaration.from_callable(client=client, callable=run_python.run_python),
        types.FunctionDeclaration.from_callable(client=client, callable=run_google_search.run_google_search),
        types.FunctionDeclaration.from_callable(client=client, callable=git_push.commit_and_push),
    ])
    config = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tools=[agent_tools] if tool_use_allowed else []
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
                    function_output += part.function_call.args['code'] + "\nOutput:\n"
                    function_output += run_python.run_python(part.function_call.args['code'])
                if part.function_call.name == "run_google_search":
                    function_output += run_google_search.run_google_search(part.function_call.args['query'])
                if part.function_call.name == "commit_and_push":
                    function_output += git_push.commit_and_push(part.function_call.args['message'])

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