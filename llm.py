from google import genai
from google.genai import types
import os
import json
import inspect
from pathlib import Path
from config import GEMINI_API_KEY as GEMINI_API_KEY
from config import model as model
import history
import memory as memory

# Memory Retriever file located alongside this module
MEMORY_RETRIEVER_FILE = Path(__file__).parent / "agent_instructions/memory_retriever.md"

# Intent file located alongside this module
INTENT_FILE = Path(__file__).parent / "agent_instructions/intent.md"

# Manager file located alongside this module
MANAGER_FILE = Path(__file__).parent / "agent_instructions/manager.md"

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


def generate_response(user_message: str, cron_job: bool) -> str:

    exit_string = ""
    default_temperature = 0.1
    temperature = default_temperature
    history.append_history(role="user", text=user_message)

    if not cron_job:
        relevant_memories = memory.get_default_store().search_memories(history.get_conversation_history(), limit=5)
        relevant_memories_text = "\n\n".join([f"Memory: {memory.text}\nMetadata: {json.dumps(memory.metadata)}" for memory in relevant_memories])
        try:
            memory_retriever_response = _run_model_api(history.get_conversation_history() + relevant_memories_text, MEMORY_RETRIEVER_FILE.read_text(encoding="utf-8"), tool_use_allowed=False, force_tool=False, temperature=default_temperature)
            if memory_retriever_response != "<NO_RELEVANT_MEMORIES>":
                history.append_history(role="MemoryRetriever", text=memory_retriever_response)
        except Exception as e:
            print(f"Error generating memory retriever response: {e}")

        intent_response = ""
        try:
            intent_response = _run_model_api(history.get_conversation_history(), INTENT_FILE.read_text(encoding="utf-8"), tool_use_allowed=True, force_tool=False, temperature=default_temperature)
        except Exception as e:
            print(f"Error generating intent response: {e}")

        if intent_response != "<complex>":
            history.append_history(role="IntentClassifier", text=intent_response)
            if not cron_job:
                memory_extractor_response = _run_model_api(history.get_conversation_history(), MEMORY_EXTRACTOR_FILE.read_text(encoding="utf-8"), tool_use_allowed=False, force_tool=False, temperature=default_temperature)
                if memory_extractor_response.strip() != "<NO_MEMORY>" and memory_extractor_response.strip() != "":
                    memory.get_default_store().write_memory(memory_extractor_response)
                    history.append_history(role="MemoryExtractor", text=memory_extractor_response)
            return intent_response

    while True:
        # clear the 'sub-agents/execution_order.json' file at the start of each loop iteration
        if EXECUTION_ORDER_FILE.exists():
            try:
                EXECUTION_ORDER_FILE.unlink()
            except Exception as e:
                print(f"Error clearing execution order file: {e}")

        manager_response = ""
        # Get the manager's response based on the conversation history and the new user message
        try:
            manager_response = _run_model_api(history.get_conversation_history(), MANAGER_FILE.read_text(encoding="utf-8"), tool_use_allowed=True, force_tool=True, temperature=temperature)
            history.append_history(role="Manager", text=manager_response)
        except Exception as e:
            print(f"Error generating manager response: {e}")

        # read the execution order from the .json file
        if not EXECUTION_ORDER_FILE.exists():
            print("No execution order file found. ")
            continue
        else:
            try:
                with EXECUTION_ORDER_FILE.open("r", encoding="utf-8") as f:
                    execution_order_dict = json.load(f)
            except Exception as e:
                print(f"Error reading execution order file: {e}")
                continue

        for agent in execution_order_dict['sub_agents']:
            sub_agent_response = ""
            try:
                sub_agent_response = _run_model_api(history.get_conversation_history() + "\n\n" + agent['instruction'], system_instructions=SUB_AGENT_FILE.read_text(encoding="utf-8"), tool_use_allowed=True, force_tool=False, temperature=temperature)
                history.append_history(role=agent['task_name'], text=sub_agent_response)
                if sub_agent_response.strip() == "<CANNOT_PROCEED>":
                    break
            except Exception as e:
                print(f"Error generating response for sub-agent '{agent['task_name']}': {e}")

        if sub_agent_response.strip() == "<CANNOT_PROCEED>":
            continue

        try:
            exit_string = _run_model_api(history.get_conversation_history() + "\n\nThe user's original message was: " + user_message, REVIEWER_FILE.read_text(encoding="utf-8"), tool_use_allowed=False, force_tool=False, temperature=default_temperature)
            history.append_history(role="Reviewer", text=exit_string)
        except Exception as e:
            print(f"Error generating reviewer response: {e}")
        if exit_string == "<yes>":
            break
        else:
            if temperature < 2.0:
                temperature += 0.1

    text_response = ""
    try:
        text_response = _run_model_api(history.get_conversation_history(), TEXTER_FILE.read_text(encoding="utf-8"), tool_use_allowed=False, force_tool=False, temperature=default_temperature)
        history.append_history(role="Texter", text=text_response)
    except Exception as e:
        print(f"Error generating texter response: {e}")

    if not cron_job:
        relevant_memories_history = "\n\n".join([f"Memory: {memory.text}" for memory in memory.get_default_store().search_memories(history.get_conversation_history(), limit=10)])
        try:
            memory_extractor_response = _run_model_api("History: " + history.get_conversation_history() + "\n\nRelevant memories: \n\n" + relevant_memories_history, MEMORY_EXTRACTOR_FILE.read_text(encoding="utf-8"), tool_use_allowed=False, force_tool=False, temperature=default_temperature)
            if memory_extractor_response.strip() != "<NO_MEMORY>":
                memory.get_default_store().write_memory(memory_extractor_response)
                history.append_history(role="MemoryExtractor", text=memory_extractor_response)
        except Exception as e:
            print(f"Error generating memory extractor response: {e}")
    return text_response


def _get_function_declarations(client: genai.Client = None) -> list[types.FunctionDeclaration]:
    """
    Helper function to return a list of available tools for the agent to use, based on the functions defined in the skills directory.
    """
    # get all modules in the skills directory and return only functions defined in those modules
    function_declarations = []
    skills_dir = Path(__file__).parent / "skills"
    for skill_file in skills_dir.glob("*.py"):
        module_name = skill_file.stem
        try:
            module = __import__(f"skills.{module_name}", fromlist=[module_name])
            for _, attr in inspect.getmembers(module, inspect.isfunction):
                if attr.__module__ == module.__name__ and not attr.__name__.startswith("_"):
                    function_declarations.append(types.FunctionDeclaration.from_callable(client=client, callable=attr))
        except Exception as e:
            print(f"Error importing skill module '{module_name}': {e}")
    return function_declarations


def _get_skill(function_name: str, function_args: dict) -> str:
    """Helper function to execute a tool function based on its name and arguments, and return the output as a string."""
    function_output = ""
    for skill_file in (Path(__file__).parent / "skills").glob("*.py"):
        module_name = skill_file.stem
        module = __import__(f"skills.{module_name}", fromlist=[module_name])
        for _, attr in inspect.getmembers(module, inspect.isfunction):
            if attr.__module__ == module.__name__ and attr.__name__ == function_name:
                result = attr(function_args[next(iter(function_args))])
                function_output += result
    return function_output


def _run_model_api(text: str, system_instructions: str, tool_use_allowed: bool = True, force_tool: bool = False, temperature: float = 1) -> str:
    """
    Helper function to call the model API with the given text and system instructions, and return the generated response.
    text: the input text to generate a response for
    system_instructions: the system instructions to provide to the model for this generation
    tool_use_allowed: whether to allow the model to use tools for this generation (default: True)
    temperature: the temperature to use for this generation (default: 1)
    """
    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

    client = genai.Client()
    agent_tools = types.Tool(function_declarations=_get_function_declarations(client=client))
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=["run_python"]
        )
    )
    config = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tool_config=tool_config if force_tool else None,
        tools=[agent_tools] if tool_use_allowed else [],
        temperature=temperature,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=force_tool)
    )

    function_output = ""

    while True:
        try:
            response = client.models.generate_content(
                model=model,
                config=config,
                contents=text,
            )
            break
        except Exception as e:
            print(f"Error calling model API: {e}")
        print("Retrying...")

    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                function_output += _get_skill(part.function_call.name, part.function_call.args)
                    
    output = ""

    if function_output == "":
        output = response.text or ""
    else:
        output = response.text or ""
        output += "\n\n" + function_output
    return output