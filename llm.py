from google import genai
from google.genai import types
import os
import re
from pathlib import Path
from credentials import GEMINI_API_KEY
import tools

# Instructions file located alongside this module
INSTRUCTIONS_FILE = Path(__file__).parent / "agent_instructions/instructions.md"

# Reviewer file located alongside this module
REVIEWER_FILE = Path(__file__).parent / "agent_instructions/reviewer.md"

# Cleaner file located alongside this module
CLEANER_FILE = Path(__file__).parent / "agent_instructions/cleaner.md"

# Path to Hugging Face cache directory, where models and tokenizers are stored after download.
# /Users/dylanrapanan/.cache/huggingface/hub

# Global flag indicating whether the LLM is currently running (True) or idle (False)
llm_running = False

def generate_response(user_message: str, max_loops: int = 10, verbose: bool = True) -> str:
    global llm_running

    llm_running = True
    cleaned_output = ""
    loops = 0

    tools.append_history("user", user_message)

    while loops < max_loops:
        messages = tools.get_conversation_history_text()

        prompt = f"\n\n.{user_message}"

        # Finally add the current user prompt as the latest user message
        messages += f"\n\nuser: {prompt}"

        try:
            # Generate text from the model
            text = _run_model_api(messages, INSTRUCTIONS_FILE.read_text(encoding="utf-8"), verbose=verbose)

            tools.append_history("llm", text)

            # Build a chat-formatted review prompt: include the assistant's output
            # and ask for a yes/no review using roles the model expects.
            tools.append_history("user", REVIEWER_FILE.read_text(encoding="utf-8"))
            review = _run_model_api(tools.get_conversation_history_text(), REVIEWER_FILE.read_text(encoding="utf-8"), verbose=verbose)

            tools.append_history("llm", review)

            if "<yes>" in review.strip().lower():
                break
        except Exception as e:
            # If any error occurs during generation or code execution, print it and return an error message.
            print(f"\nError in generate_response: {e}")
            text = f"Error in generate_response: {e}"
            tools.append_history("llm", text)

        loops += 1

    cleaned_prompt = CLEANER_FILE.read_text(encoding="utf-8") if CLEANER_FILE.exists() else ""
    cleaned_prompt += user_message
    tools.append_history("user", cleaned_prompt)
    cleaned_output = _run_model_api(tools.get_conversation_history_text(), CLEANER_FILE.read_text(encoding="utf-8"), verbose=verbose).strip()
    tools.append_history("llm", cleaned_output)

    llm_running = False
    return cleaned_output


def _run_model_api(text, system_instructions, verbose: bool = False) -> str:
    # set the environment variable for the Gemini API key if it's not already set, using the value from credentials.py
    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

    client = genai.Client()
    config = types.GenerateContentConfig(
        system_instruction=system_instructions
        )
    print("Generating response from Gemini API... ")

    response = client.models.generate_content_stream(
        model="gemini-3-flash-preview", 
        config=config,
        contents=text,
    )

    output = ""
    for chunk in response:
        output += chunk.text
        if verbose:
            print(chunk.text, end='')

    if len(output) > 0:
        output += tools.run_python(output, verbose=verbose)

    print("\nFinished generating.")
    return output.strip()