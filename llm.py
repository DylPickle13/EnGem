from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import re
import os
from pathlib import Path
from credentials import HF_TOKEN
import tools

# Instructions file located alongside this module
INSTRUCTIONS_FILE = Path(__file__).parent / "agent_instructions/instructions.md"

# Reviewer file located alongside this module
REVIEWER_FILE = Path(__file__).parent / "agent_instructions/reviewer.md"

# Cleaner file located alongside this module
CLEANER_FILE = Path(__file__).parent / "agent_instructions/cleaner.md"

# Model configuration
MODEL_NAME = "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"

# Path to Hugging Face cache directory, where models and tokenizers are stored after download.
# /Users/dylanrapanan/.cache/huggingface/hub

# Model and tokenizer are None until first use to avoid allocating memory at import
model = None
tokenizer = None
# Global flag indicating whether the LLM is currently running (True) or idle (False)
llm_running = False

def generate_response(user_message: str, max_tokens: int = -1, max_loops: int = 3, verbose: bool = True) -> str:

    cleaned_output = ""
    loops = 0

    tools.append_history("user", user_message)

    while loops < max_loops:
        messages = tools.get_conversation_history()

        prompt = INSTRUCTIONS_FILE.read_text(encoding="utf-8") if INSTRUCTIONS_FILE.exists() else ""
        prompt += f"\n\n{user_message}"

        if loops > 0:
            prompt += f"\n\nThe assistant's previous response was unsatisfactory. Please try again and provide a better response based on the conversation history and instructions. "

        # Finally add the current user prompt as the latest user message
        messages.append({"role": "user", "content": prompt})

        try:
            # Generate text from the model
            text = _run_model(messages, max_tokens=max_tokens, verbose=verbose)

            # Perform tool calls (search and code execution) on the generated text, which may append additional content to the result.
            text += tools.search(text)
            text += tools.run_python(text, verbose=verbose)

            tools.append_history("llm", text)

            # Build a chat-formatted review prompt: include the assistant's output
            # and ask for a yes/no review using roles the model expects.
            tools.append_history("user", REVIEWER_FILE.read_text(encoding="utf-8") if REVIEWER_FILE.exists() else "")
            review = _run_model(tools.get_conversation_history(), max_tokens=-1, verbose=verbose)

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
    cleaned_output = _run_model(tools.get_conversation_history(), max_tokens=-1, verbose=verbose).strip()
    tools.append_history("llm", cleaned_output)

    return cleaned_output

def _run_model(messages, max_tokens: int = -1, verbose: bool = False) -> str:
    # Ensure model+tokenizer are loaded (lazy load on first call)
    global model, tokenizer, llm_running

    text = ""

    # Mark LLM as running for the duration of this generation.
    llm_running = True
    print("Generating response... ")
    try:
        if model is None or tokenizer is None:
            # Ensure Hugging Face token is available to the loader. We don't
            # overwrite existing environment variables, but set them if
            # missing so private models or rate-limited access works.
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)
            os.environ.setdefault("HF_TOKEN", HF_TOKEN)
            model, tokenizer = load(MODEL_NAME)

        # Make a sampler with temperature = 0.5 for some randomness in generation (temperature=0.0 would be deterministic)
        sampler = make_sampler(temp=0.5)

        # Apply the chat template to the messages, which formats them in a way the model expects.
        generation_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        pieces = []
        # Stream and collect tokens as they're produced so the terminal shows progress
        for response in stream_generate(model, tokenizer, generation_prompt, max_tokens=max_tokens, sampler=sampler):
            text_piece = response.text
            if text_piece:
                if verbose:
                    print(text_piece, end="", flush=True)
                pieces.append(text_piece)

        text = "".join(pieces).strip()
    except Exception as e:
        print(f"\nError in _run_model: {e}")
        return f"Error in model generation: {e}"
    
    llm_running = False
    print("\nFinished generating.")
    return text