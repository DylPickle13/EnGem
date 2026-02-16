from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import re
import os
from pathlib import Path
from credentials import HF_TOKEN
import tools

# Instructions file located alongside this module
INSTRUCTIONS_FILE = Path(__file__).parent / "instructions.md"

# Model configuration
MODEL_NAME = "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"

# Path to Hugging Face cache directory, where models and tokenizers are stored after download.
# /Users/dylanrapanan/.cache/huggingface/hub

# Model and tokenizer are None until first use to avoid allocating memory at import
model = None
tokenizer = None
# Global flag indicating whether the LLM is currently running (True) or idle (False)
llm_running = False

def generate_response(user_message: str, max_tokens: int = -1, loops: int = 2, verbose: bool = True) -> str:

    messages = tools.get_conversation_history()

    prompt = INSTRUCTIONS_FILE.read_text(encoding="utf-8") if INSTRUCTIONS_FILE.exists() else ""
    prompt += f"\n\n{user_message}"

    # Finally add the current user prompt as the latest user message
    messages.append({"role": "user", "content": prompt})

    # Ensure model+tokenizer are loaded (lazy load on first call)
    global model, tokenizer, llm_running
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

        # Make a sampler with temperature = 0.7 and min_p = 0.05 to balance creativity and coherence.
        sampler = make_sampler(temp=0.25, top_p=0.9)

        # Apply the chat template to the messages, which formats them in a way the model expects.
        generation_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        pieces = []
        # Stream and print tokens as they're produced so the terminal shows progress
        for response in stream_generate(model, tokenizer, generation_prompt, max_tokens=max_tokens, sampler=sampler):
            text = response.text
            if text:
                if verbose:
                    print(text, end="", flush=True)
            pieces.append(text)

        full_text = "".join(pieces)

        # Final cleanup: strip leading/trailing whitespace
        result_text = full_text.strip()

        # Perform tool calls (search and code execution) on the generated text, which may append additional content to the result.
        result_text += tools.search(result_text)
        result_text += tools.run_python(result_text, verbose=verbose)

        # remove any tool calls from the final output before returning including the tags (e.g. <search>...</search>)
        result_text = re.sub(r"<search>.*?</search>", "", result_text, flags=re.DOTALL | re.IGNORECASE)
        result_text = re.sub(r"<think>.*?</think>", "", result_text, flags=re.DOTALL | re.IGNORECASE)
    
        return result_text
    except Exception as e:
        # If any error occurs during generation or code execution, print it and return an error message.
        print(f"\nError in generate_response: {e}")
    finally:
        # Always mark the LLM as not running when the function exits
        llm_running = False
        print("Finished generating... ")