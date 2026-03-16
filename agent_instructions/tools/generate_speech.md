# generate_speech tool instructions

Use this tool for single-speaker text-to-speech generation.

What this tool does:
- Converts prompt text into speech audio.
- Uses Gemini TTS with a prebuilt voice (default Kore).
- Saves a WAV file in generated_files/.
- Returns absolute file path on success.
- Returns an error string prefixed with error: on failure.

How to call it well:
- Provide the exact script to be spoken.
- Include punctuation for pacing and emphasis.
- Keep text clean and free of markdown artifacts.
- If voice style matters, mention desired tone in the script wording.

Important constraints:
- Output is mono 16-bit PCM WAV.
- Empty/invalid speech data returns an error string.
- Validate that returned path exists before downstream use.

High quality prompting pattern:
- Final script exactly as it should be spoken.
- Target tone and pace implied via punctuation and phrasing.