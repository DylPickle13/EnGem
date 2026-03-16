# generate_video tool instructions

Use this tool to generate a short video clip from text.

What this tool does:
- Calls Gemini Veo video generation with 16:9 aspect ratio.
- Polls operation status until complete.
- Downloads and saves an MP4 file in generated_files/.
- Returns absolute video path on success.
- Returns an error string prefixed with error: on failure.

How to call it well:
- Provide a concrete scene description with subject, setting, and motion.
- Specify camera behavior (static, dolly, pan, aerial, close-up).
- Include mood, lighting, and style direction.
- Keep narrative concise and visually explicit.

Important constraints:
- This is asynchronous and may take time.
- Failures are returned as error strings.
- If failed, tighten or simplify the prompt and retry.

High quality prompting pattern:
- Scene setup.
- Motion details.
- Camera direction.
- Visual style.
- Desired outcome focus.