# generate_image tool instructions

Use this tool to create a single image from a text prompt.

What this tool does:
- Calls Gemini image generation with 16:9 aspect ratio.
- Saves one PNG image in generated_files/.
- Returns the absolute file path on success.
- Returns an empty string on failure.

How to call it well:
- Describe subject, style, composition, lighting, and camera framing.
- Include quality constraints (clean background, no text, realistic, etc.) when needed.
- Request one clear output concept per call.

Important constraints:
- Produces one image per call.
- Empty-string output means generation or save failed.
- If output is empty, refine prompt and retry.

High quality prompting pattern:
- Subject.
- Visual style.
- Composition/framing.
- Mood and lighting.
- Any strict exclusions.