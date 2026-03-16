# generate_image tool instructions

Use this tool to generate an image and save it in generated_files/.

Call contract (important):
- The tool accepts one string argument named prompt.
- For advanced control, set prompt to a JSON object string.
- Always include a top-level prompt field inside the JSON.
- Use mid-level image_size to reduce costs, unless stated otherwise in the user request.

Required JSON shape:
{
	"prompt": "Image description",
	"model": "gemini-3.1-flash-image-preview",
	"aspect_ratio": "16:9",
	"image_size": "2K",
	"person_generation": "allow_all",
	"output_mime_type": "image/png",
	"output_compression_quality": 95,
	"response_modalities": ["TEXT", "IMAGE"],
	"temperature": 0.7,
	"seed": 123,
	"use_google_search": true,
	"reference_images": [
		{"path": "generated_files/ref_subject.png"},
		{"path": "generated_files/ref_style.png"}
	]
}

Supported values and limits:
- aspect_ratio: 1:1, 1:4, 1:8, 2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 21:9.
- image_size: 512, 1K, 2K, 4K.
- person_generation: allow_all, allow_adult, dont_allow.
- output_mime_type: image/png, image/jpeg, image/webp.
- response_modalities: TEXT and/or IMAGE.
- reference_images: up to 14 images for composition, identity, and style guidance.

Reference image formats:
- Each reference image can be provided as:
	- {"path": "local/path.png"}
	- {"uri": "gs://...", "mime_type": "image/png"}
	- {"image_base64": "...", "mime_type": "image/png"}

Prompt quality guidance:
- Include subject, composition, camera/framing, style, lighting, and mood.
- For text-in-image, quote the exact text and describe typography/layout.
- For exclusions, describe desired clean scene rather than only saying no X.

Behavior:
- Tool returns absolute image path on success.
- Tool returns empty string on failure.
- If generation fails, simplify prompt and reduce constraints, then retry.
 - If the prompt was a JSON object, the tool will return the output path followed by a \`JSON_INPUT:\` marker and the parsed JSON payload (path is on the first line for backwards compatibility).