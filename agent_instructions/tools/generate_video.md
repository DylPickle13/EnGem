# generate_video tool instructions

Use this tool to generate a video with Veo and save it in generated_files/.

Call contract (important):
- The tool accepts one string argument named prompt.
- For advanced control, set prompt to a JSON object string.
- Always include a top-level prompt field inside the JSON.
- Use mid-level resolution to reduce costs, unless stated otherwise in the user request.

Required JSON shape:
{
	"prompt": "Video description",
	"model": "veo-3.1-fast-generate-preview",
	"aspect_ratio": "16:9",
	"resolution": "720p",
	"duration_seconds": 8,
	"number_of_videos": 1,
	"fps": 24,
	"seed": 123,
	"person_generation": "allow_all",
	"negative_prompt": "undesired elements",
	"enhance_prompt": true,
	"generate_audio": true,
	"first_image": {
		"path": "generated_files/frame_start.png"
	},
	"last_frame": {
		"path": "generated_files/frame_end.png"
	},
	"reference_images": [
		{
			"path": "generated_files/ref1.png",
			"reference_type": "asset"
		},
		{
			"path": "generated_files/ref2.png",
			"reference_type": "style"
		}
	],
	"video": {
		"path": "generated_files/previous_video.mp4"
	}
}

Supported values and limits:
- aspect_ratio: 16:9 or 9:16.
- resolution: 720p, 1080p, or 4k.
- duration_seconds: 4, 6, or 8.
- person_generation: allow_all, allow_adult, dont_allow.
- reference_images: up to 3 images.
- reference_type: asset or style.
- first_image and last_frame can be local file path, gcs uri, or base64 image payload.

When to include image/video controls:
- Use first_image (or image) for image-to-video start frame.
- Use last_frame to constrain the ending frame interpolation.
- Use reference_images for identity/style preservation.
- Use video for Veo extension workflows when continuing a prior clip.

Prompt quality guidance:
- Include subject, action, style, camera movement/composition, and ambiance.
- If audio matters, include dialogue in quotes and explicit SFX/ambient cues.

Behavior:
- Tool returns absolute video path on success.
- Tool returns error: ... on failure.
- If generation fails, simplify prompt and/or reduce constraints (fewer references, default resolution).
 - If the prompt was a JSON object, the tool will return the output path followed by a \`JSON_INPUT:\` marker and the parsed JSON payload (path is on the first line for backwards compatibility).