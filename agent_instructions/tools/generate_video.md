# generate_video tool instructions

Use this tool to generate a video with Veo and save it in generated_files/.

Call contract (important):
- The tool accepts one string argument named prompt.
- For advanced control, set prompt to a JSON object string.
- Prefer Gemini API JSON field names (camelCase). Snake_case aliases remain supported for backward compatibility.
- Always include a top-level prompt field inside the JSON.
- Use mid-level resolution to reduce costs, unless stated otherwise in the user request.

Preferred JSON shape (Gemini API style):
{
	"prompt": "Video description",
	"model": "veo-3.1-lite-generate-preview",
	"aspectRatio": "16:9",
	"resolution": "720p",
	"durationSeconds": 8,
	"numberOfVideos": 1,
	"negativePrompt": "undesired elements",
	"enhancePrompt": true,
	"firstImage": {
		"path": "generated_files/frame_start.png"
	},
	"lastFrame": {
		"path": "generated_files/frame_end.png"
	},
	"referenceImages": [
		{
			"path": "generated_files/ref1.png",
			"referenceType": "asset"
		},
		{
			"path": "generated_files/ref2.png",
			"referenceType": "style"
		}
	],
	"video": {
		"uri": "<video-uri-from-previous-generation>"
	}
}

Supported aliases:
- aspectRatio or aspect_ratio.
- durationSeconds or duration_seconds.
- numberOfVideos or number_of_videos.
- negativePrompt or negative_prompt.
- enhancePrompt or enhance_prompt.
- firstImage or first_image (or image).
- lastFrame or last_frame.
- referenceImages or reference_images.
- referenceType or reference_type.

Supported values and limits:
- aspectRatio/aspect_ratio: 16:9 or 9:16.
- resolution: 720p, 1080p, or 4k.
- durationSeconds/duration_seconds: 4, 6, or 8.
- durationSeconds/duration_seconds must be 8 when using referenceImages/reference_images, video extension, 1080p, or 4k.
- numberOfVideos/number_of_videos: 1.
- personGeneration/person_generation is currently not supported in this environment; omit it.
- referenceImages/reference_images: up to 3 images.
- referenceType/reference_type: asset or style.
- firstImage/first_image and lastFrame/last_frame can be local file path, gcs uri, or base64 image payload.
- video extension input should use uri only in this environment (local path/base64 video maps to encodedVideo and can fail).
- fps is not a request parameter for Veo 3.1 (output is fixed at 24fps).
- for extension requests with video input, use resolution 720p.
- do not combine video with firstImage/first_image/image; these are different generation modes.

When to include image/video controls:
- Use firstImage/first_image (or image) for image-to-video start frame.
- Use lastFrame/last_frame to constrain the ending frame interpolation.
- Use referenceImages/reference_images for identity/style preservation.
- Use video for Veo extension workflows when continuing a prior clip.

Prompt quality guidance:
- Include subject, action, style, camera movement/composition, and ambiance.
- If audio matters, include dialogue in quotes and explicit SFX/ambient cues.

Behavior:
- Tool returns absolute video path on success.
- Tool returns error: ... on failure.
- If generation fails, simplify prompt and/or reduce constraints (fewer references, default resolution).
 - If the prompt was a JSON object, the tool will return the output path followed by a \`JSON_INPUT:\` marker and the parsed JSON payload (path is on the first line for backwards compatibility).