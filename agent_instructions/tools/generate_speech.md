# generate_speech tool instructions

Use this tool for Gemini text-to-speech generation (single-speaker or multi-speaker).

Call contract (important):
- The tool accepts one primary string argument named prompt.
- For advanced control, set prompt to a JSON object string.
- Always include a top-level prompt field inside JSON mode.

Required JSON shape:
{
	"prompt": "Text to be spoken",
	"model": "gemini-2.5-flash-preview-tts",
	"voice_name": "Kore",
	"language_code": "en-US",
	"sample_rate_hz": 24000,
	"temperature": 0.7,
	"top_p": 0.95,
	"top_k": 40,
	"candidate_count": 1,
	"max_output_tokens": 2048,
	"seed": 123,
	"stop_sequences": ["<END>"],
	"speakers": [
		{"speaker": "Joe", "voice_name": "Kore"},
		{"speaker": "Jane", "voice_name": "Puck"}
	]
}

Supported behavior:
- Single speaker: provide voice_name.
- Multi-speaker: provide speakers array (up to 2); each item needs speaker and voice_name.
- If speakers is provided and valid, multi-speaker mode is used.
- Output is WAV (mono, 16-bit PCM), written to generated_files/.

Voice options:
- Use these voice profiles to match delivery style to your content:
- Zephyr (Bright): clear, lively brightness; good for upbeat intros and short promos.
- Puck (Upbeat): energetic and playful; good for social clips and youthful narration.
- Charon (Informative): calm explainer tone; good for tutorials and how-to content.
- Kore (Firm): confident, directive delivery; good for announcements and host narration.
- Fenrir (Excitable): high-energy, animated delivery; good for hype reads and teaser copy.
- Leda (Youthful): younger, lighter character; good for friendly educational reads.
- Orus (Firm): grounded authority with less intensity than hype voices; good for structured briefings.
- Aoede (Breezy): relaxed and airy flow; good for lifestyle and travel narration.
- Callirrhoe (Easy-going): conversational and low-pressure; good for casual explainers.
- Autonoe (Bright): polished brightness; good for ad reads and energetic product demos.
- Enceladus (Breathy): intimate, soft-edge texture; good for reflective or atmospheric reads.
- Iapetus (Clear): crisp articulation and neutral color; good for technical narration.
- Umbriel (Easy-going): smooth and approachable; good for long-form voiceover comfort.
- Algieba (Smooth): silky, polished tonality; good for premium brand reads.
- Despina (Smooth): gentle smoothness with easy pacing; good for calm storytelling.
- Erinome (Clear): neutral and legible delivery; good for instructional scripts.
- Algenib (Gravelly): textured/grainy character; good for dramatic or gritty lines.
- Rasalgethi (Informative): broadcast-style clarity; good for documentary and educational copy.
- Laomedeia (Upbeat): positive, friendly momentum; good for app walkthroughs and onboarding.
- Achernar (Soft): mellow and controlled; good for bedtime-style or soothing content.
- Alnilam (Firm): strong anchor voice; good for formal narration and key messages.
- Schedar (Even): balanced, neutral pacing; good for steady long-form reads.
- Gacrux (Mature): older, seasoned character; good for authoritative storytelling.
- Pulcherrima (Forward): present and assertive projection; good for direct-response lines.
- Achird (Friendly): warm and personable; good for customer-facing scripts.
- Zubenelgenubi (Casual): informal and relaxed; good for conversational podcasts.
- Vindemiatrix (Gentle): soft and kind tone; good for compassionate messaging.
- Sadachbia (Lively): animated and bright motion; good for event and promo reads.
- Sadaltager (Knowledgeable): expert-like, composed delivery; good for educational authority.
- Sulafat (Warm): rounded, inviting warmth; good for brand storytelling and reflective narration.
- Selection tips:
- For tutorials/explainers: Charon, Rasalgethi, Iapetus, Erinome, Sadaltager.
- For ads/energy: Puck, Zephyr, Fenrir, Laomedeia, Sadachbia.
- For calm/warm reads: Sulafat, Achernar, Vindemiatrix, Despina, Umbriel.
- For authoritative reads: Kore, Alnilam, Orus, Gacrux, Pulcherrima.

Prompt quality guidance:
- Provide exact transcript text.
- Include style and pacing direction in natural language when needed.
- For multi-speaker, keep speaker labels in transcript aligned with speakers entries.

Return format:
- Success (plain text prompt): absolute output path.
- Success (JSON prompt): output path, then JSON_INPUT: followed by parsed JSON payload.
- Failure: error: ...