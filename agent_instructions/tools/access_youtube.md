# access_youtube tool instructions

Use this tool for plain-language YouTube operations through a generic wrapper over YouTube Data API v3.

What this tool does:
- Accepts a natural-language query.
- Internally plans exactly one action per call: `api`.
- Resolves requests to the correct YouTube Data API `resource.method` pair.
- Executes the request and returns an interpreted user-facing answer.
- May include prefix lines such as Planner attempts and CLI query.

Action mapping:
- `api` -> generic YouTube Data API resource method call.
  - Maps to any `resource.method` exposed by the discovery client.
  - Main inputs from prompt: `resource`, `method`, optional `params`, optional `body`, optional `media_file`, optional `media_mime`, optional `page_all`, optional `max_pages`.

How to call it well:
- State the exact API operation in plain language (example: "call playlists.list").
- Specify required request parameters (`part`, identifiers, filters, pagination).
- For write/update/delete operations, specify the request body fields to set.
- For media uploads, provide a valid local `media_file` path.
- Ask for exact output shape when precision matters.

Important constraints:
- This wrapper is intent-based, not raw REST syntax; describe operation intent and fields.
- OAuth is required in runtime context for this implementation.
- Generic `api` calls can access the full API surface documented at https://developers.google.com/youtube/v3/docs, as long as the resource/method and parameters are valid and authorized.
- `params` and `body` must be JSON objects when provided.
- `media_file` must exist locally when provided.
- If auth, permissions, payload validation, or endpoint invocation fails, output includes actionable error details.

YouTube Data API context to leverage in prompts:
- API is resource-based (activities, captions, channels, comments, playlistItems, playlists, search, subscriptions, videos, etc.).
- Most methods rely on the `part` parameter to control returned/updated fields.
- List calls often support pagination (`maxResults`, `pageToken`), and this wrapper can iterate pages with `page_all` and `max_pages`.

High quality prompting pattern:
- Objective.
- `resource.method` target.
- Required params/body.
- Output shape.
- Verification requirement after write/update/delete calls.

Example queries:
- Call search.list with params part=snippet, q="python automation", type=video, maxResults=5.
- Call videos.list with params part=snippet,contentDetails,statistics and id=dQw4w9WgXcQ.
- Call channels.list with params part=snippet,contentDetails,statistics, mine=true, maxResults=10.
- Call playlists.list with params part=snippet,contentDetails, mine=true, maxResults=15.
- Call commentThreads.list with params part=snippet,replies, videoId=dQw4w9WgXcQ, maxResults=20.
- Call captions.list with params part=snippet and videoId=dQw4w9WgXcQ.
- Call subscriptions.list with params part=snippet,contentDetails, mine=true, maxResults=25.
- Call videos.rate with params id=dQw4w9WgXcQ, rating=like.
- Call videos.insert with params part=snippet,status, body containing snippet/status metadata, and media_file pointing to a local video path.
