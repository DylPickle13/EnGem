# access_youtube tool instructions

Use this tool for plain-language YouTube operations via YouTube Data API v3.

What this tool does:
- Accepts a natural-language query.
- Internally plans one of three actions: search, get details, upload.
- Executes the action and returns an interpreted answer.
- May include prefix lines such as Planner attempts and CLI query.

How to call it well:
- For search: specify query topic and result count.
- For details: provide exact video id.
- For upload: provide local file path and title, optionally description/tags/privacy.
- Ask for explicit output fields when precision matters.

Important constraints:
- Search/get/upload may require valid OAuth credentials in runtime context.
- Upload fails if local file path does not exist.
- Errors are returned with actionable details.

High quality prompting pattern:
- Operation intent (search/get/upload).
- Required identifiers/inputs.
- Expected final output format (summary, list, metadata fields).