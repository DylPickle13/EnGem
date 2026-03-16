# access_google_workspace tool instructions

Use this tool for plain-language Google Workspace operations (Drive, Docs, Sheets, Calendar, Gmail, etc.) through the gws wrapper.

What this tool does:
- Accepts a natural-language query.
- Internally plans and validates API calls/workflows.
- Executes the request and returns an interpreted user-facing answer.
- May include prefix lines such as Planner attempts and CLI query/queries.

How to call it well:
- Describe the exact workspace task and desired final output.
- Include concrete identifiers when known (document id, spreadsheet id, folder id, event id).
- For write operations, specify verification expectations.
- For read operations, specify the exact fields/content needed.

Important constraints:
- This is not raw shell access; give intent, not command syntax.
- Multi-step tasks are supported when dependencies are clear.
- If auth/permission fails, returned output includes relevant error context.

High quality prompting pattern:
- Objective.
- Inputs/ids.
- Required output shape.
- Verification requirement after any write/update/delete.