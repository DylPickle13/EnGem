# run_google_search tool instructions

Use this tool for quick web-grounded discovery and current-information lookups.

What this tool does:
- Uses Gemini Google Search grounding.
- Returns model candidate text describing findings.
- Optimized for summarized search results, not full-page scraping.

How to call it well:
- Write a focused query with topic, scope, and timeframe.
- Include constraints like region, date range, or source type when needed.
- Ask for concise synthesis points to reduce noisy output.

Important constraints:
- Output may not include complete raw source content.
- This is not a browser automation tool.
- For deep multi-source analysis, prefer deep_research.

High quality prompting pattern:
- Subject + time window + required evidence type.
- Example: "Latest policy updates on X in 2025 with key changes and dates."