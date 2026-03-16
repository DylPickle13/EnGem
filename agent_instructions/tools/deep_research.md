# deep_research tool instructions

Use this tool for complex, multi-step research that needs broader synthesis than a quick search.

What this tool does:
- Starts a background deep research interaction.
- Polls until completion.
- Returns final synthesized text (or failure status).
- Prefixes output with a research interaction id.

How to call it well:
- Provide a clear research objective and decision context.
- Include scope boundaries (timeframe, geography, domain).
- Specify expected output structure (summary, comparison table, risks, recommendations).

Important constraints:
- This tool can take longer than quick search tools.
- Best for non-trivial analysis, not simple fact lookup.
- On failure, output includes error details; revise scope and retry if needed.

High quality prompting pattern:
- Objective.
- Key questions.
- Constraints.
- Required deliverable format.