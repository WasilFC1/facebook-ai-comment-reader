# Facebook AI Comment Reader — Version 1

A simple local Streamlit application that reads the Facebook scraper Excel export and uses an OpenAI model to interpret each comment in natural language.

## Setup (Windows)

1. Install Python 3.11+.
2. Open PowerShell in this folder.
3. Create and activate a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Launch:
   ```powershell
   streamlit run app.py
   ```
6. Enter your OpenAI API key in the sidebar, paste the Facebook post description, upload the scraper `.xlsx`, and click **Analyse comments with AI**.

## What it does
- Reads the `text` column, plus `author_name`, `id`, `comment_url`, and `comment_time` when available.
- Uses the AI to classify each comment as `order`, `question`, `ambiguous`, `change`, `cancellation`, or `other`.
- Extracts items, quantities, colours and sizes where stated.
- Flags uncertain comments for review.
- Shows results and allows CSV download.

## Important limitations
- Requires internet and an OpenAI API key; API usage may incur charges.
- Sends comment text and the supplied post description to OpenAI. Review privacy requirements before processing personal data.
- Processes up to 1,999 data rows per file in this initial version.
- Analyses comments independently; it does not reconcile multiple comments by the same customer.
- Does not automatically approve orders, purchase stock, or generate a supplier order.
- AI interpretations can be wrong; verify results before acting on them.
