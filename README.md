# Stock Research Assistant

Questions answered from the public BSE filings of two Indian companies, Stylam Industries and Speciality Restaurants, with page citations. A cohort capstone project. It explains what the filings say. It is not investment advice.

- **Live app:** https://pkumarshon-equity-filings-assistant.hf.space (behind a password)
- **Model:** `gpt-6.1-terra`, through OpenAI's Responses API

## How it works

**At build time**
1. `fetch_corpus.py` and `ingest.py` collect the filings and clean their text.
2. `rechunk.py` and `embed.py` split the text into chunks and embed them.
3. `build_financials.py` builds a table of figures from the financial statements.
4. `build_ui.py` works out everything the screens show that does not depend on a question.

**While the user waits**
1. `agent.py` lets the model choose tools: look up a figure, search the filings, or get the latest share price.
2. Plain-code checks test the answer. A number that no tool returned holds the answer back, and so does advice.
3. `app.py` shows the answer with its sources.

The health check (`healthcheck.py`) is mostly fixed rules. Only check 7, on management's promises, uses a model, and the rules decide its label.

## Run it

```
pip install -r requirements.txt
python check_agent_rules.py     # the rule tests; no key needed
python app.py                   # needs the built data below and an OpenAI key
```

Keys are read from environment variables, such as `CAPSTONE_OPENAI_API_KEY`. Never put a key in a file here.

**Not in this repo:** the filings (`corpus/`), the built data (`.store/`) and the document list (`docs.csv`), which are too large or hold local folder paths. So `python app.py` will not start from a fresh clone. `fetch_corpus.py` and `ingest.py` also read a local folder that is not included. `space/` holds the app code deployed to the live app.

## Measured results

Answers vary between runs, so the single-run scores below are not fixed numbers. `gpt-6.1-terra`, 40 questions, two runs:

| Measure | Run 1 | Run 2 |
|---|---|---|
| First tool fits the question | 32/40 | 32/40 |
| Expected figure in the answer (a crude string match) | 32/39 | 29/39 |
| "Insufficient evidence" on questions not in the filings | 3/4 | 4/4 |

Repeating four of the questions three times each, with and without the price tool, gave the expected figure 7/12 and 6/12 times. Files: `agent_results_*.csv`, `agent_summary_*.txt`, `price_tool_control.json`.
