# arxiv-digest

A daily digest of the new arXiv papers that match **your** research interests, written as a Markdown file you can read in any editor, on GitHub or in Obsidian.

You describe what you care about: keywords, a sentence or two about a research idea, or authors you follow. `arxiv-digest run` then:
1. fetches the papers newly announced in your arXiv categories
2. ranks them against your interests on your own computer
3. writes the best ones, with links, abstracts and Claude's reasons, to `digests/2026-11-03.md` and `digests/2026-11-03.html`, and opens the HTML in your browser

Or run `arxiv-digest app` for the same thing in a browser window: browse past digests, run a new one, and edit your interests in forms.

> **Status: v0.3.** A local model shortlists the day's papers, and Claude keeps only the genuinely relevant ones and explains why each was chosen. Every digest is written as Markdown and as HTML, and there's a local web app. Coming next:
> - full-text summaries of the top picks
> - an MCP server so Claude Desktop and Claude Code can use the digest directly

## Install

arxiv-digest needs [uv](https://docs.astral.sh/uv/). It isn't on PyPI yet; from a clone of this repository:

```bash
uv tool install .          # puts `arxiv-digest` on your PATH
```

It runs on Python 3.11–3.14. If you only have 3.15, uv fetches a supported Python automatically. 3.15 is excluded for now because the local model's runtime (onnxruntime) has no 3.15 build yet.

## Quick start

```bash
arxiv-digest init      # a few questions: categories, output folder, interests
arxiv-digest run       # fetch, rank, write today's digest and open it
arxiv-digest app       # or do all of this in the web app
```

The first run downloads a small language model once (~90 MB) and covers arXiv's latest announcement. After that, each run picks up everything announced since the previous run (up to 7 days back), and a paper never appears in two digests.

arXiv announces new papers Sunday to Thursday at about 20:00 US Eastern, with none on Friday or Saturday nights. Running `arxiv-digest run` each morning is enough; running it more often just finds nothing new.

## Commands

| Command | What it does |
|---|---|
| `arxiv-digest init` | Create the config by answering a few questions |
| `arxiv-digest run` | Write a digest (Markdown and HTML) of the papers that are new since the last run, and open the HTML |
| `arxiv-digest run --no-open` | The same, without opening the browser |
| `arxiv-digest run --dry-run` | Print the digest instead of saving it, and remember nothing |
| `arxiv-digest run --days 3` | Cover the last 3 days of submissions (papers already shown are skipped) |
| `arxiv-digest run --include-seen` | Redo the latest announcement, including papers already shown (useful after editing interests) |
| `arxiv-digest run --no-claude` | Choose papers on this computer only, without Claude's picks and explanations |
| `arxiv-digest run --no-model` | Rank by keywords and followed authors only (no local model) |
| `arxiv-digest categories list [--all]` | Your categories, or every arXiv category |
| `arxiv-digest categories add hep-th` | Watch another category (checked against arXiv's list) |
| `arxiv-digest categories remove hep-th` | Stop watching a category |
| `arxiv-digest interests` | Show your interests |
| `arxiv-digest status` | File locations, papers seen, Claude's status, and the last run |
| `arxiv-digest app [--port 8765]` | Open the web app in your browser |

Every command accepts `--config PATH` to use a different config file.

## Configuration

`~/.config/arxiv-digest/config.toml` (written by `init`, safe to edit by hand):

```toml
categories = ["astro-ph.CO", "astro-ph.GA", "cs.LG", "stat.ML"]
output_dir = "~/Projects/arxiv-digest/digests"   # or e.g. a folder in your Obsidian vault
max_papers = 15                 # most papers in one digest
top_picks = 3                   # papers in the Top picks section

[[interest]]
name = "Simulation-based inference"
description = "Neural likelihood-free inference for estimating cosmological parameters"
keywords = ["simulation-based inference", "neural posterior estimation"]
exclude = ["quantum"]           # skip papers that mention any of these

[[interest]]
name = "Followed authors"
authors = ["Vera Rubin"]        # their new papers are always included
```

Each `[[interest]]` needs a `name` plus at least one of the following:

| Key | Meaning |
|---|---|
| `description` | A sentence or two, matched by meaning rather than exact words |
| `keywords` | Words or phrases that make a paper more relevant. Matching ignores case and accents, treats "N-body" and "N body" alike, and accepts plurals. |
| `require` | Every one of these must appear in the title or abstract |
| `exclude` | A paper containing any of these is skipped for this interest |
| `authors` | Their new papers are always included. Matching uses surname plus first initial, so "David W. Hogg", "D. Hogg" and "Hogg, David" are the same person. |

Less common settings, with their defaults:

| Setting | Default | Meaning |
|---|---|---|
| `shortlist_size` | `40` | How many papers the local model keeps for final ranking |
| `max_catchup_days` | `7` | The furthest back a run looks after a long break |
| `min_zscore` | `2.5` | Without Claude, how far above the day's average similarity a paper must be to be picked on meaning alone (2.5 ≈ the closest 1%) |
| `use_claude` | `true` | Let Claude pick the final papers and explain each one (needs [Claude Code](https://claude.com/claude-code)) |
| `claude_model` | `"sonnet"` | `"sonnet"`, `"haiku"` (lighter) or `"opus"` (most thorough), or a full model name |
| `claude_command` | `"claude"` | The command that runs Claude Code, if it isn't `claude` on your PATH |
| `claude_timeout` | `600` | Seconds to wait for Claude before falling back to the local ranking |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | The local model. `BAAI/bge-small-en-v1.5` ranks slightly better but uses about 4× the CPU. |
| `app_background` | `"default"` | The web app's background: `"default"`, `"none"`, or a path to your own image |
| `allow_unknown_categories` | `false` | Allow category codes that aren't in the bundled list, e.g. one arXiv added recently |

## How ranking works

1. **Fetch.** arXiv's API lists papers only after they're announced, so arxiv-digest asks for everything *submitted* since the newest paper it has already seen, with a 2-day overlap for papers announced late. It drops papers already shown, and keeps to arXiv's limit of one request every 3 seconds. Cross-listed papers are included. New versions of old papers are not.
2. **Score.** A small model on your CPU turns each title and abstract, and each interest, into a vector, and measures how close they are. The closeness is then compared with the rest of that day's papers (in standard deviations), so interests worded broadly or narrowly are treated fairly. Keyword hits add to the score, and followed authors put a paper at the top.
3. **Shortlist.** The shortlist keeps every followed-author paper plus the best papers for each interest, so a busy topic can't crowd out a quiet one.
4. **Pick.** Claude reads the shortlist and your interests and keeps only the papers that are genuinely relevant: at most `max_papers`, best first, and none on a quiet day. For each one it writes a short "why" and a 1–10 relevance score. Followed-author papers are always kept.
   - It runs through the Claude Code CLI (`claude -p`) with your own Claude sign-in, so no API key is needed. Each call is isolated: no tools, no saved session, and your Claude Code settings and hooks aren't loaded. The answer must match a JSON schema, and arxiv-digest checks it before use.
   - In testing, one call on a 35-paper shortlist took a few seconds.
   - **Signing in:** arxiv-digest never handles your Claude login. The `claude` command uses the sign-in Claude Code already stored when you signed in once (`claude auth login`, or `/login` inside Claude Code). `arxiv-digest status` and the web app show whether you're signed in. That check runs `claude auth status` and uses none of your Claude usage.
   - **If you're not signed in,** runs skip Claude with a note saying how to sign in, and the web app shows a **Sign in to Claude** button. The button runs `claude auth login`, which opens Claude's sign-in page in your browser.
   - If Claude Code isn't installed, or the call fails, the digest falls back to the local rule (a followed author, a keyword hit, or similarity well above the day's average) and says so at the top.

**What leaves your computer:** requests to arXiv; the one-time model download from Hugging Face; and, when Claude is used, your interests plus the shortlisted titles, authors and abstracts, sent to Anthropic through Claude Code. With `--no-claude`, ranking stays entirely on your computer.

## Where files live

| What | Where | Override |
|---|---|---|
| Config | `~/.config/arxiv-digest/config.toml` | `ARXIV_DIGEST_CONFIG`, `--config` |
| Database: papers seen, cached vectors, run history | `~/.local/share/arxiv-digest/` | `ARXIV_DIGEST_HOME` |
| Model cache | `~/.cache/arxiv-digest/models/` | `ARXIV_DIGEST_HOME` |
| Digests | `output_dir` in the config. `init` suggests the `digests/` folder of this project when run from a clone (git-ignored, so digests are never committed), otherwise `~/arxiv-digest`. | |

These follow the XDG conventions (`XDG_CONFIG_HOME` etc. are respected) on every platform, macOS included. `ARXIV_DIGEST_HOME` puts the database and model cache under one folder, which is handy for testing.

## Development

```bash
uv sync
uv run pytest           # fast and offline: fake arXiv responses and a fake model
uv run ruff check . && uv run ruff format --check .
```

## Web app

`arxiv-digest app` starts a small web server on your computer and opens it in your browser:

- **Start page:** a greeting with your latest digest's numbers (new, shortlisted, picked) and a **Run today's digest** button. Below it, your interests appear as coloured chips: click one to edit it in a side panel, or click **+ Add interest**. **Save** keeps the changes; **Save and re-run latest announcement** also shows today's papers again, chosen for the updated interests.
- **Digests:** every digest you've made, newest first. Each one opens in the same layout as the HTML file, with older/newer links.
- **Run:** a live tracker follows each step (Fetch, Shortlist, Claude, Done) with counts and a progress bar, and the new digest opens when it finishes. The full log is one click away.
- **Settings:** add, edit or remove interests (description, keywords, authors, required and excluded words), choose categories from arXiv's full list, and set the digest size and Claude model. Your config is checked before it's saved, so a mistake can't break it. Comments in the file are kept, except inside `[[interest]]` tables.

**Background:** the app shows a night-sky image behind dark, translucent panels: a dark indigo sky with a soft copper and teal light trail. Set `app_background = "none"` in the config to turn it off, or point it at your own `.jpg`, `.png` or `.webp` file. The image is original artwork drawn entirely by code (`tools/make_background.py`, no photos involved). Regenerate it, or try another seed, with `uv run --with pillow python tools/make_background.py src/arxiv_digest/assets/background.jpg 12`.

The server only listens on `127.0.0.1`. It rejects requests addressed to any other host name, which blocks DNS-rebinding attacks from other sites open in your browser. Every form also carries a random token, so other websites can't submit changes for you. Stop it with Ctrl+C.

## Roadmap

1. ✅ Core: config, fetching, local ranking, Markdown digest
2. ✅ Claude ranking through the Claude Code CLI (`claude -p`): picks only relevant papers, up to the cap, with a "why it was chosen" line for each
3. ✅ HTML digests (same content as the Markdown; math rendered with KaTeX) and the local web app
4. Deep read: Claude reads the full text of the top picks and summarizes the method and results
5. MCP server, so Claude can browse, search and explain your digest
6. PyPI release and a Claude Code plugin

## License

[MIT](LICENSE) © 2026 Matt Wiemann.

The web app bundles the fonts [Inter](https://rsms.me/inter/), [Space Grotesk](https://github.com/floriankarsten/space-grotesk) and [JetBrains Mono](https://www.jetbrains.com/lp/mono/), each under the [SIL Open Font License 1.1](src/arxiv_digest/assets/fonts/).
