# arxiv-digest

A daily digest of the new arXiv papers that match **your** research interests, written as a Markdown file you can read in any editor, on GitHub or in Obsidian.

You describe what you care about: keywords, a sentence or two about a research idea, or authors you follow. `arxiv-digest run` then:
1. fetches the papers newly announced in your arXiv categories
2. ranks them against your interests on your own computer
3. writes the best ones, with links, abstracts and Claude's reasons, to `digests/2026-11-03.md` and `digests/2026-11-03.html`, and opens the HTML in your browser

Or run `arxiv-digest app` for the same thing in a browser window: browse past digests, run a new one, and edit your interests in forms.

> **Status: v0.4.** A local model shortlists the day's papers. Claude keeps only the genuinely relevant ones, explains why each was chosen, and reads the top picks in full to summarise their method and results. Every digest is written as Markdown and as HTML. There's a local web app, and an MCP server so Claude Code and Claude Desktop can use your digests directly.

## Install

You need Python 3.11–3.14. For Claude's picks, explanations and deep reads, you also need [Claude Code](https://claude.com/claude-code) signed in with your Claude account. arxiv-digest isn't on PyPI yet, so install it from a clone of this repository with any one of:

```bash
uv tool install .            # with uv (recommended: it picks a supported Python for you)
pipx install .               # with pipx
python3 -m pip install .     # with plain pip, ideally in a virtual environment
```

uv isn't required. It's just the easiest option. If your `python3` is 3.15 (or older than 3.11), pip and pipx will refuse to install, because the local model's runtime (onnxruntime) has no 3.15 build yet. In that case either:
- use uv, which fetches a supported Python automatically, or
- point pipx at a supported one: `pipx install --python python3.13 .`

Contributors use uv to run the tests (see [Development](#development)).

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
| `arxiv-digest run --no-deep-read` | Skip Claude's full-text read of the top picks |
| `arxiv-digest run --no-model` | Rank by keywords and followed authors only (no local model) |
| `arxiv-digest categories list [--all]` | Your categories, or every arXiv category |
| `arxiv-digest categories add hep-th` | Watch another category (checked against arXiv's list) |
| `arxiv-digest categories remove hep-th` | Stop watching a category |
| `arxiv-digest interests` | Show your interests |
| `arxiv-digest status` | File locations, papers seen, Claude's status, and the last run |
| `arxiv-digest mcp install` / `uninstall` | Make arxiv-digest's tools available in Claude Code and Claude Desktop (see [Use it from Claude](#use-it-from-claude-mcp)) |
| `arxiv-digest app [--port 8765]` | Run the web app in this terminal and open it in your browser |
| `arxiv-digest open` | Open the web app, starting it in the background if needed (what the launcher entry runs) |
| `arxiv-digest shortcut install` / `remove` | Add or remove "arxiv-digest" in your system's app launcher |

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
| `claude_command` | `"claude"` | The command that runs Claude Code. If it isn't on your PATH, arxiv-digest also checks the usual install locations (`~/.local/bin`, Homebrew, npm, nvm and others) and your login shell. Set a full path only if Claude Code still isn't found. |
| `claude_timeout` | `600` | Seconds to wait for Claude before falling back to the local ranking |
| `deep_reads` | `3` | How many top picks Claude reads in full (0 turns deep reads off; never more than `top_picks`) |
| `deep_read_model` | same as `claude_model` | The model for deep reads, e.g. `"opus"` for the most thorough reading |
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
   - **Claude account only.** arxiv-digest uses your Claude account (a Claude subscription signed in to Claude Code) and never API keys or pay-as-you-go API credits:
     - Claude Code normally prefers an API key found in the environment over your account sign-in, and bills that key's credits. So arxiv-digest removes `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and the Bedrock/Vertex/Foundry switches from the environment it runs Claude Code in. A key set for another tool can't cause "Credit balance is too low" errors.
     - Only a Claude-account sign-in counts. An Anthropic Console (API billing) sign-in is treated as not signed in.
   - **Logging in.** arxiv-digest never handles your login; Claude Code stores it. `arxiv-digest status` and the web app show which account is signed in. That check runs `claude auth status` and uses none of your Claude usage.
     - The app has a **Log in to a Claude account** button beside the status, which runs `claude auth login --claudeai` and opens Claude's login page. Use it to sign in, or to switch accounts.
     - When you're not signed in, runs skip Claude with a note saying how to log in.
   - If Claude Code isn't installed, or the call fails, the digest falls back to the local rule (a followed author, a keyword hit, or similarity well above the day's average) and says so at the top.

5. **Deep read.** For the top picks (3 by default), arxiv-digest downloads the full text. It uses arXiv's HTML version where there is one, which keeps the math as LaTeX, and otherwise the PDF's text, leaving out the references. Claude reads each paper and writes three short parts, shown in a "Deep read" panel on the paper's card:
   - **Method:** what the authors did
   - **Results:** what they found, with the key numbers
   - **Why it matters:** why it matters for your interest, with the main caveat

   Reads run in parallel and took about 20 seconds for three papers in testing. Each read is cached, so re-running a digest doesn't read the same paper twice. Very long papers are cut to about 120,000 characters. If a paper can't be downloaded or read, the digest says so and carries on.

**What leaves your computer:**
- requests to arXiv, including full-text downloads for deep reads
- the one-time model download from Hugging Face
- when Claude is used, sent to Anthropic through Claude Code: your interests, the shortlisted titles, authors and abstracts, and the full text of the papers that get a deep read

With `--no-claude`, ranking stays entirely on your computer.

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

## Use it from Claude (MCP)

```bash
arxiv-digest mcp install
```

This adds arxiv-digest's tools to **Claude Code**, for all your projects (`claude mcp add --scope user`), and, if it's installed, **Claude Desktop** (macOS and Windows; restart Claude Desktop afterwards). Then you can ask Claude things like:

- "What's in my latest arXiv digest? Summarise the top picks."
- "Any new papers this week on emulators for weak lensing?"
- "Read 2611.02627 and compare its method with my top pick."
- "Add an interest for intrinsic alignments, with the keyword *IA*."

The tools:

| Tool | What it does |
|---|---|
| `list_digests`, `get_digest` | Your digests, with Claude's reasons and deep reads |
| `search_papers` | Searches the last ~90 days of papers in your categories by meaning, including ones that weren't picked |
| `get_paper` | A paper's details by arXiv id or link, and optionally its full text, in parts |
| `todays_candidates` | The latest announcement, shortlisted on your computer, for the Claude you're talking to to judge (nothing is marked as seen) |
| `run_digest` | Runs a digest like the Run button |
| `list_interests`, `add_interest`, `update_interest`, `remove_interest` | Manage your interests (checked before saving; comments in the file are kept) |

The server runs on your computer over stdio. Claude starts it; it opens no network port. Only `run_digest` calls Claude itself, through Claude Code with your Claude account, exactly like the Run button. The other tools leave the thinking to the Claude you're already talking to. Paper text in the results is marked as untrusted data. Claude Code asks your permission before using each tool, unless you allow it. `arxiv-digest mcp uninstall` removes it.

## Web app

### Opening it without a terminal

The first time you open the web app, it offers to add itself to your Applications (macOS), Start menu (Windows) or app launcher (Linux) with one click. You can also do it from a terminal:

```bash
arxiv-digest shortcut install
```

This adds **arxiv-digest**, with a test-tube icon, to your app launcher:

| Platform | Where it goes | Tested |
|---|---|---|
| Linux | `~/.local/share/applications`, so it appears in GNOME, KDE, Omarchy and similar launchers | yes |
| macOS | `~/Applications/arxiv-digest.app` (Spotlight, Launchpad) | no |
| Windows | the Start menu | no |

Clicking it starts the app in the background, with no terminal window, and opens it in your browser. If it's already running, clicking just opens it again.

A background app stops by itself 10 minutes after you close its last tab, never during a run. You can also stop it any time with **Quit** in the top bar.

The entry runs the same Python environment you ran `shortcut install` from. If you reinstall arxiv-digest somewhere else, run `shortcut install` again. Remove the entry with `arxiv-digest shortcut remove`.

### What's in it

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
4. ✅ Deep read: Claude reads the full text of the top picks and summarises the method and results
5. ✅ MCP server, so Claude can browse, search and explain your digests
6. PyPI release and a Claude Code plugin

## License

[MIT](LICENSE) © 2026 Matt Wiemann.

The web app bundles the fonts [Inter](https://rsms.me/inter/), [Space Grotesk](https://github.com/floriankarsten/space-grotesk) and [JetBrains Mono](https://www.jetbrains.com/lp/mono/), each under the [SIL Open Font License 1.1](src/arxiv_digest/assets/fonts/).
