<p align="center">
  <img src="src/arxiv_digest/assets/icons/arxiv-digest.svg" alt="The arxiv-digest icon, a test tube" width="128">
</p>

<h1 align="center">arxiv-digest</h1>

<p align="center">
  A daily digest of the new arXiv papers that match <b>your</b> research interests,<br>
  shortlisted by a small local model, then further ranked and explained by Claude.
</p>

**Install (recommended).** On macOS or Linux, open a terminal and run:

```bash
curl -LsSf https://raw.githubusercontent.com/SampsonML/arxiv-digest/main/install.sh | sh
```

## Install

The command above is the recommended way to install, on macOS and Linux. **On Windows**, open PowerShell and run

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/SampsonML/arxiv-digest/main/install.ps1 | iex"
```

**arxiv-digest** is now in your applications, with a test-tube icon: Launchpad and Spotlight on macOS, the Start menu on Windows, and your app launcher on Linux. The installer also opens it in your browser. The first time, choose your arXiv categories and describe your interests, then press **Run today's digest**.

What the installer does ([read it first](install.sh) if you like):
1. If you don't have [uv](https://docs.astral.sh/uv/), it asks to install it. uv sets up a supported Python (3.11–3.14) for arxiv-digest without changing your system's Python.
2. It downloads arxiv-digest from this repository and installs it in its own environment, with the exact package versions in `uv.lock`. The `arxiv-digest` command goes in `~/.local/bin`.
3. It adds the app to your applications, checks whether Claude Code is signed in, and opens the app.

Good to know:
- **Claude.** For Claude's picks, reasons and deep reads, you need [Claude Code](https://claude.com/claude-code) signed in with your Claude account; API keys are never used. The app shows whether you're signed in and has a **Log in** button. Without Claude Code, arxiv-digest still works, choosing papers on your computer only.
- **Runs on** macOS 14 or later on Apple silicon, Linux (x86-64 or ARM) and Windows 10/11. Intel Macs aren't supported, because the local model's runtime (onnxruntime) is no longer built for them.

**Update:** press **Quit** in the app, then run the install command again.

**Uninstall:**

```bash
arxiv-digest shortcut remove     # the app entry
arxiv-digest mcp uninstall       # only if you added the MCP server
uv tool uninstall arxiv-digest
```

Your settings and history are in `~/.config/arxiv-digest` and `~/.local/share/arxiv-digest`; delete those too to remove everything.

## What it does

You describe what you care about: keywords, a sentence or two about a research idea, or authors you follow. Each time you run a digest, arxiv-digest:
1. fetches the papers newly announced in your arXiv categories
2. ranks them against your interests on your own computer
3. shows the best ones in the app, with links, abstracts and Claude's reasons

Digests stay in the app, where you can browse every one you've made. Nothing is written to your disk unless you ask: to keep a digest as a file, press **Save digest** at the top of it and choose **As HTML** (a web page with working links and rendered math) or **As Markdown** (for any editor, GitHub or Obsidian). Your browser downloads it.

> **Status: v0.5.** A local model shortlists the day's papers. Claude keeps only the genuinely relevant ones, explains why each was chosen, and reads the top picks in full to summarise their method and results. Any digest can be saved as Markdown or HTML. There's a local web app, and an MCP server so Claude Code and Claude Desktop can use your digests directly.

## From the terminal

Everything the app does also works from the command line:

```bash
arxiv-digest run       # fetch, rank, and open today's digest in the app
arxiv-digest status    # file locations, Claude's sign-in and the last run
```

If your terminal says `arxiv-digest` isn't found, run `uv tool update-shell` and open a new terminal; that adds `~/.local/bin` to your PATH.

The first run downloads a small language model once (~90 MB) and covers arXiv's latest announcement. After that, each run picks up everything announced since the previous run (up to 7 days back), and a paper never appears in two digests.

arXiv announces new papers Sunday to Thursday at about 20:00 US Eastern, with none on Friday or Saturday nights. Running a digest each morning is enough; running it more often just finds nothing new.

## Commands

| Command | What it does |
|---|---|
| `arxiv-digest init` | Create the config by answering a few questions |
| `arxiv-digest run` | Make a digest of the papers that are new since the last run, and open it in the app |
| `arxiv-digest run --save` | The same, and also write it as Markdown and HTML files in `output_dir` |
| `arxiv-digest run --print` | Also print the digest (Markdown) in the terminal |
| `arxiv-digest run --no-open` | Don't open the app |
| `arxiv-digest run --dry-run` | Print the digest instead of keeping it, and remember nothing |
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
| `arxiv-digest setup` | Add the app to your applications, check Claude, and open the app (what the installers run) |

Every command accepts `--config PATH` to use a different config file.

## Configuration

`~/.config/arxiv-digest/config.toml` (written by `init`, safe to edit by hand):

```toml
categories = ["astro-ph.CO", "astro-ph.GA", "cs.LG", "stat.ML"]
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
| `output_dir` | `~/arxiv-digest` | Where `arxiv-digest run --save` writes Markdown and HTML files (e.g. a folder in your Obsidian vault) |
| `shortlist_size` | `40` | How many papers the local model keeps for final ranking |
| `max_catchup_days` | `7` | The furthest back a run looks after a long break |
| `min_zscore` | `2.5` | Without Claude, how far above the day's average similarity a paper must be to be picked on meaning alone (2.5 ≈ the closest 1%) |
| `use_claude` | `true` | Let Claude pick the final papers and explain each one (needs [Claude Code](https://claude.com/claude-code)) |
| `claude_model` | `"sonnet"` | `"sonnet"`, `"haiku"` (lighter) or `"opus"` (most thorough), or a full model name |
| `claude_command` | `"claude"` | The command that runs Claude Code. If it isn't on your PATH, arxiv-digest also checks the usual install locations (`~/.local/bin`, Homebrew, npm, nvm and others) and your login shell. Set a full path only if Claude Code still isn't found. |
| `claude_timeout` | `600` | Seconds to wait for Claude before falling back to the local ranking |
| `deep_reads` | `3` | How many top picks Claude reads in full (0 turns deep reads off; never more than `top_picks`) |
| `deep_read_model` | `"sonnet"` | The model for deep reads. It doesn't follow `claude_model`; set it to e.g. `"opus"` for the most thorough reading |
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

5. **Deep read.** For the top picks (3 by default), arxiv-digest downloads the full text. It uses arXiv's HTML version where there is one, which keeps the math as LaTeX, and otherwise the PDF's text, leaving out the references. Claude reads each paper and writes three parts, at least five sentences in all, shown in a "Deep read" panel on the paper's card:
   - **Method:** what the authors did (2–3 sentences)
   - **Results:** what they found, with the key numbers (2–3 sentences)
   - **Why it matters:** why it matters for your interest, with the main caveat (1–2 sentences)

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
| Digests | In the database. Files only when you save one: **Save digest** downloads it with your browser, and `run --save` writes to `output_dir` (default `~/arxiv-digest`; when you work on the code with `uv run`, the project's git-ignored `digests/` folder) | |

These follow the XDG conventions (`XDG_CONFIG_HOME` etc. are respected) on every platform, macOS included. `ARXIV_DIGEST_HOME` puts the database and model cache under one folder, which is handy for testing.

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

The installer adds **arxiv-digest**, with a test-tube icon, to your applications. If you installed it [another way](#other-ways-to-install), run `arxiv-digest setup`, or use the one-click offer the web app shows the first time it opens. (`arxiv-digest shortcut install` adds the entry without opening the app.)

| Platform | Where it goes | Tested |
|---|---|---|
| Linux | `~/.local/share/applications`, so it appears in GNOME, KDE, Omarchy and similar launchers | yes |
| macOS | `~/Applications/arxiv-digest.app` (Spotlight, Launchpad) | no |
| Windows | the Start menu | no |

Clicking it starts the app in the background, with no terminal window, and opens it in your browser. If it's already running, clicking just opens it again.

A background app stops by itself 10 minutes after you close its last tab, never during a run. You can also stop it any time with **Quit** in the top bar.

The entry runs the Python environment it was added from (the installer's is `~/.local/share/uv/tools/arxiv-digest`). If you install arxiv-digest another way, run `arxiv-digest setup` again. Remove the entry with `arxiv-digest shortcut remove`.

### What's in it

`arxiv-digest app` starts a small web server on your computer and opens it in your browser:

- **Start page:** a greeting with your latest digest's numbers (new, shortlisted, picked) and a **Run today's digest** button. Below it, your interests appear as coloured chips: click one to edit it in a side panel, or click **+ Add interest**. **Save** keeps the changes; **Save and re-run latest announcement** also shows today's papers again, chosen for the updated interests.
- **Digests:** every digest you've made, newest first, with older/newer links. **Save digest** at the top of one downloads it as HTML or Markdown.
- **Run:** a live tracker follows each step (Fetch, Shortlist, Claude, Done) with counts and a progress bar, and the new digest opens when it finishes. The full log is one click away.
- **Settings:** add, edit or remove interests (description, keywords, authors, required and excluded words), choose categories from arXiv's full list, and set the digest size and Claude model. Your config is checked before it's saved, so a mistake can't break it. Comments in the file are kept, except inside `[[interest]]` tables.

**Background:** the app shows a night-sky image behind dark, translucent panels: a dark indigo sky with a soft copper and teal light trail. Set `app_background = "none"` in the config to turn it off, or point it at your own `.jpg`, `.png` or `.webp` file. The image is original artwork drawn entirely by code (`tools/make_background.py`, no photos involved). Regenerate it, or try another seed, with `uv run --with pillow python tools/make_background.py src/arxiv_digest/assets/background.jpg 12`.

The server only listens on `127.0.0.1`. It rejects requests addressed to any other host name, which blocks DNS-rebinding attacks from other sites open in your browser. Every form also carries a random token, so other websites can't submit changes for you. Stop it with Ctrl+C.

## Other ways to install

The install command is the simplest route, but arxiv-digest is an ordinary Python package that needs Python 3.11–3.14.

**From a clone.** `./install.sh` (or `install.ps1` on Windows) in a clone of this repository does the same as the install command, using the clone's code. Run it again after `git pull` to update.

**With pipx or pip.** From a clone, either of these works:

```bash
pipx install .               # with pipx
python3 -m pip install .     # with plain pip, ideally in a virtual environment
```

Then run `arxiv-digest setup` to add the app to your applications and open it. These don't use the exact versions in `uv.lock`.

If your `python3` is 3.15 (or older than 3.11), pip and pipx refuse to install, because onnxruntime has no 3.15 build yet. In that case either:
- use uv, which fetches a supported Python automatically, or
- point pipx at a supported one: `pipx install --python python3.13 .`

## Development

```bash
uv sync
uv run pytest           # fast and offline: fake arXiv responses and a fake model
uv run ruff check . && uv run ruff format --check .
```

## License

[MIT](LICENSE) © 2026 Matt Wiemann.

The web app bundles the fonts [Inter](https://rsms.me/inter/), [Space Grotesk](https://github.com/floriankarsten/space-grotesk) and [JetBrains Mono](https://www.jetbrains.com/lp/mono/), each under the [SIL Open Font License 1.1](src/arxiv_digest/assets/fonts/).
