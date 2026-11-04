from datetime import UTC, datetime

import httpx
import numpy as np
import pytest
import soundfile
from conftest import make_paper
from typer.testing import CliRunner

from arxiv_digest import audio, cli, paths
from arxiv_digest import config as cfg
from arxiv_digest.claude_cli import ClaudeAnswer, ClaudeError, ClaudeNotSignedIn
from arxiv_digest.digest import Digest, Entry
from arxiv_digest.models import DeepRead
from arxiv_digest.store import Store


def _digest(kind="digest", entries=None):
    if entries is None:
        entries = [
            Entry(
                make_paper(
                    "2611.00001",
                    title="Emulators for $N$-body sims at $z \\sim 2$",
                    authors=("Ada Lovelace", "Grace Hopper", "Alan Turing"),
                ),
                "Fast sims",
                why="Builds an emulator for $\\sigma_8$.",
                deep_read=DeepRead("Trains a network.", "1% error.", "Faster sims.", "PDF"),
            ),
            Entry(
                make_paper("2611.00002", title="Second paper", authors=("Vera Rubin",)),
                "Followed",
                followed_authors=("Vera Rubin",),
            ),
            Entry(make_paper("2611.00003", title="Third paper?"), "Fast sims", why="Also good."),
        ]
    return Digest(
        kind=kind,
        generated=datetime(2026, 11, 3, 9, 30).astimezone(),
        categories=["astro-ph.CO"],
        interests=["Fast sims", "Followed"],
        top_picks=1,
        new_papers=616,
        shortlisted=35,
        window_start=datetime(2026, 10, 30, 19, tzinfo=UTC),
        window_end=datetime(2026, 11, 2, 19, tzinfo=UTC),
        ranked_by="Chosen by Claude (test).",
        entries=entries,
    )


class FakeSpeaker:
    """Half a second of a quiet tone per paragraph; no voice download or model."""

    name = "en_US-test-medium"

    def __init__(self, downloaded=True):
        self.downloaded = downloaded
        self.spoken: list[list[str]] = []

    def is_downloaded(self):
        return self.downloaded

    def prepare(self, log):
        self.downloaded = True

    def speak(self, paragraphs, progress):
        self.spoken.append(list(paragraphs))
        rate = 16000
        tone = (np.sin(np.linspace(0, 440 * np.pi, rate // 2)) * 3000).astype(np.int16)
        for done in range(1, len(paragraphs) + 1):
            progress(done, len(paragraphs))
        return np.tile(tone, len(paragraphs)), rate


class FakeWriter:
    """Writes one short segment per paper, or fails for the papers in `fail`."""

    label = "Claude (test)"
    model = "test"

    def __init__(self, error=None, fail=None):
        self.error = error
        self.fail = fail  # paper ids that fail; None means all of them, when error is set
        self.calls = 0

    def write(self, entry, digest):
        self.calls += 1
        if self.error and (self.fail is None or entry.paper.id in self.fail):
            raise self.error
        return [f"A podcast segment about {entry.paper.id}.", "That's the paper."]

    @staticmethod
    def segments(digest):
        return {e.paper.id: FakeWriter().write(e, digest) for e in digest.entries}


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    yield store
    store.close()


def test_speakable_says_math_in_words():
    text = audio.speakable(
        "Emulators for $N$-body sims at $z\\sim2$ with $\\sigma_8$, $10^{14}\\,M_\\odot$"
        " and **bold** <1% error ~3x faster: https://example.org/x"
    )
    assert "$" not in text and "\\" not in text and "*" not in text and "http" not in text
    assert "N-body" in text
    assert "z about 2" in text
    assert "sigma 8" in text
    assert "10 to the 14 M solar" in text
    assert "less than 1% error about 3x faster:" in text


def test_episode_follows_the_digest_page():
    digest = _digest()
    script = audio.episode_script(
        digest, {e.paper.id: [f"About {e.paper.id}."] for e in digest.entries}
    )
    assert script == [
        "Welcome to your arXiv digest for Tuesday, November 3rd. 3 papers today.",
        "Our top pick.",
        "About 2611.00001.",
        "Next, Fast sims, with one paper.",  # groups follow the config's order
        "About 2611.00003.",
        "Next, Followed, with one paper.",
        "About 2611.00002.",
        audio.CLOSING,
    ]
    weird = audio.episode_script(_digest(kind="weird"), FakeWriter.segments(digest))
    assert weird[0].startswith("Welcome to something weird, for Tuesday, November 3rd: 3 papers")
    assert weird[1] == "Our weird paper."


def test_a_paper_read_as_written():
    first, second, third = _digest().entries
    assert audio.plain_segment(first) == [
        "Emulators for N-body sims at z about 2. By Ada Lovelace and colleagues.",
        "Builds an emulator for sigma 8.",
        "Trains a network.",  # the deep read, instead of the abstract
        "1% error.",
        "Faster sims.",
    ]
    assert audio.plain_segment(second) == [
        "Second paper. It's from Vera Rubin, whom you follow.",
        "Some abstract.",
    ]
    assert audio.plain_segment(third)[0] == "Third paper? By A. Author."


def test_claude_prompt_and_answer():
    digest = _digest()
    first, second, _ = digest.entries
    prompt = audio.build_prompt(first, digest)
    assert 'The listener follows the research interest "Fast sims".' in prompt
    assert "Title: Emulators for $N$-body sims" in prompt
    assert "First author: Ada Lovelace and 2 others" in prompt
    assert "Abstract:\nSome abstract." in prompt
    assert "- Method: Trains a network." in prompt
    assert "Followed authors: Vera Rubin" in audio.build_prompt(second, digest)
    assert "something weird" in audio.build_prompt(first, _digest(kind="weird"))
    parts = dict.fromkeys(audio.PARTS, "")
    parts.update(hook="Ever wondered?", introduction="Meet **EMBER**.", results="$z=10$.")
    assert audio.parse_script(parts) == ["Ever wondered? Meet EMBER.", "z equals 10."]
    for bad in ({**parts, "results": " "}, {"paragraphs": ["text"]}, {}):
        with pytest.raises(ClaudeError):
            audio.parse_script(bad)


def test_script_writer_asks_claude_for_a_two_minute_segment():
    class FakeCLI:
        model = "sonnet"

        def ask(self, system, prompt, schema):
            assert "about two minutes" in system and schema is audio.SCHEMA
            assert "Title: Second paper" in prompt
            return ClaudeAnswer({part: f"The {part}." for part in audio.PARTS})

    digest = _digest()
    paragraphs = audio.ScriptWriter(FakeCLI()).write(digest.entries[1], digest)
    assert paragraphs[0] == "The hook. The introduction." and paragraphs[-1] == "The wrap_up."
    assert len(paragraphs) == len(audio.PARTS) - 1


def test_make_audio_writes_one_mp3_once(store, tmp_path):
    digest = _digest()
    digest_id = store.save_digest(digest, None, None)
    speaker, writer, steps = FakeSpeaker(downloaded=False), FakeWriter(), []
    result = audio.make_audio(
        digest_id,
        store,
        speaker,
        tmp_path / "audio",
        writer=writer,
        log=lambda line: None,
        progress=lambda step, details: steps.append((step, details.get("done"))),
    )
    assert soundfile.info(result.path).format == "MP3"
    assert result.script_by == "Claude (test)" and not result.cached and not result.notes
    assert writer.calls == 3  # one segment per paper
    assert speaker.spoken == [audio.episode_script(digest, FakeWriter.segments(digest))]
    assert [s for s in steps if s[0] == "script"] == [("script", n) for n in range(4)]
    assert ("voice", None) in steps and steps[-1] == ("save", None)
    assert not list((tmp_path / "audio").glob("*.part"))

    again = audio.make_audio(digest_id, store, speaker, tmp_path / "audio", writer=writer)
    assert again.cached and again.path == result.path
    assert len(speaker.spoken) == 1

    result.path.unlink()  # remade from the cached segments: Claude isn't asked again
    audio.make_audio(digest_id, store, speaker, tmp_path / "audio", writer=writer)
    assert writer.calls == 3 and len(speaker.spoken) == 2


def test_a_failed_segment_is_read_as_written(store, tmp_path):
    digest = _digest()
    digest_id = store.save_digest(digest, None, None)
    speaker = FakeSpeaker()
    writer = FakeWriter(error=ClaudeError("timed out"), fail={"2611.00003"})
    result = audio.make_audio(digest_id, store, speaker, tmp_path, writer=writer, log=print)
    assert result.notes == [
        "Claude couldn't write one segment (timed out); that paper is read as written."
    ]
    assert result.script_by == "Claude (test)"
    segments = FakeWriter.segments(digest)
    segments["2611.00003"] = audio.plain_segment(digest.entries[2])
    assert speaker.spoken[0] == audio.episode_script(digest, segments)


def test_without_a_claude_sign_in_every_paper_is_read_as_written(store, tmp_path):
    digest = _digest()
    digest_id = store.save_digest(digest, None, None)
    speaker = FakeSpeaker()
    writer = FakeWriter(error=ClaudeNotSignedIn("no"))
    result = audio.make_audio(
        digest_id, store, speaker, tmp_path, writer=writer, log=print, workers=1
    )
    assert "isn't signed in to a Claude account" in result.notes[0] and len(result.notes) == 1
    assert result.script_by is None
    plain = {e.paper.id: audio.plain_segment(e) for e in digest.entries}
    assert speaker.spoken[0] == audio.episode_script(digest, plain)

    no_claude = audio.make_audio(digest_id, store, FakeSpeaker(), tmp_path / "plain", log=print)
    assert no_claude.script_by is None and not no_claude.notes


def test_empty_or_missing_digests_have_nothing_to_read(store, tmp_path):
    empty = store.save_digest(_digest(entries=[]), None, None)
    with pytest.raises(audio.AudioError, match="no papers"):
        audio.make_audio(empty, store, FakeSpeaker(), tmp_path)
    with pytest.raises(audio.AudioError, match="no digest #99"):
        audio.make_audio(99, store, FakeSpeaker(), tmp_path)


def test_old_audio_is_pruned(tmp_path):
    for number in range(5):
        (tmp_path / f"digest-{number}-x.mp3").write_bytes(b"x")
    audio.prune(tmp_path, keep=2)
    assert len(list(tmp_path.glob("digest-*.mp3"))) == 2


def test_voice_download_goes_to_a_part_file_first(tmp_path):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=b"model" if request.url.path.endswith("x") else b"{}")

    speaker = audio.PiperSpeaker(
        "en_US-lessac-medium", tmp_path, transport=httpx.MockTransport(handler)
    )
    assert not speaker.is_downloaded()
    logged = []
    speaker.prepare(logged.append)
    assert speaker.is_downloaded()
    assert requested[1] == (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    )
    assert "Downloading the voice en_US-lessac-medium" in logged[0]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "en_US-lessac-medium.onnx",
        "en_US-lessac-medium.onnx.json",
    ]


def test_unknown_or_failed_voice_downloads_leave_nothing_behind(tmp_path):
    missing = audio.PiperSpeaker(
        "en_US-nobody-medium",
        tmp_path,
        transport=httpx.MockTransport(lambda request: httpx.Response(404)),
    )
    with pytest.raises(audio.AudioError, match="no Piper voice called en_US-nobody-medium"):
        missing.prepare(lambda m: None)

    def broken(request):
        raise httpx.ConnectError("offline", request=request)

    offline = audio.PiperSpeaker(
        "en_US-lessac-medium", tmp_path, transport=httpx.MockTransport(broken)
    )
    with pytest.raises(audio.AudioError, match="couldn't download the voice"):
        offline.prepare(lambda m: None)
    assert list(tmp_path.iterdir()) == []


def test_audio_voice_setting(tmp_path):
    interests = [cfg.Interest(name="A", keywords=["x"])]
    text = cfg.render_template(["cs.LG"], str(tmp_path), interests)
    assert cfg.parse(text, tmp_path / "c.toml").audio_voice == cfg.AUDIO_VOICE
    voice = 'audio_voice = "en_GB-alba-medium"\n'
    assert cfg.parse(voice + text, tmp_path / "c.toml").audio_voice == "en_GB-alba-medium"
    with pytest.raises(cfg.ConfigError, match="Piper voice name"):
        cfg.parse('audio_voice = "alba"\n' + text, tmp_path / "c.toml")


def test_audio_command_saves_the_newest_digest(tmp_path, monkeypatch, make_config):
    monkeypatch.setenv("ARXIV_DIGEST_HOME", str(tmp_path / "home"))
    conf = make_config()
    conf.path.write_text(cfg.render_template(conf.categories, str(conf.output_dir), conf.interests))
    store = Store(paths.database_path())
    store.save_digest(_digest(kind="weird"), None, None)
    newest = store.save_digest(_digest(), None, None)
    store.close()
    calls = []

    def fake_make(config, digest_id, *, use_claude, log):
        calls.append((digest_id, use_claude))
        cached = tmp_path / "cached.mp3"
        cached.write_bytes(b"mp3")
        return audio.AudioResult(cached, _digest(), seconds=120, notes=["A note"])

    monkeypatch.setattr(audio, "make_audio_with_config", fake_make)
    result = CliRunner().invoke(cli.app, ["audio", "--config", str(conf.path), "--no-claude"])
    assert result.exit_code == 0, result.output
    assert calls == [(newest, False)]
    assert (conf.output_dir / "2026-11-03.mp3").read_bytes() == b"mp3"
    assert "A note" in result.output and "(2 min)" in result.output


def test_voices_with_many_speakers(tmp_path, monkeypatch):
    assert audio.split_voice("en_GB-vctk-medium#p374") == ("en_GB-vctk-medium", "p374")
    assert audio.split_voice("en_GB-alan-medium") == ("en_GB-alan-medium", None)
    assert audio.voice_urls("en_GB-vctk-medium#p374")[".onnx"].endswith(
        "en/en_GB/vctk/medium/en_GB-vctk-medium.onnx"
    )
    digest = _digest()
    name = audio.audio_file(tmp_path, 1, digest, "en_GB-vctk-medium#p374").name
    assert "#" not in name and "en_GB-vctk-medium-p374" in name

    class Chunk:
        audio_int16_array = np.ones(100, dtype=np.int16)

    class FakeVoice:
        class config:  # noqa: N801 - mirrors Piper's attribute
            sample_rate = 16000
            speaker_id_map = {"p326": 0, "p374": 7}

        def __init__(self):
            self.speaker_ids = []

        def synthesize(self, text, syn_config=None):
            self.speaker_ids.append(syn_config.speaker_id if syn_config else None)
            return [Chunk()]

    voice = FakeVoice()
    monkeypatch.setattr("piper.PiperVoice.load", lambda *args, **kwargs: voice)
    audio.PiperSpeaker("en_GB-vctk-medium#p374", tmp_path).speak(["Hi."], lambda d, t: None)
    assert voice.speaker_ids == [7]
    with pytest.raises(audio.AudioError, match="has no speaker 'p999'.*p326, p374"):
        audio.PiperSpeaker("en_GB-vctk-medium#p999", tmp_path).speak(["Hi."], lambda d, t: None)


def test_every_choice_in_the_picker_is_a_valid_voice(tmp_path):
    interests = [cfg.Interest(name="A", keywords=["x"])]
    text = cfg.render_template(["cs.LG"], str(tmp_path), interests)
    assert set(audio.VOICE_CHOICES) == {
        (accent, gender)
        for accent in ("British", "American", "Australian")
        for gender in ("male", "female")
    }
    for voice in audio.VOICE_CHOICES.values():
        line = f'audio_voice = "{voice}"\n'
        assert cfg.parse(line + text, tmp_path / "c.toml").audio_voice == voice
    with pytest.raises(cfg.ConfigError, match="Piper voice name"):
        cfg.parse('audio_voice = "en_GB-vctk-medium#p 1"\n' + text, tmp_path / "c.toml")


def test_previews_are_made_once(tmp_path):
    speaker = FakeSpeaker()
    path = audio.voice_preview("en_GB-vctk-medium#p374", tmp_path, speaker=speaker)
    assert path.name == "en_GB-vctk-medium-p374.mp3" and soundfile.info(path).format == "MP3"
    assert speaker.spoken == [[audio.PREVIEW_TEXT]]
    assert audio.voice_preview("en_GB-vctk-medium#p374", tmp_path, speaker=speaker) == path
    assert len(speaker.spoken) == 1
