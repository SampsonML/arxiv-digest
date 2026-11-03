from pathlib import Path

from arxiv_digest import paths


def _project(tmp_path: Path, name: str = "arxiv-digest") -> Path:
    project = tmp_path / "proj"
    (project / "src" / "arxiv_digest").mkdir(parents=True)
    (project / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0"\n')
    return project


def test_source_checkout_detects_a_clone(tmp_path):
    project = _project(tmp_path)
    assert paths.source_checkout(project / "src" / "arxiv_digest") == project.resolve()


def test_source_checkout_ignores_installed_copies_and_other_projects(tmp_path):
    installed = tmp_path / "venv" / "lib" / "site-packages" / "arxiv_digest"
    installed.mkdir(parents=True)
    assert paths.source_checkout(installed) is None
    other = _project(tmp_path / "x", name="something-else")
    assert paths.source_checkout(other / "src" / "arxiv_digest") is None


def test_display_uses_tilde_for_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert paths.display(tmp_path / "Projects" / "arxiv-digest" / "digests") == (
        "~/Projects/arxiv-digest/digests"
    )
    assert paths.display(Path("/srv/digests")) == "/srv/digests"


def test_default_output_dir(monkeypatch, tmp_path):
    project = tmp_path / "proj"
    monkeypatch.setattr(paths, "source_checkout", lambda: project)
    assert paths.default_output_dir() == str(project / "digests")
    monkeypatch.setattr(paths, "source_checkout", lambda: None)
    assert paths.default_output_dir() == "~/arxiv-digest"
