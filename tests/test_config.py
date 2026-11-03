from pathlib import Path

import pytest

from arxiv_digest import config as cfg

GOOD = """
categories = ["astro-ph.co", "cs.LG", "cs.LG"]
output_dir = "digests"

[[interest]]
name = "Fast simulations"
description = "Neural emulators for cosmology"
keywords = ["emulator"]
exclude = "quantum"

[[interest]]
name = "People"
authors = ["Vera Rubin"]
"""


def test_parse_valid_config(tmp_path):
    conf = cfg.parse(GOOD, tmp_path / "config.toml")
    assert conf.categories == ["astro-ph.CO", "cs.LG"]  # canonical spelling, no duplicates
    assert conf.output_dir == tmp_path / "digests"  # relative to the config file
    assert [i.name for i in conf.interests] == ["Fast simulations", "People"]
    assert conf.interests[0].exclude == ["quantum"]  # a single string is accepted
    assert conf.max_papers == 15 and conf.top_picks == 3
    assert conf.interests[0].query_text == (
        "Fast simulations. Neural emulators for cosmology. Keywords: emulator"
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('categories = ["astro-ph.CM"]\n[[interest]]\nname="x"\nkeywords=["y"]', "Did you mean"),
        ('categories = []\n[[interest]]\nname="x"\nkeywords=["y"]', "empty"),
        ('categories = ["cs.LG"]', "No interests"),
        ('categories = ["cs.LG"]\n[[interest]]\nname="x"', "needs a `description`"),
        ('categories = ["cs.LG"]\ncolour = 1\n[[interest]]\nname="x"\nkeywords=["y"]', "colour"),
        ('categories = ["cs.LG"]\n[[interest]]\nname="x"\nkeyword=["y"]', "keyword"),
        (
            'categories = ["cs.LG"]\nmax_papers = 0\n[[interest]]\nname="x"\nkeywords=["y"]',
            "between",
        ),
        (
            'categories = ["cs.LG"]\n[[interest]]\nname="A"\nkeywords=["y"]'
            '\n[[interest]]\nname="a"\nkeywords=["z"]',
            "names must differ",
        ),
        ("categories = [", "not valid TOML"),
    ],
)
def test_parse_errors_explain_the_problem(tmp_path, text, message):
    with pytest.raises(cfg.ConfigError, match=message):
        cfg.parse(text, tmp_path / "config.toml")


def test_unknown_category_allowed_when_opted_in(tmp_path):
    text = 'allow_unknown_categories = true\ncategories = ["new.XY"]\n'
    text += '[[interest]]\nname="x"\nkeywords=["y"]'
    assert cfg.parse(text, tmp_path / "c.toml").categories == ["new.XY"]


def test_template_round_trips(tmp_path):
    interests = [
        cfg.Interest(name='Quotes "and" ü', description="d", keywords=["a", "b"]),
        cfg.Interest(name="People", authors=["Vera Rubin"]),
    ]
    text = cfg.render_template(["cs.LG"], "~/arxiv-digest", interests)
    conf = cfg.parse(text, tmp_path / "config.toml")
    assert conf.interests[0].name == 'Quotes "and" ü'
    assert conf.output_dir == Path("~/arxiv-digest").expanduser()


def test_add_and_remove_categories_keep_comments(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("# my comment\n" + GOOD)
    conf, added = cfg.add_categories(path, ["hep-th", "cs.lg", "hep-th"])
    assert added == ["hep-th"]
    assert conf.categories == ["astro-ph.CO", "cs.LG", "hep-th"]
    assert "# my comment" in path.read_text()
    conf, removed = cfg.remove_categories(path, ["cs.LG"])
    assert removed == ["cs.LG"]
    assert conf.categories == ["astro-ph.CO", "hep-th"]
    with pytest.raises(cfg.ConfigError, match="Did you mean"):
        cfg.add_categories(path, ["hep-tx"])
    with pytest.raises(cfg.ConfigError, match="at least one"):
        cfg.remove_categories(path, ["astro-ph.CO", "hep-th"])
