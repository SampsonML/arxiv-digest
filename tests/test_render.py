from conftest import make_paper

from arxiv_digest.render import _authors, _safe
from arxiv_digest.shortlist import Candidate, Match


def test_safe_escapes_tag_like_text_but_not_math():
    assert _safe("Fewer <b>than</b> stars at $z<b$") == "Fewer &lt;b>than&lt;/b> stars at $z<b$"
    assert _safe("A 3<5 comparison") == "A 3<5 comparison"


def test_authors_are_truncated_but_followed_ones_always_shown():
    names = tuple(f"Person {chr(65 + i)}" for i in range(10)) + ("Vera Rubin",)
    match = Match("People", 1.0, None, (), ("Vera Rubin",))
    text = _authors(Candidate(make_paper("1", authors=names), (match,)))
    assert text.startswith("Person A, Person B")
    assert "**Vera Rubin** et al. (11 authors)" in text
    assert "Person J" not in text
