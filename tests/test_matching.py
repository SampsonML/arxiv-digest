from arxiv_digest.matching import author_key, find_terms, match_authors, normalize


def test_normalize_strips_accents_case_and_punctuation():
    assert normalize("Gödel's  N-Body\tsimulation!") == "godel s n body simulation"


def test_terms_match_whole_words_with_plurals_and_hyphen_variants():
    text = normalize("We train neural emulators for N body simulations of lensing.")
    assert find_terms(text, ["emulator", "N-body", "simulation", "lens", "neural network"]) == [
        "emulator",
        "N-body",
        "simulation",
    ]


def test_terms_do_not_match_inside_words():
    assert find_terms(normalize("Galaxy lensing"), ["lens", "axy"]) == []


def test_author_key_formats():
    assert author_key("David W. Hogg") == ("hogg", "d")
    assert author_key("Hogg, David") == ("hogg", "d")
    assert author_key("D. Hogg Jr.") == ("hogg", "d")
    assert author_key("Plato") == ("plato", "")
    assert author_key("  ") is None


def test_match_authors_uses_surname_and_initial():
    paper = ["D. W. Hogg", "José Núñez", "Ana Rubin"]
    assert match_authors(paper, ["David Hogg", "Jose Nunez", "Vera Rubin", "Smith"]) == [
        "David Hogg",
        "Jose Nunez",
    ]


def test_match_authors_surname_only_when_no_first_name():
    assert match_authors(["Ana Rubin"], ["Rubin"]) == ["Rubin"]
