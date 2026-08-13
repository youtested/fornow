"""Edit tool replacer cascade tests: near-miss oldStrings get salvaged."""

from opencode_py.tools.edit import _edit


SAMPLE = '''def hello():
    print("Hello, World!")

def world():
    return 42
'''


def test_exact_match_still_works(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), '    print("Hello, World!")', "    print('hi')")
    assert r.get("error") is None
    assert "    print('hi')" in p.read_text()


def test_trailing_space_in_old_string_salvaged(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), '    print("Hello, World!") ', "    print('hi')")
    assert r.get("error") is None
    assert "    print('hi')" in p.read_text()


def test_wrong_indentation_salvaged(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), '      print("Hello, World!")', "      print('hi')")
    assert r.get("error") is None
    assert "      print('hi')" in p.read_text()


def test_crlf_file_salvaged(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE.replace("\n", "\r\n"))
    r = _edit(str(p), "def world():", "def world():  # v2")
    assert r.get("error") is None
    assert "def world():  # v2" in p.read_text()


def test_blank_line_context_removed(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), '    print("Hello, World!")\n\ndef world():', '    print("Hello, World!")\ndef world():')
    assert r.get("error") is None
    assert "\n\n" not in p.read_text()


def test_ambiguous_fallback_still_errors(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), "def ", "fn ")
    assert r.get("error") is True
    assert "multiple matches" in r["output"].lower()


def test_unmatched_still_errors(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(SAMPLE)
    r = _edit(str(p), "totally not in the file", "x")
    assert r.get("error") is True
    assert "not found" in r["output"]


def test_fallback_replace_all(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("  a\n   a\n    a\n")
    r = _edit(str(p), " a", "b", True)
    assert r.get("error") is None
    assert p.read_text() == " b\n  b\n   b\n"