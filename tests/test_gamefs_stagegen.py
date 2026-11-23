from stagegen.extract import iostore


def test_gamefs_stagegen_containers_are_explicit_opt_in(monkeypatch):
    seen = []

    class FakeContainer:
        def __init__(self, path):
            seen.append(path)
            self.files = {}
            self.chunk_index = {}

    def fake_glob(pattern):
        # os.path.join() uses backslashes on Windows and forward slashes on
        # POSIX. Normalize only the fixture's pattern matching so this test
        # exercises the same GameFS behaviour on both platforms.
        pattern = pattern.replace("\\", "/")
        if pattern.endswith("~mods/*.utoc"):
            return [
                r"C:\\Game\\Paks\\~mods\\third-party.utoc",
                r"C:\\Game\\Paks\\~mods\\pakchunk99-Test_Stagegen_9_P.utoc",
            ]
        if pattern.endswith("*.utoc"):
            return [r"C:\\Game\\Paks\\pakchunk0-Windows.utoc"]
        return []

    monkeypatch.setattr(iostore.glob, "glob", fake_glob)
    monkeypatch.setattr(iostore, "Container", FakeContainer)

    iostore.GameFS(r"C:\\Game\\Paks")
    assert len(seen) == 1
    assert "Stagegen" not in seen[0]

    seen.clear()
    iostore.GameFS(r"C:\\Game\\Paks", include_mods=True)
    assert len(seen) == 2
    assert all("Stagegen" not in p for p in seen)

    seen.clear()
    iostore.GameFS(r"C:\\Game\\Paks", include_stagegen=True)
    assert len(seen) == 2
    assert any("Stagegen" in p for p in seen)
    assert all("third-party" not in p for p in seen)
