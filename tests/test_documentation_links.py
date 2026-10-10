"""A local evidence file must not conceal a missing GitHub link target."""
from scripts.check_documentation import broken_links


def test_tracks_files_directories_and_encoded_paths(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a b.md").write_text("# A\n")
    index = tmp_path / "README.md"
    index.write_text("[guide](docs/a%20b.md#any-anchor) [folder](docs/)\n"
                     "[external](https://example.com/no-network) [local](#heading)\n")
    assert broken_links(tmp_path, [index], {"docs/a b.md"}) == []


def test_ignored_existing_and_missing_targets_both_fail(tmp_path):
    (tmp_path / "local.json").write_text("{}")
    index = tmp_path / "README.md"
    index.write_text("[local](local.json)\n[missing](absent.md)\n")
    errors = broken_links(tmp_path, [index], set())
    assert len(errors) == 2
    assert "README.md:1: missing tracked target: local.json" in errors
    assert "README.md:2: missing tracked target: absent.md" in errors


def test_examples_in_fences_are_not_links_and_outside_paths_fail(tmp_path):
    index = tmp_path / "README.md"
    index.write_text("```markdown\n[example](invented.md)\n```\n[outside](../other.md)\n")
    errors = broken_links(tmp_path, [index], set())
    assert len(errors) == 1
    assert "README.md:4: outside repository" in errors[0]
