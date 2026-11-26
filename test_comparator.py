import pytest

import comparator


def test_normalize_mapping():
    data = {"alpha": {"requirements": ["one"]}}

    assert comparator.normalize_entries(data) == [
        {"name": "alpha", "requirements": ["one"]}
    ]


def test_rejects_non_string_name():
    with pytest.raises(ValueError, match="string name"):
        comparator.normalize_entries([{"name": 7}])


def test_comparison_outputs(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    names = tmp_path / "names.txt"
    requirements = tmp_path / "requirements.txt"

    first.write_text("- name: alpha\n  requirements: [old]\n", encoding="utf-8")
    second.write_text("- name: beta\n  requirements: [new]\n", encoding="utf-8")

    comparator.main(first, second, names, requirements)

    assert names.read_text(encoding="utf-8") == (
        f"Element names only in {first}:\n- alpha\n"
        f"\nElement names only in {second}:\n- beta\n"
    )
    assert requirements.read_text(encoding="utf-8") == (
        f"Requirements only in {first}:\n- alpha,old\n"
        f"\nRequirements only in {second}:\n- beta,new\n"
    )


def test_identical_files(tmp_path):
    source = tmp_path / "source.yaml"
    names = tmp_path / "names.txt"
    requirements = tmp_path / "requirements.txt"
    source.write_text("- name: alpha\n  requirements: [one]\n", encoding="utf-8")

    comparator.main(source, source, names, requirements)

    assert names.read_text(encoding="utf-8") == (
        "NO DIFFERENCES IN REGARDS TO ELEMENT NAMES\n"
    )
    assert requirements.read_text(encoding="utf-8") == (
        "NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS\n"
    )