import json

import pytest

from blt_visualizer.patching import (
    MAX_PROMPT_BYTES,
    analyze_prompt,
    next_byte_entropies,
)


def test_entropy_patches_reconstruct_utf8_prompt():
    prompt = "hello café ☕"
    result = analyze_prompt(prompt, strategy="entropy")

    rebuilt = bytes(
        byte.value
        for patch in result.patches
        for byte in result.bytes[patch.start : patch.end]
    )
    assert rebuilt == prompt.encode("utf-8")
    assert result.byte_count > len(prompt)
    assert 1 <= result.patch_count <= result.byte_count


def test_entropy_patches_respect_size_constraints_except_final_patch():
    result = analyze_prompt(
        "byte models allocate compute at surprising transitions.",
        strategy="entropy",
        threshold_percentile=0.70,
        min_patch_size=2,
        max_patch_size=8,
    )

    assert all(2 <= patch.length <= 8 for patch in result.patches[:-1])
    assert 1 <= result.patches[-1].length <= 8
    assert any(
        "entropy" in patch.boundary_reason
        for patch in result.patches[1:]
    )


def test_fixed_patching_has_expected_boundaries():
    result = analyze_prompt("abcdefghij", strategy="fixed", fixed_patch_size=4)

    assert [patch.length for patch in result.patches] == [4, 4, 2]
    assert [patch.start for patch in result.patches] == [0, 4, 8]


def test_whitespace_patching_ends_after_spaces():
    result = analyze_prompt("one two three", strategy="whitespace")

    assert [patch.text for patch in result.patches] == ["one␠", "two␠", "three"]


def test_analysis_is_json_serializable():
    result = analyze_prompt("BLT", strategy="entropy")

    json.dumps(result.to_dict())


def test_empty_and_oversized_prompts_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        analyze_prompt("")
    with pytest.raises(ValueError, match="at or below"):
        analyze_prompt("x" * (MAX_PROMPT_BYTES + 1))


def test_entropies_align_one_to_one_with_bytes():
    data = "☕".encode("utf-8")
    entropies = next_byte_entropies(data)

    assert len(entropies) == len(data)
    assert all(0 <= value <= 8 for value in entropies)

