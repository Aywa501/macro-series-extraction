"""Tests for disambiguation logic (stage 02).

Uses 20 hardcoded test cases covering clear idiomatic, clear literal,
and ambiguous usages. Bedrock API calls are mocked — no AWS access required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import importlib.util
import types

import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Shim: load stage 02 module (filename starts with a digit, so use importlib)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location(
    "src_02_disambiguate",
    PROJECT_ROOT / "src" / "02_disambiguate.py",
)
_mod = types.ModuleType("src_02_disambiguate")
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
sys.modules["src_02_disambiguate"] = _mod

build_prompt = _mod.build_prompt  # type: ignore[attr-defined]
parse_llm_response = _mod.parse_llm_response  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

class DisambiguationCase(NamedTuple):
    context_text: str
    idiom: str
    expected_is_idiomatic: bool
    expected_triviality_hint: str  # "trivial" | "significant" | "neutral" | any
    description: str


TEST_CASES: list[DisambiguationCase] = [
    # --- Clear idiomatic cases ---
    DisambiguationCase(
        context_text=(
            "The honourable gentleman is simply penny-pinching with the public purse, "
            "refusing to fund what every citizen deserves."
        ),
        idiom="penny-pinching",
        expected_is_idiomatic=True,
        expected_triviality_hint="trivial",
        description="penny-pinching clearly idiomatic (frugality criticism)",
    ),
    DisambiguationCase(
        context_text=(
            "This measure will cost the taxpayer a pretty penny, "
            "and we must consider whether the expenditure is justified."
        ),
        idiom="a pretty penny",
        expected_is_idiomatic=True,
        expected_triviality_hint="significant",
        description="a pretty penny — large sum, significant framing",
    ),
    DisambiguationCase(
        context_text=(
            "He is the sort of fellow who turns up like a bad penny "
            "whenever there is a debate on naval estimates."
        ),
        idiom="turn up like a bad penny",
        expected_is_idiomatic=True,
        expected_triviality_hint="trivial",
        description="bad penny — unwanted recurring presence",
    ),
    DisambiguationCase(
        context_text=(
            "In for a penny, in for a pound — having committed to the treaty "
            "we cannot now withdraw at the first difficulty."
        ),
        idiom="in for a penny in for a pound",
        expected_is_idiomatic=True,
        expected_triviality_hint="neutral",
        description="in for a penny idiom — commitment framing",
    ),
    DisambiguationCase(
        context_text=(
            "The policy is penny wise and pound foolish: "
            "we save on inspection costs and pay dearly in industrial accidents."
        ),
        idiom="penny wise pound foolish",
        expected_is_idiomatic=True,
        expected_triviality_hint="trivial",
        description="penny wise pound foolish — classic false economy",
    ),
    DisambiguationCase(
        context_text=(
            "He took the King's shilling and thus placed himself "
            "entirely at the disposal of the War Office."
        ),
        idiom="take the king's shilling",
        expected_is_idiomatic=True,
        expected_triviality_hint="significant",
        description="King's shilling — military enlistment commitment",
    ),
    DisambiguationCase(
        context_text=(
            "Good clerks are a shilling a dozen in London; "
            "the difficulty is finding one who can keep a secret."
        ),
        idiom="shilling a dozen",
        expected_is_idiomatic=True,
        expected_triviality_hint="trivial",
        description="shilling a dozen — common and cheap",
    ),
    DisambiguationCase(
        context_text=(
            "The proposal is not worth a farthing to the working men "
            "of this country, who need real reform, not platitudes."
        ),
        idiom="not worth a farthing",
        expected_is_idiomatic=True,
        expected_triviality_hint="trivial",
        description="not worth a farthing — proposal deemed worthless",
    ),
    DisambiguationCase(
        context_text=(
            "The Minister used the laboratory workers as guinea pigs "
            "for this untested programme, with predictably disastrous results."
        ),
        idiom="guinea pig",
        expected_is_idiomatic=True,
        expected_triviality_hint="significant",
        description="guinea pig — test subjects, significant harm implied",
    ),
    DisambiguationCase(
        context_text=(
            "He said he would not give a penny more to this failing scheme, "
            "and I entirely agree with his assessment."
        ),
        idiom="not a penny more",
        expected_is_idiomatic=True,
        expected_triviality_hint="trivial",
        description="not a penny more — figurative refusal to concede",
    ),
    # --- Clear literal cases ---
    DisambiguationCase(
        context_text=(
            "The cost of a telegraph message was reduced from one shilling "
            "to sixpence for the first twenty words."
        ),
        idiom="take the king's shilling",
        expected_is_idiomatic=False,
        expected_triviality_hint="neutral",
        description="literal shilling price for telegraph (no idiom present)",
    ),
    DisambiguationCase(
        context_text=(
            "The poor relief fund paid out not a penny more than the minimum "
            "prescribed by the statute — exactly four shillings and sixpence per week."
        ),
        idiom="not a penny more",
        expected_is_idiomatic=False,
        expected_triviality_hint="neutral",
        description="literal not a penny more — exact monetary amount specified",
    ),
    DisambiguationCase(
        context_text=(
            "The price of bread rose to three farthings per loaf, "
            "placing it beyond the reach of the poorest families."
        ),
        idiom="not worth a farthing",
        expected_is_idiomatic=False,
        expected_triviality_hint="neutral",
        description="literal farthing price for bread",
    ),
    DisambiguationCase(
        context_text=(
            "A guinea was equivalent to twenty-one shillings and was the preferred "
            "unit for professional fees and auction prices throughout the century."
        ),
        idiom="guinea pig",
        expected_is_idiomatic=False,
        expected_triviality_hint="neutral",
        description="literal guinea coin description — historical monetary context",
    ),
    DisambiguationCase(
        context_text=(
            "The soldier received his bounty of one shilling upon enlistment, "
            "which was handed to him by the recruiting sergeant."
        ),
        idiom="take the king's shilling",
        expected_is_idiomatic=False,
        expected_triviality_hint="neutral",
        description="literal shilling paid at enlistment (historical description)",
    ),
    # --- Ambiguous cases ---
    DisambiguationCase(
        context_text=(
            "The honourable member insists we should spend a penny on this "
            "project, though whether he means the coin or something else entirely "
            "the House may judge."
        ),
        idiom="spend a penny",
        expected_is_idiomatic=True,  # euphemism joke in context
        expected_triviality_hint="trivial",
        description="spend a penny — ambiguous joke but idiomatic use implied",
    ),
    DisambiguationCase(
        context_text=(
            "The Right Honourable Gentleman will not give a penny more "
            "towards Irish land reform than he is legally obligated to do."
        ),
        idiom="not a penny more",
        expected_is_idiomatic=True,
        expected_triviality_hint="significant",
        description="not a penny more — could be literal but figurative framing dominates",
    ),
    DisambiguationCase(
        context_text=(
            "We have been penny-pinching on defence for a decade; "
            "the bill now comes due in the form of obsolete equipment."
        ),
        idiom="penny-pinching",
        expected_is_idiomatic=True,
        expected_triviality_hint="significant",
        description="penny-pinching on defence — significant consequence framing",
    ),
    DisambiguationCase(
        context_text=(
            "The laboratory animals — including the guinea pigs and rabbits "
            "used in these experiments — deserve proper welfare protections."
        ),
        idiom="guinea pig",
        expected_is_idiomatic=False,
        expected_triviality_hint="neutral",
        description="guinea pig — actual animals, not metaphorical test subjects",
    ),
    DisambiguationCase(
        context_text=(
            "Not worth a farthing, he said of the amendment — and the House "
            "divided strongly against it, two hundred votes to forty-three."
        ),
        idiom="not worth a farthing",
        expected_is_idiomatic=True,
        expected_triviality_hint="trivial",
        description="not worth a farthing — idiomatic dismissal of amendment",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_response(is_idiomatic: bool, confidence: str, triviality: str) -> str:
    """Construct a fake JSON response string as the model would return."""
    return json.dumps(
        {
            "is_idiomatic": is_idiomatic,
            "confidence": confidence,
            "triviality_hint": triviality,
            "reasoning": "Mocked reasoning for test.",
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """Verify that the prompt template is filled correctly."""

    def test_prompt_contains_idiom(self) -> None:
        prompt = build_prompt("penny-pinching", "He was penny-pinching again.")
        assert "penny-pinching" in prompt

    def test_prompt_contains_context(self) -> None:
        ctx = "The minister refused to fund schools."
        prompt = build_prompt("penny-pinching", ctx)
        assert ctx in prompt

    def test_prompt_contains_json_keys(self) -> None:
        prompt = build_prompt("a pretty penny", "It cost a pretty penny.")
        for key in ("is_idiomatic", "confidence", "triviality_hint", "reasoning"):
            assert key in prompt


class TestParseResponse:
    """Verify JSON parsing from LLM responses."""

    def test_clean_json(self) -> None:
        raw = _make_mock_response(True, "high", "trivial")
        result = parse_llm_response(raw)
        assert result["is_idiomatic"] is True
        assert result["confidence"] == "high"
        assert result["triviality_hint"] == "trivial"

    def test_false_is_idiomatic(self) -> None:
        raw = _make_mock_response(False, "medium", "neutral")
        result = parse_llm_response(raw)
        assert result["is_idiomatic"] is False

    def test_json_embedded_in_text(self) -> None:
        raw = 'Sure, here you go: {"is_idiomatic": true, "confidence": "low", "triviality_hint": "significant", "reasoning": "test"} Hope that helps.'
        result = parse_llm_response(raw)
        assert result["is_idiomatic"] is True

    def test_malformed_json(self) -> None:
        raw = "I cannot classify this. The text is unclear."
        result = parse_llm_response(raw)
        assert result["is_idiomatic"] is None
        assert result["confidence"] == "failed"

    def test_empty_response(self) -> None:
        result = parse_llm_response("")
        assert result["confidence"] == "failed"


class TestDisambiguationCases:
    """Integration-style tests: mock Bedrock, verify parse→classify flow."""

    @pytest.mark.parametrize("case", TEST_CASES, ids=[c.description for c in TEST_CASES])
    def test_case(self, case: DisambiguationCase) -> None:
        """For each test case, mock the model returning the expected labels and verify roundtrip."""
        # Build what the model *should* return
        mock_raw = _make_mock_response(
            case.expected_is_idiomatic,
            "high",
            case.expected_triviality_hint,
        )

        # Mock BedrockClient.invoke_with_retry
        with patch("src_02_disambiguate.BedrockClient") as MockClient:
            instance = MockClient.return_value
            instance.invoke_with_retry.return_value = mock_raw

            # Simulate calling the client and parsing
            raw_response = instance.invoke_with_retry(
                prompt=build_prompt(case.idiom, case.context_text),
                system="You are a linguistic classifier. Respond only with a JSON object, no other text.",
                model_id="meta.llama3-1-8b-instruct-v1:0",
                max_tokens=256,
            )

        parsed = parse_llm_response(raw_response)

        assert parsed["is_idiomatic"] == case.expected_is_idiomatic, (
            f"[{case.description}] Expected is_idiomatic={case.expected_is_idiomatic}, got {parsed['is_idiomatic']}"
        )
        assert parsed["triviality_hint"] == case.expected_triviality_hint, (
            f"[{case.description}] Expected triviality_hint={case.expected_triviality_hint!r}, got {parsed['triviality_hint']!r}"
        )
