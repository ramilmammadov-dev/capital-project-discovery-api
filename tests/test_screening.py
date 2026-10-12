"""Screening: routing policy, heuristic screener and the Jev screener against a fake client."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capex.config import AppConfig, ScreeningConfig
from capex.costs import CostLedger
from capex.screening import build_screener
from capex.screening.base import recall_guard, route
from capex.screening.heuristic import HeuristicScreener, PassThroughScreener
from capex.screening.jev import JevScreener
from capex.storage.cache import FileCache

from .helpers import make_chunk, make_document, table_chunk

CFG = ScreeningConfig()
TEXT_CHUNK = make_chunk(
    "[[PDF page 2]]\nThe City reviews its policies every year and keeps reserves.", pages=(2,), seq=2, est_rows=0
)
GLOSSARY = make_chunk(
    "[[PDF page 1]]\nGlossary of Terms and Acronyms\nFTE - full-time equivalent\nGFOA - Government Finance Officers "
    "Association\nDefinitions of budget terms\nTable of contents",
    pages=(1,), seq=9, est_rows=0,
)
LOW = ({"explicit": 0.02, "possible": 0.03, "context_only": 0.1, "none": 0.85},
       {"extract": 0.02, "extract_with_context": 0.03, "skip": 0.95})


class TestRoute:
    def test_high_probability_is_extracted(self):
        presence = {"explicit": 0.8, "possible": 0.1, "context_only": 0.05, "none": 0.05}
        need = {"extract": 0.8, "extract_with_context": 0.1, "skip": 0.1}
        decision, reason, guard, p = route(presence, need, "project_listing", TEXT_CHUNK, CFG)
        assert (decision, guard) == ("extract", False) and p == pytest.approx(0.9)
        assert "p_project=0.90" in reason

    def test_context_need_selects_extract_with_context(self):
        presence = {"explicit": 0.7, "possible": 0.1}
        need = {"extract": 0.4, "extract_with_context": 0.5, "skip": 0.1}
        assert route(presence, need, "project_listing", TEXT_CHUNK, CFG)[0] == "extract_with_context"

    def test_plausible_probability_is_still_extracted(self):
        presence = {"explicit": 0.1, "possible": 0.1, "none": 0.8}
        need = {"extract": 0.1, "extract_with_context": 0.1, "skip": 0.8}
        decision, reason, guard, _p = route(presence, need, "administrative", TEXT_CHUNK, CFG)
        assert decision == "extract" and guard is False and "plausible" in reason

    def test_a_screener_that_does_not_say_skip_never_skips(self):
        presence = {"explicit": 0.02, "possible": 0.03, "none": 0.95}
        need = {"extract": 0.3, "extract_with_context": 0.3, "skip": 0.4}
        assert route(presence, need, "administrative", TEXT_CHUNK, CFG)[0] != "skip"

    def test_low_probability_with_the_recall_guard_is_extracted_with_guard_flag(self):
        decision, reason, guard, _ = route(*LOW, "administrative", table_chunk(8), CFG)
        assert (decision, guard) == ("extract", True) and "recall guard" in reason

    def test_low_probability_without_guard_is_skipped(self):
        decision, _reason, guard, p = route(*LOW, "administrative", TEXT_CHUNK, CFG)
        assert (decision, guard) == ("skip", False) and p == pytest.approx(0.05)

    def test_guard_can_be_disabled(self):
        config = ScreeningConfig(recall_guard=False)
        assert route(*LOW, "administrative", table_chunk(8), config)[0] == "skip"

    def test_recall_guard_signals(self):
        assert recall_guard(table_chunk(8)) is True
        assert recall_guard(TEXT_CHUNK) is False
        ids_and_money = make_chunk("[[PDF page 1]]\nST85100001 Harbor $1,000,000\nST85100002 Cedar $2,000,000")
        assert recall_guard(ids_and_money.model_copy(update={"est_rows": 0})) is True


class TestHeuristicScreener:
    async def test_project_table_is_extracted_and_glossary_is_skipped(self):
        screener = HeuristicScreener(CFG)
        table, glossary = await screener.screen([table_chunk(10), GLOSSARY], make_document(), "Org")
        assert table.decision == "extract" and table.presence == "explicit"
        assert table.content_type == "project_listing" and table.p_project > 0.9
        assert table.provider == "heuristic" and table.model == "heuristic-v1"
        assert glossary.decision == "skip" and glossary.content_type == "administrative"
        assert glossary.p_project < CFG.plausible_threshold and glossary.guard is False

    async def test_results_follow_input_order_and_chunk_ids(self):
        chunks = [table_chunk(6, doc_id=f"d{i}") for i in range(3)]
        results = await HeuristicScreener(CFG).screen(chunks, make_document(), "Org")
        assert [r.chunk_id for r in results] == [c.chunk_id for c in chunks]

    async def test_pass_through_extracts_everything(self):
        results = await PassThroughScreener().screen([GLOSSARY, TEXT_CHUNK], make_document(), "Org")
        assert [r.decision for r in results] == ["extract", "extract"]
        assert all(r.provider == "none" and r.p_project == 1.0 for r in results)


def jev_response(explicit=0.9, tokens=12_345, model="jev-1.13.0"):
    presence = {"explicit": explicit, "possible": 0.05, "context_only": 0.03, "none": max(0.0, 0.92 - explicit)}
    content = {"project_listing": 0.8, "administrative": 0.2}
    need = {"extract": 0.9, "extract_with_context": 0.05, "skip": 0.05}
    choice = lambda probs: SimpleNamespace(probabilities=probs)  # noqa: E731
    return SimpleNamespace(
        choices={"presence": choice(presence), "content_type": choice(content), "need": choice(need)},
        model=model, usage=SimpleNamespace(input_tokens=tokens, output_tokens=0),
    )


class FakeJevClient:
    def __init__(self, respond=None):
        self.respond = respond or (lambda state: jev_response())
        self.calls = []

    async def system_one(self, state, questions, model):
        self.calls.append({"state": state, "questions": questions, "model": model})
        return self.respond(state)


def make_jev(tmp_path, client, config=None):
    config = config or AppConfig()
    ledger = CostLedger(config)
    return JevScreener(config, "fake-key", ledger, FileCache(tmp_path / "cache"), client=client), ledger


class TestJevScreener:
    async def test_probabilities_are_routed_and_cost_is_recorded(self, tmp_path):
        client = FakeJevClient()
        screener, ledger = make_jev(tmp_path, client)
        [result] = await screener.screen([table_chunk(8)], make_document(), "Org")
        assert result.provider == "jev" and result.model == "jev-1.13.0"
        assert result.decision == "extract" and result.presence == "explicit"
        assert result.content_type == "project_listing" and result.need == "extract"
        assert result.p_project == pytest.approx(0.95) and result.cached is False and result.error is None
        assert len(client.calls) == 1
        assert client.calls[0]["model"] == "jev-1.13.0"
        assert set(client.calls[0]["questions"]) == {"presence", "content_type", "need"}
        assert "ST85100000" in client.calls[0]["state"]["text"]
        assert "FY2027-2031 CIP" in client.calls[0]["state"]["document"] and "Org" in client.calls[0]["state"]["document"]
        usage = ledger.records[0]
        assert (usage.stage, usage.task, usage.provider) == ("screening", "jev_screening", "jev")
        assert usage.input_tokens == 12_345 and usage.output_tokens == 0
        assert usage.cost_usd == pytest.approx(12_345 * 0.042 / 1e6, rel=1e-6)
        assert usage.input_price_per_mtok == pytest.approx(0.042) and usage.estimated is False

    async def test_second_call_is_served_from_cache(self, tmp_path):
        client = FakeJevClient()
        screener, ledger = make_jev(tmp_path, client)
        chunk = table_chunk(8)
        await screener.screen([chunk], make_document(), "Org")
        [again] = await screener.screen([chunk], make_document(), "Org")
        assert len(client.calls) == 1 and len(ledger.records) == 1
        assert again.cached is True and again.decision == "extract"
        assert ledger.cache_hits["screening"] == 1

    async def test_cache_is_shared_but_routing_is_recomputed_with_current_thresholds(self, tmp_path):
        client = FakeJevClient(lambda state: jev_response(explicit=0.3))
        screener, _ = make_jev(tmp_path, client)
        chunk = TEXT_CHUNK
        [first] = await screener.screen([chunk], make_document(), "Org")
        strict = AppConfig()
        strict.screening.extract_threshold = 0.99
        strict.screening.plausible_threshold = 0.98
        other, _ = make_jev(tmp_path, client, strict)
        [second] = await other.screen([chunk], make_document(), "Org")
        assert len(client.calls) == 1 and second.cached is True
        assert first.decision == "extract"
        assert second.decision in ("extract", "skip")  # new thresholds applied to cached probabilities
        assert "p_project=0.35" in first.reason

    async def test_client_exception_falls_back_to_heuristic_with_error_set(self, tmp_path):
        def boom(state):
            raise RuntimeError("jev is down")

        client = FakeJevClient(boom)
        screener, ledger = make_jev(tmp_path, client)
        results = await screener.screen([table_chunk(8), GLOSSARY], make_document(), "Org")
        assert [r.provider for r in results] == ["heuristic", "heuristic"]
        assert all(r.error and r.error.startswith("jev error: RuntimeError: jev is down") for r in results)
        assert all("heuristic fallback after Jev error" in r.reason for r in results)
        assert [r.decision for r in results] == ["extract", "skip"]  # recall is not lost
        assert len(ledger.records) == 2 and not any(rec.success for rec in ledger.records)
        assert ledger.records[0].error.startswith("RuntimeError")

    async def test_only_failing_chunks_fall_back(self, tmp_path):
        def respond(state):
            if "Glossary" in state["text"]:
                raise ValueError("bad chunk")
            return jev_response()

        screener, _ = make_jev(tmp_path, FakeJevClient(respond))
        good, bad = await screener.screen([table_chunk(8), GLOSSARY], make_document(), "Org")
        assert good.provider == "jev" and good.error is None
        assert bad.provider == "heuristic" and "ValueError" in (bad.error or "")


class TestBuildScreener:
    def test_jev_without_key_falls_back_to_heuristic_with_a_warning(self, tmp_path):
        config = AppConfig()
        screener, warnings = build_screener(config, CostLedger(config), FileCache(tmp_path / "c"), "jev")
        assert isinstance(screener, HeuristicScreener)
        assert warnings and "TYPESAFE_API_KEY" in warnings[0]

    def test_modes(self, tmp_path):
        config = AppConfig()
        args = (config, CostLedger(config), FileCache(tmp_path / "c"))
        assert isinstance(build_screener(*args, "heuristic")[0], HeuristicScreener)
        assert isinstance(build_screener(*args, "none")[0], PassThroughScreener)
        config.screening.provider = "heuristic"
        assert isinstance(build_screener(*args)[0], HeuristicScreener)
