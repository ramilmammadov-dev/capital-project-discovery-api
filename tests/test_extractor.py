"""Extractor with a scripted fake LLM client (no network, no keys)."""

from __future__ import annotations

import re

from capex.costs import CostLedger
from capex.extraction.extractor import Extractor, split_chunk
from capex.llm import LLMError, LLMRegistry
from capex.storage.cache import FileCache

from .helpers import (
    FakeLLM,
    content_of,
    llm_invalid,
    llm_ok,
    llm_truncated,
    make_document,
    projects_from_user,
    table_chunk,
)

ORG = "City of Exampleville"


def always_ok(call):
    return llm_ok(projects_from_user(call["user"]), call["model"])


def make_extractor(app_config, tmp_path, handler=always_ok, *, cache=None):
    fake = FakeLLM(handler)
    registry = LLMRegistry(app_config)
    registry.register("anthropic", fake)
    ledger = CostLedger(app_config)
    extractor = Extractor(app_config, registry, ledger, cache or FileCache(tmp_path / "cache"))
    return extractor, fake, ledger


class TestNormalExtraction:
    async def test_json_becomes_validated_records(self, app_config, tmp_path):
        extractor, fake, ledger = make_extractor(app_config, tmp_path)
        outcome = await extractor.extract(table_chunk(8), make_document(), ORG)
        assert outcome.error is None and outcome.split is False and outcome.escalated is False
        assert outcome.raw_count == 8 and len(outcome.records) == 8
        assert sum(outcome.rejected.values()) == 0
        rec = outcome.records[0]
        assert rec.project_id == "ST85100000"
        assert rec.estimated_budget == 1_000_000
        assert rec.source_pages == [4]
        assert rec.model == app_config.models.extraction.model
        assert rec.excerpt_verified and rec.name_verified
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["schema_name"] == "capital_projects" and call["stage"] == "extraction"
        assert call["model"] == app_config.models.extraction.model
        assert f"ORGANIZATION: {ORG}" in call["user"] and "CONTENT:\n[[PDF page 4]]" in call["user"]
        assert "projects" in call["schema"]["properties"]
        assert len(ledger.records) == 1 and ledger.records[0].cost_usd > 0

    async def test_invented_projects_are_rejected_and_counted(self, app_config, tmp_path):
        def handler(call):
            data = projects_from_user(call["user"])
            data["projects"].append({"project_name": "Atlantis Underwater Dome", "extraction_confidence": 0.9})
            return llm_ok(data, call["model"])

        extractor, _, _ = make_extractor(app_config, tmp_path, handler)
        outcome = await extractor.extract(table_chunk(8), make_document(), ORG)
        assert len(outcome.records) == 8
        assert outcome.rejected["name_not_in_source"] == 1
        assert outcome.raw_count == 9


class TestCaching:
    async def test_second_run_makes_zero_calls(self, app_config, tmp_path):
        extractor, fake, ledger = make_extractor(app_config, tmp_path)
        chunk, document = table_chunk(8), make_document()
        first = await extractor.extract(chunk, document, ORG)
        assert len(fake.calls) == 1
        again = await extractor.extract(chunk, document, ORG)
        assert len(fake.calls) == 1
        assert ledger.cache_hits["extraction"] == 1
        assert [r.project_id for r in again.records] == [r.project_id for r in first.records]

    async def test_cache_survives_a_new_extractor_on_the_same_cache_dir(self, app_config, tmp_path):
        cache_dir = FileCache(tmp_path / "shared")
        e1, f1, _ = make_extractor(app_config, tmp_path, cache=cache_dir)
        await e1.extract(table_chunk(8), make_document(), ORG)
        e2, f2, l2 = make_extractor(app_config, tmp_path, cache=FileCache(tmp_path / "shared"))
        outcome = await e2.extract(table_chunk(8), make_document(), ORG)
        assert len(f1.calls) == 1 and len(f2.calls) == 0
        assert len(outcome.records) == 8 and l2.records == []

    async def test_changed_document_metadata_is_a_cache_miss(self, app_config, tmp_path):
        extractor, fake, _ = make_extractor(app_config, tmp_path)
        chunk = table_chunk(8)
        await extractor.extract(chunk, make_document(title="CIP A"), ORG)
        await extractor.extract(chunk, make_document(title="CIP B"), ORG)
        assert len(fake.calls) == 2

    async def test_concurrent_identical_requests_are_coalesced(self, app_config, tmp_path):
        import asyncio

        extractor, fake, _ = make_extractor(app_config, tmp_path)
        chunk, document = table_chunk(8), make_document()
        results = await asyncio.gather(*(extractor.extract(chunk, document, ORG) for _ in range(3)))
        assert len(fake.calls) == 1
        assert all(len(r.records) == 8 for r in results)


class TestSplitting:
    async def test_truncated_first_call_splits_and_both_halves_are_extracted(self, app_config, tmp_path):
        state = {"n": 0}

        def handler(call):
            state["n"] += 1
            if state["n"] == 1:
                return llm_truncated(call["model"])
            return llm_ok(projects_from_user(call["user"]), call["model"])

        extractor, fake, _ = make_extractor(app_config, tmp_path, handler)
        chunk = table_chunk(10, two_pages=True)
        outcome = await extractor.extract(chunk, make_document(), ORG)
        assert outcome.split is True and outcome.error is None
        assert len(fake.calls) == 3
        assert outcome.escalated is False and extractor.escalations == 0 and extractor.splits == 1
        ids = sorted(r.project_id for r in outcome.records)
        assert ids == [f"ST851{i:05d}" for i in range(10)]
        assert {r.chunk_id for r in outcome.records} == {f"{chunk.chunk_id}a", f"{chunk.chunk_id}b"}
        # records keep the right page after the split (rows 0-4 on page 4, rows 5-9 on page 5)
        pages = {r.project_id: r.source_pages for r in outcome.records}
        assert all(pages[f"ST851{i:05d}"] == [4] for i in range(5))
        assert all(pages[f"ST851{i:05d}"] == [5] for i in range(5, 10))
        # each half was sent as its own, smaller prompt
        sizes = [len(c["user"]) for c in fake.calls]
        assert sizes[1] < sizes[0] and sizes[2] < sizes[0]

    async def test_second_run_reuses_the_cached_halves(self, app_config, tmp_path):
        state = {"n": 0}

        def handler(call):
            state["n"] += 1
            if "ST85100009" in call["user"] and "ST85100000" in call["user"]:  # the whole chunk
                return llm_truncated(call["model"])
            return llm_ok(projects_from_user(call["user"]), call["model"])

        extractor, fake, _ = make_extractor(app_config, tmp_path, handler)
        chunk = table_chunk(10, two_pages=True)
        await extractor.extract(chunk, make_document(), ORG)
        before = len(fake.calls)
        outcome = await extractor.extract(chunk, make_document(), ORG)
        assert len(outcome.records) == 10
        # only the (uncacheable) truncated whole-chunk attempt is repeated; the halves come from the cache
        assert len(fake.calls) - before == 1

    async def test_dense_table_with_low_recall_is_split_and_the_better_result_kept(self, app_config, tmp_path):
        def handler(call):
            whole = "ST85100000" in call["user"] and "ST85100009" in call["user"]
            return llm_ok(projects_from_user(call["user"], limit=2 if whole else None), call["model"])

        extractor, fake, _ = make_extractor(app_config, tmp_path, handler)
        outcome = await extractor.extract(table_chunk(10), make_document(), ORG)
        assert outcome.split is True
        assert len(outcome.records) == 10
        assert len(fake.calls) == 3

    async def test_persistent_failure_reports_an_error_and_no_records(self, app_config, tmp_path):
        extractor, fake, _ = make_extractor(app_config, tmp_path, lambda call: llm_invalid(call["model"]))
        outcome = await extractor.extract(table_chunk(10), make_document(), ORG)
        assert outcome.records == []
        assert outcome.error and "extraction failed" in outcome.error
        assert len(fake.calls) > 3  # whole chunk, halves, quarters, then the escalation model

    async def test_llm_errors_are_contained(self, app_config, tmp_path):
        def handler(call):
            raise LLMError("provider is down")

        extractor, _, _ = make_extractor(app_config, tmp_path, handler)
        outcome = await extractor.extract(table_chunk(10), make_document(), ORG)
        assert outcome.records == [] and outcome.error


class TestEscalation:
    async def test_hallucinating_cheap_model_is_escalated_once(self, app_config, tmp_path):
        def handler(call):
            if call["model"] == app_config.models.escalation.model:
                return llm_ok(projects_from_user(call["user"]), call["model"])
            data = projects_from_user(call["user"], limit=4)
            data["projects"] += [{"project_name": f"Imaginary Project {n}", "extraction_confidence": 0.9} for n in range(4)]
            return llm_ok(data, call["model"])

        extractor, fake, _ = make_extractor(app_config, tmp_path, handler)
        outcome = await extractor.extract(table_chunk(6), make_document(), ORG)
        assert outcome.escalated is True and extractor.escalations == 1
        assert app_config.models.escalation.model in outcome.models
        assert len(outcome.records) == 6
        assert {r.model for r in outcome.records} == {app_config.models.escalation.model}
        assert sum(c["model"] == app_config.models.escalation.model for c in fake.calls) == 1

    async def test_escalation_is_capped_per_job(self, app_config, tmp_path):
        app_config.extraction.max_escalations_per_job = 0

        def handler(call):
            data = projects_from_user(call["user"], limit=4)
            data["projects"] += [{"project_name": f"Imaginary Project {n}", "extraction_confidence": 0.9} for n in range(4)]
            return llm_ok(data, call["model"])

        extractor, fake, _ = make_extractor(app_config, tmp_path, handler)
        outcome = await extractor.extract(table_chunk(6), make_document(), ORG)
        assert outcome.escalated is False and extractor.escalations == 0
        assert all(c["model"] == app_config.models.extraction.model for c in fake.calls)


class TestSplitChunk:
    def test_short_chunks_are_not_split(self):
        chunk = table_chunk(2)
        assert split_chunk(chunk) == [chunk]

    def test_split_at_a_page_marker_keeps_every_row_once(self):
        chunk = table_chunk(10, two_pages=True)
        first, second = split_chunk(chunk)
        assert first.text.startswith("[[PDF page 4]]") and second.text.startswith("[[PDF page 5]]")
        assert first.pages == [4] and second.pages == [5]
        rows = [*re.findall(r"ST851\d{5}", first.text), *re.findall(r"ST851\d{5}", second.text)]
        assert rows == [f"ST851{i:05d}" for i in range(10)]
        assert (first.chunk_id, second.chunk_id) == (f"{chunk.chunk_id}a", f"{chunk.chunk_id}b")
        assert first.content_hash != second.content_hash != chunk.content_hash
        assert first.est_rows + second.est_rows == chunk.est_rows

    def test_mid_page_split_repeats_the_page_marker_on_the_second_half(self):
        chunk = table_chunk(12)  # a single page: the cut falls between two rows
        first, second = split_chunk(chunk)
        assert first.text.splitlines()[0] == "[[PDF page 4]]"
        assert second.text.splitlines()[0] == "[[PDF page 4]]"
        assert second.pages == [4]
        body = "\n".join([*first.text.splitlines()[1:], *second.text.splitlines()[1:]])
        assert body == "\n".join(chunk.text.splitlines()[1:])

    def test_content_of_helper_reads_the_content_section(self):
        assert content_of("A\n\nCONTENT:\nbody") == "body"
