"""Paired-evaluation effectiveness, recorded PER CHANNEL (§9).

THE PROBLEM THIS REPLACES. The historical implementation recorded ONE boolean,
`paired_eval_effective`, and its own warning already admitted the truth: the
seed was forwarded only on a subset of outer calls, and depth>1 / inner
`llm()` / `llm_batch()` calls were unpaired. A single optimistic boolean cannot
express that, so a formal run could report "paired" while most of its sampling
was unpaired.

REQUIRED FIELDS (§9). Six, mirroring the channels that actually exist:

    pairing_requested          the operator asked for pairing
    pairing_outer_effective    depth-1 outer generation carried a seed
    pairing_inner_effective    inner llm()/llm_batch() carried a seed
    pairing_executor_effective the executor honoured the seed (backend capability)
    pairing_test_effective     held-out evaluation was paired
    pairing_limitations        human-readable reasons for anything not effective

FAIL CLOSED (§9.4). `assert_formal_pairing(...)` is called at formal startup: if
pairing was requested and any channel is silently inert, the run refuses to
start. The alternative -- starting and discovering it in the analysis -- is what
makes an SRI number uninterpretable.

MATCHED-INSTANCE FALLBACK (§9.5). A provider with no seed API is not a dead end:
the evaluator can still use matched frozen instances plus repeated evaluation.
That is recorded as a limitation and the manifest must NOT claim full CRN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from meta_n.sri.protocol import ProtocolError

PAIRING_SCHEMA_VERSION = 1

# The channels a formal run can pair, in the order they matter to stability.
CHANNELS = ("outer", "inner", "executor", "test")


@dataclass
class PairingRecord:
    """§9's six required manifest fields, plus the derivation trail."""

    pairing_requested: bool = False
    pairing_outer_effective: bool = False
    pairing_inner_effective: bool = False
    pairing_executor_effective: bool = False
    pairing_test_effective: bool = False
    pairing_limitations: List[str] = field(default_factory=list)
    schema_version: int = PAIRING_SCHEMA_VERSION
    # extra, non-required context that makes the record auditable
    crn_claimed: bool = False
    model: Optional[str] = None
    backend: Optional[str] = None

    def __post_init__(self) -> None:
        self.pairing_requested = bool(self.pairing_requested)
        for c in CHANNELS:
            setattr(self, "pairing_{}_effective".format(c),
                    bool(getattr(self, "pairing_{}_effective".format(c))))
        self.pairing_limitations = [str(x) for x in self.pairing_limitations]
        # A record that claims CRN while a channel is inert is self-contradictory.
        if self.crn_claimed and not self.all_effective:
            raise ProtocolError(
                "PairingRecord claims full CRN (crn_claimed=True) while {} "
                "channel(s) are inert; set crn_claimed=False and record the "
                "limitation instead (§9)".format(
                    [c for c in CHANNELS if not self.channel_effective(c)]))

    def channel_effective(self, channel: str) -> bool:
        if channel not in CHANNELS:
            raise ProtocolError("unknown pairing channel {!r}".format(channel))
        return bool(getattr(self, "pairing_{}_effective".format(channel)))

    @property
    def all_effective(self) -> bool:
        return all(self.channel_effective(c) for c in CHANNELS)

    @property
    def any_effective(self) -> bool:
        return any(self.channel_effective(c) for c in CHANNELS)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def manifest_fields(self) -> Dict[str, Any]:
        """Exactly the six required keys, for stamping into a run manifest."""
        d = self.to_dict()
        return {k: d[k] for k in (
            "pairing_requested", "pairing_outer_effective",
            "pairing_inner_effective", "pairing_executor_effective",
            "pairing_test_effective", "pairing_limitations")}


def build_record(*, requested: bool, model: Optional[str],
                 backend: Optional[str],
                 outer_seeded: bool, inner_seeded: bool,
                 executor_honours: bool, test_seeded: bool,
                 extra_limitations: Sequence[str] = (),
                 matched_instance_fallback: bool = False) -> PairingRecord:
    """Assemble the record from what each channel ACTUALLY did.

    `matched_instance_fallback` records §9.5: the provider lacks a seed API, so
    matched frozen instances plus repeats were used instead. It is a legitimate
    mode and is labelled, never dressed up as CRN.
    """
    lim: List[str] = []
    if requested and not executor_honours:
        lim.append("provider has no per-request seed API: the seed is dropped at "
                   "the wire, so sampling luck is NOT correlated across arms")
    if requested and not outer_seeded:
        lim.append("outer depth-1 generation did not carry a seed")
    if requested and not inner_seeded:
        lim.append("inner llm()/llm_batch() calls did not carry a seed "
                   "(depth>1 sampling is unpaired)")
    if requested and not test_seeded:
        lim.append("held-out evaluation did not carry a seed")
    if matched_instance_fallback:
        lim.append("matched-instance repeated evaluation used in place of "
                   "per-request seeding; full CRN is not claimed")
    lim.extend(str(x) for x in extra_limitations)

    rec = PairingRecord(
        pairing_requested=bool(requested),
        pairing_outer_effective=bool(requested and outer_seeded
                                     and executor_honours),
        pairing_inner_effective=bool(requested and inner_seeded
                                     and executor_honours),
        pairing_executor_effective=bool(requested and executor_honours),
        pairing_test_effective=bool(requested and test_seeded
                                    and executor_honours),
        pairing_limitations=lim,
        crn_claimed=False,
        model=model, backend=backend)
    return rec


def assert_formal_pairing(record: PairingRecord, *, formal: bool) -> None:
    """§9.4: formal startup fails when requested pairing is silently inert."""
    if not formal or not record.pairing_requested:
        return
    inert = [c for c in CHANNELS if not record.channel_effective(c)]
    if inert:
        raise ProtocolError(
            "formal pairing requested but channel(s) {} are INERT. Refusing to "
            "start: a run whose sampling is only partly paired cannot support a "
            "paired SRI claim. Limitations: {}".format(
                inert, record.pairing_limitations))


def merge_records(records: Sequence[PairingRecord]) -> PairingRecord:
    """Worst-case merge across seeds: a channel is effective only if EVERY seed says so.

    Reporting the best seed's channel would let one lucky seed hide that the
    others were unpaired.
    """
    if not records:
        raise ProtocolError("no pairing records to merge")
    out = PairingRecord(
        pairing_requested=any(r.pairing_requested for r in records),
        pairing_outer_effective=all(r.pairing_outer_effective for r in records),
        pairing_inner_effective=all(r.pairing_inner_effective for r in records),
        pairing_executor_effective=all(r.pairing_executor_effective
                                       for r in records),
        pairing_test_effective=all(r.pairing_test_effective for r in records),
        pairing_limitations=sorted({x for r in records
                                    for x in r.pairing_limitations}),
        model=records[0].model, backend=records[0].backend)
    return out


# --------------------------------------------------------------------------
# acceptance self-test (§14 item 16)
# --------------------------------------------------------------------------

def self_test() -> int:
    print("pairing.py acceptance")
    print("=" * 68)

    # a fully paired run
    good = build_record(requested=True, model="m", backend="azure",
                        outer_seeded=True, inner_seeded=True,
                        executor_honours=True, test_seeded=True)
    assert good.all_effective and not good.pairing_limitations
    assert_formal_pairing(good, formal=True)
    print("OK  all channels       every channel effective, formal start allowed")

    # §14.16: effectiveness is truthful at depth > 1 -- the inner channel is the
    # one the old single boolean lied about.
    partial = build_record(requested=True, model="m", backend="relay",
                           outer_seeded=True, inner_seeded=False,
                           executor_honours=True, test_seeded=True)
    assert partial.pairing_outer_effective is True
    assert partial.pairing_inner_effective is False
    assert any("inner llm()" in x for x in partial.pairing_limitations)
    try:
        assert_formal_pairing(partial, formal=True)
        raise AssertionError("a formally-inert inner channel was accepted")
    except ProtocolError as e:
        assert "INERT" in str(e)
    print("OK  truthful at depth>1 inner channel reported INERT even though the "
          "outer channel is paired; formal start refused")

    # non-formal mode keeps the historical permissive behaviour
    assert_formal_pairing(partial, formal=False)
    print("OK  non-formal         the same record does not block a non-formal run")

    # §9.5: matched-instance fallback is labelled, and CRN is NOT claimed
    fb = build_record(requested=True, model="m", backend="relay",
                      outer_seeded=False, inner_seeded=False,
                      executor_honours=False, test_seeded=False,
                      matched_instance_fallback=True)
    assert fb.crn_claimed is False
    assert any("matched-instance" in x for x in fb.pairing_limitations)
    assert any("no per-request seed API" in x for x in fb.pairing_limitations)
    print("OK  matched-instance   fallback labelled; full CRN explicitly not "
          "claimed")

    # claiming CRN with an inert channel is self-contradictory
    try:
        PairingRecord(pairing_requested=True, pairing_outer_effective=True,
                      pairing_inner_effective=False,
                      pairing_executor_effective=True,
                      pairing_test_effective=True, crn_claimed=True)
        raise AssertionError("a CRN claim with an inert channel was accepted")
    except ProtocolError as e:
        assert "crn_claimed" in str(e)
    print("OK  crn claim guard    claiming CRN while a channel is inert refused")

    # merge is worst-case, not best-case
    merged = merge_records([good, partial])
    assert merged.pairing_outer_effective is True
    assert merged.pairing_inner_effective is False
    print("OK  worst-case merge   one unpaired seed makes the merged inner "
          "channel INERT (a lucky seed cannot hide it)")

    # manifest fields are exactly the six required keys
    mf = good.manifest_fields()
    assert sorted(mf) == sorted([
        "pairing_requested", "pairing_outer_effective",
        "pairing_inner_effective", "pairing_executor_effective",
        "pairing_test_effective", "pairing_limitations"]), sorted(mf)
    print("OK  manifest fields    the six required keys are present verbatim")

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
