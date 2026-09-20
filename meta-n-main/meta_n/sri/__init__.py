"""SRI formal experimental protocol (design: 2026-09-20).

Modules:
    protocol.py           profiles, manifests, drift validation, root bundles
    ledger.py             append-only proposal-slot ledger
    transition_audit.py   canonical post-freeze depth-2 -> depth-3 audit
    metrics.py            offline aggregation of frozen artifacts
"""

from meta_n.sri.protocol import (  # noqa: F401
    EXTENDED10,
    LEDGER_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    OPERATIONAL_ALLOWLIST,
    PRIMARY6,
    PROFILE_SCHEMA_VERSION,
    ProtocolError,
    RunManifest,
    SRIProfile,
    TREATMENT_FIELDS,
    assert_no_protocol_drift,
    assert_treatment_parity,
    canonical_json,
    format_effective_config_table,
    load_profile,
    resolve_effective_config,
    root_bundle_sha256,
    sha256_file,
    sha256_of,
)

__all__ = [
    "EXTENDED10", "PRIMARY6", "ProtocolError", "RunManifest", "SRIProfile",
    "assert_no_protocol_drift", "assert_treatment_parity", "canonical_json",
    "format_effective_config_table", "load_profile",
    "resolve_effective_config", "root_bundle_sha256", "sha256_file",
    "sha256_of", "TREATMENT_FIELDS", "OPERATIONAL_ALLOWLIST",
    "PROFILE_SCHEMA_VERSION", "MANIFEST_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
]
