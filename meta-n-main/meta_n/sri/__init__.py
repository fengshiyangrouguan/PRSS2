"""SRI formal experimental protocol (design: 2026-09-20).

Modules:
    protocol.py           profiles, manifests, drift validation, root bundles
    ledger.py             append-only proposal-slot ledger
    hooks.py              orchestrator-side slot open/close + material capture
    transition_audit.py   canonical post-freeze depth-2 -> depth-3 audit
    metrics.py            final selection, per-run and cross-seed aggregation
    pairing.py            per-channel paired-evaluation effectiveness (§9)
"""

from meta_n.sri.protocol import (  # noqa: F401
    CONFIG_KEY_TO_CLI,
    EXTENDED10,
    LEDGER_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    NEGATABLE_BOOL_CONFIG_KEYS,
    OPERATIONAL_ALLOWLIST,
    POSITIVE_ONLY_BOOL_CONFIG_KEYS,
    PRIMARY6,
    PROFILE_SCHEMA_VERSION,
    ProtocolError,
    RunManifest,
    SRIProfile,
    TREATMENT_FIELDS,
    assert_no_protocol_drift,
    assert_treatment_parity,
    canonical_json,
    collect_root_bundle,
    environment_fingerprint,
    format_effective_config_table,
    git_revision,
    hash_tree,
    load_profile,
    pinned_run_config,
    render_pinned_flags,
    resolve_effective_config,
    root_bundle_sha256,
    sha256_file,
    sha256_of,
    verify_run_config,
)

__all__ = [
    "EXTENDED10", "PRIMARY6", "ProtocolError", "RunManifest", "SRIProfile",
    "assert_no_protocol_drift", "assert_treatment_parity", "canonical_json",
    "collect_root_bundle", "environment_fingerprint",
    "format_effective_config_table", "git_revision", "hash_tree",
    "load_profile", "pinned_run_config", "render_pinned_flags",
    "resolve_effective_config", "root_bundle_sha256", "sha256_file",
    "sha256_of", "verify_run_config", "TREATMENT_FIELDS",
    "OPERATIONAL_ALLOWLIST", "PROFILE_SCHEMA_VERSION", "MANIFEST_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION", "CONFIG_KEY_TO_CLI", "NEGATABLE_BOOL_CONFIG_KEYS",
    "POSITIVE_ONLY_BOOL_CONFIG_KEYS",
]
