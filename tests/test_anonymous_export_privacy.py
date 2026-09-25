import gzip

import pytest

from paper.export_submission import (
    ANONYMOUS_COMMIT,
    audit_bytes,
    sanitize_export_metadata,
)


COMMIT = "5c2576e2a98db331ab0f90540119cacecde2ef07"


def test_nested_git_provenance_is_removed_from_exported_data():
    exported = sanitize_export_metadata({
        "snapshot_commit": COMMIT,
        "snapshot_revision": COMMIT,
        "rows": [{"git_revision": COMMIT,
                  "novelty_snapshot_commit": COMMIT,
                  "normalized_sha256": "a" * 64}],
    })

    assert exported == {
        "snapshot_commit": ANONYMOUS_COMMIT,
        "rows": [{"novelty_snapshot_commit": ANONYMOUS_COMMIT,
                  "normalized_sha256": "a" * 64}],
    }


@pytest.mark.parametrize("name,compress", [("data.json", False), ("data.json.gz", True)])
def test_privacy_screen_rejects_git_hashes_in_plain_and_compressed_entries(name, compress):
    payload = f'{{"revision":"{COMMIT}"}}'.encode()
    if compress:
        payload = gzip.compress(payload)
    with pytest.raises(AssertionError, match="Privacy screen rejected"):
        audit_bytes(name, payload)
