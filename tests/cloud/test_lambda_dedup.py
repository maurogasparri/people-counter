"""Tests for Lambda dedup L3 (inter-camera deduplication).

Uses a mock DynamoDB table to test the dedup logic without AWS.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.cloud.lambda_dedup import deduplicate_hashes, handler


class MockConditionalCheckFailedException(Exception):
    pass


def _make_mock_table():
    """Create a mock DynamoDB table with in-memory store."""
    store = {}
    table = MagicMock()

    # Mock the exception class path
    client = MagicMock()
    client.exceptions.ConditionalCheckFailedException = (
        MockConditionalCheckFailedException
    )
    table.meta.client = client

    def put_item(**kwargs):
        key = (kwargs["Item"]["store_date"], kwargs["Item"]["hash"])
        if key in store:
            raise MockConditionalCheckFailedException("Already exists")
        store[key] = kwargs["Item"]

    table.put_item = put_item
    table._store = store  # Expose for assertions
    return table


class TestDeduplicateHashes:
    @patch("src.cloud.lambda_dedup._get_table")
    def test_all_new(self, mock_get_table):
        table = _make_mock_table()
        mock_get_table.return_value = table

        result = deduplicate_hashes(
            "store-001", "2026-03-30", ["hash_a", "hash_b", "hash_c"], "cam-01"
        )
        assert result["new_count"] == 3
        assert result["duplicate_count"] == 0

    @patch("src.cloud.lambda_dedup._get_table")
    def test_all_duplicates(self, mock_get_table):
        table = _make_mock_table()
        mock_get_table.return_value = table

        # First call — all new
        deduplicate_hashes(
            "store-001", "2026-03-30", ["hash_a", "hash_b"], "cam-01"
        )

        # Second call with same hashes from different camera
        result = deduplicate_hashes(
            "store-001", "2026-03-30", ["hash_a", "hash_b"], "cam-02"
        )
        assert result["new_count"] == 0
        assert result["duplicate_count"] == 2

    @patch("src.cloud.lambda_dedup._get_table")
    def test_mixed_new_and_duplicate(self, mock_get_table):
        table = _make_mock_table()
        mock_get_table.return_value = table

        deduplicate_hashes(
            "store-001", "2026-03-30", ["hash_a"], "cam-01"
        )

        result = deduplicate_hashes(
            "store-001", "2026-03-30", ["hash_a", "hash_b"], "cam-02"
        )
        assert result["new_count"] == 1  # hash_b is new
        assert result["duplicate_count"] == 1  # hash_a already exists

    @patch("src.cloud.lambda_dedup._get_table")
    def test_different_dates_are_independent(self, mock_get_table):
        table = _make_mock_table()
        mock_get_table.return_value = table

        deduplicate_hashes(
            "store-001", "2026-03-30", ["hash_a"], "cam-01"
        )

        # Same hash, different date → should be new
        result = deduplicate_hashes(
            "store-001", "2026-03-31", ["hash_a"], "cam-01"
        )
        assert result["new_count"] == 1

    @patch("src.cloud.lambda_dedup._get_table")
    def test_different_stores_are_independent(self, mock_get_table):
        table = _make_mock_table()
        mock_get_table.return_value = table

        deduplicate_hashes(
            "store-001", "2026-03-30", ["hash_a"], "cam-01"
        )

        # Same hash, different store → should be new
        result = deduplicate_hashes(
            "store-002", "2026-03-30", ["hash_a"], "cam-01"
        )
        assert result["new_count"] == 1

    @patch("src.cloud.lambda_dedup._get_table")
    def test_empty_hashes(self, mock_get_table):
        table = _make_mock_table()
        mock_get_table.return_value = table

        result = deduplicate_hashes(
            "store-001", "2026-03-30", [], "cam-01"
        )
        assert result["new_count"] == 0
        assert result["duplicate_count"] == 0


class TestHandler:
    @patch("src.cloud.lambda_dedup._get_table")
    def test_valid_event(self, mock_get_table):
        table = _make_mock_table()
        mock_get_table.return_value = table

        event = {
            "device_id": "store-001-cam-01",
            "store_id": "store-001",
            "date": "2026-03-30",
            "data": {
                "hashes": ["h1", "h2", "h3"],
                "protocol": "wifi",
            },
        }

        result = handler(event, None)
        assert result["statusCode"] == 200
        assert result["body"]["new_count"] == 3

    @patch("src.cloud.lambda_dedup._get_table")
    def test_empty_hashes_event(self, mock_get_table):
        table = _make_mock_table()
        mock_get_table.return_value = table

        event = {
            "device_id": "store-001-cam-01",
            "data": {"hashes": []},
        }

        result = handler(event, None)
        assert result["statusCode"] == 200
        assert result["body"]["new_count"] == 0

    def test_malformed_event(self):
        result = handler({}, None)
        assert result["statusCode"] == 500
