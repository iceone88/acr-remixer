# SPDX-FileCopyrightText: 2026 hiro98 (vale981) and the ACR Remixer contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from stagegen import toolchain


def test_blender_checksum_manifest_parses_hash_before_filename():
    expected = "a" * 64
    manifest = f"{expected}  {toolchain.BLENDER_ARCHIVE}\n".encode()
    assert toolchain._read_expected_sha256(manifest, toolchain.BLENDER_ARCHIVE) == expected


def test_blender_checksum_manifest_parses_parenthesized_format():
    expected = "b" * 64
    manifest = f"SHA256 ({toolchain.BLENDER_ARCHIVE}) = {expected}\n".encode()
    assert toolchain._read_expected_sha256(manifest, toolchain.BLENDER_ARCHIVE) == expected


def test_blender_checksum_manifest_fails_closed_when_file_missing():
    with pytest.raises(toolchain.ToolchainError, match="no SHA-256 entry"):
        toolchain._read_expected_sha256(b"c" * 64 + b"  other.zip\n", toolchain.BLENDER_ARCHIVE)


def test_safe_extract_rejects_path_traversal(tmp_path: Path):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "must not escape")
    target = tmp_path / "extract"
    target.mkdir()
    with pytest.raises(toolchain.ToolchainError, match="Unsafe path"):
        toolchain._safe_extract(archive_path, target)
    assert not (tmp_path / "outside.txt").exists()


def test_safe_extract_extracts_valid_blender_like_archive(tmp_path: Path):
    archive_path = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("blender-5.2.5-windows-x64/blender.exe", b"fixture")
    target = tmp_path / "extract"
    target.mkdir()
    unpacked = toolchain._safe_extract(archive_path, target)
    assert unpacked == len(b"fixture")
    assert (target / "blender-5.2.5-windows-x64" / "blender.exe").read_bytes() == b"fixture"


def test_pinned_blender_urls_are_official_release_urls():
    assert toolchain.BLENDER_ARCHIVE_URL.startswith("https://download.blender.org/release/Blender5.2/")
    assert toolchain.BLENDER_CHECKSUM_URL.startswith("https://download.blender.org/release/Blender5.2/")
    assert toolchain.BLENDER_ARCHIVE.endswith("windows-x64.zip")
