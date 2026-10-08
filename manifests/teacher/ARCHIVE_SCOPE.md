# Local evidence archive

This bundle preserves the explicitly selected local files. It does not establish model quality, training completion, or faithful paper reproduction.

`FILES_MANIFEST.jsonl` contains one SHA256 for each stored payload file. `ORIGIN_PATH_MAP.jsonl` additionally maps deduplicated archive members to their stored equivalent. `EXCLUSIONS.jsonl` records all omitted files and container archives with sizes and reasons. `NESTED_ARCHIVE_AUDIT.jsonl` records unique, duplicate, and rejected members of ZIP/TAR inputs. Source files were not modified. Teacher original ZIP and new server evidence are separate assets.

`SHA256SUMS.txt` covers every bundled file except itself; the external verification record hashes this checksum file and the final ZIP.
