# Security and Disclosure Audit

Current candidate includes **intentional, author-authorized source-linked metadata and frozen OOF records**. These are not claimed anonymous. Exact permitted files, columns, counts and hashes are in [RELEASE_DATA_POLICY.json](../../data/RELEASE_DATA_POLICY.json); field scope is explained in [the disclosure review](../data/DATA_DISCLOSURE_REVIEW.md).

Only the source-patient-pattern rule is excepted for byte-identical reviewed files. Credentials, personal/server paths, email patterns and private-key rules remain active in both ordinary text and decompressed gzip content. Unknown gzip, changed reviewed files, symlinks, unlisted files and oversized unapproved files fail. Bounded decompression and exact content SHA/schema/count checks prevent treating an arbitrary archive as approved metadata. The policy is a review record, not a cryptographic author signature or ethics certificate.

No source image, cache, checkpoint, weight, direct-identity/contact field, new clinical attribute, raw server log, manuscript or private absolute-path map is included. Author name/ORCID and legitimate upstream copyright notices are deliberate metadata. Synthetic tests contain artificial fixtures only.

The historical reachable-file scan was completed before the authorized data addition; its baseline and exclusions remain in [REPOSITORY_FINAL_CHECK.json](REPOSITORY_FINAL_CHECK.json). It does not cover commit author metadata, deleted/unreachable objects or every possible identifier format. No history was rewritten. Current source-linked rows are declared data, not retroactively described as absent from history.

For the current public repository, inspect the exact reviewed commit and successful CI result before merging or issuing a Release. Heuristic scanning does not guarantee absence of every secret or establish legal permission, irreversible anonymity, IRB suitability or pathology validity. Public visibility was verified on 28 October 2026; Release and archive registration remain separate author actions.
