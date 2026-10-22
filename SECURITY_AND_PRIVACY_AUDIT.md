# Security and privacy audit

The candidate tree was read and scanned; .gitignore was not treated as sanitization. No old .git directory/history, raw image/cache, checkpoint/weight, raw patient prediction, patient ID, source-image hash list, credential file, email correspondence, ethics inquiry or full server log is included. Exact internal mapping/diffs/original-source snapshots remain outside this ZIP.

Heuristic checks found no private server/account paths, source patient-ID patterns, email strings, private-key markers, GitHub tokens, cloud access keys or credential-value assignments. No symlink or file over5MiB was found. Whole-artifact hashes are scientific provenance, not individual-image or patient records. Only aggregate results and artificial SYNTH fixtures are included. The supplied author name and public ORCID are deliberate citation metadata.

This scan does not prove absence of every secret or certify legal/ethics suitability. The release-file decision table and image manifest intentionally withhold linkable metadata pending author review. An author must resolve copyright, institutional rights, image credits and private-data release scope before public posting.

Run scripts/check_public_package.py on the entire candidate before release and verify SHA256_MANIFEST.txt. All ZIP members were checked against that exact tree; internal_review is excluded.

Follow-up disclosure review inspected original schemas privately. New public files contain field names, aggregate fold counts and source-file hashes only, not patient/source/UID/hash-list values or notebook outputs. Original patient-level rows remain withheld pending final review. Heuristic scanning does not establish legal permission or anonymity.
