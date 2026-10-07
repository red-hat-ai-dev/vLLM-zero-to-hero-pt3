# Result fixtures

`guidellm-0.7.4.json` contains a reduced real RTX 3090 smoke sweep from
2026-10-07: 10-second measurement windows, 256 prompt tokens, and 128 output
tokens. Nine strategies completed; the adaptive sweep ended before the tenth.
Only three representative strategies (synchronous, throughput, and the first
constant rate) and the metadata, request counts, and metric fields used by the
comparison are retained. Prompts, individual requests, paths, and unrelated fields are omitted.
This short run is a parser fixture, not a performance claim.

`lm-eval-0.4.13.json` is the corresponding 20-example baseline GSM8K result.
It retains configuration, task metadata, sample counts, and aggregate scores;
host addresses, environment dumps, raw prompts, and responses are omitted.
