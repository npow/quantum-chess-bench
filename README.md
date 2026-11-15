# Quantum Chess Bench

Run LLMs against each other under a frozen quantum-chess protocol, with legal actions supplied by an authoritative Rust engine. Each model and reasoning configuration is a separate player. Python 3.10+ and Rust 1.99+ are required. Python uses only the standard library.

## Run locally

```sh
python3 bench.py
python3 bench.py --position
python3 bench.py --audit --output runs/smoke
```

The default smoke run plays two short random-player games without API calls. `--position` prints the example quantum opening, piece identities, possible classical boards, and every legal action. First invocation builds the Rust engine.

## Run models

1. Copy `models.example.json` to a new configuration file. Replace model IDs and set each player's `parameters` for its provider's reasoning/token controls if needed. `base_url` defaults to `https://openrouter.ai/api/v1`; any compatible Chat Completions endpoint can be used. `api_key_env` names the environment variable containing its key. Never put credentials in configuration.
2. Export the key, then run:

```sh
export OPENROUTER_API_KEY='your-key'
python3 bench.py --config models.json --output runs/models-v1
```

3. Inspect `runs/models-v1/standings.json`, then verify:

```sh
python3 bench.py --audit --output runs/models-v1
```

Re-running the same command resumes completed games. Configuration, prompts, runner, engine source, and build metadata are hashed; changes require a new output directory. An interrupted game restarts from its opening; prior request logs are retained so billed calls remain visible. Do not run two processes against the same output directory.

## Protocol v1

- Every pair of players plays `pairs` opening pairs, swapping colors. Openings cycle through the configured list; defaults start from the normal position. All openings are validated before model calls. The example includes an opening with four split pieces to exercise quantum play.
- Each model receives the shared rules, full current state, piece identities and halves, possible-board FENs, move history, and full legal action list, including collapse choices. Separate conversations retain earlier exchanges up to `context_chars` (a character budget, not a token budget); the latest full state is always supplied. Opponent identity, ratings, and engine suggestions are absent.
- The API requires a structured `submit_move` tool call containing the exact legal action. JSON move objects and bare actions are also parsed for compatible endpoints. Illegal/malformed replies retry without advancing; three rejected replies in a turn forfeit by default.
- Checkmate/stalemate take precedence. Three repetitions of the position during played turns or 150 accepted half-moves after the opening draw. The rules engine implements this repository's deterministic variant: the mover chooses minimal collapses. See [RULES.md](RULES.md). Standard chess fifty-move/insufficient-material adjudication is not added.
- Network/auth/provider failures stop the run as incomplete; they never count as losses. The default HTTP timeout is 120 seconds. Use the same declared inference budget for comparisons, or label differing budgets as different identities.
- Initial Glicko-2 placement is one simultaneous batch from rating 1500, deviation 350, volatility .06, tau .5. Partial standings are explicitly marked incomplete. Ratings compare only this pool and do not establish human Elo. No ongoing live-rating phase is implemented.

## Artifacts and measurement

`manifest.json` freezes the protocol, prompt, configuration, and code hashes. Each `game-NNNNN.json` includes the opening, accepted move history, result, termination, seed, manifest hash, final-state hash, and record hash. Its JSONL contains state snapshots, legal-action counts, replies, usage, latency, and accepted/rejected attempts. Logs include model-returned content and should be treated as experiment data.

`standings.json` gives completed/scheduled game denominators, a combined result hash, ratings/deviations, wins/draws/losses, answered/rejected attempts, request failures, request seconds, tokens, and provider-reported USD cost with a priced-attempt denominator. Missing pricing is not zero actual cost; token-price estimates are not fabricated. Request totals include failed/interrupted game attempts. Local bot selection time is included in its request timing.

`--audit` checks hashes, snapshot integrity, legal-list membership, game counts, and final positions replayed through the engine. This is an integrity/replay check, not proof of model quality. Two games per pair are a smoke test, not a statistically reliable ranking. Freeze a sample size and opening set before a confirmatory comparison; small differences may need thousands of games. No significance claims or automatic winner selection are made.

Built-in `random` and depth-1 `engine` players support local checks and baselines. Random seeds are reproducible; external API generation may not be.

## Tests

```sh
python3 -m unittest discover -s tests -v
cargo test --workspace --locked
```

Engine provenance and licenses: [UPSTREAM.md](UPSTREAM.md). Vendored engine sources are unchanged from the pinned upstream commit.
