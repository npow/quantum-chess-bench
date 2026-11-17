# Quantum Chess Bench

A Rust CLI for LLM, random, and built-in search quantum chess players. The vendored Rust rules engine generates legal moves and applies accepted actions directly. Model requests use an OpenAI-compatible Chat Completions endpoint with a required `submit_move` tool call.

## Run

Requires Rust 1.99+. Copy `models.example.json` to `models.json`, set provider model IDs, and export the API key named in each player's `api_key_env`:

```sh
cp models.example.json models.json
export OPENROUTER_API_KEY=...
cargo run --release -- models.json runs/first
```

The endpoint defaults to OpenRouter. Set `base_url` per player for a compatible endpoint. Each player can set `kind` to `llm` (default), `random`, or `engine`; the latter uses the built-in search player and does not need an API key. Config supports reproducible `seed`, `openings`, `pairs`, `max_plies`, `max_attempts`, `timeout_seconds`, and `context_chars` settings. The CLI arguments are config path and output directory.

Each pair plays once with each color. Invalid actions retry up to `max_attempts`; exhaustion forfeits. Checkmate, stalemate, threefold repetition, and the move limit end games. API errors stop the run and are logged as incomplete attempts, never as forfeits. Each game has a JSON record and JSONL turn log. `manifest.json` hashes the configuration, prompt, runner, and engine sources. Re-running resumes completed games after integrity checks; changed configuration or code requires a new output directory. `--audit <output-dir>` replays and checks saved records; `--position` prints a sample quantum position.

## State and rules

Each model turn receives the board rendering, move history, side to move, check state, split-piece summary, possible-board count, and all legal quantum move strings, including collapse choices. See [RULES.md](RULES.md) for the rules implemented by the vendored engine.

## Verification status

The repository has seven end-to-end tests and no unit tests. `cargo test --locked` runs them against a local mock provider or the built-in players. `cargo test --workspace --locked` also builds the vendored Rust crates.

Seventy-seven Kani harnesses prove bounded properties of the runner, notation, board engine, quantum collapse search, and rating code. The collapse proofs exhaust every one-split and two-split bad-state truth table and verify that the emitted masks are exactly the smallest valid cubes. Other harnesses cover single-byte move membership, config validation predicates for player kinds, credentials, names, models, parameters, openings, and positive integers, including typed config and player records, the config-facts boundary, composed runtime numeric projection, and top-level error precedence, player validation, all five numeric options, opening structure, and LLM field-error precedence, opponent enumeration, cyclic opening selection, successful-result gating, color-swapped assignments, and exact pair/repetition coverage for schedules with up to five players and four repetitions; notation parsing and formatting primitives; slider rays for every square, direction, and blocker mask; board index/key updates after placing and removing any piece kind on any square, double pawn pushes for either color and every file, promotion, en passant, and castling; the `Position::states` output for fixed one-, two-, and three-split positions, plus direct production-vector checks for one- and two-split positions; the production state visitor dispatching every five-split mask exactly once and in order for every validity pattern; the production selection-to-board helper for five split pieces and one unsplit piece with symbolic square locations, live/dead halves, arbitrary colors and piece kinds, and all 32 masks; successor-mask classification against every reachable-key subset for three valid boards; candidate selection against an independent uniqueness specification for every active subset of 64 piece IDs; `apply_plan` moving both halves and marking a captured piece dead; dead-half handling in board construction including out-of-range remembered squares; allocation-free selection checks for two pieces including dead halves and three pieces with symbolic squares across all masks and an unsplit blocker; occupancy behavior for dead and out-of-range halves; rejection of castling through check; color, piece-letter, and square-coordinate mappings; Glicko outcome scores and clock-speed thresholds for all u32 clock inputs; and preservation of rating and volatility over three fixed empty-period inputs. Kani over-approximates floating-point `sqrt`, so that last harness does not prove the deviation-growth result.

Run the general proofs with:

```sh
cargo kani --harness move_gate_accepts_only_listed_actions
cargo kani --harness verification::player_config_validation_obeys_error_precedence_for_every_field_combination
cargo kani --harness verification::config_validation_preserves_top_level_error_precedence
cargo kani --harness verification::config_facts_combine_player_numeric_and_opening_checks
cargo kani --harness verification::llm_field_validation_matches_ordered_requirements
cargo kani --harness verification::color_swapped_schedule_pairs_reverse_every_assignment
cargo kani --harness verification::schedule_opponent_enumerates_each_unordered_player_pair
cargo kani --harness verification::schedule_opponent_maps_every_valid_offset_into_the_remaining_players
cargo kani --harness verification::optional_positive_integer_matches_presence_and_range_spec
cargo kani --harness verification::numeric_config_options_reject_only_first_present_invalid_value
cargo kani --harness verification::typed_numeric_projection_preserves_defaults_and_present_values
cargo kani --harness verification::runtime_options_projection_combines_all_settings
cargo kani --harness verification::inline_credential_rejection_matches_field_presence
cargo kani --harness verification::player_name_uniqueness_matches_exact_bytes_for_two_prior_names
cargo kani --harness verification::three_byte_player_kind_parser_recognizes_only_llm
cargo kani --harness verification::six_byte_player_kind_parser_recognizes_random_and_engine
cargo kani --harness verification::player_kind_parser_defaults_to_llm_and_rejects_other_lengths
cargo kani --harness verification::api_key_environment_selection_and_missing_key_result_are_exact
cargo kani --harness verification::required_model_length_rejects_only_missing_or_empty_strings
cargo kani --harness verification::optional_parameters_accept_absence_or_object_values
cargo kani --harness verification::opening_array_requires_positive_length
cargo kani --harness verification::opening_entry_requires_an_array_containing_only_strings
cargo kani --harness verification::openings_config_validation_matches_structure_precedence
cargo kani --harness verification::schedule_pair_visitation_matches_ordered_pair_repetitions
cargo kani --harness verification::opening_selection_is_cyclic_and_stays_in_bounds
cargo kani --harness verification::success_gate_runs_record_write_only_for_completed_game
cargo kani --package qchess --harness placing_then_removing_one_piece_restores_empty_board_indexes
cargo kani --package qchess --harness slider_ray_stops_on_first_blocker
cargo kani --package qchess --harness color_piece_and_square_mappings_round_trip
cargo kani --package qchess --harness legal_double_pawn_push_preserves_board_indexes_and_en_passant
cargo kani --package qchess --harness legal_kingside_castle_moves_both_pieces_and_preserves_indexes
cargo kani --package qchess --harness kingside_castling_is_denied_when_a_transit_square_is_attacked
cargo kani --package qchess --harness legal_en_passant_capture_removes_the_pawn_from_its_actual_square
cargo kani --package qchess --harness legal_pawn_promotion_updates_kind_and_all_board_indexes
cargo kani --package qchess --harness states_enumerates_both_halves_of_one_split_piece
cargo kani --package qchess --harness states_classifies_two_split_piece_collisions_exactly
cargo kani --package qchess --harness states_builds_exact_boards_for_three_piece_position
cargo kani --package qchess --harness board_for_selection_matches_symbolic_five_piece_spec
cargo kani --package qchess --harness board_for_selection_ignores_dead_half_square
cargo kani --package qchess --harness state_visitor_dispatches_all_five_split_masks_in_order
cargo kani --package qchess --harness one_split_board_builder_matches_unsplit_blocker_model
cargo kani --package qchess --harness one_split_states_materializes_both_boards_with_real_board_builder
cargo kani --package qchess --harness two_split_states_separates_valid_and_colliding_boards
cargo kani --package qchess --harness apply_plan_moves_both_halves_and_marks_captured_piece_dead
cargo kani --package qchess --harness successor_classifier_matches_every_three_key_reachability_pattern
cargo kani --package qchess --harness candidate_accumulator_matches_pairwise_unique_piece_spec
cargo kani --package qchess --harness symbolic_three_piece_selection_matches_pairwise_collision_spec
cargo kani --package qchess --harness symbolic_two_piece_selection_matches_square_collision_spec
cargo kani --package qchess --harness symbolic_selection_accounts_for_unsplit_pieces_and_dead_halves
cargo kani --package qchess --harness occupancy_transition_matches_collision_spec_for_all_inputs
cargo kani --package glicko2 --harness outcome_scores_are_bounded_and_flip_is_an_involution
cargo kani --package glicko2 --harness empty_period_preserves_rating_and_volatility
cargo kani --package glicko2 --harness clock_speed_matches_threshold_spec_for_all_u32_inputs
```

Run the bounded collapse truth-table proofs with:

```sh
cargo kani --package qchess --harness all_one_split_truth_tables_match_the_minimal_collapse_spec --default-unwind 5
cargo kani --package qchess --harness two_split_truth_tables_00_to_03_match_the_minimal_collapse_spec --default-unwind 5
cargo kani --package qchess --harness two_split_truth_tables_04_to_07_match_the_minimal_collapse_spec --default-unwind 5
cargo kani --package qchess --harness two_split_truth_tables_08_to_11_match_the_minimal_collapse_spec --default-unwind 5
cargo kani --package qchess --harness two_split_truth_tables_12_to_15_match_the_minimal_collapse_spec --default-unwind 5
```

Formal verification remains incomplete: these proofs do not establish correctness of all chess rules, quantum collapse behavior, the full Glicko update algorithm, network handling, persistence, or the complete tournament. In particular, the Glicko empty-period harness checks three fixed inputs and omits the deviation calculation. Kani's [floating-point model](https://model-checking.github.io/kani/rust-feature-support.html) over-approximates `sqrt`. Records use SHA-256 hashes for corruption detection, but are not signed and do not provide tamper evidence.

See [FORMAL_VERIFICATION.md](FORMAL_VERIFICATION.md) for the proof coverage map and prioritized remaining obligations.

The upstream engine sources and licenses are documented in [UPSTREAM.md](UPSTREAM.md).
