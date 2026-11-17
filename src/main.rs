use qchess::{Check, Color, Game, Status, ai};
use reqwest::blocking::Client;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    env, fs,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const PROTOCOL: &str = "quantum-chess-bench-v1";
const CONFIG_NUMERIC_KEYS: [&str; 5] = [
    "pairs",
    "max_plies",
    "max_attempts",
    "timeout_seconds",
    "context_chars",
];
const SYSTEM_PROMPT: &str = concat!(
    "You play quantum chess. Win using your own reasoning. The engine is the authority. ",
    "Select exactly one string from legal_moves, including any collapse suffix. ",
    "Use submit_move to submit exactly one legal move. Do not use external engines.\n\n",
    include_str!("../RULES.md")
);

#[derive(Clone)]
struct ScheduledGame {
    white: usize,
    black: usize,
    opening: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlayerKind {
    Llm,
    Random,
    Engine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LlmFieldError {
    None,
    MissingModel,
    MissingApiKey,
    InvalidParameters,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlayerConfigError {
    MissingName,
    InlineCredentials,
    DuplicateName,
    UnsupportedKind,
    LlmFields(LlmFieldError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpeningsConfigError {
    NotArray,
    Empty,
    InvalidEntry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConfigValidationError {
    MissingPlayers,
    TooFewPlayers,
    Player {
        index: usize,
        error: PlayerConfigError,
    },
    PlayerIndexOutOfRange(usize),
    NumericOption(usize),
    Openings(OpeningsConfigError),
}

#[derive(Clone, Copy)]
struct ConfigValidationFacts {
    players_present: bool,
    player_count: usize,
    first_player_error: Option<(usize, PlayerConfigError)>,
    numeric_options: [Option<Option<u64>>; 5],
    openings_is_null: bool,
    openings_is_array: bool,
    openings_has_entries: bool,
    all_opening_entries_are_valid: bool,
}

struct ValidatedRuntimeOptions {
    pair_repetitions: usize,
    max_plies: usize,
    max_attempts: usize,
    timeout_seconds: u64,
    context_chars: usize,
    seed: u64,
}

struct ValidatedPlayer {
    name: String,
    kind: PlayerKind,
    model: String,
    api_key_env: String,
    base_url: String,
    parameters: Map<String, Value>,
}

struct ValidatedConfig {
    players: Vec<ValidatedPlayer>,
    runtime: ValidatedRuntimeOptions,
    openings: Vec<Vec<String>>,
}

fn llm_field_error(
    model_is_nonempty: bool,
    api_key_is_set: bool,
    parameters_are_valid: bool,
) -> LlmFieldError {
    if !model_is_nonempty {
        LlmFieldError::MissingModel
    } else if !api_key_is_set {
        LlmFieldError::MissingApiKey
    } else if !parameters_are_valid {
        LlmFieldError::InvalidParameters
    } else {
        LlmFieldError::None
    }
}

fn player_config_error(
    name_is_nonempty: bool,
    has_inline_credentials: bool,
    name_is_unique: bool,
    kind: Option<PlayerKind>,
    llm_error: LlmFieldError,
) -> Option<PlayerConfigError> {
    if !name_is_nonempty {
        Some(PlayerConfigError::MissingName)
    } else if has_inline_credentials {
        Some(PlayerConfigError::InlineCredentials)
    } else if !name_is_unique {
        Some(PlayerConfigError::DuplicateName)
    } else {
        match kind {
            None => Some(PlayerConfigError::UnsupportedKind),
            Some(PlayerKind::Llm) if llm_error != LlmFieldError::None => {
                Some(PlayerConfigError::LlmFields(llm_error))
            }
            Some(PlayerKind::Llm | PlayerKind::Random | PlayerKind::Engine) => None,
        }
    }
}

fn color_swapped_pair(first: usize, second: usize) -> [(usize, usize); 2] {
    [(first, second), (second, first)]
}

fn schedule_opponent(player_count: usize, white: usize, offset: usize) -> Option<usize> {
    let remaining = player_count.checked_sub(white)?.checked_sub(1)?;
    (offset < remaining).then(|| white + 1 + offset)
}

fn visit_schedule_pairs<E>(
    player_count: usize,
    pair_repetitions: usize,
    mut visit: impl FnMut(usize, usize, usize) -> Result<(), E>,
) -> Result<(), E> {
    for white in 0..player_count {
        let mut opponent_offset = 0;
        while let Some(black) = schedule_opponent(player_count, white, opponent_offset) {
            for repetition in 0..pair_repetitions {
                visit(white, black, repetition)?;
            }
            opponent_offset += 1;
        }
    }
    Ok(())
}

fn opening_index(repetition: usize, opening_count: usize) -> Option<usize> {
    (opening_count > 0).then(|| repetition % opening_count)
}

fn execute_on_success<T, E, R>(
    result: Result<T, E>,
    action: impl FnOnce(T) -> Result<R, E>,
) -> Result<Result<R, E>, E> {
    result.map(action)
}

fn optional_positive_integer(value: Option<Option<u64>>) -> bool {
    match value {
        None => true,
        Some(Some(number)) => number > 0,
        Some(None) => false,
    }
}

fn positive_u64_or_default(value: Option<Option<u64>>, default: u64) -> Option<u64> {
    match value {
        None => Some(default),
        Some(Some(0) | None) => None,
        Some(Some(number)) => Some(number),
    }
}

fn optional_u64_or_default(value: Option<Option<u64>>, default: u64) -> u64 {
    value.flatten().unwrap_or(default)
}

fn project_runtime_options(
    numeric_options: [Option<Option<u64>>; 5],
    seed: Option<Option<u64>>,
) -> Option<ValidatedRuntimeOptions> {
    Some(ValidatedRuntimeOptions {
        pair_repetitions: positive_u64_or_default(numeric_options[0], 1)? as usize,
        max_plies: positive_u64_or_default(numeric_options[1], 150)? as usize,
        max_attempts: positive_u64_or_default(numeric_options[2], 3)? as usize,
        timeout_seconds: positive_u64_or_default(numeric_options[3], 120)?,
        context_chars: positive_u64_or_default(numeric_options[4], 60_000)? as usize,
        seed: optional_u64_or_default(seed, 1),
    })
}

fn first_invalid_numeric_option(options: [Option<Option<u64>>; 5]) -> Option<usize> {
    options
        .into_iter()
        .position(|option| !optional_positive_integer(option))
}

fn player_name_is_unique<'a>(
    candidate: &[u8],
    mut prior_names: impl Iterator<Item = &'a [u8]>,
) -> bool {
    prior_names.all(|prior| prior != candidate)
}

fn has_inline_api_credentials(api_key_present: bool, api_key_value_present: bool) -> bool {
    api_key_present || api_key_value_present
}

fn parse_player_kind(kind: Option<&[u8]>) -> Option<PlayerKind> {
    match kind {
        None | Some(b"llm") => Some(PlayerKind::Llm),
        Some(b"random") => Some(PlayerKind::Random),
        Some(b"engine") => Some(PlayerKind::Engine),
        Some(_) => None,
    }
}

fn api_key_environment_name(configured: Option<&str>) -> &str {
    configured.unwrap_or("OPENROUTER_API_KEY")
}

fn player_base_url(configured: Option<&str>) -> &str {
    configured.unwrap_or("https://openrouter.ai/api/v1")
}

fn missing_environment_key(name: &str, is_present: bool) -> Option<&str> {
    (!is_present).then_some(name)
}

fn required_nonempty_string(length: Option<usize>) -> bool {
    length.is_some_and(|length| length > 0)
}

fn optional_object_value(is_present: bool, is_object: bool) -> bool {
    !is_present || is_object
}

fn opening_array_length_is_valid(length: usize) -> bool {
    length > 0
}

fn opening_entry_is_valid(is_array: bool, all_moves_are_strings: bool) -> bool {
    is_array && all_moves_are_strings
}

fn openings_config_error(
    is_null: bool,
    is_array: bool,
    has_entries: bool,
    all_entries_are_valid: bool,
) -> Option<OpeningsConfigError> {
    if is_null {
        None
    } else if !is_array {
        Some(OpeningsConfigError::NotArray)
    } else if !has_entries {
        Some(OpeningsConfigError::Empty)
    } else if !all_entries_are_valid {
        Some(OpeningsConfigError::InvalidEntry)
    } else {
        None
    }
}

fn config_validation_error(
    players_present: bool,
    player_count: usize,
    first_player_error: Option<(usize, PlayerConfigError)>,
    first_numeric_error: Option<usize>,
    openings_error: Option<OpeningsConfigError>,
) -> Option<ConfigValidationError> {
    if !players_present {
        Some(ConfigValidationError::MissingPlayers)
    } else if player_count < 2 {
        Some(ConfigValidationError::TooFewPlayers)
    } else if let Some((index, error)) = first_player_error {
        if index < player_count {
            Some(ConfigValidationError::Player { index, error })
        } else {
            Some(ConfigValidationError::PlayerIndexOutOfRange(index))
        }
    } else if let Some(index) = first_numeric_error {
        Some(ConfigValidationError::NumericOption(index))
    } else {
        openings_error.map(ConfigValidationError::Openings)
    }
}

fn validate_config_facts(facts: ConfigValidationFacts) -> Option<ConfigValidationError> {
    let first_numeric_error = first_invalid_numeric_option(facts.numeric_options);
    let opening_error = openings_config_error(
        facts.openings_is_null,
        facts.openings_is_array,
        facts.openings_has_entries,
        facts.all_opening_entries_are_valid,
    );
    config_validation_error(
        facts.players_present,
        facts.player_count,
        facts.first_player_error,
        first_numeric_error,
        opening_error,
    )
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash(value: &Value) -> String {
    hash_bytes(&serde_json::to_vec(value).expect("JSON values always serialize"))
}

fn write_json(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn write_event(log: &mut File, event: &Value) -> Result<(), Box<dyn std::error::Error>> {
    serde_json::to_writer(&mut *log, event)?;
    log.write_all(b"\n")?;
    log.flush()?;
    Ok(())
}

fn status_name(status: &Status) -> &'static str {
    match status {
        Status::Ongoing => "ongoing",
        Status::Stalemate => "stalemate",
        Status::Checkmate {
            winner: Color::White,
        } => "white_win",
        Status::Checkmate {
            winner: Color::Black,
        } => "black_win",
    }
}

fn state(game: &Game) -> Value {
    let mut legal: Vec<String> = if game.status() == Status::Ongoing {
        ai::children(game, true)
            .into_iter()
            .map(|child| child.notation)
            .collect()
    } else {
        Vec::new()
    };
    legal.sort();
    legal.dedup();
    let pieces = game
        .pieces()
        .into_iter()
        .map(|piece| {
            json!({
                "id": piece.id,
                "color": piece.color.name(),
                "split": piece.split,
                "origin": piece.origin.name(),
                "halves": piece.halves.into_iter().map(|half| json!({
                    "square": half.square,
                    "kind": half.kind.name(),
                    "alive": half.alive,
                })).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let check = match game.check() {
        Check::None => "none",
        Check::Quantum => "quantum",
        Check::Full => "full",
    };
    json!({
        "status": status_name(&game.status()),
        "to_move": game.to_move().name(),
        "check": check,
        "board": game.board_text(),
        "pieces": pieces,
        "legal_moves": legal,
        "state_count": game.state_count(),
        "position_key": format!("{:016x}", game.position_key()),
        "history": game.history(),
        "possible_boards_fen": game.states_fen(),
    })
}

fn parse_move(message: &Value) -> Option<String> {
    if let Some(calls) = message["tool_calls"].as_array() {
        if calls.len() != 1 || calls[0]["function"]["name"] != "submit_move" {
            return None;
        }
        let arguments = calls[0]["function"]["arguments"].as_str()?;
        return serde_json::from_str::<Value>(arguments)
            .ok()?
            .get("move")?
            .as_str()
            .map(str::to_owned);
    }
    let content = message["content"].as_str()?;
    if let Ok(value) = serde_json::from_str::<Value>(content) {
        if let Some(movement) = value["move"].as_str() {
            return Some(movement.to_owned());
        }
    }
    let content = content.trim();
    (!content.is_empty()).then(|| content.to_owned())
}

fn call(
    client: &Client,
    player: &ValidatedPlayer,
    messages: &[Value],
    timeout: u64,
) -> Result<(Value, Value), String> {
    let key_name = &player.api_key_env;
    let key = env::var(key_name).map_err(|_| format!("missing environment variable {key_name}"))?;
    let mut body = player.parameters.clone();
    body.insert("model".into(), json!(player.model));
    body.insert("messages".into(), json!(messages));
    body.insert("stream".into(), json!(false));
    body.insert(
        "tools".into(),
        json!([{"type":"function","function":{"name":"submit_move","description":"Submit exactly one move from legal_moves.","parameters":{"type":"object","properties":{"move":{"type":"string"}},"required":["move"],"additionalProperties":false}}}]),
    );
    body.insert(
        "tool_choice".into(),
        json!({"type":"function","function":{"name":"submit_move"}}),
    );
    let response: Value = client
        .post(format!("{}/chat/completions", player.base_url))
        .bearer_auth(key)
        .json(&Value::Object(body))
        .timeout(Duration::from_secs(timeout))
        .send()
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?
        .json()
        .map_err(|error| error.to_string())?;
    let message = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .filter(|message| message.is_object())
        .ok_or_else(|| "provider response is missing choices[0].message".to_owned())?;
    Ok((message.clone(), response["usage"].clone()))
}

fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn validate_config(config: &Value) -> Result<ValidatedConfig, String> {
    let players = config["players"].as_array();
    let player_count = players.map_or(0, Vec::len);
    let mut names: Vec<String> = Vec::new();
    let mut first_player_error = None;
    if let Some(players) = players.filter(|_| player_count >= 2) {
        for (index, player) in players.iter().enumerate() {
            let name = player["name"].as_str().filter(|name| !name.is_empty());
            let inline_credentials = has_inline_api_credentials(
                player.get("api_key").is_some(),
                player.get("api_key_value").is_some(),
            );
            let name_is_unique = player_name_is_unique(
                name.unwrap_or_default().as_bytes(),
                names.iter().map(|existing| existing.as_bytes()),
            );
            let kind = parse_player_kind(player["kind"].as_str().map(str::as_bytes));
            let should_check_llm_fields = name.is_some()
                && !inline_credentials
                && name_is_unique
                && kind == Some(PlayerKind::Llm);
            let llm_error = if should_check_llm_fields {
                let key_name = api_key_environment_name(player["api_key_env"].as_str());
                let parameters = player.get("parameters");
                llm_field_error(
                    required_nonempty_string(player["model"].as_str().map(str::len)),
                    missing_environment_key(key_name, env::var_os(key_name).is_some()).is_none(),
                    optional_object_value(
                        parameters.is_some(),
                        parameters.is_some_and(Value::is_object),
                    ),
                )
            } else {
                LlmFieldError::None
            };
            if let Some(error) = player_config_error(
                name.is_some(),
                inline_credentials,
                name_is_unique,
                kind,
                llm_error,
            ) {
                first_player_error = Some((index, error));
                break;
            }
            names.push(name.expect("validated names are non-empty").to_owned());
        }
    }
    let numeric_options = CONFIG_NUMERIC_KEYS.map(|key| config.get(key).map(Value::as_u64));
    let openings = config.get("openings").unwrap_or(&Value::Null);
    let opening_list = openings.as_array();
    let all_entries_are_valid = opening_list.is_some_and(|entries| {
        entries.iter().all(|opening| {
            let moves = opening.as_array();
            opening_entry_is_valid(
                moves.is_some(),
                moves.is_some_and(|moves| moves.iter().all(Value::is_string)),
            )
        })
    });
    let validation_error = validate_config_facts(ConfigValidationFacts {
        players_present: players.is_some(),
        player_count,
        first_player_error,
        numeric_options,
        openings_is_null: openings.is_null(),
        openings_is_array: opening_list.is_some(),
        openings_has_entries: opening_list
            .is_some_and(|entries| opening_array_length_is_valid(entries.len())),
        all_opening_entries_are_valid: all_entries_are_valid,
    });
    match validation_error {
        None => {}
        Some(ConfigValidationError::MissingPlayers) => {
            return Err("missing players array".into());
        }
        Some(ConfigValidationError::TooFewPlayers) => {
            return Err("provide at least two players".into());
        }
        Some(ConfigValidationError::Player { index, error }) => {
            let player = &players.expect("validated player arrays are present")[index];
            let name = player["name"].as_str().unwrap_or_default();
            match error {
                PlayerConfigError::MissingName => {
                    return Err("every player needs a non-empty name".into());
                }
                PlayerConfigError::InlineCredentials => {
                    return Err("store API credentials in environment variables and reference them with api_key_env".into());
                }
                PlayerConfigError::DuplicateName => {
                    return Err("player names must be unique".into());
                }
                PlayerConfigError::UnsupportedKind => {
                    return Err(format!("unsupported player kind for {name}"));
                }
                PlayerConfigError::LlmFields(LlmFieldError::MissingModel) => {
                    return Err(format!("LLM player {name} requires a model"));
                }
                PlayerConfigError::LlmFields(LlmFieldError::MissingApiKey) => {
                    let key_name = api_key_environment_name(player["api_key_env"].as_str());
                    return Err(format!("missing environment variable {key_name}"));
                }
                PlayerConfigError::LlmFields(LlmFieldError::InvalidParameters) => {
                    return Err(format!("parameters for {name} must be an object"));
                }
                PlayerConfigError::LlmFields(LlmFieldError::None) => {
                    unreachable!("successful LLM fields cannot be a validation error");
                }
            }
        }
        Some(ConfigValidationError::PlayerIndexOutOfRange(_)) => {
            return Err("internal validator produced an out-of-range player index".into());
        }
        Some(ConfigValidationError::NumericOption(index)) => {
            let key = CONFIG_NUMERIC_KEYS[index];
            return Err(format!("{key} must be a positive integer"));
        }
        Some(ConfigValidationError::Openings(OpeningsConfigError::NotArray)) => {
            return Err("openings must be an array".into());
        }
        Some(ConfigValidationError::Openings(OpeningsConfigError::Empty)) => {
            return Err("openings must not be empty".into());
        }
        Some(ConfigValidationError::Openings(OpeningsConfigError::InvalidEntry)) => {
            return Err("every opening must be an array of move strings".into());
        }
    }
    let raw_players = players.expect("successful configuration has a players array");
    let players = raw_players
        .iter()
        .map(|player| ValidatedPlayer {
            name: player["name"]
                .as_str()
                .expect("validated player names are non-empty")
                .to_owned(),
            kind: parse_player_kind(player["kind"].as_str().map(str::as_bytes))
                .expect("validated player kinds are supported"),
            model: player["model"].as_str().unwrap_or_default().to_owned(),
            api_key_env: api_key_environment_name(player["api_key_env"].as_str()).to_owned(),
            base_url: player_base_url(player["base_url"].as_str())
                .trim_end_matches('/')
                .to_owned(),
            parameters: player["parameters"]
                .as_object()
                .cloned()
                .unwrap_or_default(),
        })
        .collect();
    let runtime = project_runtime_options(numeric_options, config.get("seed").map(Value::as_u64))
        .expect("invalid numeric options already failed validation");
    let openings = if openings.is_null() {
        vec![Vec::new()]
    } else {
        opening_list
            .expect("successful openings value is null or an array")
            .iter()
            .map(|opening| {
                opening
                    .as_array()
                    .expect("validated opening is an array")
                    .iter()
                    .map(|movement| {
                        movement
                            .as_str()
                            .expect("validated opening move is a string")
                            .to_owned()
                    })
                    .collect()
            })
            .collect()
    };
    Ok(ValidatedConfig {
        players,
        runtime,
        openings,
    })
}

fn make_schedule(config: &ValidatedConfig) -> Result<Vec<ScheduledGame>, String> {
    let openings = &config.openings;
    let mut schedule = Vec::new();
    visit_schedule_pairs(
        config.players.len(),
        config.runtime.pair_repetitions,
        |white, black, repetition| {
            let opening_number = opening_index(repetition, openings.len())
                .expect("openings validated or defaulted to one empty opening");
            let opening = openings[opening_number].clone();
            let mut validation_game = Game::new();
            for movement in &opening {
                validation_game
                    .play(movement)
                    .map_err(|error| format!("invalid opening {movement:?}: {error}"))?;
            }
            let [(white_a, black_a), (white_b, black_b)] = color_swapped_pair(white, black);
            schedule.push(ScheduledGame {
                white: white_a,
                black: black_a,
                opening: opening.clone(),
            });
            schedule.push(ScheduledGame {
                white: white_b,
                black: black_b,
                opening,
            });
            Ok::<(), String>(())
        },
    )?;
    Ok(schedule)
}

fn context_size(messages: &[Value]) -> usize {
    serde_json::to_vec(messages).map_or(usize::MAX, |bytes| bytes.len())
}

#[allow(clippy::too_many_arguments)]
fn play_game(
    client: &Client,
    players: &[ValidatedPlayer],
    config: &ValidatedConfig,
    scheduled: &ScheduledGame,
    game_number: usize,
    manifest_hash: &str,
    log_path: &Path,
) -> Result<Value, Box<dyn std::error::Error>> {
    let white = &players[scheduled.white];
    let black = &players[scheduled.black];
    let players_by_color = [white, black];
    let max_plies = config.runtime.max_plies;
    let max_attempts = config.runtime.max_attempts;
    let timeout = config.runtime.timeout_seconds;
    let context_chars = config.runtime.context_chars;
    let seed = config.runtime.seed.wrapping_add(game_number as u64);
    let mut random_state = seed;
    let mut game = Game::new();
    for movement in &scheduled.opening {
        game.play(movement)?;
    }
    let mut chats = [Vec::<Value>::new(), Vec::<Value>::new()];
    let mut records = Vec::new();
    let mut seen = HashMap::<String, usize>::new();
    let (termination_name, score);
    let final_state;
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    loop {
        let current_state = state(&game);
        match current_state["status"].as_str().unwrap_or("") {
            "white_win" => {
                termination_name = "white_win";
                score = 1.0;
                final_state = current_state;
                break;
            }
            "black_win" => {
                termination_name = "black_win";
                score = 0.0;
                final_state = current_state;
                break;
            }
            "stalemate" => {
                termination_name = "stalemate";
                score = 0.5;
                final_state = current_state;
                break;
            }
            _ => {}
        }
        let position_key = current_state["position_key"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        let repetitions = seen.entry(position_key).or_default();
        *repetitions += 1;
        if *repetitions >= 3 {
            termination_name = "threefold_repetition";
            score = 0.5;
            final_state = current_state;
            break;
        }
        if game.history().len().saturating_sub(scheduled.opening.len()) >= max_plies {
            termination_name = "move_limit";
            score = 0.5;
            final_state = current_state;
            break;
        }

        let side = usize::from(game.to_move() == Color::Black);
        let player = players_by_color[side];
        let mut chat = std::mem::take(&mut chats[side]);
        while context_size(&chat) > context_chars && chat.len() >= 2 {
            chat.drain(..2);
        }
        chat.push(json!({"role":"user","content":current_state.to_string()}));
        let legal = current_state["legal_moves"].as_array().unwrap();
        if legal.is_empty() {
            return Err("ongoing game state has no legal moves".into());
        }
        let mut accepted = false;

        for attempt in 1..=max_attempts {
            let started = Instant::now();
            let response = match player.kind {
                PlayerKind::Random => {
                    let index = (next_random(&mut random_state) % legal.len() as u64) as usize;
                    Ok((
                        json!({"role":"assistant","content":legal[index]}),
                        json!({}),
                    ))
                }
                PlayerKind::Engine => {
                    let choice = ai::best_move(
                        &game,
                        ai::Config {
                            depth: 1,
                            time: Some(Duration::from_millis(500)),
                            ..Default::default()
                        },
                    );
                    Ok((
                        choice.map_or(
                            Value::Null,
                            |choice| json!({"role":"assistant","content":choice.notation}),
                        ),
                        json!({}),
                    ))
                }
                PlayerKind::Llm => {
                    let mut messages = vec![json!({"role":"system","content":SYSTEM_PROMPT})];
                    messages.extend(chat.clone());
                    call(client, player, &messages, timeout)
                }
            };
            let (message, usage) = match response {
                Ok(response) => response,
                Err(error) => {
                    let event = json!({
                        "ply": game.history().len(), "player": player.name, "attempt": attempt,
                        "state_hash": hash(&current_state), "state": current_state,
                        "legal_count": legal.len(), "error": "request_failed",
                        "error_detail": error, "elapsed_seconds": started.elapsed().as_secs_f64(),
                    });
                    write_event(&mut log, &event)?;
                    return Err(format!("request failed for {}", player.name).into());
                }
            };
            let movement = parse_move(&message);
            let legal_bytes = legal
                .iter()
                .filter_map(Value::as_str)
                .map(str::as_bytes)
                .collect::<Vec<_>>();
            let is_legal = movement
                .as_ref()
                .is_some_and(|candidate| is_legal_move(candidate.as_bytes(), &legal_bytes));
            chat.push(json!({
                "role":"assistant",
                "content":message["content"].as_str().map(str::to_owned)
                    .unwrap_or_else(|| serde_json::to_string(&message["tool_calls"]).unwrap_or_default()),
            }));
            let event = json!({
                "ply": game.history().len(), "player": player.name, "attempt": attempt,
                "state_hash": hash(&current_state), "legal_count": legal.len(),
                "state": current_state, "response": message, "usage": usage,
                "move": movement, "accepted": is_legal,
                "elapsed_seconds": started.elapsed().as_secs_f64(),
            });
            write_event(&mut log, &event)?;
            records.push(event);
            if is_legal {
                let movement = movement.expect("legality implies a move");
                game.play(&movement)?;
                accepted = true;
                break;
            }
            chat.push(json!({
                "role":"user",
                "content":format!("Rejected move. Attempt {attempt}. Board unchanged; opponent has not moved. Select from legal_moves."),
            }));
        }
        chats[side] = chat;
        if !accepted {
            termination_name = "invalid_move_forfeit";
            score = if side == 0 { 0.0 } else { 1.0 };
            final_state = state(&game);
            break;
        }
    }

    Ok(json!({
        "game": game_number, "manifest_hash": manifest_hash,
        "white": white.name, "black": black.name, "score": score,
        "termination": termination_name, "opening": scheduled.opening,
        "moves": game.history(), "seed": seed,
        "final_state_hash": hash(&final_state), "records": records,
    }))
}

fn ratings(players: &[ValidatedPlayer], results: &[Value]) -> Vec<Value> {
    let mut rows = players.iter().map(|player| {
        let name = player.name.as_str();
        let games: Vec<(glicko2::Rating, glicko2::Outcome)> = results.iter().filter_map(|result| {
            let score = result["score"].as_f64()?;
            let score = if result["white"].as_str() == Some(name) {
                score
            } else if result["black"].as_str() == Some(name) {
                1.0 - score
            } else {
                return None;
            };
            let outcome = if score == 1.0 {
                glicko2::Outcome::Win
            } else if score == 0.0 {
                glicko2::Outcome::Loss
            } else {
                glicko2::Outcome::Draw
            };
            Some((glicko2::Rating::default(), outcome))
        }).collect();
        let rating = glicko2::update(glicko2::Rating::default(), &games, &glicko2::Config {
            tau: 0.5, min_deviation: 0.0, max_deviation: 350.0,
        });
        json!({"name":name,"rating":rating.rating,"deviation":rating.deviation,"games":games.len()})
    }).collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right["rating"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&left["rating"].as_f64().unwrap_or(0.0))
    });
    rows
}

fn metrics(
    output: &Path,
    players: &[ValidatedPlayer],
    results: &[Value],
) -> Result<Value, Box<dyn std::error::Error>> {
    let mut rows = Map::new();
    for player in players {
        let name = &player.name;
        rows.insert(
            name.to_owned(),
            json!({
                "wins":0,"draws":0,"losses":0,"answered_attempts":0,"rejected_attempts":0,
                "request_failures":0,"request_seconds":0.0,"reported_tokens":0,
                "reported_cost_usd":0.0,"priced_attempts":0,
            }),
        );
    }
    for result in results {
        for (name, score) in [
            (
                result["white"].as_str().unwrap_or(""),
                result["score"].as_f64().unwrap_or(0.5),
            ),
            (
                result["black"].as_str().unwrap_or(""),
                1.0 - result["score"].as_f64().unwrap_or(0.5),
            ),
        ] {
            let column = if score == 1.0 {
                "wins"
            } else if score == 0.0 {
                "losses"
            } else {
                "draws"
            };
            if let Some(value) = rows.get_mut(name).and_then(Value::as_object_mut) {
                value[column] = json!(value[column].as_u64().unwrap_or(0) + 1);
            }
        }
    }
    for entry in fs::read_dir(output)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl") {
            continue;
        }
        let file = File::open(path)?;
        for line in BufReader::new(file).lines() {
            let event: Value = serde_json::from_str(&line?)?;
            let name = event["player"].as_str().unwrap_or("");
            let Some(row) = rows.get_mut(name).and_then(Value::as_object_mut) else {
                continue;
            };
            row["request_seconds"] = json!(
                row["request_seconds"].as_f64().unwrap_or(0.0)
                    + event["elapsed_seconds"].as_f64().unwrap_or(0.0)
            );
            if event.get("error").is_some() {
                row["request_failures"] = json!(row["request_failures"].as_u64().unwrap_or(0) + 1);
                continue;
            }
            row["answered_attempts"] = json!(row["answered_attempts"].as_u64().unwrap_or(0) + 1);
            if event["accepted"] == false {
                row["rejected_attempts"] =
                    json!(row["rejected_attempts"].as_u64().unwrap_or(0) + 1);
            }
            row["reported_tokens"] = json!(
                row["reported_tokens"].as_u64().unwrap_or(0)
                    + event["usage"]["total_tokens"].as_u64().unwrap_or(0)
            );
            if let Some(cost) = event["usage"]["cost"].as_f64() {
                row["reported_cost_usd"] =
                    json!(row["reported_cost_usd"].as_f64().unwrap_or(0.0) + cost);
                row["priced_attempts"] = json!(row["priced_attempts"].as_u64().unwrap_or(0) + 1);
            }
        }
    }
    Ok(Value::Object(rows))
}

fn checked_record(mut record: Value, manifest_hash: &str) -> Result<Value, String> {
    let expected = record["record_hash"]
        .as_str()
        .ok_or("saved game has no record hash")?
        .to_owned();
    record
        .as_object_mut()
        .ok_or("saved game is not an object")?
        .remove("record_hash");
    if hash(&record) != expected || record["manifest_hash"] != manifest_hash {
        return Err("saved game failed integrity or manifest check".into());
    }
    if let Some(events) = record["records"].as_array() {
        for event in events {
            if event.get("state").is_some() && hash(&event["state"]) != event["state_hash"] {
                return Err("saved turn state failed hash check".into());
            }
            if event["accepted"] == true
                && !event["state"]["legal_moves"]
                    .as_array()
                    .is_some_and(|moves| moves.iter().any(|movement| movement == &event["move"]))
            {
                return Err("accepted move was not listed as legal".into());
            }
        }
    }
    record["record_hash"] = json!(expected);
    Ok(record)
}

fn manifest(config: &Value) -> Value {
    let mut identity = json!({
        "protocol": PROTOCOL,
        "config": config,
        "system_prompt": SYSTEM_PROMPT,
        "engine_source_hash": env!("QBENCH_ENGINE_SOURCE_HASH"),
        "runner_hash": env!("QBENCH_RUNNER_HASH"),
        "build_hash": env!("QBENCH_BUILD_HASH"),
    });
    identity["hash"] = json!(hash(&identity));
    identity
}

fn standings(
    output: &Path,
    players: &[ValidatedPlayer],
    results: &[Value],
    scheduled: usize,
    manifest_hash: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(json!({
        "protocol": PROTOCOL,
        "completed": results.len(),
        "scheduled": scheduled,
        "placement_complete": results.len() == scheduled,
        "results_hash": hash(&json!(results)),
        "manifest_hash": manifest_hash,
        "ratings": ratings(players, results),
        "metrics": metrics(output, players, results)?,
    }))
}

fn run(config_path: PathBuf, output: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let config: Value = serde_json::from_slice(&fs::read(config_path)?)?;
    let validated = validate_config(&config)?;
    let players = &validated.players;
    let schedule = execute_on_success(
        make_schedule(&validated).map_err(|error| {
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
                as Box<dyn std::error::Error>
        }),
        |schedule| {
            fs::create_dir_all(&output)?;
            Ok::<_, Box<dyn std::error::Error>>(schedule)
        },
    )??;
    let manifest = manifest(&config);
    let manifest_path = output.join("manifest.json");
    if manifest_path.exists() {
        let existing: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        if existing != manifest {
            return Err("run configuration or code changed; use a new output directory".into());
        }
    } else {
        write_json(&manifest_path, &manifest)?;
    }
    let manifest_hash = manifest["hash"].as_str().unwrap_or("").to_owned();
    let client = Client::new();
    let mut results = Vec::new();
    for (game_number, scheduled) in schedule.iter().enumerate() {
        let game_path = output.join(format!("game-{game_number:05}.json"));
        let record = if game_path.exists() {
            let record = serde_json::from_slice(&fs::read(&game_path)?)?;
            checked_record(record, &manifest_hash)?
        } else {
            println!(
                "Game {}/{}: {} vs {}",
                game_number + 1,
                schedule.len(),
                players[scheduled.white].name,
                players[scheduled.black].name
            );
            let record = execute_on_success(
                play_game(
                    &client,
                    players,
                    &validated,
                    scheduled,
                    game_number,
                    &manifest_hash,
                    &output.join(format!("game-{game_number:05}.jsonl")),
                ),
                |mut record| {
                    record["record_hash"] = json!(hash(&record));
                    write_json(&game_path, &record)?;
                    Ok(record)
                },
            )??;
            record
        };
        results.push(record);
        let summary = standings(&output, &players, &results, schedule.len(), &manifest_hash)?;
        write_json(&output.join("standings.json"), &summary)?;
    }
    if results.is_empty() {
        let summary = standings(&output, &players, &results, schedule.len(), &manifest_hash)?;
        write_json(&output.join("standings.json"), &summary)?;
    }
    print!("{}", fs::read_to_string(output.join("standings.json"))?);
    Ok(())
}

fn audit(output: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let manifest: Value = serde_json::from_slice(&fs::read(output.join("manifest.json"))?)?;
    let manifest_hash = manifest["hash"]
        .as_str()
        .ok_or("manifest has no hash")?
        .to_owned();
    let mut identity = manifest.clone();
    identity
        .as_object_mut()
        .ok_or("manifest is not an object")?
        .remove("hash");
    if hash(&identity) != manifest_hash {
        return Err("manifest hash mismatch".into());
    }
    for (field, current) in [
        ("engine_source_hash", env!("QBENCH_ENGINE_SOURCE_HASH")),
        ("runner_hash", env!("QBENCH_RUNNER_HASH")),
        ("build_hash", env!("QBENCH_BUILD_HASH")),
    ] {
        if manifest[field].as_str() != Some(current) {
            return Err(format!("current code does not match manifest {field}").into());
        }
    }
    let mut paths = fs::read_dir(&output)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("game-") && name.ends_with(".json"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    let mut records = Vec::new();
    for path in paths {
        let record: Value = serde_json::from_slice(&fs::read(&path)?)?;
        let record = checked_record(record, &manifest_hash)?;
        let mut game = Game::new();
        for movement in record["moves"]
            .as_array()
            .ok_or("record has no move list")?
        {
            game.play(movement.as_str().ok_or("move is not a string")?)?;
        }
        if hash(&state(&game)) != record["final_state_hash"] {
            return Err(format!("final state replay mismatch: {}", path.display()).into());
        }
        let log_path = path.with_extension("jsonl");
        if !log_path.exists() {
            return Err(format!("turn log missing: {}", log_path.display()).into());
        }
        for line in BufReader::new(File::open(log_path)?).lines() {
            let event: Value = serde_json::from_str(&line?)?;
            if event.get("state").is_some() && hash(&event["state"]) != event["state_hash"] {
                return Err("turn snapshot hash mismatch".into());
            }
            if event["accepted"] == true
                && !event["state"]["legal_moves"]
                    .as_array()
                    .is_some_and(|moves| moves.iter().any(|m| m == &event["move"]))
            {
                return Err("accepted action outside legal list".into());
            }
        }
        records.push(record);
    }
    let standings: Value = serde_json::from_slice(&fs::read(output.join("standings.json"))?)?;
    if standings["results_hash"] != hash(&json!(records))
        || standings["completed"].as_u64() != Some(records.len() as u64)
    {
        return Err("standings sample count/hash mismatch".into());
    }
    println!(
        "Verified {}/{} game records and final-position replays.",
        records.len(),
        standings["scheduled"]
    );
    Ok(())
}

fn position() -> Result<(), Box<dyn std::error::Error>> {
    let mut game = Game::new();
    for movement in ["e3/4", "e5", "Nf3", "Nc6", "Bb5/c4", "Ne7/f6", "0-0/"] {
        game.play(movement)?;
    }
    println!("{}", serde_json::to_string_pretty(&state(&game))?);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("--audit") => audit(PathBuf::from(
            args.next().unwrap_or_else(|| "runs/rust".into()),
        ))?,
        Some("--position") => position()?,
        config => {
            let config = PathBuf::from(config.unwrap_or("models.json"));
            let output = PathBuf::from(args.next().unwrap_or_else(|| "runs/rust".into()));
            run(config, output)?;
        }
    }
    Ok(())
}

#[cfg(kani)]
mod verification {
    #[kani::proof]
    #[kani::unwind(6)]
    fn config_facts_combine_player_numeric_and_opening_checks() {
        let players_present: bool = kani::any();
        let player_count: usize = kani::any();
        let player_index: usize = kani::any();
        let player_error_selector: u8 = kani::any();
        let player_error = match player_error_selector % 8 {
            0 => None,
            1 => Some(super::PlayerConfigError::MissingName),
            2 => Some(super::PlayerConfigError::InlineCredentials),
            3 => Some(super::PlayerConfigError::DuplicateName),
            4 => Some(super::PlayerConfigError::UnsupportedKind),
            5 => Some(super::PlayerConfigError::LlmFields(
                super::LlmFieldError::MissingModel,
            )),
            6 => Some(super::PlayerConfigError::LlmFields(
                super::LlmFieldError::MissingApiKey,
            )),
            _ => Some(super::PlayerConfigError::LlmFields(
                super::LlmFieldError::InvalidParameters,
            )),
        }
        .map(|error| (player_index, error));
        let numeric_options: [Option<Option<u64>>; 5] = kani::any();
        let openings_is_null: bool = kani::any();
        let openings_is_array: bool = kani::any();
        let openings_has_entries: bool = kani::any();
        let all_opening_entries_are_valid: bool = kani::any();
        let facts = super::ConfigValidationFacts {
            players_present,
            player_count,
            first_player_error: player_error,
            numeric_options,
            openings_is_null,
            openings_is_array,
            openings_has_entries,
            all_opening_entries_are_valid,
        };
        let numeric_error = super::first_invalid_numeric_option(numeric_options);
        let opening_error = super::openings_config_error(
            openings_is_null,
            openings_is_array,
            openings_has_entries,
            all_opening_entries_are_valid,
        );

        match super::validate_config_facts(facts) {
            None => {
                assert!(players_present && player_count >= 2);
                assert!(player_error.is_none());
                assert!(numeric_error.is_none());
                assert!(opening_error.is_none());
            }
            Some(super::ConfigValidationError::MissingPlayers) => {
                assert!(!players_present);
            }
            Some(super::ConfigValidationError::TooFewPlayers) => {
                assert!(players_present && player_count < 2);
            }
            Some(super::ConfigValidationError::Player { index, error }) => {
                assert!(players_present && player_count >= 2);
                assert!(index < player_count);
                assert_eq!(player_error, Some((index, error)));
            }
            Some(super::ConfigValidationError::PlayerIndexOutOfRange(index)) => {
                assert!(players_present && player_count >= 2);
                assert!(index >= player_count);
                assert_eq!(
                    player_error.map(|(player_index, _)| player_index),
                    Some(index)
                );
            }
            Some(super::ConfigValidationError::NumericOption(index)) => {
                assert!(players_present && player_count >= 2);
                assert!(player_error.is_none());
                assert_eq!(numeric_error, Some(index));
            }
            Some(super::ConfigValidationError::Openings(error)) => {
                assert!(players_present && player_count >= 2);
                assert!(player_error.is_none());
                assert!(numeric_error.is_none());
                assert_eq!(opening_error, Some(error));
            }
        }
    }

    #[kani::proof]
    fn config_validation_preserves_top_level_error_precedence() {
        let players_present: bool = kani::any();
        let player_count: usize = kani::any();
        let player_index: usize = kani::any();
        let player_error_selector: u8 = kani::any();
        let player_error = match player_error_selector % 8 {
            0 => None,
            1 => Some(super::PlayerConfigError::MissingName),
            2 => Some(super::PlayerConfigError::InlineCredentials),
            3 => Some(super::PlayerConfigError::DuplicateName),
            4 => Some(super::PlayerConfigError::UnsupportedKind),
            5 => Some(super::PlayerConfigError::LlmFields(
                super::LlmFieldError::MissingModel,
            )),
            6 => Some(super::PlayerConfigError::LlmFields(
                super::LlmFieldError::MissingApiKey,
            )),
            _ => Some(super::PlayerConfigError::LlmFields(
                super::LlmFieldError::InvalidParameters,
            )),
        }
        .map(|error| (player_index, error));
        let numeric_error: Option<usize> = kani::any();
        let opening_error_selector: u8 = kani::any();
        let opening_error = match opening_error_selector % 4 {
            0 => None,
            1 => Some(super::OpeningsConfigError::NotArray),
            2 => Some(super::OpeningsConfigError::Empty),
            _ => Some(super::OpeningsConfigError::InvalidEntry),
        };
        match super::config_validation_error(
            players_present,
            player_count,
            player_error,
            numeric_error,
            opening_error,
        ) {
            None => {
                assert!(players_present && player_count >= 2);
                assert!(player_error.is_none());
                assert!(numeric_error.is_none());
                assert!(opening_error.is_none());
            }
            Some(super::ConfigValidationError::MissingPlayers) => {
                assert!(!players_present);
            }
            Some(super::ConfigValidationError::TooFewPlayers) => {
                assert!(players_present && player_count < 2);
            }
            Some(super::ConfigValidationError::Player { index, error }) => {
                assert!(players_present && player_count >= 2);
                assert!(index < player_count);
                assert_eq!(player_error, Some((index, error)));
            }
            Some(super::ConfigValidationError::PlayerIndexOutOfRange(index)) => {
                assert!(players_present && player_count >= 2);
                assert!(index >= player_count);
                assert_eq!(
                    player_error.map(|(player_index, _)| player_index),
                    Some(index)
                );
            }
            Some(super::ConfigValidationError::NumericOption(index)) => {
                assert!(players_present && player_count >= 2);
                assert!(player_error.is_none());
                assert_eq!(numeric_error, Some(index));
            }
            Some(super::ConfigValidationError::Openings(error)) => {
                assert!(players_present && player_count >= 2);
                assert!(player_error.is_none());
                assert!(numeric_error.is_none());
                assert_eq!(opening_error, Some(error));
            }
        }
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn numeric_config_options_reject_only_first_present_invalid_value() {
        let options: [Option<Option<u64>>; 5] = kani::any();
        match super::first_invalid_numeric_option(options) {
            None => {
                for option in options {
                    assert!(super::optional_positive_integer(option));
                }
            }
            Some(index) => {
                assert!(index < options.len());
                assert!(!super::optional_positive_integer(options[index]));
                for option in &options[..index] {
                    assert!(super::optional_positive_integer(*option));
                }
            }
        }
    }

    #[kani::proof]
    fn typed_numeric_projection_preserves_defaults_and_present_values() {
        let value: Option<Option<u64>> = kani::any();
        let default: u64 = kani::any();
        let positive_projection = super::positive_u64_or_default(value, default);
        let optional_projection = super::optional_u64_or_default(value, default);
        match value {
            None => {
                assert_eq!(positive_projection, Some(default));
                assert_eq!(optional_projection, default);
            }
            Some(None) => {
                assert_eq!(positive_projection, None);
                assert_eq!(optional_projection, default);
            }
            Some(Some(0)) => {
                assert_eq!(positive_projection, None);
                assert_eq!(optional_projection, 0);
            }
            Some(Some(number)) => {
                assert!(number > 0);
                assert_eq!(positive_projection, Some(number));
                assert_eq!(optional_projection, number);
            }
        }
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn runtime_options_projection_combines_all_settings() {
        let numeric_options: [Option<Option<u64>>; 5] = kani::any();
        let seed: Option<Option<u64>> = kani::any();
        let projected = super::project_runtime_options(numeric_options, seed);
        let all_numeric_options_are_valid = numeric_options
            .into_iter()
            .all(super::optional_positive_integer);
        assert_eq!(projected.is_some(), all_numeric_options_are_valid);
        if let Some(runtime) = projected {
            let defaults = [1, 150, 3, 120, 60_000];
            let expected = std::array::from_fn::<_, 5, _>(|index| match numeric_options[index] {
                None => defaults[index],
                Some(Some(value)) => value,
                Some(None) => unreachable!(),
            });
            assert_eq!(runtime.pair_repetitions, expected[0] as usize);
            assert_eq!(runtime.max_plies, expected[1] as usize);
            assert_eq!(runtime.max_attempts, expected[2] as usize);
            assert_eq!(runtime.timeout_seconds, expected[3]);
            assert_eq!(runtime.context_chars, expected[4] as usize);
            assert_eq!(runtime.seed, super::optional_u64_or_default(seed, 1));
        }
    }

    #[kani::proof]
    fn openings_config_validation_matches_structure_precedence() {
        let is_null: bool = kani::any();
        let is_array: bool = kani::any();
        let has_entries: bool = kani::any();
        let all_entries_are_valid: bool = kani::any();
        match super::openings_config_error(is_null, is_array, has_entries, all_entries_are_valid) {
            None => {
                assert!(is_null || (is_array && has_entries && all_entries_are_valid));
            }
            Some(super::OpeningsConfigError::NotArray) => {
                assert!(!is_null && !is_array);
            }
            Some(super::OpeningsConfigError::Empty) => {
                assert!(!is_null && is_array && !has_entries);
            }
            Some(super::OpeningsConfigError::InvalidEntry) => {
                assert!(!is_null && is_array && has_entries && !all_entries_are_valid);
            }
        }
    }

    #[kani::proof]
    fn player_config_validation_obeys_error_precedence_for_every_field_combination() {
        let name_is_nonempty: bool = kani::any();
        let has_inline_credentials: bool = kani::any();
        let name_is_unique: bool = kani::any();
        let kind_selector: u8 = kani::any();
        let kind = match kind_selector % 4 {
            0 => Some(super::PlayerKind::Llm),
            1 => Some(super::PlayerKind::Random),
            2 => Some(super::PlayerKind::Engine),
            _ => None,
        };
        let error_selector: u8 = kani::any();
        let llm_error = match error_selector % 4 {
            0 => super::LlmFieldError::None,
            1 => super::LlmFieldError::MissingModel,
            2 => super::LlmFieldError::MissingApiKey,
            _ => super::LlmFieldError::InvalidParameters,
        };
        match super::player_config_error(
            name_is_nonempty,
            has_inline_credentials,
            name_is_unique,
            kind,
            llm_error,
        ) {
            None => {
                assert!(name_is_nonempty && !has_inline_credentials && name_is_unique);
                match kind {
                    Some(super::PlayerKind::Llm) => {
                        assert_eq!(llm_error, super::LlmFieldError::None);
                    }
                    Some(super::PlayerKind::Random | super::PlayerKind::Engine) => {}
                    None => panic!("an unsupported kind must fail validation"),
                }
            }
            Some(super::PlayerConfigError::MissingName) => {
                assert!(!name_is_nonempty);
            }
            Some(super::PlayerConfigError::InlineCredentials) => {
                assert!(name_is_nonempty && has_inline_credentials);
            }
            Some(super::PlayerConfigError::DuplicateName) => {
                assert!(name_is_nonempty && !has_inline_credentials && !name_is_unique);
            }
            Some(super::PlayerConfigError::UnsupportedKind) => {
                assert!(name_is_nonempty && !has_inline_credentials && name_is_unique);
                assert_eq!(kind, None);
            }
            Some(super::PlayerConfigError::LlmFields(error)) => {
                assert!(name_is_nonempty && !has_inline_credentials && name_is_unique);
                assert_eq!(kind, Some(super::PlayerKind::Llm));
                assert_eq!(error, llm_error);
                assert_ne!(error, super::LlmFieldError::None);
            }
        }
    }

    #[kani::proof]
    fn llm_field_validation_matches_ordered_requirements() {
        let model_is_nonempty: bool = kani::any();
        let api_key_is_set: bool = kani::any();
        let parameters_are_valid: bool = kani::any();
        match super::llm_field_error(model_is_nonempty, api_key_is_set, parameters_are_valid) {
            super::LlmFieldError::None => {
                assert!(model_is_nonempty && api_key_is_set && parameters_are_valid);
            }
            super::LlmFieldError::MissingModel => {
                assert!(!model_is_nonempty);
            }
            super::LlmFieldError::MissingApiKey => {
                assert!(model_is_nonempty && !api_key_is_set);
            }
            super::LlmFieldError::InvalidParameters => {
                assert!(model_is_nonempty && api_key_is_set && !parameters_are_valid);
            }
        }
    }

    #[kani::proof]
    fn success_gate_runs_record_write_only_for_completed_game() {
        let game_completes: bool = kani::any();
        let mut action_ran = false;
        let result =
            super::execute_on_success(if game_completes { Ok(()) } else { Err(()) }, |_| {
                action_ran = true;
                Ok::<(), ()>(())
            });
        assert_eq!(action_ran, game_completes);
        assert_eq!(result.is_ok(), game_completes);
        if game_completes {
            assert!(result.unwrap().is_ok());
        }
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn schedule_pair_visitation_matches_ordered_pair_repetitions() {
        let player_count: u8 = kani::any();
        let pair_repetitions: u8 = kani::any();
        kani::assume((2..=5).contains(&player_count));
        kani::assume(pair_repetitions <= 4);

        let n = player_count as usize;
        let repetitions = pair_repetitions as usize;
        let mut seen = 0u64;
        let mut callback_count = 0usize;
        let result = super::visit_schedule_pairs(n, repetitions, |white, black, repetition| {
            assert!(white < black);
            assert!(black < n);
            assert!(repetition < repetitions);
            if white < black && black < n && repetition < repetitions {
                let pair_rank = white * (2 * n - white - 1) / 2 + (black - white - 1);
                let slot = pair_rank * repetitions + repetition;
                assert!(slot < 64);
                if slot < 64 {
                    let bit = 1u64 << slot;
                    assert_eq!(seen & bit, 0);
                    seen |= bit;
                }
            }
            callback_count += 1;
            Ok::<(), ()>(())
        });
        assert!(result.is_ok());

        let expected_count = n * (n - 1) / 2 * repetitions;
        assert_eq!(callback_count, expected_count);
        let expected_seen = if expected_count == 0 {
            0
        } else {
            (1u64 << expected_count) - 1
        };
        assert_eq!(seen, expected_seen);
    }

    #[kani::proof]
    fn opening_selection_is_cyclic_and_stays_in_bounds() {
        let repetition: u8 = kani::any();
        let opening_count: u8 = kani::any();
        if opening_count == 0 {
            assert_eq!(
                super::opening_index(repetition as usize, opening_count as usize),
                None
            );
        } else if let Some(index) =
            super::opening_index(repetition as usize, opening_count as usize)
        {
            assert!(index < opening_count as usize);
            assert_eq!(index, (repetition % opening_count) as usize);
        } else {
            panic!("positive opening count must produce an index");
        }
    }

    #[kani::proof]
    fn required_model_length_rejects_only_missing_or_empty_strings() {
        let length: Option<usize> = kani::any();
        assert_eq!(
            super::required_nonempty_string(length),
            matches!(length, Some(value) if value > 0)
        );
    }

    #[kani::proof]
    fn optional_parameters_accept_absence_or_object_values() {
        let is_present: bool = kani::any();
        let is_object: bool = kani::any();
        assert_eq!(
            super::optional_object_value(is_present, is_object),
            !is_present || is_object
        );
    }

    #[kani::proof]
    fn opening_array_requires_positive_length() {
        let length: usize = kani::any();
        assert_eq!(super::opening_array_length_is_valid(length), length > 0);
    }

    #[kani::proof]
    fn opening_entry_requires_an_array_containing_only_strings() {
        let is_array: bool = kani::any();
        let all_moves_are_strings: bool = kani::any();
        assert_eq!(
            super::opening_entry_is_valid(is_array, all_moves_are_strings),
            is_array && all_moves_are_strings
        );
    }

    #[kani::proof]
    fn three_byte_player_kind_parser_recognizes_only_llm() {
        let bytes: [u8; 3] = kani::any();
        let expected = if bytes == *b"llm" {
            Some(super::PlayerKind::Llm)
        } else {
            None
        };
        assert_eq!(super::parse_player_kind(Some(&bytes)), expected);
    }

    #[kani::proof]
    fn six_byte_player_kind_parser_recognizes_random_and_engine() {
        let bytes: [u8; 6] = kani::any();
        let expected = if bytes == *b"random" {
            Some(super::PlayerKind::Random)
        } else if bytes == *b"engine" {
            Some(super::PlayerKind::Engine)
        } else {
            None
        };
        assert_eq!(super::parse_player_kind(Some(&bytes)), expected);
    }

    #[kani::proof]
    fn player_kind_parser_defaults_to_llm_and_rejects_other_lengths() {
        assert_eq!(super::parse_player_kind(None), Some(super::PlayerKind::Llm));
        assert_eq!(super::parse_player_kind(Some(b"")), None);
        assert_eq!(super::parse_player_kind(Some(b"ll")), None);
        assert_eq!(super::parse_player_kind(Some(b"LLM")), None);
        assert_eq!(super::parse_player_kind(Some(b"other")), None);
    }

    #[kani::proof]
    fn api_key_environment_selection_and_missing_key_result_are_exact() {
        let is_present: bool = kani::any();
        let configured_name = super::api_key_environment_name(Some("TEST_PROVIDER_KEY"));
        let missing = super::missing_environment_key(configured_name, is_present);
        assert_eq!(missing.is_none(), is_present);
        if let Some(missing_name) = missing {
            assert!(std::ptr::eq(missing_name, configured_name));
        }
    }

    #[kani::proof]
    fn player_base_url_uses_configured_value_or_openrouter_default() {
        assert_eq!(super::player_base_url(None), "https://openrouter.ai/api/v1");
        assert_eq!(
            super::player_base_url(Some("http://localhost:8080/v1")),
            "http://localhost:8080/v1"
        );
    }

    #[kani::proof]
    fn player_name_uniqueness_matches_exact_bytes_for_two_prior_names() {
        let candidate: [u8; 8] = kani::any();
        let first: [u8; 8] = kani::any();
        let second: [u8; 8] = kani::any();
        let prior_names: [&[u8]; 2] = [&first, &second];
        let unique = super::player_name_is_unique(&candidate, prior_names.iter().copied());
        assert_eq!(unique, candidate != first && candidate != second);
    }

    #[kani::proof]
    fn inline_credential_rejection_matches_field_presence() {
        let api_key_present: bool = kani::any();
        let api_key_value_present: bool = kani::any();
        assert_eq!(
            super::has_inline_api_credentials(api_key_present, api_key_value_present),
            api_key_present || api_key_value_present
        );
    }

    #[kani::proof]
    fn optional_positive_integer_matches_presence_and_range_spec() {
        let value: Option<Option<u64>> = kani::any();
        let valid = super::optional_positive_integer(value);
        match value {
            None => {
                assert!(valid);
            }
            Some(None) => {
                assert!(!valid);
            }
            Some(Some(0)) => {
                assert!(!valid);
            }
            Some(Some(number)) => {
                assert!(number > 0);
                assert!(valid);
            }
        }
    }

    #[kani::proof]
    fn schedule_opponent_maps_every_valid_offset_into_the_remaining_players() {
        let player_count: usize = kani::any();
        let white: usize = kani::any();
        let offset: usize = kani::any();
        if let (Some(remaining_players), Some(remaining)) = (
            player_count.checked_sub(white),
            player_count
                .checked_sub(white)
                .and_then(|n| n.checked_sub(1)),
        ) && remaining_players > 0
            && offset < remaining
        {
            let black = super::schedule_opponent(player_count, white, offset).unwrap();
            assert!(white < black);
            assert!(black < player_count);
            assert_eq!(black - white - 1, offset);
        } else {
            assert_eq!(super::schedule_opponent(player_count, white, offset), None);
        }
    }

    #[kani::proof]
    fn schedule_opponent_enumerates_each_unordered_player_pair() {
        let player_count: usize = kani::any();
        let white: usize = kani::any();
        let black: usize = kani::any();
        kani::assume(white < black && black < player_count);
        assert_eq!(
            super::schedule_opponent(player_count, white, black - white - 1),
            Some(black)
        );
    }

    #[kani::proof]
    fn color_swapped_schedule_pairs_reverse_every_assignment() {
        let first: usize = kani::any();
        let second: usize = kani::any();
        let [(white_a, black_a), (white_b, black_b)] = super::color_swapped_pair(first, second);

        assert_eq!((white_a, black_a), (first, second));
        assert_eq!((white_b, black_b), (second, first));
        assert_eq!(white_a, black_b);
        assert_eq!(black_a, white_b);
    }

    #[kani::proof]
    fn move_gate_accepts_only_listed_actions() {
        let candidate: u8 = kani::any();
        let legal_a: u8 = kani::any();
        let legal_b: u8 = kani::any();
        let candidate_bytes = [candidate];
        let legal_a_bytes = [legal_a];
        let legal_b_bytes = [legal_b];
        let result =
            super::is_legal_move(&candidate_bytes, &[&legal_a_bytes[..], &legal_b_bytes[..]]);
        assert_eq!(result, candidate == legal_a || candidate == legal_b);
    }
}

fn is_legal_move(candidate: &[u8], legal: &[&[u8]]) -> bool {
    legal.iter().any(|move_bytes| *move_bytes == candidate)
}
