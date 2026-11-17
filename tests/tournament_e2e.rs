use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    process::Command,
    thread,
};

#[test]
fn cli_runs_color_swapped_tournament_against_mock_provider() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let mut count = 0;
        while count < 7 {
            let (mut stream, _) = listener.accept().unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0; 4096];
            loop {
                let n = stream.read(&mut chunk).unwrap();
                if n == 0 {
                    break;
                }
                bytes.extend_from_slice(&chunk[..n]);
                let head_end = bytes.windows(4).position(|w| w == b"\r\n\r\n");
                if let Some(end) = head_end {
                    let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
                    let length = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length: "))
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0);
                    if bytes.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            let request = String::from_utf8_lossy(&bytes);
            assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
            assert!(request.contains("submit_move"));
            assert!(
                request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer test-key")
            );
            let split = request.find("\r\n\r\n").unwrap() + 4;
            let req: Value = serde_json::from_str(&request[split..]).unwrap();
            assert!(req["model"] == "mock-alpha" || req["model"] == "mock-beta");
            assert_eq!(req["temperature"], 0);
            let user_message_count = req["messages"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|message| message["role"] == "user")
                .count();
            assert_eq!(user_message_count, if count == 1 { 2 } else { 1 });
            let state = req["messages"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|message| message["role"] == "user")
                .filter_map(|message| message["content"].as_str())
                .find_map(|content| serde_json::from_str::<Value>(content).ok())
                .filter(|state| state["legal_moves"].is_array())
                .unwrap();
            let movement = if count == 0 {
                "illegal-action"
            } else {
                state["legal_moves"][0].as_str().unwrap()
            };
            let arguments = json!({"move":movement}).to_string();
            let body = json!({"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call1","type":"function","function":{"name":"submit_move","arguments":arguments}}]}}],"usage":{"total_tokens":3,"cost":0.0001}}).to_string();
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
            count += 1;
        }
        count
    });

    let dir = std::env::temp_dir().join(format!("qbench-e2e-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let config = dir.join("config.json");
    fs::write(&config, json!({"players":[
        {"name":"alpha","model":"mock-alpha","base_url":format!("http://{address}/v1"),"api_key_env":"QB_E2E_KEY","parameters":{"temperature":0}},
        {"name":"beta","model":"mock-beta","base_url":format!("http://{address}/v1"),"api_key_env":"QB_E2E_KEY","parameters":{"temperature":0}}
    ],"pairs":1,"max_plies":3,"max_attempts":2,"timeout_seconds":5,"context_chars":1}).to_string()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg(&config)
        .arg(dir.join("out"))
        .env("QB_E2E_KEY", "test-key")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(server.join().unwrap(), 7);
    let out = dir.join("out");
    let standings: Value =
        serde_json::from_slice(&fs::read(out.join("standings.json")).unwrap()).unwrap();
    assert_eq!(standings["completed"], 2);
    assert_eq!(standings["scheduled"], 2);
    assert_eq!(standings["placement_complete"], true);
    assert_eq!(standings["metrics"]["alpha"]["answered_attempts"], 4);
    assert_eq!(standings["metrics"]["alpha"]["rejected_attempts"], 1);
    assert_eq!(standings["metrics"]["alpha"]["reported_tokens"], 12);
    assert_eq!(standings["metrics"]["alpha"]["priced_attempts"], 4);
    let manifest = fs::read_to_string(out.join("manifest.json")).unwrap();
    assert!(!manifest.contains("test-key"));
    let first: Value =
        serde_json::from_slice(&fs::read(out.join("game-00000.json")).unwrap()).unwrap();
    assert_eq!(first["white"], "alpha");
    assert_eq!(first["moves"].as_array().unwrap().len(), 3);
    assert_eq!(first["moves"][0], first["records"][1]["move"]);
    assert_eq!(first["records"].as_array().unwrap().len(), 4);
    assert_eq!(first["records"][0]["accepted"], false);
    assert_eq!(first["records"][1]["accepted"], true);
    assert_eq!(first["score"], 0.5);
    let second: Value =
        serde_json::from_slice(&fs::read(out.join("game-00001.json")).unwrap()).unwrap();
    assert_eq!(second["white"], "beta");
    assert_eq!(second["black"], "alpha");
    assert_eq!(second["score"], 0.5);
    let first_bytes = fs::read(out.join("game-00000.json")).unwrap();
    let resumed = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg(&config)
        .arg(&out)
        .env("QB_E2E_KEY", "test-key")
        .output()
        .unwrap();
    assert!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert_eq!(first_bytes, fs::read(out.join("game-00000.json")).unwrap());
    fs::remove_file(out.join("standings.json")).unwrap();
    let resummarized = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg(&config)
        .arg(&out)
        .env("QB_E2E_KEY", "test-key")
        .output()
        .unwrap();
    assert!(
        resummarized.status.success(),
        "{}",
        String::from_utf8_lossy(&resummarized.stderr)
    );
    let rebuilt: Value =
        serde_json::from_slice(&fs::read(out.join("standings.json")).unwrap()).unwrap();
    assert_eq!(rebuilt["completed"], 2);
    let audit = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg("--audit")
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        audit.status.success(),
        "{}",
        String::from_utf8_lossy(&audit.stderr)
    );
    assert!(String::from_utf8_lossy(&audit.stdout).contains("Verified 2/2 game records"));
    let mut changed: Value = serde_json::from_slice(&fs::read(&config).unwrap()).unwrap();
    changed["max_plies"] = json!(2);
    fs::write(&config, changed.to_string()).unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg(&config)
        .arg(&out)
        .env("QB_E2E_KEY", "test-key")
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_runs_seeded_random_and_search_players_without_provider_keys() {
    let dir = std::env::temp_dir().join(format!("qbench-agents-e2e-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let config = dir.join("config.json");
    fs::write(
        &config,
        json!({
            "players": [
                {"name":"random","kind":"random"},
                {"name":"search","kind":"engine"}
            ],
            "pairs": 3,
            "seed": 42,
            "max_plies": 2,
            "max_attempts": 1,
            "timeout_seconds": 5,
            "openings": [["e4"], ["d4"]]
        })
        .to_string(),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg(&config)
        .arg(dir.join("out"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = dir.join("out");
    let first: Value =
        serde_json::from_slice(&fs::read(out.join("game-00000.json")).unwrap()).unwrap();
    let second: Value =
        serde_json::from_slice(&fs::read(out.join("game-00001.json")).unwrap()).unwrap();
    assert_eq!(first["moves"].as_array().unwrap().len(), 3);
    assert_eq!(second["moves"].as_array().unwrap().len(), 3);
    assert_eq!(first["opening"], json!(["e4"]));
    assert_eq!(second["opening"], json!(["e4"]));
    assert_eq!(first["termination"], "move_limit");
    assert_eq!(second["termination"], "move_limit");
    assert_eq!(first["seed"], 42);
    assert_eq!(second["seed"], 43);
    assert_eq!(first["records"][0]["player"], "search");
    assert_eq!(first["records"][1]["player"], "random");
    assert_eq!(second["records"][0]["player"], "random");
    assert_eq!(second["records"][1]["player"], "search");
    for game_number in 2..6 {
        let record: Value = serde_json::from_slice(
            &fs::read(out.join(format!("game-{game_number:05}.json"))).unwrap(),
        )
        .unwrap();
        let expected_opening = if game_number < 4 { "d4" } else { "e4" };
        assert_eq!(record["opening"], json!([expected_opening]));
    }
    let repeated = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg(&config)
        .arg(dir.join("repeat"))
        .output()
        .unwrap();
    assert!(repeated.status.success());
    let repeat_first: Value =
        serde_json::from_slice(&fs::read(dir.join("repeat/game-00000.json")).unwrap()).unwrap();
    assert_eq!(
        first["records"][0]["move"],
        repeat_first["records"][0]["move"]
    );
    let position = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg("--position")
        .output()
        .unwrap();
    assert!(position.status.success());
    let state: Value = serde_json::from_slice(&position.stdout).unwrap();
    assert_eq!(state["history"].as_array().unwrap().len(), 7);
    assert!(!state["legal_moves"].as_array().unwrap().is_empty());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn provider_failure_is_saved_as_incomplete_not_a_forfeit() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut bytes = Vec::new();
        let mut chunk = [0; 4096];
        loop {
            let n = stream.read(&mut chunk).unwrap();
            if n == 0 {
                break;
            }
            bytes.extend_from_slice(&chunk[..n]);
            let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
            let length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                break;
            }
        }
        let body = "{\"error\":\"provider unavailable\"}";
        write!(stream, "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
    });
    let dir = std::env::temp_dir().join(format!("qbench-failure-e2e-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let config = dir.join("config.json");
    fs::write(&config, json!({"players":[
        {"name":"alpha","model":"mock-alpha","base_url":format!("http://{address}/v1"),"api_key_env":"QB_E2E_KEY"},
        {"name":"beta","model":"mock-beta","base_url":format!("http://{address}/v1"),"api_key_env":"QB_E2E_KEY"}
    ],"pairs":1,"max_plies":1,"max_attempts":1,"timeout_seconds":5}).to_string()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg(&config)
        .arg(dir.join("out"))
        .env("QB_E2E_KEY", "failure-test-key")
        .output()
        .unwrap();
    assert!(!output.status.success());
    server.join().unwrap();
    let out = dir.join("out");
    assert!(!out.join("game-00000.json").exists());
    let event: Value =
        serde_json::from_str(&fs::read_to_string(out.join("game-00000.jsonl")).unwrap()).unwrap();
    assert_eq!(event["error"], "request_failed");
    assert!(!event.to_string().contains("failure-test-key"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_rejects_duplicate_names_and_inline_api_credentials() {
    let dir = std::env::temp_dir().join(format!("qbench-config-e2e-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let cases = [
        (
            "duplicate-names",
            json!({"players":[
                {"name":"same","kind":"random"},
                {"name":"same","kind":"engine"}
            ]}),
            "player names must be unique",
        ),
        (
            "inline-api-key",
            json!({"players":[
                {"name":"alpha","kind":"random","api_key":"sentinel-secret"},
                {"name":"beta","kind":"engine"}
            ]}),
            "store API credentials in environment variables",
        ),
        (
            "inline-api-key-value",
            json!({"players":[
                {"name":"alpha","kind":"random","api_key_value":"sentinel-secret"},
                {"name":"beta","kind":"engine"}
            ]}),
            "store API credentials in environment variables",
        ),
        (
            "unknown-player-kind",
            json!({"players":[
                {"name":"alpha","kind":"wizard"},
                {"name":"beta","kind":"engine"}
            ]}),
            "unsupported player kind",
        ),
        (
            "missing-model",
            json!({"players":[
                {"name":"alpha","kind":"llm"},
                {"name":"beta","kind":"engine"}
            ]}),
            "LLM player alpha requires a model",
        ),
        (
            "empty-model",
            json!({"players":[
                {"name":"alpha","kind":"llm","model":""},
                {"name":"beta","kind":"engine"}
            ]}),
            "LLM player alpha requires a model",
        ),
        (
            "non-object-parameters",
            json!({"players":[
                {"name":"alpha","kind":"llm","model":"mock","api_key_env":"QB_CONFIG_VALIDATION_KEY","parameters":[]},
                {"name":"beta","kind":"engine"}
            ]}),
            "parameters for alpha must be an object",
        ),
        (
            "empty-openings",
            json!({"players":[
                {"name":"alpha","kind":"random"},
                {"name":"beta","kind":"engine"}
            ],"openings":[]}),
            "openings must not be empty",
        ),
        (
            "non-array-openings",
            json!({"players":[
                {"name":"alpha","kind":"random"},
                {"name":"beta","kind":"engine"}
            ],"openings":"invalid"}),
            "openings must be an array",
        ),
        (
            "non-array-opening-entry",
            json!({"players":[
                {"name":"alpha","kind":"random"},
                {"name":"beta","kind":"engine"}
            ],"openings":[{}]}),
            "every opening must be an array of move strings",
        ),
        (
            "non-string-opening-move",
            json!({"players":[
                {"name":"alpha","kind":"random"},
                {"name":"beta","kind":"engine"}
            ],"openings":[[17]]}),
            "every opening must be an array of move strings",
        ),
        (
            "illegal-opening-move",
            json!({"players":[
                {"name":"alpha","kind":"random"},
                {"name":"beta","kind":"engine"}
            ],"openings":[["not-a-move"]]}),
            "invalid opening",
        ),
    ];

    for (name, config_value, expected_error) in cases {
        let config = dir.join(format!("{name}.json"));
        fs::write(&config, config_value.to_string()).unwrap();
        let output_dir = dir.join(name);
        let output = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
            .arg(&config)
            .arg(&output_dir)
            .env("QB_CONFIG_VALIDATION_KEY", "test-key")
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "{name} unexpectedly passed validation"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(expected_error), "{stderr}");
        assert!(!stderr.contains("sentinel-secret"));
        if name == "illegal-opening-move" {
            assert!(
                !output_dir.exists(),
                "opening validation failure created tournament output"
            );
        }
    }

    let missing_env_config = dir.join("missing-env.json");
    fs::write(
        &missing_env_config,
        json!({"players":[
            {"name":"alpha","kind":"llm","model":"mock","api_key_env":"QB_CONFIG_E2E_MISSING_KEY"},
            {"name":"beta","kind":"random"}
        ]})
        .to_string(),
    )
    .unwrap();
    let missing_env = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg(&missing_env_config)
        .arg(dir.join("missing-env"))
        .env_remove("QB_CONFIG_E2E_MISSING_KEY")
        .output()
        .unwrap();
    assert!(!missing_env.status.success());
    assert!(
        String::from_utf8_lossy(&missing_env.stderr)
            .contains("missing environment variable QB_CONFIG_E2E_MISSING_KEY")
    );

    let default_env_config = dir.join("default-env.json");
    fs::write(
        &default_env_config,
        json!({"players":[
            {"name":"alpha","kind":"llm","model":"mock"},
            {"name":"beta","kind":"engine"}
        ]})
        .to_string(),
    )
    .unwrap();
    let default_env = Command::new(env!("CARGO_BIN_EXE_qbench-engine"))
        .arg(&default_env_config)
        .arg(dir.join("default-env"))
        .env_remove("OPENROUTER_API_KEY")
        .output()
        .unwrap();
    assert!(!default_env.status.success());
    assert!(
        String::from_utf8_lossy(&default_env.stderr)
            .contains("missing environment variable OPENROUTER_API_KEY")
    );

    fs::remove_dir_all(dir).unwrap();
}
