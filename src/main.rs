use qchess::{Color, Game, Status, ai};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
fn handle(v: Value) -> Result<Value, String> {
    if v["op"] == "ratings" {
        let rows = v["results"].as_array().ok_or("missing results")?;
        let names = v["names"].as_array().ok_or("missing names")?;
        let out: Vec<_> = names
            .iter()
            .map(|name| {
                let mut games = Vec::new();
                for row in rows {
                    let score = row["score"].as_f64().unwrap();
                    let score = if row["white"] == *name {
                        score
                    } else if row["black"] == *name {
                        1.0 - score
                    } else {
                        continue;
                    };
                    let outcome = if score == 1.0 {
                        glicko2::Outcome::Win
                    } else if score == 0.0 {
                        glicko2::Outcome::Loss
                    } else {
                        glicko2::Outcome::Draw
                    };
                    games.push((glicko2::Rating::default(), outcome));
                }
                let r = glicko2::update(
                    glicko2::Rating::default(),
                    &games,
                    &glicko2::Config {
                        tau: 0.5,
                        min_deviation: 0.0,
                        max_deviation: 350.0,
                    },
                );
                json!({"name": name, "rating":r.rating,"deviation":r.deviation,"games":games.len()})
            })
            .collect();
        return Ok(json!(out));
    }
    let history: Vec<String> =
        serde_json::from_value(v["history"].clone()).map_err(|e| e.to_string())?;
    let mut g = if let Some(fen) = v["fen"].as_str() {
        Game::from_fen(fen)
    } else {
        Ok(Game::new())
    }
    .map_err(|e| e.to_string())?;
    for m in &history {
        g.play(m).map_err(|e| e.to_string())?;
    }
    if v["op"] == "choose" {
        let c = ai::best_move(
            &g,
            ai::Config {
                depth: 1,
                time: Some(std::time::Duration::from_millis(500)),
                ..Default::default()
            },
        );
        return Ok(json!({"move":c.map(|c| c.notation)}));
    }
    let status = match g.status() {
        Status::Ongoing => "ongoing",
        Status::Stalemate => "stalemate",
        Status::Checkmate {
            winner: Color::White,
        } => "white_win",
        Status::Checkmate {
            winner: Color::Black,
        } => "black_win",
    };
    let mut legal: Vec<_> = if status == "ongoing" {
        ai::children(&g, true)
            .into_iter()
            .map(|c| c.notation)
            .collect()
    } else {
        vec![]
    };
    legal.sort();
    legal.dedup();
    let pieces: Vec<_> = g.pieces().into_iter().map(|p| json!({"id":p.id,"color":format!("{:?}",p.color).to_lowercase(),"split":p.split,"halves":p.halves.into_iter().map(|h|json!({"square":h.square,"kind":format!("{:?}",h.kind),"alive":h.alive})).collect::<Vec<_>>()})).collect();
    Ok(
        json!({"status":status,"to_move":format!("{:?}",g.to_move()).to_lowercase(),"check":format!("{:?}",g.check()),"board":g.board_text(),"pieces":pieces,"legal_moves":legal,"state_count":g.state_count(),"position_key":format!("{:016x}",g.position_key()),"history":history,"possible_boards_fen":g.states_fen()}),
    )
}
fn main() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let result = line
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
            .and_then(handle);
        let out = match result {
            Ok(v) => json!({"ok":v}),
            Err(e) => json!({"error":e}),
        };
        println!("{out}");
        io::stdout().flush().unwrap();
    }
}
