use qchess::{Game, Kind, notation};

fn assert_round_trip(intent: notation::Intent) -> String {
    let text = notation::format(&intent);
    let parsed = notation::parse(&text).expect("formatted intent parses");
    assert_eq!(parsed.intent, intent);
    assert!(parsed.collapse.is_none());
    text
}

#[test]
fn public_notation_api_round_trips_and_game_accepts_formatted_moves() {
    let castle = assert_round_trip(notation::Intent::Castle {
        side: 0,
        split: false,
    });
    let split_castle = assert_round_trip(notation::Intent::Castle {
        side: 1,
        split: true,
    });
    let hinted_knight = assert_round_trip(notation::Intent::Move {
        kind: Kind::Knight,
        hint: notation::Hint {
            file: Some(1),
            rank: Some(0),
        },
        to: 18,
        promo: None,
        second: None,
    });
    let split_promotion = assert_round_trip(notation::Intent::Move {
        kind: Kind::Pawn,
        hint: notation::Hint::default(),
        to: 60,
        promo: Some(Kind::Queen),
        second: Some(notation::Second::To {
            hint: notation::Hint::default(),
            to: 60,
            promo: Some(Kind::Knight),
        }),
    });
    let stay = assert_round_trip(notation::Intent::Move {
        kind: Kind::Bishop,
        hint: notation::Hint::default(),
        to: 20,
        promo: None,
        second: Some(notation::Second::Stay),
    });

    assert_eq!(castle, "0-0");
    assert_eq!(split_castle, "0-0-0/");
    assert_eq!(hinted_knight, "Nb1c3");
    assert_eq!(split_promotion, "e8=Q/N");
    assert_eq!(stay, "Be3/");

    assert_eq!(
        notation::parse("e3/4").unwrap().intent,
        notation::Intent::Move {
            kind: Kind::Pawn,
            hint: notation::Hint::default(),
            to: 20,
            promo: None,
            second: Some(notation::Second::To {
                hint: notation::Hint::default(),
                to: 28,
                promo: None,
            }),
        }
    );
    assert_eq!(
        notation::parse("e3/d").unwrap().intent,
        notation::Intent::Move {
            kind: Kind::Pawn,
            hint: notation::Hint::default(),
            to: 20,
            promo: None,
            second: Some(notation::Second::To {
                hint: notation::Hint::default(),
                to: 19,
                promo: None,
            }),
        }
    );
    assert_eq!(
        notation::parse("Bb5/Bc4").unwrap().intent,
        notation::Intent::Move {
            kind: Kind::Bishop,
            hint: notation::Hint::default(),
            to: 33,
            promo: None,
            second: Some(notation::Second::To {
                hint: notation::Hint::default(),
                to: 26,
                promo: None,
            }),
        }
    );

    for malformed in ["e9", "Nf0", "", "e4//d5"] {
        assert!(
            notation::parse(malformed).is_err(),
            "accepted {malformed:?}"
        );
    }

    let mut game = Game::new();
    let first_move = assert_round_trip(notation::Intent::Move {
        kind: Kind::Pawn,
        hint: notation::Hint::default(),
        to: 20,
        promo: None,
        second: None,
    });
    let result = game
        .play(&first_move)
        .expect("formatted legal move is playable");
    assert_eq!(result.notation, first_move);
}

#[test]
fn public_game_plays_both_legal_branches_of_a_split_knight() {
    let mut game = Game::new();
    game.play("Nb1c3/a3")
        .expect("the knight splits onto c3 and a3");
    game.play("a7a6").expect("Black makes a legal reply");
    game.play("Nd5/b5")
        .expect("both knight halves move along legal branches");

    let boards = game.states_fen();
    assert_eq!(boards.len(), 2);
    assert_ne!(boards[0], boards[1]);
    assert_eq!(game.history().len(), 3);
}

#[test]
fn public_game_reports_all_pieces_in_an_ambiguous_move() {
    let mut game = Game::from_fen("7k/8/8/8/8/1N6/2N5/K7 w - - 0 1")
        .expect("minimal position with two knights");
    let error = game.play("Nd4").expect_err("two knights can reach d4");

    let message = error.to_string();
    assert!(message.contains("ambiguous:"));
    assert!(message.contains("b3, c2"));
    assert!(message.contains("d4"));
    assert!(game.history().is_empty());
}
