//! Quantum algebraic notation.
//!
//! Ordinary SAN plus a `/` that splits a piece into two halves:
//!
//! | input         | meaning                                                      |
//! |---------------|--------------------------------------------------------------|
//! | `Nf3`, `exd5` | whole piece (both halves) moves, as in ordinary chess        |
//! | `Bb5/c4`      | halves go to b5 and c4                                        |
//! | `e3/4`        | shorthand: second square keeps the file (e3 and e4)          |
//! | `Nxd5/`       | one half moves to d5, the other half stays where it is       |
//! | `0-0/`        | quantum castling: rook moves, half the king castles          |
//! | `e8=Q/N`      | promote into two different quantum pieces on e8              |
//! | `Ncd5/eg6`    | disambiguation may be given for each half                    |
//!
//! Trailing `+`, `#`, `?` and `!` (e.g. `Qh5+?` for "Check?") are ignored.
//! When a move needs collapsing and there is more than one minimal way, the
//! mover may choose by appending `{...}` naming the half each collapsed piece
//! keeps: `{d6}`, `{Nc3}`, or `{xBd5}` for "the half captured on d5".

use crate::QError;
use crate::board::{KING_SIDE, QUEEN_SIDE};
use crate::types::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Hint {
    pub file: Option<u8>,
    pub rank: Option<u8>,
}

impl Hint {
    pub fn matches(&self, s: Sq) -> bool {
        self.file.is_none_or(|f| f == file_of(s)) && self.rank.is_none_or(|r| r == rank_of(s))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Second {
    /// `X/`: the other half stays put.
    Stay,
    /// `X/Y`: the other half goes to `to`.
    To {
        hint: Hint,
        to: Sq,
        promo: Option<Kind>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Intent {
    Castle {
        side: usize,
        split: bool,
    },
    Move {
        kind: Kind,
        hint: Hint,
        to: Sq,
        promo: Option<Kind>,
        second: Option<Second>,
    },
}

/// A collapse choice: keep the half of a piece that is on `sq`
/// (or that was captured on `sq` when `dead`). `num` picks the piece by its
/// number when several quantum pieces of that kind share the square, as in
/// `{N1d2}` (0 names an unnumbered piece).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollapseSpec {
    pub kind: Option<Kind>,
    pub num: Option<u8>,
    pub sq: Sq,
    pub dead: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parsed {
    pub intent: Intent,
    pub collapse: Option<Vec<CollapseSpec>>,
}

/// Write a move back as notation that [`parse`] reads as the same intent:
/// pieces by letter (none for pawns), the disambiguation given, then the
/// squares, e.g. `Nb1c3`, `Bc1d2/e3`, `e7e8=Q/N`, `0-0/`.
pub fn format(intent: &Intent) -> String {
    let mut out = String::new();
    format_into(intent, &mut out);
    out
}

trait NotationSink {
    fn push_char(&mut self, ch: char);
    fn push_str(&mut self, text: &str);
}

impl NotationSink for String {
    fn push_char(&mut self, ch: char) {
        self.push(ch);
    }

    fn push_str(&mut self, text: &str) {
        String::push_str(self, text);
    }
}

fn format_into(intent: &Intent, out: &mut impl NotationSink) {
    fn hint(h: &Hint, out: &mut impl NotationSink) {
        if let Some(f) = h.file {
            out.push_char((b'a' + f) as char);
        }
        if let Some(r) = h.rank {
            out.push_char((b'1' + r) as char);
        }
    }
    fn promo(p: Option<Kind>, out: &mut impl NotationSink) {
        if let Some(k) = p {
            out.push_char('=');
            out.push_char(k.letter());
        }
    }
    fn square(out: &mut impl NotationSink, s: Sq) {
        out.push_char((b'a' + file_of(s)) as char);
        out.push_char((b'1' + rank_of(s)) as char);
    }

    match *intent {
        Intent::Castle { side, split } => {
            out.push_str(if side == KING_SIDE { "0-0" } else { "0-0-0" });
            if split {
                out.push_char('/');
            }
        }
        Intent::Move {
            kind,
            hint: h,
            to,
            promo: p,
            second,
        } => {
            if kind != Kind::Pawn {
                out.push_char(kind.letter());
            }
            hint(&h, out);
            square(out, to);
            promo(p, out);
            match second {
                None => {}
                Some(Second::Stay) => out.push_char('/'),
                Some(Second::To {
                    hint: h2,
                    to: t2,
                    promo: p2,
                }) => {
                    out.push_char('/');
                    if h2 == Hint::default()
                        && t2 == to
                        && let Some(k) = p2
                    {
                        // Promotion-only second half: `e8=Q/N`.
                        out.push_char(k.letter());
                    } else {
                        hint(&h2, out);
                        square(out, t2);
                        promo(p2, out);
                    }
                }
            }
        }
    }
}

fn err<T>(input: &str, why: &str) -> Result<T, QError> {
    Err(QError(format!("can't read move \"{input}\": {why}")))
}

pub fn parse(input: &str) -> Result<Parsed, QError> {
    let raw = input.trim();
    let (mv, collapse) = match raw.find('{') {
        Some(i) => {
            let rest = &raw[i + 1..];
            let Some(j) = rest.find('}') else {
                return err(raw, "missing '}'");
            };
            (raw[..i].trim(), Some(parse_collapse(raw, &rest[..j])?))
        }
        None => (raw, None),
    };
    let mv = mv.trim_end_matches(['+', '#', '?', '!']);
    if mv.is_empty() {
        return err(raw, "empty move");
    }
    let intent = parse_intent(raw, mv.as_bytes())?;
    Ok(Parsed { intent, collapse })
}

fn parse_collapse(raw: &str, body: &str) -> Result<Vec<CollapseSpec>, QError> {
    let mut out = Vec::new();
    for tok in body.split([',', ' ']).filter(|t| !t.is_empty()) {
        let mut t = tok;
        let dead = t.starts_with('x');
        if dead {
            t = &t[1..];
        }
        let kind = t.chars().next().and_then(Kind::from_letter);
        if kind.is_some() {
            t = &t[1..];
        }
        // Optional piece number: digits right before the square.
        let digits = t.bytes().take_while(u8::is_ascii_digit).count();
        let num = if kind.is_some() && digits > 0 && t.len() == digits + 2 {
            let n = t[..digits].parse::<u8>().ok();
            t = &t[digits..];
            n
        } else {
            None
        };
        let Some(s) = parse_sq(t) else {
            return err(raw, &format!("bad collapse choice \"{tok}\""));
        };
        if kind.is_some() && digits > 0 && num.is_none() {
            return err(raw, &format!("bad collapse choice \"{tok}\""));
        }
        out.push(CollapseSpec {
            kind,
            num,
            sq: s,
            dead,
        });
    }
    if out.is_empty() {
        return err(raw, "empty collapse choice {}");
    }
    Ok(out)
}

fn parse_intent(raw: &str, mv: &[u8]) -> Result<Intent, QError> {
    parse_intent_core(mv)
        .map_err(|error| QError(format!("can't read move \"{raw}\": {}", error.message())))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntentError {
    CastleSuffix,
    Destination,
    NonPawnPromotion,
    Second(SecondError),
}

impl IntentError {
    fn message(self) -> &'static str {
        match self {
            Self::CastleSuffix => "castling can only be followed by '/'",
            Self::Destination => "expected a destination square like e4",
            Self::NonPawnPromotion => "only pawns promote",
            Self::Second(SecondError::MissingSquare) => "expected a square after '/'",
            Self::Second(SecondError::Invalid) => "can't read the part after '/'",
        }
    }
}

fn parse_intent_core(mv: &[u8]) -> Result<Intent, IntentError> {
    if starts_normalized(mv, b"0-0-0") {
        return parse_castle_suffix(&mv[5..], QUEEN_SIDE);
    }
    if starts_normalized(mv, b"0-0") {
        return parse_castle_suffix(&mv[3..], KING_SIDE);
    }

    let (main, second) = match mv.iter().position(|&byte| byte == b'/') {
        Some(index) => (&mv[..index], Some(&mv[index + 1..])),
        None => (mv, None),
    };

    parse_move_core(main, second)
}

fn parse_castle_suffix(suffix: &[u8], side: usize) -> Result<Intent, IntentError> {
    match suffix {
        b"" => Ok(Intent::Castle { side, split: false }),
        b"/" => Ok(Intent::Castle { side, split: true }),
        _ => Err(IntentError::CastleSuffix),
    }
}

fn parse_move_core(main: &[u8], second: Option<&[u8]>) -> Result<Intent, IntentError> {
    let Some((mut chars, mut len)) = filtered_bytes(main) else {
        return Err(IntentError::Destination);
    };
    let kind = match (len > 0)
        .then(|| Kind::from_letter(chars[0] as char))
        .flatten()
    {
        Some(k) if k != Kind::Pawn => {
            chars.copy_within(1..len, 0);
            len -= 1;
            k
        }
        _ => Kind::Pawn,
    };
    let promo = take_promo(&chars, &mut len);
    let (hint, to) = match split_square(&chars[..len]) {
        Some(v) => v,
        None => return Err(IntentError::Destination),
    };
    if promo.is_some() && kind != Kind::Pawn {
        return Err(IntentError::NonPawnPromotion);
    }

    let second = match second {
        None => None,
        Some(s) => Some(parse_second_core(s, to).map_err(IntentError::Second)?),
    };
    Ok(Intent::Move {
        kind,
        hint,
        to,
        promo,
        second,
    })
}

/// Filter SAN separators into a fixed-size buffer; accepted destination
/// syntax has at most a piece letter, promotion, two hints, and a square.
fn filtered_bytes(input: &[u8]) -> Option<([u8; 8], usize)> {
    let mut bytes = [0; 8];
    let mut len = 0;
    for byte in input
        .iter()
        .copied()
        .filter(|b| !matches!(b, b'x' | b'-' | b':'))
    {
        if len == bytes.len() {
            return None;
        }
        bytes[len] = byte;
        len += 1;
    }
    Some((bytes, len))
}

fn starts_normalized(input: &[u8], pattern: &[u8]) -> bool {
    input.len() >= pattern.len()
        && input[..pattern.len()]
            .iter()
            .zip(pattern)
            .all(|(&actual, &expected)| (if actual == b'O' { b'0' } else { actual }) == expected)
}

/// Strip a trailing promotion (`=Q` or bare `Q`).
fn take_promo(chars: &[u8; 8], len: &mut usize) -> Option<Kind> {
    let last = *chars.get(len.checked_sub(1)?)?;
    let k = Kind::from_letter(last as char).filter(|k| !matches!(k, Kind::Pawn | Kind::King))?;
    *len -= 1;
    if *len > 0 && chars[*len - 1] == b'=' {
        *len -= 1;
    }
    Some(k)
}

/// `[file][rank]square` → (hint, square).
fn split_square(chars: &[u8]) -> Option<(Hint, Sq)> {
    if chars.len() < 2 || chars.len() > 4 {
        return None;
    }
    let n = chars.len();
    let to = crate::types::parse_sq_bytes(&chars[n - 2..])?;
    let mut hint = Hint::default();
    for &c in &chars[..n - 2] {
        match c {
            b'a'..=b'h' if hint.file.is_none() => hint.file = Some(c - b'a'),
            b'1'..=b'8' if hint.rank.is_none() => hint.rank = Some(c - b'1'),
            _ => return None,
        }
    }
    Some((hint, to))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SecondError {
    MissingSquare,
    Invalid,
}

fn parse_second_core(s: &[u8], first: Sq) -> Result<Second, SecondError> {
    let Some((mut chars, mut len)) = filtered_bytes(s) else {
        return Err(SecondError::Invalid);
    };
    if len == 0 {
        // `e4/` stays; `e4/x` or `e4/-` names a second move without a square.
        return if s.is_empty() {
            Ok(Second::Stay)
        } else {
            Err(SecondError::MissingSquare)
        };
    }
    // Promotion-only: `e8=Q/N` or `e8=Q/=N`.
    if let Some(k) = promotion_only(&chars[..len]) {
        return Ok(Second::To {
            hint: Hint::default(),
            to: first,
            promo: Some(k),
        });
    }
    parse_second_destination(&mut chars, &mut len, first)
}

fn parse_second_destination(
    chars: &mut [u8; 8],
    len: &mut usize,
    first: Sq,
) -> Result<Second, SecondError> {
    // Optional repeated piece letter: `Bb5/Bc4`.
    strip_repeated_piece_letter(chars, len);
    let promo = take_promo(chars, len);
    let to = match &chars[..*len] {
        [byte] => shorthand_square(*byte, first).ok_or(SecondError::Invalid)?,
        [] if promo.is_some() => first,
        _ => match split_square(&chars[..*len]) {
            Some((hint, to)) => return Ok(Second::To { hint, to, promo }),
            None => return Err(SecondError::Invalid),
        },
    };
    Ok(Second::To {
        hint: Hint::default(),
        to,
        promo,
    })
}

fn strip_repeated_piece_letter(chars: &mut [u8; 8], len: &mut usize) {
    if *len > 0 && chars[0].is_ascii_uppercase() {
        chars.copy_within(1..*len, 0);
        *len -= 1;
    }
}

fn shorthand_square(byte: u8, first: Sq) -> Option<Sq> {
    match byte {
        b'a'..=b'h' => Some(sq(byte - b'a', rank_of(first))),
        b'1'..=b'8' => Some(sq(file_of(first), byte - b'1')),
        _ => None,
    }
}

fn promotion_only(chars: &[u8]) -> Option<Kind> {
    let mut promotion = None;
    let mut count = 0;
    for &byte in chars {
        if byte == b'=' {
            continue;
        }
        count += 1;
        if count > 1 {
            return None;
        }
        promotion = Kind::from_letter(byte as char);
    }
    if count != 1 {
        return None;
    }
    let kind = promotion?;
    matches!(kind, Kind::Queen | Kind::Rook | Kind::Bishop | Kind::Knight).then_some(kind)
}

#[cfg(kani)]
mod verification {
    use super::{Hint, Intent, NotationSink, Second, format_into};
    use crate::types::Kind;

    struct Buffer {
        bytes: [u8; 32],
        len: usize,
    }

    impl Buffer {
        fn new() -> Self {
            Self {
                bytes: [0; 32],
                len: 0,
            }
        }
    }

    impl NotationSink for Buffer {
        fn push_char(&mut self, ch: char) {
            assert!(ch.is_ascii() && self.len < self.bytes.len());
            self.bytes[self.len] = ch as u8;
            self.len += 1;
        }

        fn push_str(&mut self, text: &str) {
            for ch in text.chars() {
                self.push_char(ch);
            }
        }
    }

    fn assert_format(intent: Intent, expected: &[u8]) {
        let mut notation = Buffer::new();
        format_into(&intent, &mut notation);
        assert_eq!(notation.len, expected.len());
        assert_eq!(&notation.bytes[..notation.len], expected);
    }

    #[kani::proof]
    fn castle_parser_accepts_every_zero_or_letter_spelling_and_split_suffix() {
        for mask in 0..4 {
            for split in [false, true] {
                let mut notation = [b'0', b'-', b'0', b'/', 0, 0];
                if mask & 1 != 0 {
                    notation[0] = b'O';
                }
                if mask & 2 != 0 {
                    notation[2] = b'O';
                }
                assert_eq!(super::starts_normalized(&notation[..3], b"0-0"), true);
                assert_eq!(
                    super::parse_castle_suffix(
                        if split { &b"/"[..] } else { &b""[..] },
                        crate::board::KING_SIDE
                    ),
                    Ok(Intent::Castle {
                        side: crate::board::KING_SIDE,
                        split
                    })
                );
            }
        }
        for mask in 0..8 {
            for split in [false, true] {
                let mut notation = [b'0', b'-', b'0', b'-', b'0', b'/', 0, 0];
                if mask & 1 != 0 {
                    notation[0] = b'O';
                }
                if mask & 2 != 0 {
                    notation[2] = b'O';
                }
                if mask & 4 != 0 {
                    notation[4] = b'O';
                }
                assert_eq!(super::starts_normalized(&notation[..5], b"0-0-0"), true);
                assert_eq!(
                    super::parse_castle_suffix(
                        if split { &b"/"[..] } else { &b""[..] },
                        crate::board::QUEEN_SIDE
                    ),
                    Ok(Intent::Castle {
                        side: crate::board::QUEEN_SIDE,
                        split
                    })
                );
            }
        }
    }

    #[kani::proof]
    fn representative_intents_format_to_their_canonical_notation() {
        assert_format(
            Intent::Castle {
                side: crate::board::KING_SIDE,
                split: false,
            },
            b"0-0",
        );
        assert_format(
            Intent::Castle {
                side: crate::board::QUEEN_SIDE,
                split: true,
            },
            b"0-0-0/",
        );
        assert_format(
            Intent::Move {
                kind: Kind::Knight,
                hint: Hint {
                    file: Some(1),
                    rank: Some(0),
                },
                to: 18,
                promo: None,
                second: None,
            },
            b"Nb1c3",
        );
        assert_format(
            Intent::Move {
                kind: Kind::Pawn,
                hint: Hint::default(),
                to: 60,
                promo: Some(Kind::Queen),
                second: Some(Second::To {
                    hint: Hint::default(),
                    to: 60,
                    promo: Some(Kind::Knight),
                }),
            },
            b"e8=Q/N",
        );
        assert_format(
            Intent::Move {
                kind: Kind::Bishop,
                hint: Hint::default(),
                to: 20,
                promo: None,
                second: Some(Second::Stay),
            },
            b"Be3/",
        );
    }

    #[kani::proof]
    fn destination_square_formats_correctly_for_every_board_square() {
        let square: u8 = kani::any();
        kani::assume(square < 64);
        let intent = Intent::Move {
            kind: Kind::Pawn,
            hint: Hint::default(),
            to: square,
            promo: None,
            second: None,
        };
        let mut notation = Buffer::new();
        format_into(&intent, &mut notation);
        assert_eq!(notation.len, 2);
        assert_eq!(notation.bytes[0], b'a' + square % 8);
        assert_eq!(notation.bytes[1], b'1' + square / 8);
        assert_eq!(
            crate::types::parse_sq_bytes(&notation.bytes[..notation.len]),
            Some(square)
        );
    }

    fn assert_square_split(
        bytes: &[u8],
        hint_file: Option<u8>,
        hint_rank: Option<u8>,
        destination_file: u8,
        destination_rank: u8,
    ) {
        assert_eq!(
            super::split_square(bytes),
            Some((
                Hint {
                    file: hint_file,
                    rank: hint_rank,
                },
                crate::types::sq(destination_file, destination_rank)
            ))
        );
    }

    #[kani::proof]
    fn repeated_piece_prefix_removal_matches_ascii_uppercase_for_all_lengths() {
        let original: [u8; 8] = kani::any();
        let mut actual = original;
        let mut len: usize = kani::any();
        kani::assume(len <= actual.len());

        let original_len = len;
        let remove = len > 0 && original[0].is_ascii_uppercase();
        super::strip_repeated_piece_letter(&mut actual, &mut len);
        assert_eq!(
            len,
            if remove {
                original_len - 1
            } else {
                original_len
            }
        );
        for index in 0..8 {
            if remove && index < len {
                assert_eq!(actual[index], original[index + 1]);
            } else {
                assert_eq!(actual[index], original[index]);
            }
        }
    }

    #[kani::proof]
    fn take_promo_classifies_every_byte_with_optional_equals() {
        let byte: u8 = kani::any();
        let expected =
            Kind::from_letter(byte as char).filter(|kind| !matches!(kind, Kind::Pawn | Kind::King));

        let mut bare = [0u8; 8];
        bare[0] = byte;
        let mut bare_len = 1;
        assert_eq!(super::take_promo(&bare, &mut bare_len), expected);
        assert_eq!(bare_len, if expected.is_some() { 0 } else { 1 });

        let mut marked = [0u8; 8];
        marked[0] = b'=';
        marked[1] = byte;
        let mut marked_len = 2;
        assert_eq!(super::take_promo(&marked, &mut marked_len), expected);
        assert_eq!(marked_len, if expected.is_some() { 0 } else { 2 });
    }

    #[kani::proof]
    fn split_square_decodes_destinations_without_hints() {
        let file: u8 = kani::any();
        let rank: u8 = kani::any();
        kani::assume(file < 8 && rank < 8);
        assert_square_split(&[b'a' + file, b'1' + rank], None, None, file, rank);
    }

    #[kani::proof]
    fn split_square_decodes_every_file_hint_and_destination() {
        let hint_file: u8 = kani::any();
        let destination_file: u8 = kani::any();
        let destination_rank: u8 = kani::any();
        kani::assume(hint_file < 8 && destination_file < 8 && destination_rank < 8);
        assert_square_split(
            &[
                b'a' + hint_file,
                b'a' + destination_file,
                b'1' + destination_rank,
            ],
            Some(hint_file),
            None,
            destination_file,
            destination_rank,
        );
    }

    #[kani::proof]
    fn split_square_decodes_every_rank_hint_and_destination() {
        let hint_rank: u8 = kani::any();
        let destination_file: u8 = kani::any();
        let destination_rank: u8 = kani::any();
        kani::assume(hint_rank < 8 && destination_file < 8 && destination_rank < 8);
        assert_square_split(
            &[
                b'1' + hint_rank,
                b'a' + destination_file,
                b'1' + destination_rank,
            ],
            None,
            Some(hint_rank),
            destination_file,
            destination_rank,
        );
    }

    #[kani::proof]
    fn split_square_decodes_every_file_rank_hint_and_destination() {
        let hint_file: u8 = kani::any();
        let hint_rank: u8 = kani::any();
        let destination_file: u8 = kani::any();
        let destination_rank: u8 = kani::any();
        kani::assume(
            hint_file < 8 && hint_rank < 8 && destination_file < 8 && destination_rank < 8,
        );
        assert_square_split(
            &[
                b'a' + hint_file,
                b'1' + hint_rank,
                b'a' + destination_file,
                b'1' + destination_rank,
            ],
            Some(hint_file),
            Some(hint_rank),
            destination_file,
            destination_rank,
        );
    }

    #[kani::proof]
    fn second_shorthand_decodes_every_valid_file_and_rank() {
        let first: u8 = kani::any();
        let file: u8 = kani::any();
        let rank: u8 = kani::any();
        kani::assume(first < 64 && file < 8 && rank < 8);
        assert_eq!(
            super::shorthand_square(b'a' + file, first),
            Some(crate::types::sq(file, crate::types::rank_of(first)))
        );
        assert_eq!(
            super::shorthand_square(b'1' + rank, first),
            Some(crate::types::sq(crate::types::file_of(first), rank))
        );
    }

    #[kani::proof]
    fn second_shorthand_rejects_every_other_byte() {
        let first: u8 = kani::any();
        let byte: u8 = kani::any();
        kani::assume(first < 64);
        let expected = match byte {
            b'a'..=b'h' => Some(crate::types::sq(byte - b'a', crate::types::rank_of(first))),
            b'1'..=b'8' => Some(crate::types::sq(crate::types::file_of(first), byte - b'1')),
            _ => None,
        };
        assert_eq!(super::shorthand_square(byte, first), expected);
    }

    #[kani::proof]
    fn promotion_only_second_half_accepts_each_promotable_piece() {
        let promotion = match kani::any::<u8>() % 4 {
            0 => Kind::Queen,
            1 => Kind::Rook,
            2 => Kind::Bishop,
            _ => Kind::Knight,
        };
        let letter = promotion.letter() as u8;
        assert_eq!(super::promotion_only(&[letter]), Some(promotion));
        assert_eq!(super::promotion_only(&[b'=', letter]), Some(promotion));
        assert_eq!(super::promotion_only(&[letter, b'=']), Some(promotion));
    }

    #[kani::proof]
    fn promotion_only_classifies_every_two_byte_input() {
        let first: u8 = kani::any();
        let second: u8 = kani::any();
        let candidate = if first == b'=' && second != b'=' {
            Some(second)
        } else if second == b'=' && first != b'=' {
            Some(first)
        } else {
            None
        };
        let expected = candidate
            .and_then(|byte| Kind::from_letter(byte as char))
            .filter(|kind| matches!(kind, Kind::Queen | Kind::Rook | Kind::Bishop | Kind::Knight));
        assert_eq!(super::promotion_only(&[first, second]), expected);
    }
}
