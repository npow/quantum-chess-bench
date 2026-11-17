//! Basic chess vocabulary: colors, piece kinds and squares.

/// Side to move / owner of a piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn other(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    pub(crate) fn idx(self) -> usize {
        match self {
            Color::White => 0,
            Color::Black => 1,
        }
    }

    pub(crate) fn home_rank(self) -> u8 {
        match self {
            Color::White => 0,
            Color::Black => 7,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Color::White => "white",
            Color::Black => "black",
        }
    }
}

/// Kind of a (half of a) piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl Kind {
    /// SAN letter (uppercase). Pawns use `P` where a letter is needed.
    pub fn letter(self) -> char {
        match self {
            Kind::Pawn => 'P',
            Kind::Knight => 'N',
            Kind::Bishop => 'B',
            Kind::Rook => 'R',
            Kind::Queen => 'Q',
            Kind::King => 'K',
        }
    }

    pub fn from_letter(c: char) -> Option<Kind> {
        Some(match c {
            'P' => Kind::Pawn,
            'N' => Kind::Knight,
            'B' => Kind::Bishop,
            'R' => Kind::Rook,
            'Q' => Kind::Queen,
            'K' => Kind::King,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Pawn => "pawn",
            Kind::Knight => "knight",
            Kind::Bishop => "bishop",
            Kind::Rook => "rook",
            Kind::Queen => "queen",
            Kind::King => "king",
        }
    }

    pub(crate) fn fen_char(self, color: Color) -> char {
        let c = self.letter();
        match color {
            Color::White => c,
            Color::Black => c.to_ascii_lowercase(),
        }
    }
}

/// Square index 0..64, `a1 = 0`, `b1 = 1`, ..., `h8 = 63`.
pub type Sq = u8;

pub fn sq(file: u8, rank: u8) -> Sq {
    rank * 8 + file
}

pub fn file_of(s: Sq) -> u8 {
    s % 8
}

pub fn rank_of(s: Sq) -> u8 {
    s / 8
}

pub fn sq_name(s: Sq) -> String {
    format!("{}{}", (b'a' + file_of(s)) as char, rank_of(s) + 1)
}

pub fn parse_sq(t: &str) -> Option<Sq> {
    parse_sq_bytes(t.as_bytes())
}

pub(crate) fn parse_sq_bytes(b: &[u8]) -> Option<Sq> {
    if b.len() != 2 {
        return None;
    }
    let f = b[0].wrapping_sub(b'a');
    let r = b[1].wrapping_sub(b'1');
    (f < 8 && r < 8).then(|| sq(f, r))
}

/// Step from a square by a file/rank delta, staying on the board.
pub(crate) fn offset(s: Sq, df: i8, dr: i8) -> Option<Sq> {
    let f = file_of(s) as i8 + df;
    let r = rank_of(s) as i8 + dr;
    ((0..8).contains(&f) && (0..8).contains(&r)).then(|| sq(f as u8, r as u8))
}

#[cfg(kani)]
mod verification {
    use super::{Color, Kind, file_of, parse_sq_bytes, rank_of, sq};

    #[kani::proof]
    fn color_piece_and_square_mappings_round_trip() {
        let color = if kani::any::<bool>() {
            Color::White
        } else {
            Color::Black
        };
        assert_eq!(color.other().other(), color);
        assert!(color.idx() < 2);

        let kind = match kani::any::<u8>() % 6 {
            0 => Kind::Pawn,
            1 => Kind::Knight,
            2 => Kind::Bishop,
            3 => Kind::Rook,
            4 => Kind::Queen,
            _ => Kind::King,
        };
        assert_eq!(Kind::from_letter(kind.letter()), Some(kind));

        let square: u8 = kani::any();
        kani::assume(square < 64);
        assert!(file_of(square) < 8);
        assert!(rank_of(square) < 8);
        assert_eq!(sq(file_of(square), rank_of(square)), square);
    }

    #[kani::proof]
    fn square_parser_handles_every_two_byte_input() {
        let bytes: [u8; 2] = kani::any();
        let expected = if (b'a'..=b'h').contains(&bytes[0]) && (b'1'..=b'8').contains(&bytes[1]) {
            Some((bytes[1] - b'1') * 8 + bytes[0] - b'a')
        } else {
            None
        };
        assert_eq!(parse_sq_bytes(&bytes), expected);
    }

    #[kani::proof]
    fn square_parser_rejects_lengths_other_than_two() {
        let bytes: [u8; 3] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= bytes.len() && len != 2);
        assert_eq!(parse_sq_bytes(&bytes[..len]), None);
    }
}
