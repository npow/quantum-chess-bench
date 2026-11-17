#![allow(clippy::needless_range_loop)]
//! Quantum positions: pieces made of two halves, the set of possible
//! classical board states, and the move/collapse rules from the sheet.
//!
//! Model
//! -----
//! Every piece has two halves. An ordinary piece has both halves on the same
//! square. A split piece has halves on different squares (or one captured,
//! or two promotion kinds). The possible board states are every way of picking
//! one half per split piece (2^n of them).
//!
//! Playing a move:
//! 1. Work out which halves move where (from the notation) and the classical
//!    moves ("branches") this stands for in each state.
//! 2. Move the halves. Enemy halves on a destination square are captured.
//! 3. `reachable` = every board you get by playing a branch legally (regular
//!    chess rules, own king safe) from a previous state.
//! 4. A resulting state that isn't reachable is impossible, so pieces must
//!    collapse until none is left. That is rule 3 on the sheet. Rule 2 says use
//!    as few collapses as possible, and the mover chooses between equally small
//!    options.
//!
//! The sheet's consequences then fall out automatically: moving through a
//! quantum piece collapses it away, pawn captures take both halves, a
//! half-capture forces a collapse, and quantum check must be resolved.

use crate::QError;
use crate::board::*;
use crate::notation::{CollapseSpec, Hint, Intent, Second};
use crate::types::*;
#[cfg(feature = "verus")]
use vstd::prelude::*;

/// Upper limit on simultaneously split pieces (2^16 = 65,536 board states).
pub const MAX_SPLIT: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Half {
    pub sq: Sq,
    pub kind: Kind,
    /// `false` once captured; `sq` then remembers where.
    pub alive: bool,
    pub moved: bool,
}

impl Half {
    fn same(&self, o: &Half) -> bool {
        // Two captured halves are the same outcome: the piece is gone.
        (!self.alive && !o.alive)
            || (self.sq == o.sq && self.kind == o.kind && self.alive == o.alive)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub color: Color,
    pub h: [Half; 2],
    /// Kind the piece started the game as (a promoted pawn stays `Pawn`).
    pub origin: Kind,
    /// Number telling same-kind pieces apart (pawns 1-8 by file, rooks and
    /// knights 1-2 from the a-file side); 0 for unnumbered kinds.
    pub num: u8,
}

/// A collapse choice: bit `k` of the mask fixes split piece `k` to the half
/// in bit `k` of the value.
pub type Cube = (u32, u32);

/// One way a move can be played: the collapse picked (if the mover had a
/// choice), the position, and its boards when they are already known.
pub type Outcome = (Option<Cube>, Position, Option<States>);

/// At most two distinct halves, without allocating.
#[derive(Clone, Copy)]
struct Locs {
    items: [(usize, Sq, Kind); 2],
    len: usize,
}

impl std::ops::Deref for Locs {
    type Target = [(usize, Sq, Kind)];
    fn deref(&self) -> &Self::Target {
        &self.items[..self.len]
    }
}

impl IntoIterator for Locs {
    type Item = (usize, Sq, Kind);
    type IntoIter = std::iter::Take<std::array::IntoIter<(usize, Sq, Kind), 2>>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter().take(self.len)
    }
}

impl std::hash::Hash for Piece {
    /// Everything about the piece packed into one word: positions are hashed
    /// for every move the AI looks at, and field-by-field hashing was a tenth
    /// of its time.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let half = |h: &Half| {
            h.sq as u64 | (h.kind as u64) << 6 | (h.alive as u64) << 9 | (h.moved as u64) << 10
        };
        state.write_u64(
            half(&self.h[0])
                | half(&self.h[1]) << 11
                | (self.color as u64) << 22
                | (self.origin as u64) << 23
                | (self.num as u64) << 26,
        );
    }
}

fn subscript(n: u8) -> String {
    n.to_string()
        .chars()
        .map(|c| char::from_u32(0x2080 + c.to_digit(10).unwrap_or(0)).unwrap_or(c))
        .collect()
}

impl Piece {
    /// Short name for notation when its half of kind `kind` moves, e.g.
    /// `P₅`, `R₁`, `B`, or `Qₚ₅` for a queen promoted from pawn 5.
    pub fn label(&self, kind: Kind) -> String {
        let letter = kind.letter().to_ascii_uppercase();
        match (self.num, kind == self.origin) {
            (0, _) => letter.to_string(),
            (n, true) => format!("{letter}{}", subscript(n)),
            (n, false) => format!("{letter}\u{209A}{}", subscript(n)),
        }
    }

    /// Spoken name, e.g. `white rook 2` or `white queen (pawn 5)`.
    pub fn name(&self, kind: Kind) -> String {
        match (self.num, kind == self.origin) {
            (0, _) => format!("{} {}", self.color.name(), kind.name()),
            (n, true) => format!("{} {} {n}", self.color.name(), kind.name()),
            (n, false) => format!(
                "{} {} ({} {n})",
                self.color.name(),
                kind.name(),
                self.origin.name()
            ),
        }
    }

    pub fn is_split(&self) -> bool {
        !self.h[0].same(&self.h[1])
    }
    pub fn is_dead(&self) -> bool {
        !self.h[0].alive && !self.h[1].alive
    }
    /// Distinct living (square, kind) halves with a representative index.
    fn locs(&self) -> Locs {
        let mut v = Locs {
            items: [(0, 0, Kind::Pawn); 2],
            len: 0,
        };
        for (i, h) in self.h.iter().enumerate() {
            if h.alive && !v.iter().any(|&(_, s, k)| s == h.sq && k == h.kind) {
                v.items[v.len] = (i, h.sq, h.kind);
                v.len += 1;
            }
        }
        v
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct EpInfo {
    pub id: usize,
    pub landed: Sq,
    pub target: Sq,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Position {
    pub pieces: Vec<Piece>,
    pub turn: Color,
    pub ep: Option<EpInfo>,
}

/// Board identity used to compare states: piece id + kind on every square.
type Key = u128;

/// A fast non-cryptographic hasher (FxHash) for board keys and positions.
/// The default SipHash was a fifth of the AI's run time.
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);

impl std::hash::Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for c in chunks {
            self.write_u64(u64::from_le_bytes(*c));
        }
        for &b in rest {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_u8(&mut self, n: u8) {
        self.write_u64(n as u64);
    }
    fn write_u16(&mut self, n: u16) {
        self.write_u64(n as u64);
    }
    fn write_u32(&mut self, n: u32) {
        self.write_u64(n as u64);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

pub type FastHash = std::hash::BuildHasherDefault<Fx>;

fn key(b: &Board) -> Key {
    b.key
}

#[derive(Debug)]
pub struct States {
    /// Ids of split pieces; bit `i` of a mask picks the half of `splits[i]`.
    pub splits: Vec<usize>,
    /// Valid states.
    pub list: Vec<(u32, Board)>,
    /// Masks whose half choices put two pieces on one square.
    pub invalid: Vec<u32>,
}

/// Mark successor masks that are invalid or whose board key has no legal
/// predecessor branch. `reachable` must be sorted by the caller.
fn unreachable_states(states: &States, reachable: &[Key]) -> Vec<bool> {
    let mut bad = vec![false; 1usize << states.splits.len()];
    for &mask in &states.invalid {
        bad[mask as usize] = true;
    }
    for (mask, board) in &states.list {
        if reachable.binary_search(&key(board)).is_err() {
            bad[*mask as usize] = true;
        }
    }
    bad
}

/// Run one board builder for each choice mask and dispatch valid and invalid
/// results in mask order. Used by `Position::states` after its split bound.
fn visit_state_boards<T>(
    split_count: usize,
    mut build: impl FnMut(u32) -> Option<T>,
    mut on_valid: impl FnMut(u32, T),
    mut on_invalid: impl FnMut(u32),
) {
    for mask in 0..(1u32 << split_count) {
        match build(mask) {
            Some(board) => on_valid(mask, board),
            None => on_invalid(mask),
        }
    }
}

#[cfg_attr(
    feature = "verus",
    verus_spec(result =>
        ensures
            !alive ==> result && *final(occupied) == *old(occupied),
            alive && square >= 64
                ==> !result && *final(occupied) == *old(occupied),
            alive && square < 64
                && *old(occupied) & (1u64 << square) != 0
                ==> !result && *final(occupied) == *old(occupied),
            alive && square < 64
                && *old(occupied) & (1u64 << square) == 0
                ==> result && *final(occupied) == (*old(occupied) | (1u64 << square)),
    )
)]
fn occupy_square(occupied: &mut u64, square: Sq, alive: bool) -> bool {
    if !alive {
        return true;
    }
    if square >= 64 {
        return false;
    }
    let bit = 1u64 << square;
    if *occupied & bit != 0 {
        false
    } else {
        *occupied |= bit;
        true
    }
}

/// Whether the selected half of each split piece fits beside every unsplit
/// piece. This is the allocation-free collision predicate used by `states`.
fn selected_squares_fit(pieces: &[Piece], splits: &[usize], mask: usize) -> bool {
    let mut occupied = 0u64;
    let mut fits = true;
    for piece in pieces {
        if !piece.is_split() {
            fits &= occupy_square(&mut occupied, piece.h[0].sq, piece.h[0].alive);
        }
    }
    for (bit, &id) in splits.iter().enumerate() {
        if id >= pieces.len() || bit >= usize::BITS as usize {
            return false;
        }
        let half = pieces[id].h[(mask >> bit & 1) as usize];
        fits &= occupy_square(&mut occupied, half.sq, half.alive);
    }
    fits
}

impl States {
    /// The boards left after collapse `(mask, val)` (bit `k` of `mask` fixes
    /// `splits[k]` to the half in bit `k` of `val`), indexed for the position
    /// after the collapse, whose split pieces are the others.
    fn restrict(&self, (mask, val): (u32, u32)) -> States {
        let keep: Vec<usize> = (0..self.splits.len())
            .filter(|&k| mask >> k & 1 == 0)
            .collect();
        let squeeze = |m: u32| {
            keep.iter()
                .enumerate()
                .fold(0u32, |acc, (i, &k)| acc | (m >> k & 1) << i)
        };
        States {
            splits: keep.iter().map(|&k| self.splits[k]).collect(),
            list: self
                .list
                .iter()
                .filter(|(m, _)| m & mask == val)
                .map(|(m, b)| (squeeze(*m), b.clone()))
                .collect(),
            invalid: self
                .invalid
                .iter()
                .filter(|&&m| m & mask == val)
                .map(|&m| squeeze(m))
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Action {
    To { to: Sq, promo: Option<Kind> },
    Castle(usize),
    RookTo { from: Sq, to: Sq },
}

/// A classical move that applies in the states where the moving piece's
/// half on `from` (of kind `kind`) is the real one.
#[derive(Clone, Copy, Debug)]
struct Branch {
    from: Sq,
    kind: Kind,
    action: Action,
}

struct Plan {
    piece: usize,
    /// New (square, kind) for each half, `None` = stays.
    dest: [Option<(Sq, Kind)>; 2],
    /// (rook piece, half, destination) for castling.
    rook: Option<(usize, usize, Sq)>,
    branches: Vec<Branch>,
    ep: Option<EpInfo>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateDecision {
    None,
    Unique(usize),
    Ambiguous,
}

#[derive(Default)]
struct CandidateAccumulator {
    first: Option<usize>,
    ambiguous: bool,
}

impl CandidateAccumulator {
    fn record(&mut self, piece: usize) {
        match self.first {
            Some(first) if first != piece => self.ambiguous = true,
            None => self.first = Some(piece),
            _ => {}
        }
    }

    fn decision(self) -> CandidateDecision {
        match (self.first, self.ambiguous) {
            (None, _) => CandidateDecision::None,
            (Some(_), true) => CandidateDecision::Ambiguous,
            (Some(piece), false) => CandidateDecision::Unique(piece),
        }
    }
}

/// Result of a successful move.
pub struct MoveReport {
    pub collapses: Vec<String>,
    /// The piece that moved (the king when castling).
    pub piece: usize,
    /// Number of different minimal collapse choices that were available.
    pub options: usize,
    /// Squares the moving halves left, then the squares they arrived on
    /// (including the rook's squares when castling). No duplicates.
    pub moved: Vec<Sq>,
    /// The new position's boards, when they were already worked out
    /// (nothing collapsed), so the caller needn't list them again.
    pub states: Option<States>,
}

/// Everything about a move that is known before a collapse is chosen.
/// Shared by [`Position::play`] and [`Position::collapse_options`].
struct Analysis {
    plan: Plan,
    /// The position after the move, before any collapse is applied.
    next: Position,
    /// The boards of `next`.
    states: States,
    /// Split pieces of `next`; bit `k` of a cube refers to `splits[k]`.
    splits: Vec<usize>,
    /// Minimal collapse choices as (mask, value); empty if none is needed.
    cubes: Vec<(u32, u32)>,
    /// For each branch of the plan: legal in at least one previous state.
    used: Vec<bool>,
}

impl Position {
    pub fn initial() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("valid start")
    }

    /// Classical position from FEN (placement, side, castling, en passant).
    pub fn from_fen(fen: &str) -> Result<Self, QError> {
        let bad = |w: &str| QError(format!("bad FEN: {w}"));
        let mut parts = fen.split_whitespace();
        let placement = parts.next().ok_or_else(|| bad("empty"))?;
        let turn = match parts.next().unwrap_or("w") {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(bad("side to move")),
        };
        let castling = parts.next().unwrap_or("-");
        let ep = parts.next().unwrap_or("-");

        let mut pieces = Vec::new();
        let rows: Vec<&str> = placement.split('/').collect();
        if rows.len() != 8 {
            return Err(bad("need 8 ranks"));
        }
        for (i, row) in rows.iter().enumerate() {
            let rank = 7 - i as u8;
            let mut file = 0u8;
            for c in row.chars() {
                if let Some(d) = c.to_digit(10) {
                    file += d as u8;
                    continue;
                }
                let kind =
                    Kind::from_letter(c.to_ascii_uppercase()).ok_or_else(|| bad("piece letter"))?;
                let color = if c.is_ascii_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                if file > 7 {
                    return Err(bad("rank too long"));
                }
                let s = sq(file, rank);
                let unmoved = match kind {
                    Kind::King => {
                        let (k, q) = rights_letters(color);
                        s == sq(4, color.home_rank())
                            && (castling.contains(k) || castling.contains(q))
                    }
                    Kind::Rook => {
                        let (k, q) = rights_letters(color);
                        (s == sq(7, color.home_rank()) && castling.contains(k))
                            || (s == sq(0, color.home_rank()) && castling.contains(q))
                    }
                    _ => false,
                };
                let h = Half {
                    sq: s,
                    kind,
                    alive: true,
                    moved: !unmoved,
                };
                pieces.push(Piece {
                    color,
                    h: [h, h],
                    origin: kind,
                    num: 0,
                });
                file += 1;
            }
            if file != 8 {
                return Err(bad("rank length"));
            }
        }
        if pieces.len() > 64 {
            return Err(bad("too many pieces"));
        }
        // Number pawns, rooks and knights per color from the a-file side.
        for color in [Color::White, Color::Black] {
            for kind in [Kind::Pawn, Kind::Rook, Kind::Knight] {
                let mut ids: Vec<usize> = (0..pieces.len())
                    .filter(|&i| pieces[i].color == color && pieces[i].origin == kind)
                    .collect();
                ids.sort_by_key(|&i| (file_of(pieces[i].h[0].sq), rank_of(pieces[i].h[0].sq)));
                for (n, i) in ids.into_iter().enumerate() {
                    pieces[i].num = (n + 1).min(255) as u8;
                }
            }
        }
        let mut pos = Position {
            pieces,
            turn,
            ep: None,
        };
        if ep != "-" {
            let target = parse_sq(ep).ok_or_else(|| bad("en passant square"))?;
            let landed = if turn == Color::White {
                target - 8
            } else {
                target + 8
            };
            if let Some(id) = pos
                .pieces
                .iter()
                .position(|p| p.h[0].alive && p.h[0].sq == landed && p.h[0].kind == Kind::Pawn)
            {
                pos.ep = Some(EpInfo { id, landed, target });
            }
        }
        Ok(pos)
    }

    pub fn splits(&self) -> Vec<usize> {
        (0..self.pieces.len())
            .filter(|&i| self.pieces[i].is_split())
            .collect()
    }

    /// Castling rights and the en passant target of a board whose pieces
    /// are placed; `moved[s]` says whether the half on `s` has moved.
    fn finish_board(&self, b: &mut Board, moved: &[bool; 64]) {
        for c in [Color::White, Color::Black] {
            let home = c.home_rank();
            let at = |f: u8, k: Kind| {
                let s = sq(f, home);
                !moved[s as usize]
                    && matches!(b.sq[s as usize], Some(o) if o.color == c && o.kind == k)
            };
            if at(4, Kind::King) {
                b.castle[c.idx()] = [at(7, Kind::Rook), at(0, Kind::Rook)];
            }
        }
        if let Some(e) = self.ep
            && matches!(b.sq[e.landed as usize], Some(o) if o.id as usize == e.id && o.kind == Kind::Pawn)
        {
            b.ep = Some(e.target);
        }
    }

    /// Build the classical board for one choice mask. `None` means that the
    /// selected halves overlap or contain an invalid live square.
    fn board_for_selection(&self, splits: &[usize], mask: usize) -> Option<Board> {
        let place = |board: &mut Board, moved: &mut [bool; 64], id: usize, half: Half| -> bool {
            if !half.alive {
                return true;
            }
            if half.sq >= 64 || board.sq[half.sq as usize].is_some() {
                return false;
            }
            board.place(
                half.sq,
                Occ {
                    id: id as u8,
                    color: self.pieces[id].color,
                    kind: half.kind,
                },
            );
            moved[half.sq as usize] = half.moved;
            true
        };

        if !selected_squares_fit(&self.pieces, splits, mask) {
            return None;
        }
        let mut board = Board::empty(self.turn);
        let mut moved = [true; 64];
        for (id, piece) in self.pieces.iter().enumerate() {
            if !piece.is_split() && !place(&mut board, &mut moved, id, piece.h[0]) {
                return None;
            }
        }
        for (bit, &id) in splits.iter().enumerate() {
            if !place(
                &mut board,
                &mut moved,
                id,
                self.pieces[id].h[(mask >> bit) & 1],
            ) {
                return None;
            }
        }
        self.finish_board(&mut board, &moved);
        Some(board)
    }

    /// Every way of picking one half per split piece: the boards where no
    /// two pieces share a square, and the masks where they do. The pieces
    /// that aren't split are placed once, on a base board every state
    /// starts from.
    pub fn states(&self) -> Result<States, QError> {
        let splits = self.splits();
        if splits.len() > MAX_SPLIT {
            return Err(QError(format!(
                "too many pieces in superposition ({}; the limit is {MAX_SPLIT})",
                splits.len()
            )));
        }
        let mut list = Vec::with_capacity(1 << splits.len());
        let mut invalid = Vec::new();
        visit_state_boards(
            splits.len(),
            |mask| self.board_for_selection(&splits, mask as usize),
            |mask, board| list.push((mask, board)),
            |mask| invalid.push(mask),
        );
        Ok(States {
            splits,
            list,
            invalid,
        })
    }

    // ------------------------------------------------------------------
    // Playing moves
    // ------------------------------------------------------------------

    /// Work out a move without collapsing anything: the resulting position
    /// and every minimal collapse the mover could choose between.
    fn analyze(&self, prev: &States, intent: &Intent) -> Result<Analysis, QError> {
        let plan = self.plan(prev, intent)?;
        // Every board reachable by a legal classical branch from a previous state.
        let mut reach: Vec<Key> = Vec::with_capacity(prev.list.len() * plan.branches.len());
        let mut real_move = false;
        let mut used = vec![false; plan.branches.len()];
        for (_, b) in &prev.list {
            for (bi, br) in plan.branches.iter().enumerate() {
                let here = matches!(b.sq[br.from as usize],
                    Some(o) if o.id as usize == plan.piece && o.kind == br.kind);
                if !here {
                    continue;
                }
                let m = match br.action {
                    Action::To { to, promo } => b.find_move(br.from, to, promo),
                    Action::Castle(side) => b.find_castle(br.from, side),
                    Action::RookTo { from, to } => b.find_move(from, to, None),
                };
                if let Some(m) = m {
                    reach.push(b.key_after(m));
                    real_move = true;
                    used[bi] = true;
                }
            }
        }
        if !real_move {
            return Err(QError(
                "illegal move: not legal in any possible board state".into(),
            ));
        }

        reach.sort_unstable();
        let next = self.apply_plan(&plan);
        let ns = next.states()?;
        let n = ns.splits.len();
        let bad = unreachable_states(&ns, &reach);
        if bad.iter().all(|&x| x) {
            return Err(QError("illegal move: no consistent outcome".into()));
        }
        let cubes = if bad.iter().any(|&x| x) {
            minimal_collapses(n, &bad)
        } else {
            Vec::new()
        };
        Ok(Analysis {
            plan,
            next,
            splits: ns.splits.clone(),
            states: ns,
            cubes,
            used,
        })
    }

    /// Destination squares among `only` (a bitboard) that the side to
    /// move's piece on `from` can legally reach in at least one possible
    /// state (castling and promotion included). Sorted and deduplicated. A
    /// square already found legal on one board isn't tested on the others.
    pub fn targets(&self, st: &States, from: Sq, only: u64) -> Vec<Sq> {
        let mut found = 0u64;
        for (_, b) in &st.list {
            b.each_pseudo(from, &mut |m| {
                let bit = 1u64 << m.to;
                if only & bit != 0 && found & bit == 0 && b.is_legal(m) {
                    found |= bit;
                }
            });
        }
        let mut out = Vec::with_capacity(found.count_ones() as usize);
        while found != 0 {
            out.push(found.trailing_zeros() as Sq);
            found &= found - 1;
        }
        out
    }

    /// The minimal collapse choices for a move, as `{...}` specs that
    /// [`Position::play`] accepts. Empty when the move needs no collapse.
    pub fn collapse_options(&self, prev: &States, intent: &Intent) -> Result<Vec<String>, QError> {
        let a = self.analyze(prev, intent)?;
        Ok(a.cubes
            .iter()
            .map(|c| a.next.describe_spec(&a.splits, *c))
            .collect())
    }

    /// Play a move. Returns the new position and what collapsed.
    pub fn play(
        &self,
        prev: &States,
        intent: &Intent,
        choice: Option<&[CollapseSpec]>,
        quiet: bool,
    ) -> Result<(Position, MoveReport), QError> {
        let mut a = self.analyze(prev, intent)?;
        let cube = if a.cubes.is_empty() {
            if choice.is_some() {
                return Err(QError("no collapse is needed for this move".into()));
            }
            None
        } else {
            Some(match choice {
                Some(spec) => {
                    let cube = a.next.cube_from_spec(&a.splits, spec)?;
                    if !a.cubes.contains(&cube) {
                        let opts: Vec<String> = a
                            .cubes
                            .iter()
                            .map(|c| a.next.describe_spec(&a.splits, *c))
                            .collect();
                        return Err(QError(format!(
                            "that collapse isn't a valid minimal choice here; options: {}",
                            opts.join(" ")
                        )));
                    }
                    cube
                }
                None => a.next.rank_collapses(&a.splits, &a.cubes, a.plan.piece),
            })
        };
        let (next, mut report, changed) = self.finish(&a, intent, cube, quiet);
        report.states = Self::child_states(&mut a, cube, changed, &next);
        Ok((next, report))
    }

    /// The boards of the position a move produced, taken from its analysis
    /// instead of listed again: all of them when nothing collapsed, the ones
    /// the collapse kept otherwise. `None` when tidying the position changed
    /// it (rare), so they must be listed afresh.
    fn child_states(
        a: &mut Analysis,
        cube: Option<(u32, u32)>,
        changed: bool,
        next: &Position,
    ) -> Option<States> {
        if changed {
            return None;
        }
        let st = match cube {
            None => std::mem::replace(
                &mut a.states,
                States {
                    splits: Vec::new(),
                    list: Vec::new(),
                    invalid: Vec::new(),
                },
            ),
            Some(c) => a.states.restrict(c),
        };
        #[cfg(debug_assertions)]
        {
            let fresh = next.states().expect("states");
            assert_eq!(fresh.splits, st.splits);
            assert_eq!(fresh.invalid, st.invalid);
            assert_eq!(fresh.list.len(), st.list.len());
            for ((m1, b1), (m2, b2)) in fresh.list.iter().zip(&st.list) {
                assert_eq!(
                    (m1, b1.key, b1.castle, b1.ep, b1.turn),
                    (m2, b2.key, b2.castle, b2.ep, b2.turn)
                );
            }
        }
        #[cfg(not(debug_assertions))]
        let _ = next;
        Some(st)
    }

    /// Every way a move can be played: one entry when it needs no collapse
    /// or only one, otherwise one per minimal collapse the mover could
    /// choose, with the collapse ("cube") that picks it. The move is worked
    /// out once, for search; [`Position::describe_cube`] names a cube.
    pub fn play_each(&self, prev: &States, intent: &Intent) -> Result<Vec<Outcome>, QError> {
        let mut a = self.analyze(prev, intent)?;
        if a.cubes.len() <= 1 {
            let cube = a.cubes.first().copied();
            let (next, _, changed) = self.finish(&a, intent, cube, true);
            let states = Self::child_states(&mut a, cube, changed, &next);
            return Ok(vec![(None, next, states)]);
        }
        let cubes = a.cubes.clone();
        Ok(cubes
            .into_iter()
            .map(|c| {
                let (next, _, changed) = self.finish(&a, intent, Some(c), true);
                let states = Self::child_states(&mut a, Some(c), changed, &next);
                (Some(c), next, states)
            })
            .collect())
    }

    /// The `{...}` that picks collapse `cube` of a move, as
    /// [`Position::play_each`] returned it.
    pub fn describe_cube(
        &self,
        prev: &States,
        intent: &Intent,
        cube: (u32, u32),
    ) -> Result<String, QError> {
        let a = self.analyze(prev, intent)?;
        Ok(a.next.describe_spec(&a.splits, cube))
    }

    /// Apply the chosen collapse (if any) to an analysed move. The flag says
    /// whether tidying the result changed it (then the analysis's boards
    /// don't describe it).
    fn finish(
        &self,
        a: &Analysis,
        intent: &Intent,
        cube: Option<(u32, u32)>,
        quiet: bool,
    ) -> (Position, MoveReport, bool) {
        let plan = &a.plan;
        let mut next = a.next.clone();
        let mut report = MoveReport {
            collapses: Vec::new(),
            options: a.cubes.len().max(1),
            piece: plan.piece,
            moved: if quiet {
                Vec::new()
            } else {
                self.moved_squares(plan)
            },
            states: None,
        };
        let mut mover_collapsed = false;
        if let Some(chosen) = cube {
            for (k, &id) in a.splits.iter().enumerate() {
                if chosen.0 >> k & 1 == 1 {
                    let hi = (chosen.1 >> k & 1) as usize;
                    mover_collapsed |= id == plan.piece;
                    if !quiet {
                        report.collapses.push(next.describe_collapse(id, hi));
                    }
                    next.pieces[id].h[1 - hi] = next.pieces[id].h[hi];
                }
            }
        }
        // Name the implicit collapse of a whole-piece move, unless the
        // mover's collapse is already listed above.
        if !quiet
            && !mover_collapsed
            && let Some(c) = self.whole_move_collapse(intent, plan, &a.used)
        {
            report.collapses.insert(0, c);
        }
        let changed = next.normalize();
        (next, report, changed)
    }

    /// Squares left and squares reached by the halves a plan moves.
    fn moved_squares(&self, plan: &Plan) -> Vec<Sq> {
        let mut from = Vec::new();
        let mut to = Vec::new();
        for hi in 0..2 {
            if let Some((t, _)) = plan.dest[hi] {
                from.push(self.pieces[plan.piece].h[hi].sq);
                to.push(t);
            }
        }
        if let Some((r, rh, t)) = plan.rook {
            from.push(self.pieces[r].h[rh].sq);
            to.push(t);
        }
        let mut out: Vec<Sq> = Vec::new();
        for s in from.into_iter().chain(to) {
            if !out.contains(&s) {
                out.push(s);
            }
        }
        out
    }

    fn plan(&self, prev: &States, intent: &Intent) -> Result<Plan, QError> {
        let me = self.turn;
        match *intent {
            Intent::Castle { side, split } => {
                let home = me.home_rank();
                let e = sq(4, home);
                let king = self
                    .pieces
                    .iter()
                    .position(|p| {
                        p.color == me && p.h.iter().any(|h| h.alive && h.kind == Kind::King)
                    })
                    .ok_or_else(|| QError("no king".into()))?;
                let p = &self.pieces[king];
                let hc = (0..2)
                    .find(|&i| {
                        p.h[i].alive && p.h[i].sq == e && p.h[i].kind == Kind::King && !p.h[i].moved
                    })
                    .ok_or_else(|| {
                        QError("illegal move: the king has no half that can still castle".into())
                    })?;
                let (rf, rt, kt) = if side == KING_SIDE {
                    (7, 5, 6)
                } else {
                    (0, 3, 2)
                };
                let corner = sq(rf, home);
                let (rook, rh) = self
                    .pieces
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.color == me)
                    .find_map(|(id, p)| {
                        (0..2)
                            .find(|&i| {
                                p.h[i].alive
                                    && p.h[i].sq == corner
                                    && p.h[i].kind == Kind::Rook
                                    && !p.h[i].moved
                            })
                            .map(|i| (id, i))
                    })
                    .ok_or_else(|| QError("illegal move: no unmoved rook to castle with".into()))?;
                let ho = 1 - hc;
                let mut dest = [None, None];
                dest[hc] = Some((sq(kt, home), Kind::King));
                let mut branches = vec![Branch {
                    from: e,
                    kind: Kind::King,
                    action: Action::Castle(side),
                }];
                if split {
                    let other = p.h[ho];
                    if !other.alive {
                        return Err(QError(
                            "illegal move: can't quantum castle with a half-captured king".into(),
                        ));
                    }
                    branches.push(Branch {
                        from: other.sq,
                        kind: Kind::King,
                        action: Action::RookTo {
                            from: corner,
                            to: sq(rt, home),
                        },
                    });
                } else if p.h[ho].alive {
                    dest[ho] = Some((sq(kt, home), Kind::King));
                }
                Ok(Plan {
                    piece: king,
                    dest,
                    rook: Some((rook, rh, sq(rt, home))),
                    branches,
                    ep: None,
                })
            }
            Intent::Move {
                kind,
                hint,
                to,
                promo,
                second,
            } => self.plan_move(prev, kind, hint, to, promo, second),
        }
    }

    fn reach(prev: &States, id: usize, from: Sq, kind: Kind, to: Sq, promo: Option<Kind>) -> bool {
        prev.list.iter().any(|(_, b)| {
            matches!(b.sq[from as usize], Some(o) if o.id as usize == id && o.kind == kind)
                && b.find_move(from, to, promo).is_some()
        })
    }

    fn plan_move(
        &self,
        prev: &States,
        kind: Kind,
        hint: Hint,
        to: Sq,
        promo: Option<Kind>,
        second: Option<Second>,
    ) -> Result<Plan, QError> {
        let me = self.turn;
        // Candidate piece ids, without allocating on the ordinary path.
        let mut candidates = CandidateAccumulator::default();
        for (id, p) in self.pieces.iter().enumerate() {
            if p.color != me {
                continue;
            }
            let locs = p.locs();
            let reaches_first = locs.iter().any(|&(_, from, piece_kind)| {
                piece_kind == kind
                    && hint.matches(from)
                    && Self::reach(prev, id, from, piece_kind, to, promo)
            });
            if !reaches_first {
                continue;
            }
            // A split move like `Ne7/f6` must identify one piece that can
            // reach both destinations. The branch pairing is checked below.
            let reaches_second = match second {
                Some(Second::To {
                    hint: hint2,
                    to: to2,
                    promo: promo2,
                }) => locs.iter().any(|&(_, from, piece_kind)| {
                    piece_kind == kind
                        && hint2.matches(from)
                        && Self::reach(prev, id, from, piece_kind, to2, promo2)
                }),
                _ => true,
            };
            if !reaches_second {
                continue;
            }
            candidates.record(id);
        }
        let piece = match candidates.decision() {
            CandidateDecision::None => {
                return Err(QError(format!(
                    "illegal move: no {} can move to {}",
                    kind.name(),
                    sq_name(to)
                )));
            }
            CandidateDecision::Ambiguous => {
                let mut from = Vec::new();
                for (id, p) in self.pieces.iter().enumerate() {
                    if p.color != me {
                        continue;
                    }
                    let locs = p.locs();
                    let reaches_second = match second {
                        Some(Second::To {
                            hint: hint2,
                            to: to2,
                            promo: promo2,
                        }) => locs.iter().any(|&(_, square, piece_kind)| {
                            piece_kind == kind
                                && hint2.matches(square)
                                && Self::reach(prev, id, square, piece_kind, to2, promo2)
                        }),
                        _ => true,
                    };
                    if reaches_second {
                        from.extend(locs.iter().filter_map(|&(_, square, piece_kind)| {
                            (piece_kind == kind
                                && hint.matches(square)
                                && Self::reach(prev, id, square, piece_kind, to, promo))
                            .then(|| sq_name(square))
                        }));
                    }
                }
                return Err(QError(format!(
                    "ambiguous: {}s on {} can all reach {}",
                    kind.name(),
                    from.join(", "),
                    sq_name(to)
                )));
            }
            CandidateDecision::Unique(piece) => piece,
        };
        let p = &self.pieces[piece];
        let locs = p.locs();
        let new_kind = |k: Kind, dst: Sq, pr: Option<Kind>| {
            let last = if me == Color::White { 7 } else { 0 };
            if k == Kind::Pawn && rank_of(dst) == last {
                pr.unwrap_or(Kind::Queen)
            } else {
                k
            }
        };
        let to_branch = |(_, s, k): (usize, Sq, Kind), dst: Sq, pr: Option<Kind>| Branch {
            from: s,
            kind: k,
            action: Action::To { to: dst, promo: pr },
        };
        let mut dest = [None, None];
        let mut branches = Vec::new();

        match second {
            None => {
                for hi in 0..2 {
                    let h = p.h[hi];
                    if h.alive {
                        dest[hi] = Some((to, new_kind(h.kind, to, promo)));
                    }
                }
                for l in locs.iter() {
                    branches.push(to_branch(*l, to, promo));
                }
            }
            Some(Second::Stay) => {
                // On the boards where the staying half was the real one,
                // nothing would have moved: not a legal chess move. Only
                // quantum castling leaves half the king home, because the
                // rook moves on those boards.
                return Err(QError(
                    "illegal move: a piece can't leave half of itself behind (only quantum castling, 0-0/ or 0-0-0/, keeps half the king home)".into(),
                ));
            }
            Some(Second::To {
                hint: hint2,
                to: to2,
                promo: promo2,
            }) => {
                if to2 == to && promo2 == promo {
                    return Err(QError(
                        "both halves can't make the same move; drop the '/'".into(),
                    ));
                }
                if !p.is_split() {
                    let l = locs[0];
                    if !hint2.matches(l.1) || !Self::reach(prev, piece, l.1, l.2, to2, promo2) {
                        return Err(QError(format!(
                            "illegal move: the {} can't also reach {}",
                            kind.name(),
                            sq_name(to2)
                        )));
                    }
                    dest[0] = Some((to, new_kind(l.2, to, promo)));
                    dest[1] = Some((to2, new_kind(l.2, to2, promo2)));
                    branches.push(to_branch(l, to, promo));
                    branches.push(to_branch(l, to2, promo2));
                } else {
                    if locs.len() < 2 {
                        return Err(QError(
                            "illegal move: can't split a piece that is partly captured".into(),
                        ));
                    }
                    let (a, b) = (locs[0], locs[1]);
                    let ok = |x: (usize, Sq, Kind), y: (usize, Sq, Kind)| {
                        x.2 == kind
                            && y.2 == kind
                            && hint.matches(x.1)
                            && hint2.matches(y.1)
                            && Self::reach(prev, piece, x.1, x.2, to, promo)
                            && Self::reach(prev, piece, y.1, y.2, to2, promo2)
                    };
                    let (x, y) = match (ok(a, b), ok(b, a)) {
                        (true, false) => (a, b),
                        (false, true) => (b, a),
                        (true, true) => {
                            return Err(QError(format!(
                                "ambiguous: either half could go to {}; disambiguate like {}{}/{}{}",
                                sq_name(to),
                                sq_name(a.1),
                                sq_name(to),
                                sq_name(b.1),
                                sq_name(to2)
                            )));
                        }
                        (false, false) => {
                            return Err(QError(format!(
                                "illegal move: the halves can't reach {} and {}",
                                sq_name(to),
                                sq_name(to2)
                            )));
                        }
                    };
                    dest[x.0] = Some((to, new_kind(x.2, to, promo)));
                    dest[y.0] = Some((to2, new_kind(y.2, to2, promo2)));
                    branches.push(to_branch(x, to, promo));
                    branches.push(to_branch(y, to2, promo2));
                }
            }
        }

        // En passant becomes available after a two-square pawn push.
        let mut ep = None;
        for hi in 0..2 {
            if let Some((t, _)) = dest[hi] {
                let f = p.h[hi].sq;
                // Only a real double push: from the pawn's own second rank,
                // two squares forward (not, say, an impossible branch of a
                // whole-piece move that happens to span two ranks).
                let (start, fwd) = if p.color == Color::White {
                    (1, 3)
                } else {
                    (6, 4)
                };
                if p.h[hi].kind == Kind::Pawn
                    && file_of(f) == file_of(t)
                    && rank_of(f) == start
                    && rank_of(t) == fwd
                {
                    ep = Some(EpInfo {
                        id: piece,
                        landed: t,
                        target: (f + t) / 2,
                    });
                }
            }
        }
        Ok(Plan {
            piece,
            dest,
            rook: None,
            branches,
            ep,
        })
    }

    fn apply_plan(&self, plan: &Plan) -> Position {
        let me = self.turn;
        let mut n = self.clone();
        n.turn = me.other();
        n.ep = plan.ep;
        let kill_at = |n: &mut Position, s: Sq| {
            for q in n.pieces.iter_mut().filter(|q| q.color != me) {
                for h in q.h.iter_mut() {
                    if h.alive && h.sq == s {
                        h.alive = false;
                    }
                }
            }
        };
        for hi in 0..2 {
            if let Some((to, kind)) = plan.dest[hi] {
                let old = self.pieces[plan.piece].h[hi];
                n.pieces[plan.piece].h[hi] = Half {
                    sq: to,
                    kind,
                    alive: true,
                    moved: true,
                };
                kill_at(&mut n, to);
                if let Some(e) = self.ep
                    && old.kind == Kind::Pawn
                    && file_of(old.sq) != file_of(to)
                    && e.target == to
                {
                    for h in n.pieces[e.id].h.iter_mut() {
                        if h.alive && h.sq == e.landed {
                            h.alive = false;
                        }
                    }
                }
            }
        }
        if let Some((r, rh, to)) = plan.rook {
            let corner = self.pieces[r].h[rh].sq;
            for h in n.pieces[r]
                .h
                .iter_mut()
                .filter(|h| h.alive && h.sq == corner)
            {
                h.sq = to;
                h.moved = true;
            }
        }
        n
    }

    /// Merge halves that ended up identical. Says whether anything changed.
    fn normalize(&mut self) -> bool {
        let mut changed = false;
        for p in &mut self.pieces {
            if !p.is_split() {
                let moved = p.h[0].moved || p.h[1].moved;
                let merged = Half { moved, ..p.h[0] };
                changed |= p.h[0] != merged || p.h[1] != merged;
                p.h = [merged, merged];
            }
        }
        if let Some(e) = self.ep {
            let p = &self.pieces[e.id];
            if !p.h.iter().any(|h| h.alive && h.sq == e.landed) {
                self.ep = None;
                changed = true;
            }
        }
        changed
    }

    fn cube_from_spec(
        &self,
        splits: &[usize],
        spec: &[CollapseSpec],
    ) -> Result<(u32, u32), QError> {
        let (mut mask, mut val) = (0u32, 0u32);
        for c in spec {
            let mut hits = Vec::new();
            for (k, &id) in splits.iter().enumerate() {
                for hi in 0..2 {
                    let h = self.pieces[id].h[hi];
                    if h.sq == c.sq
                        && h.alive != c.dead
                        && c.kind.is_none_or(|kd| kd == h.kind)
                        && c.num.is_none_or(|n| n == self.pieces[id].num)
                    {
                        hits.push((k, hi));
                    }
                }
            }
            hits.dedup_by_key(|x| x.0);
            let &[(k, hi)] = hits.as_slice() else {
                return Err(QError(format!(
                    "collapse choice {}{}{}{} doesn't name exactly one quantum piece",
                    if c.dead { "x" } else { "" },
                    c.kind.map(|k| k.letter().to_string()).unwrap_or_default(),
                    c.num.map(|n| n.to_string()).unwrap_or_default(),
                    sq_name(c.sq)
                )));
            };
            mask |= 1 << k;
            val |= (hi as u32) << k;
        }
        Ok((mask, val))
    }

    /// `{...}` token keeping half `hi` of piece `id`. The piece number is
    /// added only when another quantum piece has a matching half there.
    fn spec_token(&self, splits: &[usize], id: usize, hi: usize) -> String {
        let h = self.pieces[id].h[hi];
        let clash = splits.iter().any(|&o| {
            o != id
                && self.pieces[o]
                    .h
                    .iter()
                    .any(|x| x.sq == h.sq && x.kind == h.kind && x.alive == h.alive)
        });
        format!(
            "{}{}{}{}",
            if h.alive { "" } else { "x" },
            h.kind.letter(),
            if clash {
                self.pieces[id].num.to_string()
            } else {
                String::new()
            },
            sq_name(h.sq)
        )
    }

    fn describe_spec(&self, splits: &[usize], cube: (u32, u32)) -> String {
        let toks: Vec<String> = splits
            .iter()
            .enumerate()
            .filter(|(k, _)| cube.0 >> k & 1 == 1)
            .map(|(k, &id)| self.spec_token(splits, id, (cube.1 >> k & 1) as usize))
            .collect();
        format!("{{{}}}", toks.join(","))
    }

    /// Moving a split piece as a whole (`Nd5`, no `/`) when only one of its
    /// halves can make the move collapses it onto that half first. The piece
    /// ends up unsplit, so this does not show up among the cubes; name it.
    fn whole_move_collapse(&self, intent: &Intent, plan: &Plan, used: &[bool]) -> Option<String> {
        if !matches!(intent, Intent::Move { second: None, .. }) {
            return None;
        }
        let p = &self.pieces[plan.piece];
        if p.locs().len() != 2 {
            return None;
        }
        let live: Vec<usize> = (0..plan.branches.len()).filter(|&i| used[i]).collect();
        let [only] = live[..] else { return None };
        let from = plan.branches[only].from;
        let hi = (0..2).find(|&h| p.h[h].alive && p.h[h].sq == from)?;
        Some(self.describe_collapse(plan.piece, hi))
    }

    fn describe_collapse(&self, id: usize, hi: usize) -> String {
        let p = &self.pieces[id];
        let h = p.h[hi];
        if h.alive {
            format!("{} collapsed to {}", p.name(h.kind), sq_name(h.sq))
        } else {
            format!(
                "{} collapsed to captured (on {})",
                p.name(h.kind),
                sq_name(h.sq)
            )
        }
    }

    /// Default choice among minimal collapses: capture as much as possible
    /// for the mover, then avoid collapsing the piece that just moved.
    fn rank_collapses(&self, splits: &[usize], cubes: &[(u32, u32)], mover: usize) -> (u32, u32) {
        let me = self.turn.other(); // `self` is the post-move position
        let score = |c: &(u32, u32)| {
            let mut enemy_dead = 0i32;
            let mut touches_mover = 0i32;
            for (id, p) in self.pieces.iter().enumerate() {
                let mut h = p.h;
                if let Some(k) = splits.iter().position(|&s| s == id)
                    && c.0 >> k & 1 == 1
                {
                    let hi = (c.1 >> k & 1) as usize;
                    h = [p.h[hi], p.h[hi]];
                    if id == mover {
                        touches_mover = 1;
                    }
                }
                if p.color != me {
                    enemy_dead += h.iter().filter(|x| !x.alive).count() as i32;
                }
            }
            (enemy_dead, -touches_mover)
        };
        let mut best = cubes[0];
        for c in &cubes[1..] {
            if score(c) > score(&best) {
                best = *c;
            }
        }
        best
    }
}

fn rights_letters(c: Color) -> (char, char) {
    match c {
        Color::White => ('K', 'Q'),
        Color::Black => ('k', 'q'),
    }
}

/// All smallest "cubes" (fixed bits `mask` set to `val`) that exclude every
/// bad state while keeping at least one good state.
fn minimal_collapses(n: usize, bad: &[bool]) -> Vec<(u32, u32)> {
    let mut cubes = Vec::new();
    enumerate_minimal_collapses(n, bad, |cube| cubes.push(cube));
    cubes
}

fn enumerate_minimal_collapses(n: usize, bad: &[bool], mut emit: impl FnMut((u32, u32))) -> bool {
    if n > 32 || n >= usize::BITS as usize || bad.len() < (1usize << n) {
        return false;
    }
    let total = 1usize << n;
    if !(0..total).any(|state| !bad[state]) {
        return false;
    }
    // Only bits that influence badness can matter.
    let mut relevant = [0usize; 32];
    let mut r = 0;
    for bit in 0..n {
        let mut influences = false;
        for state in 0..total {
            if bad[state] != bad[state ^ (1usize << bit)] {
                influences = true;
                break;
            }
        }
        if influences {
            relevant[r] = bit;
            r += 1;
        }
    }
    for k in 0..=r {
        let mut found = false;
        for sub in 0u64..(1u64 << r) {
            if sub.count_ones() as usize != k {
                continue;
            }
            let mut mask = 0u32;
            for (i, &bit) in relevant[..r].iter().enumerate() {
                if sub >> i & 1 == 1 {
                    mask |= 1u32 << bit;
                }
            }
            for v in 0u64..(1u64 << k) {
                // Spread v over the bits of mask.
                let mut val = 0u32;
                let mut vi = 0;
                for bit in 0..n {
                    if mask >> bit & 1 == 1 {
                        val |= ((v >> vi & 1) as u32) << bit;
                        vi += 1;
                    }
                }
                let mut excludes_bad = true;
                let mut keeps_good = false;
                for state in 0..total {
                    if state as u32 & mask == val {
                        if bad[state] {
                            excludes_bad = false;
                            break;
                        }
                        keeps_good = true;
                    }
                }
                if excludes_bad && keeps_good {
                    emit((mask, val));
                    found = true;
                }
            }
        }
        if found {
            return true;
        }
    }
    // Unreachable when at least one good state exists: fixing every relevant
    // bit to a good state's values always emits a cube.
    let g = (0..total)
        .find(|&state| !bad[state])
        .expect("good state checked above") as u32;
    let mut mask = 0u32;
    for &bit in &relevant[..r] {
        mask |= 1u32 << bit;
    }
    emit((mask, g & mask));
    true
}

#[cfg(kani)]
mod verification {
    #[kani::proof]
    fn one_split_board_builder_matches_unsplit_blocker_model() {
        use super::{Half, Piece, Position};
        use crate::board::Occ;
        use crate::types::{Color, Kind};

        let first: u8 = kani::any();
        let second: u8 = kani::any();
        kani::assume(first < 64 && second < 64 && first != second);
        let half = |sq| Half {
            sq,
            kind: Kind::Rook,
            alive: true,
            moved: true,
        };
        let position = Position {
            pieces: vec![
                Piece {
                    color: Color::White,
                    h: [half(first), half(second)],
                    origin: Kind::Rook,
                    num: 1,
                },
                Piece {
                    color: Color::Black,
                    h: [half(second), half(second)],
                    origin: Kind::Rook,
                    num: 1,
                },
            ],
            turn: Color::White,
            ep: None,
        };
        for mask in 0..2 {
            let selected = if mask == 0 { first } else { second };
            let expected = selected != second;
            let board = position.board_for_selection(&[0], mask);
            assert_eq!(board.is_some(), expected);
            if let Some(board) = board {
                assert_eq!(
                    board.sq[selected as usize],
                    Some(Occ {
                        id: 0,
                        color: Color::White,
                        kind: Kind::Rook,
                    })
                );
                assert_eq!(
                    board.sq[second as usize],
                    Some(Occ {
                        id: 1,
                        color: Color::Black,
                        kind: Kind::Rook,
                    })
                );
            }
        }
    }

    #[kani::proof]
    fn one_split_states_materializes_both_boards_with_real_board_builder() {
        use super::{Half, Piece, Position};
        use crate::board::Occ;
        use crate::types::{Color, Kind};

        let first: u8 = 8;
        let second: u8 = 9;
        let half = |sq, kind| Half {
            sq,
            kind,
            alive: true,
            moved: true,
        };
        let position = Position {
            pieces: vec![Piece {
                color: Color::White,
                h: [half(first, Kind::Rook), half(second, Kind::Rook)],
                origin: Kind::Rook,
                num: 1,
            }],
            turn: Color::White,
            ep: None,
        };
        let states = position.states().unwrap();
        assert_eq!(states.splits, vec![0]);
        assert!(states.invalid.is_empty());
        assert_eq!(states.list.len(), 2);
        for (index, selected) in [first, second].into_iter().enumerate() {
            let (mask, board) = &states.list[index];
            assert_eq!(*mask, index as u32);
            assert_eq!(board.turn, Color::White);
            assert_eq!(
                board.sq[selected as usize],
                Some(Occ {
                    id: 0,
                    color: Color::White,
                    kind: Kind::Rook,
                })
            );
            assert_eq!(board.by_color, [1u64 << selected, 0]);
            assert_eq!(board.by_kind[Kind::Rook as usize], 1u64 << selected);
            assert_eq!(board.by_kind[Kind::Bishop as usize], 0);
            assert_eq!(board.by_kind[Kind::Pawn as usize], 0);
            assert_eq!(board.by_kind[Kind::Knight as usize], 0);
            assert_eq!(board.by_kind[Kind::Queen as usize], 0);
            assert_eq!(board.by_kind[Kind::King as usize], 0);
        }
    }

    #[kani::proof]
    fn two_split_states_separates_valid_and_colliding_boards() {
        use super::{Half, Piece, Position};
        use crate::board::Occ;
        use crate::types::{Color, Kind};

        let half = |sq, kind| Half {
            sq,
            kind,
            alive: true,
            moved: true,
        };
        let position = Position {
            pieces: vec![
                Piece {
                    color: Color::White,
                    h: [half(8, Kind::Rook), half(9, Kind::Rook)],
                    origin: Kind::Rook,
                    num: 1,
                },
                Piece {
                    color: Color::Black,
                    h: [half(9, Kind::Bishop), half(10, Kind::Bishop)],
                    origin: Kind::Bishop,
                    num: 0,
                },
            ],
            turn: Color::White,
            ep: None,
        };

        let states = position.states().unwrap();
        assert_eq!(states.splits, vec![0, 1]);
        assert_eq!(states.invalid, vec![1]);
        assert_eq!(states.list.len(), 3);
        for (index, (mask, white_square, black_square)) in
            [(0, 8, 9), (2, 8, 10), (3, 9, 10)].into_iter().enumerate()
        {
            let (actual_mask, board) = &states.list[index];
            assert_eq!(*actual_mask, mask);
            assert_eq!(
                board.sq[white_square],
                Some(Occ {
                    id: 0,
                    color: Color::White,
                    kind: Kind::Rook,
                })
            );
            assert_eq!(
                board.sq[black_square],
                Some(Occ {
                    id: 1,
                    color: Color::Black,
                    kind: Kind::Bishop,
                })
            );
            assert_eq!(board.by_color, [1u64 << white_square, 1u64 << black_square]);
            assert_eq!(board.by_kind[Kind::Rook as usize], 1u64 << white_square);
            assert_eq!(board.by_kind[Kind::Bishop as usize], 1u64 << black_square);
        }
    }

    #[kani::proof]
    fn successor_classifier_matches_every_three_key_reachability_pattern() {
        use super::{States, unreachable_states};
        use crate::board::Board;
        use crate::types::Color;

        let reachable_mask: u8 = kani::any();
        kani::assume(reachable_mask < 8);
        let board_with_key = |board_key| {
            let mut board = Board::empty(Color::White);
            board.key = board_key;
            board
        };
        let states = States {
            splits: vec![0, 1],
            list: vec![
                (0, board_with_key(10)),
                (1, board_with_key(20)),
                (2, board_with_key(30)),
            ],
            invalid: vec![3],
        };
        let reachable = [
            if reachable_mask & 1 != 0 { 10 } else { 11 },
            if reachable_mask & 2 != 0 { 20 } else { 21 },
            if reachable_mask & 4 != 0 { 30 } else { 31 },
        ];

        let bad = unreachable_states(&states, &reachable);
        assert_eq!(bad.len(), 4);
        for index in 0..3 {
            assert_eq!(bad[index], reachable_mask >> index & 1 == 0);
        }
        assert!(bad[3]);
    }

    #[kani::proof]
    fn candidate_accumulator_matches_pairwise_unique_piece_spec() {
        use super::{CandidateAccumulator, CandidateDecision};

        let piece_ids: [u8; 64] = kani::any();
        let active: u64 = kani::any();

        let mut candidates = CandidateAccumulator::default();
        for index in 0..piece_ids.len() {
            if active >> index & 1 != 0 {
                candidates.record(piece_ids[index] as usize);
            }
        }
        let decision = candidates.decision();

        if active == 0 {
            assert_eq!(decision, CandidateDecision::None);
        } else {
            let mut found_first = false;
            let mut first_id = 0u8;
            for index in 0..piece_ids.len() {
                if active >> index & 1 != 0 && !found_first {
                    first_id = piece_ids[index];
                    found_first = true;
                }
            }
            assert!(found_first);
            let mut has_distinct_ids = false;
            for index in 0..piece_ids.len() {
                has_distinct_ids |= active >> index & 1 != 0 && piece_ids[index] != first_id;
            }
            if has_distinct_ids {
                assert_eq!(decision, CandidateDecision::Ambiguous);
            } else {
                assert_eq!(decision, CandidateDecision::Unique(first_id as usize));
            }
        }
    }

    #[kani::proof]
    fn apply_plan_moves_both_halves_and_marks_captured_piece_dead() {
        use super::{Action, Branch, Half, Piece, Plan, Position};
        use crate::types::{Color, Kind};

        let half = |sq| Half {
            sq,
            kind: Kind::Knight,
            alive: true,
            moved: true,
        };
        let position = Position {
            pieces: vec![
                Piece {
                    color: Color::White,
                    h: [half(10), half(10)],
                    origin: Kind::Knight,
                    num: 1,
                },
                Piece {
                    color: Color::Black,
                    h: [Half {
                        kind: Kind::Rook,
                        ..half(17)
                    }; 2],
                    origin: Kind::Rook,
                    num: 0,
                },
            ],
            turn: Color::White,
            ep: None,
        };
        let plan = Plan {
            piece: 0,
            dest: [Some((17, Kind::Knight)), Some((17, Kind::Knight))],
            rook: None,
            branches: vec![Branch {
                from: 10,
                kind: Kind::Knight,
                action: Action::To {
                    to: 17,
                    promo: None,
                },
            }],
            ep: None,
        };
        let next = position.apply_plan(&plan);
        assert_eq!(next.turn, Color::Black);
        assert_eq!(next.ep, None);
        assert_eq!(next.pieces[0].h[0].sq, 17);
        assert_eq!(next.pieces[0].h[1].sq, 17);
        assert!(next.pieces[0].h[0].alive && next.pieces[0].h[1].alive);
        assert_eq!(next.pieces[0].h[0].kind, Kind::Knight);
        assert_eq!(next.pieces[0].h[1].kind, Kind::Knight);
        assert!(next.pieces[0].h[0].moved && next.pieces[0].h[1].moved);
        assert!(!next.pieces[1].h[0].alive && !next.pieces[1].h[1].alive);
        assert_eq!(next.pieces[1].h[0].sq, 17);
        assert_eq!(next.pieces[1].h[1].sq, 17);
    }

    #[kani::proof]
    #[kani::unwind(33)]
    fn state_visitor_dispatches_all_five_split_masks_in_order() {
        let valid_masks: u32 = kani::any();
        let mut seen_valid = 0u32;
        let mut seen_invalid = 0u32;
        let mut last_valid = 0u32;
        let mut last_invalid = 0u32;
        let mut has_valid = false;
        let mut has_invalid = false;
        super::visit_state_boards(
            5,
            |mask| ((valid_masks >> mask) & 1 != 0).then_some(mask),
            |mask, value| {
                assert_eq!(mask, value);
                assert_eq!((seen_valid >> mask) & 1, 0);
                assert!(!has_valid || mask > last_valid);
                seen_valid |= 1u32 << mask;
                last_valid = mask;
                has_valid = true;
            },
            |mask| {
                assert_eq!((seen_invalid >> mask) & 1, 0);
                assert!(!has_invalid || mask > last_invalid);
                seen_invalid |= 1u32 << mask;
                last_invalid = mask;
                has_invalid = true;
            },
        );
        assert_eq!(seen_valid, valid_masks);
        assert_eq!(seen_invalid, !valid_masks);
        assert_eq!(seen_valid & seen_invalid, 0);
    }

    fn assert_one_split_case(bad: [bool; 2], expected: u8) {
        let mut emitted = [false; 4];
        let found = super::enumerate_minimal_collapses(1, &bad, |(mask, value)| {
            emitted[(mask * 2 + value) as usize] = true;
        });
        let mut actual = 0u8;
        for (index, cube) in emitted.into_iter().enumerate() {
            if cube {
                actual |= 1 << index;
            }
        }
        assert_eq!(found, expected != 0);
        assert_eq!(actual, expected);
    }

    #[kani::proof]
    fn all_one_split_truth_tables_match_the_minimal_collapse_spec() {
        assert_one_split_case([false, false], 0b0001);
        assert_one_split_case([false, true], 0b0100);
        assert_one_split_case([true, false], 0b1000);
        assert_one_split_case([true, true], 0);
    }

    fn assert_two_split_case(bad: [bool; 4]) {
        let mut emitted = 0u16;
        let found = super::enumerate_minimal_collapses(2, &bad, |(mask, value)| {
            emitted |= 1u16 << (mask * 4 + value);
        });
        if bad[0] && bad[1] && bad[2] && bad[3] {
            assert!(!found);
            assert_eq!(emitted, 0);
            return;
        }
        assert!(found);
        let mut minimum_fixed_bits = 3;
        for mask in 0..4u32 {
            for value in 0..4u32 {
                if value & !mask != 0 {
                    continue;
                }
                let mut touches_bad = false;
                let mut keeps_good = false;
                for state in 0..4u32 {
                    if state & mask == value {
                        if bad[state as usize] {
                            touches_bad = true;
                        } else {
                            keeps_good = true;
                        }
                    }
                }
                if keeps_good && !touches_bad {
                    minimum_fixed_bits = minimum_fixed_bits.min(mask.count_ones());
                }
            }
        }
        assert!(minimum_fixed_bits <= 2);
        for mask in 0..4u32 {
            for value in 0..4u32 {
                let mut touches_bad = false;
                let mut keeps_good = false;
                for state in 0..4u32 {
                    if state & mask == value {
                        if bad[state as usize] {
                            touches_bad = true;
                        } else {
                            keeps_good = true;
                        }
                    }
                }
                let is_minimal = value & !mask == 0
                    && keeps_good
                    && !touches_bad
                    && mask.count_ones() == minimum_fixed_bits;
                let was_emitted = emitted & (1u16 << (mask * 4 + value)) != 0;
                assert_eq!(was_emitted, is_minimal);
            }
        }
    }

    #[kani::proof]
    fn two_split_truth_tables_00_to_03_match_the_minimal_collapse_spec() {
        assert_two_split_case([false, false, false, false]);
        assert_two_split_case([false, false, false, true]);
        assert_two_split_case([false, false, true, false]);
        assert_two_split_case([false, false, true, true]);
    }

    #[kani::proof]
    fn two_split_truth_tables_04_to_07_match_the_minimal_collapse_spec() {
        assert_two_split_case([false, true, false, false]);
        assert_two_split_case([false, true, false, true]);
        assert_two_split_case([false, true, true, false]);
        assert_two_split_case([false, true, true, true]);
    }

    #[kani::proof]
    fn two_split_truth_tables_08_to_11_match_the_minimal_collapse_spec() {
        assert_two_split_case([true, false, false, false]);
        assert_two_split_case([true, false, false, true]);
        assert_two_split_case([true, false, true, false]);
        assert_two_split_case([true, false, true, true]);
    }

    #[kani::proof]
    fn two_split_truth_tables_12_to_15_match_the_minimal_collapse_spec() {
        assert_two_split_case([true, true, false, false]);
        assert_two_split_case([true, true, false, true]);
        assert_two_split_case([true, true, true, false]);
        assert_two_split_case([true, true, true, true]);
    }

    #[kani::proof]
    fn states_enumerates_both_halves_of_one_split_piece() {
        use super::{Half, Piece, Position};
        use crate::board::Occ;
        use crate::types::{Color, Kind};

        let first: u8 = 20;
        let second: u8 = 45;
        let half = |sq, moved| Half {
            sq,
            kind: Kind::Rook,
            alive: true,
            moved,
        };
        let position = Position {
            pieces: vec![Piece {
                color: Color::White,
                h: [half(first, false), half(second, true)],
                origin: Kind::Rook,
                num: 1,
            }],
            turn: Color::White,
            ep: None,
        };

        let states = position.states().unwrap();
        assert_eq!(states.splits, vec![0]);
        assert!(states.invalid.is_empty());
        assert_eq!(states.list.len(), 2);
        let mut seen = 0u8;
        for (mask, board) in states.list {
            assert!(mask == 0 || mask == 1);
            seen |= 1 << mask;
            let square = if mask == 0 { first } else { second };
            assert_eq!(
                board.sq[square as usize],
                Some(Occ {
                    id: 0,
                    color: Color::White,
                    kind: Kind::Rook,
                })
            );
            assert_eq!(board.by_color[Color::White.idx()], 1u64 << square);
            assert_eq!(board.by_color[Color::Black.idx()], 0);
            assert_eq!(board.by_kind[Kind::Rook as usize], 1u64 << square);
        }
        assert_eq!(seen, 0b11);
    }

    #[kani::proof]
    fn states_classifies_two_split_piece_collisions_exactly() {
        use super::{Half, Piece, Position};
        use crate::board::Occ;
        use crate::types::{Color, Kind};

        let half = |sq| Half {
            sq,
            kind: Kind::Rook,
            alive: true,
            moved: true,
        };
        let position = Position {
            pieces: vec![
                Piece {
                    color: Color::White,
                    h: [half(20), half(21)],
                    origin: Kind::Rook,
                    num: 1,
                },
                Piece {
                    color: Color::White,
                    h: [half(21), half(22)],
                    origin: Kind::Rook,
                    num: 2,
                },
            ],
            turn: Color::White,
            ep: None,
        };

        let states = position.states().unwrap();
        assert_eq!(states.splits, vec![0, 1]);
        assert_eq!(states.invalid, vec![1]);
        assert_eq!(states.list.len(), 3);
        let mut seen = 0u8;
        for (mask, board) in states.list {
            assert!(mask == 0 || mask == 2 || mask == 3);
            seen |= 1 << mask;
            let first = if mask & 1 == 0 { 20 } else { 21 };
            let second = if mask & 2 == 0 { 21 } else { 22 };
            assert_eq!(
                board.sq[first],
                Some(Occ {
                    id: 0,
                    color: Color::White,
                    kind: Kind::Rook,
                })
            );
            assert_eq!(
                board.sq[second],
                Some(Occ {
                    id: 1,
                    color: Color::White,
                    kind: Kind::Rook,
                })
            );
            assert_eq!(
                board.by_color[Color::White.idx()],
                (1u64 << first) | (1u64 << second)
            );
        }
        assert_eq!(seen, 0b1101);
    }

    #[kani::proof]
    fn states_builds_exact_boards_for_three_piece_position() {
        use super::{Half, Piece, Position};
        use crate::board::Occ;
        use crate::types::{Color, Kind};

        let squares = [0, 1, 1, 2, 2, 3, 4];

        let half = |sq| Half {
            sq,
            kind: Kind::Rook,
            alive: true,
            moved: true,
        };
        let pieces = vec![
            Piece {
                color: Color::White,
                h: [half(squares[0]), half(squares[1])],
                origin: Kind::Rook,
                num: 1,
            },
            Piece {
                color: Color::Black,
                h: [half(squares[2]), half(squares[3])],
                origin: Kind::Rook,
                num: 1,
            },
            Piece {
                color: Color::White,
                h: [half(squares[4]), half(squares[5])],
                origin: Kind::Rook,
                num: 2,
            },
            Piece {
                color: Color::Black,
                h: [half(squares[6]), half(squares[6])],
                origin: Kind::Rook,
                num: 2,
            },
        ];
        let position = Position {
            pieces,
            turn: Color::White,
            ep: None,
        };
        let states = position.states().unwrap();
        assert_eq!(states.splits, vec![0, 1, 2]);

        let mut next_valid = 0;
        let mut next_invalid = 0;
        for mask in 0u32..8 {
            let selected = [
                squares[if mask & 1 == 0 { 0 } else { 1 }],
                squares[if mask & 2 == 0 { 2 } else { 3 }],
                squares[if mask & 4 == 0 { 4 } else { 5 }],
                squares[6],
            ];
            let valid = (0..selected.len())
                .all(|i| (i + 1..selected.len()).all(|j| selected[i] != selected[j]));
            if !valid {
                assert_eq!(states.invalid[next_invalid], mask);
                next_invalid += 1;
                continue;
            }

            let (actual_mask, board) = &states.list[next_valid];
            assert_eq!(*actual_mask, mask);
            let mut white = 0u64;
            let mut black = 0u64;
            for (id, &square) in selected.iter().enumerate() {
                let color = if id % 2 == 0 {
                    Color::White
                } else {
                    Color::Black
                };
                assert_eq!(
                    board.sq[square as usize],
                    Some(Occ {
                        id: id as u8,
                        color,
                        kind: Kind::Rook,
                    })
                );
                if color == Color::White {
                    white |= 1u64 << square;
                } else {
                    black |= 1u64 << square;
                }
            }
            assert_eq!(board.by_color, [white, black]);
            assert_eq!(board.by_kind[Kind::Rook as usize], white | black);
            next_valid += 1;
        }
        assert_eq!(next_valid, states.list.len());
        assert_eq!(next_invalid, states.invalid.len());
    }

    #[kani::proof]
    fn board_for_selection_matches_symbolic_five_piece_spec() {
        use super::{Half, Piece, Position};
        use crate::board::Occ;
        use crate::types::{Color, Kind};

        let squares: [u8; 11] = kani::any();
        let color_bits: [bool; 6] = kani::any();
        let kind_codes: [u8; 11] = kani::any();
        let half_alive: [bool; 10] = kani::any();
        let mask: u8 = kani::any();
        kani::assume(squares[10] < 64);
        kani::assume(kind_codes.iter().all(|&kind| kind < 6));
        kani::assume(mask < 32);
        for piece in 0..5 {
            let first = piece * 2;
            let second = first + 1;
            kani::assume(half_alive[first] || half_alive[second]);
            kani::assume(!half_alive[first] || squares[first] < 64);
            kani::assume(!half_alive[second] || squares[second] < 64);
            kani::assume(
                !half_alive[first]
                    || !half_alive[second]
                    || squares[first] != squares[second]
                    || kind_codes[first] != kind_codes[second],
            );
        }

        let color_at = |id| {
            if color_bits[id] {
                Color::White
            } else {
                Color::Black
            }
        };
        let kind_at = |index| match kind_codes[index] {
            0 => Kind::Pawn,
            1 => Kind::Knight,
            2 => Kind::Bishop,
            3 => Kind::Rook,
            4 => Kind::Queen,
            _ => Kind::King,
        };

        let half = |sq, kind, alive| Half {
            sq,
            kind,
            alive,
            moved: true,
        };
        let position = Position {
            pieces: vec![
                Piece {
                    color: color_at(0),
                    h: [
                        half(squares[0], kind_at(0), half_alive[0]),
                        half(squares[1], kind_at(1), half_alive[1]),
                    ],
                    origin: Kind::Pawn,
                    num: 1,
                },
                Piece {
                    color: color_at(1),
                    h: [
                        half(squares[2], kind_at(2), half_alive[2]),
                        half(squares[3], kind_at(3), half_alive[3]),
                    ],
                    origin: Kind::Pawn,
                    num: 1,
                },
                Piece {
                    color: color_at(2),
                    h: [
                        half(squares[4], kind_at(4), half_alive[4]),
                        half(squares[5], kind_at(5), half_alive[5]),
                    ],
                    origin: Kind::Pawn,
                    num: 2,
                },
                Piece {
                    color: color_at(3),
                    h: [
                        half(squares[6], kind_at(6), half_alive[6]),
                        half(squares[7], kind_at(7), half_alive[7]),
                    ],
                    origin: Kind::Pawn,
                    num: 2,
                },
                Piece {
                    color: color_at(4),
                    h: [
                        half(squares[8], kind_at(8), half_alive[8]),
                        half(squares[9], kind_at(9), half_alive[9]),
                    ],
                    origin: Kind::Pawn,
                    num: 3,
                },
                Piece {
                    color: color_at(5),
                    h: [
                        half(squares[10], kind_at(10), true),
                        half(squares[10], kind_at(10), true),
                    ],
                    origin: Kind::Pawn,
                    num: 3,
                },
            ],
            turn: Color::White,
            ep: None,
        };
        let splits = [0usize, 1, 2, 3, 4];
        let selected = [
            squares[if mask & 1 == 0 { 0 } else { 1 }],
            squares[if mask & 2 == 0 { 2 } else { 3 }],
            squares[if mask & 4 == 0 { 4 } else { 5 }],
            squares[if mask & 8 == 0 { 6 } else { 7 }],
            squares[if mask & 16 == 0 { 8 } else { 9 }],
            squares[10],
        ];
        let selected_kinds = [
            kind_at(if mask & 1 == 0 { 0 } else { 1 }),
            kind_at(if mask & 2 == 0 { 2 } else { 3 }),
            kind_at(if mask & 4 == 0 { 4 } else { 5 }),
            kind_at(if mask & 8 == 0 { 6 } else { 7 }),
            kind_at(if mask & 16 == 0 { 8 } else { 9 }),
            kind_at(10),
        ];
        let selected_alive = [
            half_alive[if mask & 1 == 0 { 0 } else { 1 }],
            half_alive[if mask & 2 == 0 { 2 } else { 3 }],
            half_alive[if mask & 4 == 0 { 4 } else { 5 }],
            half_alive[if mask & 8 == 0 { 6 } else { 7 }],
            half_alive[if mask & 16 == 0 { 8 } else { 9 }],
            true,
        ];
        let valid = (0..selected.len()).all(|i| {
            (i + 1..selected.len())
                .all(|j| !selected_alive[i] || !selected_alive[j] || selected[i] != selected[j])
        });
        let board = position.board_for_selection(&splits, mask as usize);
        assert_eq!(board.is_some(), valid);
        if let Some(board) = board {
            let mut white = 0u64;
            let mut black = 0u64;
            for (id, &square) in selected.iter().enumerate() {
                if !selected_alive[id] {
                    continue;
                }
                let color = color_at(id);
                assert_eq!(
                    board.sq[square as usize],
                    Some(Occ {
                        id: id as u8,
                        color,
                        kind: selected_kinds[id]
                    })
                );
                if color == Color::White {
                    white |= 1u64 << square;
                } else {
                    black |= 1u64 << square;
                }
            }
            assert_eq!(board.by_color, [white, black]);
            for kind in [
                Kind::Pawn,
                Kind::Knight,
                Kind::Bishop,
                Kind::Rook,
                Kind::Queen,
                Kind::King,
            ] {
                let mut expected = 0u64;
                for (id, &square) in selected.iter().enumerate() {
                    if selected_alive[id] && selected_kinds[id] == kind {
                        expected |= 1u64 << square;
                    }
                }
                assert_eq!(board.by_kind[kind as usize], expected);
            }
        }
    }

    #[kani::proof]
    fn board_for_selection_ignores_dead_half_square() {
        use super::{Half, Piece, Position};
        use crate::board::Occ;
        use crate::types::{Color, Kind};

        let first_square: u8 = kani::any();
        let second_square: u8 = kani::any();
        let base_square: u8 = kani::any();
        let first_alive: bool = kani::any();
        let second_alive: bool = kani::any();
        let choose_second: bool = kani::any();
        kani::assume(first_alive || second_alive);
        kani::assume(!first_alive || first_square < 64);
        kani::assume(!second_alive || second_square < 64);
        kani::assume(base_square < 64);
        kani::assume(!first_alive || !second_alive || first_square != second_square);

        let half = |sq, alive| Half {
            sq,
            kind: Kind::Rook,
            alive,
            moved: true,
        };
        let position = Position {
            pieces: vec![
                Piece {
                    color: Color::White,
                    h: [
                        half(first_square, first_alive),
                        half(second_square, second_alive),
                    ],
                    origin: Kind::Rook,
                    num: 1,
                },
                Piece {
                    color: Color::Black,
                    h: [half(base_square, true), half(base_square, true)],
                    origin: Kind::Rook,
                    num: 1,
                },
            ],
            turn: Color::White,
            ep: None,
        };
        let splits = [0usize];
        let selected_square = if choose_second {
            second_square
        } else {
            first_square
        };
        let selected_alive = if choose_second {
            second_alive
        } else {
            first_alive
        };
        let valid = !selected_alive || selected_square != base_square;
        let board = position.board_for_selection(&splits, usize::from(choose_second));
        assert_eq!(board.is_some(), valid);
        if let Some(board) = board {
            let base = Some(Occ {
                id: 1,
                color: Color::Black,
                kind: Kind::Rook,
            });
            assert_eq!(board.sq[base_square as usize], base);
            assert_eq!(board.by_color[Color::Black.idx()], 1u64 << base_square);
            if selected_alive {
                assert_eq!(
                    board.sq[selected_square as usize],
                    Some(Occ {
                        id: 0,
                        color: Color::White,
                        kind: Kind::Rook,
                    })
                );
                assert_eq!(board.by_color[Color::White.idx()], 1u64 << selected_square);
            } else {
                assert_eq!(board.by_color[Color::White.idx()], 0);
            }
        }
    }

    #[kani::proof]
    fn symbolic_two_piece_selection_matches_square_collision_spec() {
        use super::{Half, Piece};
        use crate::types::{Color, Kind};

        let first_a: u8 = kani::any();
        let first_b: u8 = kani::any();
        let second_a: u8 = kani::any();
        let second_b: u8 = kani::any();
        kani::assume(first_a < 64 && first_b < 64 && first_a != first_b);
        kani::assume(second_a < 64 && second_b < 64 && second_a != second_b);
        let half = |sq| Half {
            sq,
            kind: Kind::Rook,
            alive: true,
            moved: true,
        };
        let pieces = [
            Piece {
                color: Color::White,
                h: [half(first_a), half(first_b)],
                origin: Kind::Rook,
                num: 1,
            },
            Piece {
                color: Color::Black,
                h: [half(second_a), half(second_b)],
                origin: Kind::Rook,
                num: 1,
            },
        ];
        let splits = [0usize, 1usize];
        for mask in 0usize..4 {
            let first = if mask & 1 == 0 { first_a } else { first_b };
            let second = if mask & 2 == 0 { second_a } else { second_b };
            assert_eq!(
                super::selected_squares_fit(&pieces, &splits, mask),
                first != second
            );
        }
    }

    #[kani::proof]
    fn symbolic_three_piece_selection_matches_pairwise_collision_spec() {
        use super::{Half, Piece};
        use crate::types::{Color, Kind};

        let squares: [u8; 7] = kani::any();
        kani::assume(squares.iter().all(|&square| square < 64));
        kani::assume(squares[0] != squares[1]);
        kani::assume(squares[2] != squares[3]);
        kani::assume(squares[4] != squares[5]);

        let half = |sq| Half {
            sq,
            kind: Kind::Rook,
            alive: true,
            moved: true,
        };
        let pieces = [
            Piece {
                color: Color::White,
                h: [half(squares[0]), half(squares[1])],
                origin: Kind::Rook,
                num: 1,
            },
            Piece {
                color: Color::Black,
                h: [half(squares[2]), half(squares[3])],
                origin: Kind::Rook,
                num: 1,
            },
            Piece {
                color: Color::White,
                h: [half(squares[4]), half(squares[5])],
                origin: Kind::Rook,
                num: 2,
            },
            Piece {
                color: Color::Black,
                h: [half(squares[6]), half(squares[6])],
                origin: Kind::Rook,
                num: 2,
            },
        ];
        let splits = [0usize, 1usize, 2usize];
        let mut expected_valid_masks = 0u32;

        for mask in 0usize..8 {
            let selected = [
                squares[if mask & 1 == 0 { 0 } else { 1 }],
                squares[if mask & 2 == 0 { 2 } else { 3 }],
                squares[if mask & 4 == 0 { 4 } else { 5 }],
                squares[6],
            ];
            let expected = (0..selected.len())
                .all(|i| (i + 1..selected.len()).all(|j| selected[i] != selected[j]));
            if expected {
                expected_valid_masks |= 1 << mask;
            }
            assert_eq!(
                super::selected_squares_fit(&pieces, &splits, mask),
                expected
            );
        }

        let mut visited_valid_masks = 0u32;
        let mut visited_invalid_masks = 0u32;
        super::visit_state_boards(
            splits.len(),
            |mask| super::selected_squares_fit(&pieces, &splits, mask as usize).then_some(mask),
            |mask, value| {
                assert_eq!(mask, value);
                assert_eq!(visited_valid_masks & (1 << mask), 0);
                visited_valid_masks |= 1 << mask;
            },
            |mask| {
                assert_eq!(visited_invalid_masks & (1 << mask), 0);
                visited_invalid_masks |= 1 << mask;
            },
        );
        assert_eq!(visited_valid_masks, expected_valid_masks);
        assert_eq!(visited_invalid_masks, !expected_valid_masks & 0xff);
        assert_eq!(visited_valid_masks & visited_invalid_masks, 0);
    }

    #[kani::proof]
    fn symbolic_selection_accounts_for_unsplit_pieces_and_dead_halves() {
        use super::{Half, Piece};
        use crate::types::{Color, Kind};

        let first_live: u8 = kani::any();
        let base_square: u8 = kani::any();
        let second_a: u8 = kani::any();
        let second_b: u8 = kani::any();
        kani::assume(first_live < 64 && base_square < 64);
        kani::assume(second_a < 64 && second_b < 64 && second_a != second_b);
        let half = |sq, alive| Half {
            sq,
            kind: Kind::Rook,
            alive,
            moved: true,
        };
        let pieces = [
            Piece {
                color: Color::White,
                h: [half(first_live, true), half(base_square, false)],
                origin: Kind::Rook,
                num: 1,
            },
            Piece {
                color: Color::Black,
                h: [half(second_a, true), half(second_b, true)],
                origin: Kind::Rook,
                num: 1,
            },
            Piece {
                color: Color::White,
                h: [half(base_square, true), half(base_square, true)],
                origin: Kind::Rook,
                num: 2,
            },
        ];
        let splits = [0usize, 1usize];
        for mask in 0usize..4 {
            let first_is_alive = mask & 1 == 0;
            let first = if first_is_alive {
                Some(first_live)
            } else {
                None
            };
            let second = if mask & 2 == 0 { second_a } else { second_b };
            let expected = second != base_square
                && first.is_none_or(|square| square != base_square && square != second);
            assert_eq!(
                super::selected_squares_fit(&pieces, &splits, mask),
                expected
            );
        }
    }

    #[kani::proof]
    fn occupancy_transition_matches_collision_spec_for_all_inputs() {
        let square: u8 = kani::any();
        let alive: bool = kani::any();
        let original: u64 = kani::any();
        let mut occupied = original;
        let fits = super::occupy_square(&mut occupied, square, alive);

        if !alive {
            assert!(fits);
            assert_eq!(occupied, original);
        } else if square >= 64 {
            assert!(!fits);
            assert_eq!(occupied, original);
        } else if original & (1u64 << square) != 0 {
            assert!(!fits);
            assert_eq!(occupied, original);
        } else {
            assert!(fits);
            assert_eq!(occupied, original | (1u64 << square));
        }
    }
}
