//! A small, dependency-free implementation of the Glicko-2 rating system,
//! with defaults modelled on lichess.
//!
//! See Mark Glickman, "Example of the Glicko-2 system".

use std::f64::consts::PI;

/// Conversion factor between the Glicko and Glicko-2 scales.
const SCALE: f64 = 173.7178;
/// Convergence tolerance of the volatility iteration.
const EPSILON: f64 = 1e-6;
/// Center of the rating scale.
const BASE_RATING: f64 = 1500.0;

/// A player's rating state on the Glicko (not Glicko-2) display scale.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rating {
    /// Rating `r`; 1500 is average.
    pub rating: f64,
    /// Rating deviation `RD`; the uncertainty of the rating.
    pub deviation: f64,
    /// Volatility `sigma`; expected fluctuation of the rating.
    pub volatility: f64,
}

impl Rating {
    /// Creates a rating from its three components.
    pub fn new(rating: f64, deviation: f64, volatility: f64) -> Self {
        Self {
            rating,
            deviation,
            volatility,
        }
    }
}

impl Default for Rating {
    /// 1500 / 350 / 0.06.
    fn default() -> Self {
        Self::new(1500.0, 350.0, 0.06)
    }
}

/// Result of a game from the point of view of one player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The player won.
    Win,
    /// The game was drawn.
    Draw,
    /// The player lost.
    Loss,
}

impl Outcome {
    /// Score of the outcome: 1.0 for a win, 0.5 for a draw, 0.0 for a loss.
    pub fn score(self) -> f64 {
        match self {
            Outcome::Win => 1.0,
            Outcome::Draw => 0.5,
            Outcome::Loss => 0.0,
        }
    }

    fn flip(self) -> Self {
        match self {
            Outcome::Win => Outcome::Loss,
            Outcome::Draw => Outcome::Draw,
            Outcome::Loss => Outcome::Win,
        }
    }
}

/// Tunable parameters of the system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// System constant `tau` constraining volatility change (lichess: 0.75).
    pub tau: f64,
    /// Lower bound for the rating deviation after an update (lichess: 45).
    pub min_deviation: f64,
    /// Upper bound for the rating deviation after an update (lichess: 500).
    pub max_deviation: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tau: 0.75,
            min_deviation: 45.0,
            max_deviation: 500.0,
        }
    }
}

fn g(phi: f64) -> f64 {
    1.0 / (1.0 + 3.0 * phi * phi / (PI * PI)).sqrt()
}

fn e(mu: f64, mu_j: f64, phi_j: f64) -> f64 {
    1.0 / (1.0 + (-g(phi_j) * (mu - mu_j)).exp())
}

/// Expected score of `a` against `b` (between 0 and 1), taking `b`'s
/// deviation into account. Note this is not exactly symmetric in general
/// (`expected_score(a, b) + expected_score(b, a)` equals 1 only when both
/// deviations are equal).
pub fn expected_score(a: Rating, b: Rating) -> f64 {
    let mu = (a.rating - BASE_RATING) / SCALE;
    let mu_j = (b.rating - BASE_RATING) / SCALE;
    e(mu, mu_j, b.deviation / SCALE)
}

/// Whether the rating is provisional (deviation > 110); lichess shows "?".
pub fn provisional(r: &Rating) -> bool {
    r.deviation > 110.0
}

/// Updates `player` after one rating period containing `games`
/// (opponent rating at the time of the game and outcome for `player`).
///
/// With no games only the deviation grows. The resulting deviation is
/// clamped to `[cfg.min_deviation, cfg.max_deviation]`.
pub fn update(player: Rating, games: &[(Rating, Outcome)], cfg: &Config) -> Rating {
    let mu = (player.rating - BASE_RATING) / SCALE;
    let phi = player.deviation / SCALE;
    let sigma = player.volatility;

    let (mu_new, phi_new, sigma_new) = if games.is_empty() {
        (mu, (phi * phi + sigma * sigma).sqrt(), sigma)
    } else {
        // Step 3: estimated variance v, and step 4: improvement delta.
        let mut v_inv = 0.0;
        let mut delta_sum = 0.0;
        for (opp, outcome) in games {
            let mu_j = (opp.rating - BASE_RATING) / SCALE;
            let phi_j = opp.deviation / SCALE;
            let gj = g(phi_j);
            let ej = e(mu, mu_j, phi_j);
            v_inv += gj * gj * ej * (1.0 - ej);
            delta_sum += gj * (outcome.score() - ej);
        }
        let v = 1.0 / v_inv;
        let delta = v * delta_sum;

        // Step 5: new volatility via the Illinois algorithm.
        let sigma_new = new_volatility(phi, sigma, v, delta, cfg.tau);

        // Steps 6-7.
        let phi_star = (phi * phi + sigma_new * sigma_new).sqrt();
        let phi_new = 1.0 / (1.0 / (phi_star * phi_star) + 1.0 / v).sqrt();
        let mu_new = mu + phi_new * phi_new * delta_sum;
        (mu_new, phi_new, sigma_new)
    };

    Rating {
        rating: mu_new * SCALE + BASE_RATING,
        deviation: (phi_new * SCALE).clamp(cfg.min_deviation, cfg.max_deviation),
        volatility: sigma_new,
    }
}

fn new_volatility(phi: f64, sigma: f64, v: f64, delta: f64, tau: f64) -> f64 {
    let a = (sigma * sigma).ln();
    let f = |x: f64| {
        let ex = x.exp();
        let d2 = phi * phi + v + ex;
        ex * (delta * delta - phi * phi - v - ex) / (2.0 * d2 * d2) - (x - a) / (tau * tau)
    };

    let mut big_a = a;
    let mut big_b = if delta * delta > phi * phi + v {
        (delta * delta - phi * phi - v).ln()
    } else {
        let mut k = 1.0;
        while f(a - k * tau) < 0.0 {
            k += 1.0;
            if k > 1000.0 {
                break;
            }
        }
        a - k * tau
    };

    let mut fa = f(big_a);
    let mut fb = f(big_b);
    let mut iterations = 0;
    while (big_b - big_a).abs() > EPSILON && iterations < 1000 {
        let c = big_a + (big_a - big_b) * fa / (fb - fa);
        let fc = f(c);
        if fc * fb <= 0.0 {
            big_a = big_b;
            fa = fb;
        } else {
            fa /= 2.0;
        }
        big_b = c;
        fb = fc;
        iterations += 1;
    }
    (big_a / 2.0).exp()
}

#[cfg(kani)]
mod verification {
    use super::{Config, Outcome, Rating, Speed, update};

    #[kani::proof]
    fn outcome_scores_are_bounded_and_flip_is_an_involution() {
        let outcome = match kani::any::<u8>() % 3 {
            0 => Outcome::Win,
            1 => Outcome::Draw,
            _ => Outcome::Loss,
        };
        assert!((0.0..=1.0).contains(&outcome.score()));
        assert_eq!(outcome.flip().flip(), outcome);
        assert_eq!(outcome.flip().score(), 1.0 - outcome.score());
    }

    #[kani::proof]
    fn empty_period_preserves_rating_and_volatility() {
        let cfg = Config::default();
        for (rating, deviation, volatility) in [
            (1500.0, 350.0, 0.06),
            (2400.0, 500.0, 0.50),
            (900.0, 1.0, 1.00),
        ] {
            let updated = update(Rating::new(rating, deviation, volatility), &[], &cfg);
            assert_eq!(updated.rating, rating);
            assert_eq!(updated.volatility, volatility);
        }
    }

    #[kani::proof]
    fn clock_speed_matches_threshold_spec_for_all_u32_inputs() {
        let initial_secs: u32 = kani::any();
        let increment_secs: u32 = kani::any();
        let estimate = u64::from(initial_secs) + 40 * u64::from(increment_secs);
        let expected = if estimate < 30 {
            Speed::UltraBullet
        } else if estimate < 180 {
            Speed::Bullet
        } else if estimate < 480 {
            Speed::Blitz
        } else if estimate < 1500 {
            Speed::Rapid
        } else {
            Speed::Classical
        };
        assert_eq!(Speed::from_clock(initial_secs, increment_secs), expected);
    }
}

/// Updates both players after a single game, each treating the game as its
/// own rating period (lichess style). Returns `(white, black)`.
pub fn update_game(
    white: Rating,
    black: Rating,
    white_outcome: Outcome,
    cfg: &Config,
) -> (Rating, Rating) {
    (
        update(white, &[(black, white_outcome)], cfg),
        update(black, &[(white, white_outcome.flip())], cfg),
    )
}

/// Game speed category, using the lichess clock estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Speed {
    /// Estimated duration under 30 seconds.
    UltraBullet,
    /// Under 180 seconds.
    Bullet,
    /// Under 480 seconds.
    Blitz,
    /// Under 1500 seconds.
    Rapid,
    /// 1500 seconds or more.
    Classical,
}

impl Speed {
    /// Classifies a clock by `initial_secs + 40 * increment_secs`.
    /// Both arguments are in seconds (a "3+2" game is `from_clock(180, 2)`).
    pub fn from_clock(initial_secs: u32, increment_secs: u32) -> Speed {
        let estimate = u64::from(initial_secs) + 40 * u64::from(increment_secs);
        match estimate {
            0..30 => Speed::UltraBullet,
            30..180 => Speed::Bullet,
            180..480 => Speed::Blitz,
            480..1500 => Speed::Rapid,
            _ => Speed::Classical,
        }
    }

    /// Lichess-style name, e.g. `"bullet"` (`"ultraBullet"` for ultra bullet).
    pub fn name(&self) -> &'static str {
        match self {
            Speed::UltraBullet => "ultraBullet",
            Speed::Bullet => "bullet",
            Speed::Blitz => "blitz",
            Speed::Rapid => "rapid",
            Speed::Classical => "classical",
        }
    }

    /// All speeds, fastest first.
    pub fn all() -> [Speed; 5] {
        [
            Speed::UltraBullet,
            Speed::Bullet,
            Speed::Blitz,
            Speed::Rapid,
            Speed::Classical,
        ]
    }
}
