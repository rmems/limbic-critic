// SPDX-License-Identifier: MIT OR Apache-2.0

//! RL critic and reward shaping.
//!
//! Translates an [`Environment`] observation into a
//! [`ModulatorVector`] of neuromodulator
//! concentrations. Two critics are provided:
//!
//! - [`SimpleCritic`] — stateless, maps the immediate objective and optional
//!   environment signals.
//! - [`TDCritic`] — stateful temporal-difference critic that tracks reward
//!   improvement over time.
//!
//! # Quick start
//!
//! ```rust
//! use limbic_critic::{Environment, SimpleCritic, TDCritic, ModulatorVector};
//!
//! /// Minimal stub environment used only for documentation examples.
//! struct StubEnv {
//!     objective: f32,
//!     surprise: f32,
//! }
//!
//! impl Environment for StubEnv {
//!     fn objective(&self) -> f32 {
//!         self.objective
//!     }
//!     fn surprise(&self) -> f32 {
//!         self.surprise
//!     }
//! }
//!
//! let env = StubEnv {
//!     objective: 0.75,
//!     surprise: 0.3,
//! };
//!
//! // Stateless mapping: dopamine ∈ [0, 1], ACh from Environment::surprise
//! let simple: ModulatorVector = SimpleCritic::assess(&env);
//! assert!((0.0..=1.0).contains(&simple.dopamine));
//! assert_eq!(simple.acetylcholine, 0.3);
//!
//! // Temporal-difference critic: dopamine ∈ [-1, 1] after tanh of EMA(TD)
//! let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
//! let first = td.assess(&env);
//! assert!((-1.0..=1.0).contains(&first.dopamine));
//! ```

use crate::environment::Environment;
use crate::modulators::ModulatorVector;

/// A stateless critic that maps immediate environment signals to neuromodulators.
///
/// `SimpleCritic` stores no history. It therefore cannot compute temporal
/// surprise on its own: acetylcholine is read directly from
/// [`Environment::surprise`] and clamped to `[0.0, 1.0]`. Use [`TDCritic`]
/// when acetylcholine should be derived from the absolute TD error
/// (`abs(td_error).tanh()`).
///
/// # Mapping
///
/// | Field | Source | Range |
/// |-------|--------|-------|
/// | `dopamine` | `env.objective()` if positive, else `0.0` | `[0.0, 1.0]` |
/// | `serotonin` | `env.volatility()` | `[0.0, 1.0]` |
/// | `acetylcholine` | `env.surprise()` | `[0.0, 1.0]` |
/// | `norepinephrine` | `env.stress()` | `[0.0, 1.0]` |
///
/// # Example
///
/// ```rust
/// use limbic_critic::{Environment, SimpleCritic};
///
/// struct StubEnv;
/// impl Environment for StubEnv {
///     fn objective(&self) -> f32 { 0.8 }
///     fn surprise(&self) -> f32 { 0.4 }
/// }
///
/// let mods = SimpleCritic::assess(&StubEnv);
/// assert_eq!(mods.dopamine, 0.8);
/// assert_eq!(mods.acetylcholine, 0.4);
/// ```
pub struct SimpleCritic;

impl SimpleCritic {
    /// Calculate neuromodulator concentrations from the current environment.
    ///
    /// # Dopamine
    ///
    /// Positive [`Environment::objective`] values are clamped to
    /// **`[0.0, 1.0]`**. Negative or zero objectives produce `dopamine = 0.0`
    /// (no negative reward signal).
    ///
    /// # Acetylcholine
    ///
    /// Taken from [`Environment::surprise`] and clamped to **`[0.0, 1.0]`**.
    /// This critic does **not** infer ACh from objective deltas.
    ///
    /// # Other fields
    ///
    /// - `serotonin` ← [`Environment::volatility`] clamped to `[0.0, 1.0]`
    /// - `norepinephrine` ← [`Environment::stress`] clamped to `[0.0, 1.0]`
    ///
    /// # Non-finite observations
    ///
    /// Current `f32` behavior is documented rather than sanitized:
    ///
    /// - A **NaN** objective is not `> 0.0`, so `dopamine` is `0.0`.
    /// - `+∞` objective clamps to `1.0`; `-∞` is treated as non-positive (`0.0`).
    /// - NaN auxiliary signals (`surprise` / `volatility` / `stress`) pass
    ///   through `clamp` as **NaN** (IEEE: NaN comparisons are false).
    /// - `±∞` auxiliary signals clamp to the nearest bound (`0.0` or `1.0`).
    pub fn assess(env: &impl Environment) -> ModulatorVector {
        let objective = env.objective();

        // Simple mapping: positive objective -> dopamine, negative -> nothing
        let dopamine = if objective > 0.0 {
            objective.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let stress = env.stress().clamp(0.0, 1.0);
        let serotonin = env.volatility().clamp(0.0, 1.0);
        let acetylcholine = env.surprise().clamp(0.0, 1.0);

        ModulatorVector {
            dopamine,
            serotonin,
            acetylcholine,
            norepinephrine: stress,
        }
    }
}

/// A stateful temporal-difference (TD) critic.
///
/// Tracks the previous objective and an exponential moving average (EMA) of
/// the TD error so that dopamine reflects *change* in reward rather than
/// absolute level. Acetylcholine is derived from surprise in the TD signal
/// (`abs(td_error).tanh()`), not from [`Environment::surprise`].
///
/// # Internal state
///
/// | Field | Meaning |
/// |-------|---------|
/// | `prev_objective` | Objective observed on the previous [`assess`](Self::assess) call; starts at `0.0`. |
/// | `ema_reward` | EMA of successive TD errors (`objective - prev_objective`); starts at `0.0`. |
/// | `alpha` | EMA learning rate, **enforced** in `(0, 1]` by [`TDCritic::new`]. Higher values weight recent TD errors more heavily. |
///
/// # Mapping
///
/// | Field | Source | Range |
/// |-------|--------|-------|
/// | `dopamine` | `ema_reward.tanh()` | **`[-1.0, 1.0]`** |
/// | `serotonin` | `env.volatility()` | `[0.0, 1.0]` |
/// | `acetylcholine` | `abs(td_error).tanh()` | `[0.0, 1.0]` |
/// | `norepinephrine` | `env.stress()` | `[0.0, 1.0]` |
///
/// # Example
///
/// ```rust
/// use limbic_critic::{Environment, TDCritic};
///
/// struct StubEnv(f32);
/// impl Environment for StubEnv {
///     fn objective(&self) -> f32 { self.0 }
/// }
///
/// let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
/// let step1 = td.assess(&StubEnv(0.0));
/// let step2 = td.assess(&StubEnv(1.0));
/// // Improvement produces a higher (more positive) dopamine signal.
/// assert!(step2.dopamine > step1.dopamine);
/// ```
pub struct TDCritic {
    prev_objective: f32,
    ema_reward: f32,
    alpha: f32, // Learning rate for the EMA
}

/// `alpha` supplied to [`TDCritic::new`] was not in `(0, 1]`.
///
/// Rejected values include `0.0`, negatives, values greater than `1.0`,
/// `NaN`, and infinities. Finite values in `(0, 1]` (including `1.0`)
/// are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidAlpha;

impl std::fmt::Display for InvalidAlpha {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TDCritic alpha must be in (0, 1] (finite, greater than 0, at most 1)")
    }
}

impl std::error::Error for InvalidAlpha {}

impl TDCritic {
    /// Create a new TD critic with the given EMA learning rate.
    ///
    /// `alpha` controls how quickly the internal EMA of TD errors adapts:
    ///
    /// - **Small `alpha`** (e.g. `0.05`) — smooth, slow reaction to changes.
    /// - **Large `alpha`** (e.g. `0.5`) — fast tracking of recent TD errors.
    /// - **`alpha = 1.0`** — no memory; the EMA equals the latest TD error.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidAlpha`] when `alpha` is not in **`(0, 1]`**.
    /// The predicate is `alpha > 0.0 && alpha <= 1.0`, which also rejects
    /// `NaN` and infinities (IEEE comparisons with `NaN` are false).
    ///
    /// Initial state on success:
    /// - `prev_objective = 0.0`
    /// - `ema_reward = 0.0`
    ///
    /// The first [`assess`](Self::assess) call therefore treats the TD error
    /// as `objective - 0.0`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use limbic_critic::TDCritic;
    ///
    /// let critic = TDCritic::new(0.2).expect("alpha in (0, 1]");
    /// // critic is ready; call assess(&env) on each time step
    /// ```
    pub fn new(alpha: f32) -> Result<Self, InvalidAlpha> {
        if !is_valid_alpha(alpha) {
            return Err(InvalidAlpha);
        }
        Ok(Self {
            prev_objective: 0.0,
            ema_reward: 0.0,
            alpha,
        })
    }

    /// Calculate neuromodulator concentrations from the TD error.
    ///
    /// # Algorithm
    ///
    /// 1. `td_error = env.objective() - prev_objective`
    /// 2. Store the current objective as `prev_objective` for the next call.
    /// 3. `acetylcholine = abs(td_error).tanh()`, clamped to `[0.0, 1.0]`.
    /// 4. Update EMA: `ema_reward ← (1 - alpha) * ema_reward + alpha * td_error`.
    /// 5. `dopamine = ema_reward.tanh()`, clamped to **`[-1.0, 1.0]`**.
    /// 6. `serotonin` / `norepinephrine` from `volatility` / `stress`, each
    ///    clamped to `[0.0, 1.0]`.
    ///
    /// Unlike [`SimpleCritic::assess`], this method mutates internal state and
    /// can produce **negative dopamine** when recent TD errors are negative
    /// (worsening outcomes).
    ///
    /// # Non-finite observations
    ///
    /// Current `f32` behavior is documented rather than sanitized:
    ///
    /// - A **NaN** or infinite objective produces a non-finite TD error.
    ///   Then `tanh` / `clamp` leave **NaN** dopamine and acetylcholine
    ///   (NaN comparisons are false), while `±∞` TD errors saturate
    ///   (`tanh(∞) = 1`, `tanh(-∞) = -1`).
    /// - Auxiliary `volatility` / `stress` follow the same clamp rules as
    ///   [`SimpleCritic::assess`]: NaN stays NaN; infinities clamp to bounds.
    ///
    /// # Parameters
    ///
    /// - `env` — environment providing the current objective (and optional
    ///   stress / volatility signals).
    pub fn assess(&mut self, env: &impl Environment) -> ModulatorVector {
        let objective = env.objective();
        let td_error = objective - self.prev_objective;
        self.prev_objective = objective;

        // Surprise / Focus calculation
        let acetylcholine = td_error.abs().tanh().clamp(0.0, 1.0);

        // Update the EMA of the reward
        self.ema_reward = (1.0 - self.alpha) * self.ema_reward + self.alpha * td_error;

        // Map the smoothed reward to dopamine
        let dopamine = self.ema_reward.tanh().clamp(-1.0, 1.0);

        let stress = env.stress().clamp(0.0, 1.0);
        let serotonin = env.volatility().clamp(0.0, 1.0);

        ModulatorVector {
            dopamine,
            serotonin,
            acetylcholine,
            norepinephrine: stress,
        }
    }
}

/// `alpha` is accepted only in the open-closed interval `(0, 1]`.
///
/// `NaN` and infinities fail the comparison and are rejected.
fn is_valid_alpha(alpha: f32) -> bool {
    alpha > 0.0 && alpha <= 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstEnv(f32);

    impl Environment for ConstEnv {
        fn objective(&self) -> f32 {
            self.0
        }
    }

    struct VolatileEnv {
        steps: Vec<f32>,
        index: usize,
    }

    impl VolatileEnv {
        fn new(values: Vec<f32>) -> Self {
            Self {
                steps: values,
                index: 0,
            }
        }
    }

    impl Environment for VolatileEnv {
        fn objective(&self) -> f32 {
            self.steps[self.index]
        }
    }

    #[test]
    fn test_simple_critic_positive_objective() {
        let env = ConstEnv(0.8);
        let mods = SimpleCritic::assess(&env);
        assert_eq!(mods.dopamine, 0.8);
        assert_eq!(mods.norepinephrine, 0.0);
    }

    #[test]
    fn test_simple_critic_negative_objective() {
        let env = ConstEnv(-0.5);
        let mods = SimpleCritic::assess(&env);
        assert_eq!(mods.dopamine, 0.0);
    }

    #[test]
    fn test_simple_critic_clamping() {
        let env = ConstEnv(5.0);
        let mods = SimpleCritic::assess(&env);
        assert_eq!(mods.dopamine, 1.0);
    }

    #[test]
    fn test_simple_critic_stress() {
        struct StressedEnv;
        impl Environment for StressedEnv {
            fn objective(&self) -> f32 {
                0.5
            }
            fn stress(&self) -> f32 {
                0.9
            }
        }
        let mods = SimpleCritic::assess(&StressedEnv);
        assert_eq!(mods.norepinephrine, 0.9);
    }

    #[test]
    fn test_simple_critic_volatility_serotonin() {
        struct SimpleVolatileEnv;
        impl Environment for SimpleVolatileEnv {
            fn objective(&self) -> f32 {
                0.5
            }
            fn volatility(&self) -> f32 {
                0.6
            }
        }
        let mods = SimpleCritic::assess(&SimpleVolatileEnv);
        assert_eq!(mods.serotonin, 0.6);
    }

    #[test]
    fn test_simple_critic_surprise_acetylcholine() {
        struct SurprisingEnv {
            surprise: f32,
        }
        impl Environment for SurprisingEnv {
            fn objective(&self) -> f32 {
                0.5
            }
            fn surprise(&self) -> f32 {
                self.surprise
            }
        }

        let mods = SimpleCritic::assess(&SurprisingEnv { surprise: 0.7 });
        assert_eq!(mods.acetylcholine, 0.7);
    }

    #[test]
    fn test_simple_critic_surprise_acetylcholine_clamping() {
        struct SurprisingEnv(f32);
        impl Environment for SurprisingEnv {
            fn objective(&self) -> f32 {
                0.0
            }
            fn surprise(&self) -> f32 {
                self.0
            }
        }

        assert_eq!(
            SimpleCritic::assess(&SurprisingEnv(-0.2)).acetylcholine,
            0.0
        );
        assert_eq!(SimpleCritic::assess(&SurprisingEnv(1.5)).acetylcholine, 1.0);
    }

    #[test]
    fn test_td_critic_no_change() {
        let env = ConstEnv(0.5);
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let mods = td.assess(&env);
        // First call: td_error = 0.5 - 0.0 = 0.5
        // ema starts at 0.0, so ema = 0.9*0.0 + 0.1*0.5 = 0.05
        // dopamine = 0.05.tanh()
        assert!((mods.dopamine - 0.05f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_td_critic_improvement() {
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let mut env = VolatileEnv::new(vec![0.0, 1.0]);

        let first = td.assess(&env);
        env.index = 1;
        let second = td.assess(&env);

        // Second call has positive improvement
        assert!(second.dopamine > first.dopamine);
    }

    #[test]
    fn test_td_critic_degradation() {
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let mut env = VolatileEnv::new(vec![1.0, 0.0]);

        let first = td.assess(&env);
        env.index = 1;
        let second = td.assess(&env);

        // Second call has negative td_error
        assert!(second.dopamine < first.dopamine);
    }

    #[test]
    fn test_td_critic_surprise() {
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let env = ConstEnv(0.5);

        let mods = td.assess(&env);
        // First td_error = 0.5, acetylcholine = 0.5.abs().tanh()
        assert!((mods.acetylcholine - 0.5f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_td_critic_volatility_serotonin() {
        struct TdVolatileEnv {
            v: f32,
        }
        impl Environment for TdVolatileEnv {
            fn objective(&self) -> f32 {
                0.0
            }
            fn volatility(&self) -> f32 {
                self.v
            }
        }
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let env = TdVolatileEnv { v: 0.6 };
        let mods = td.assess(&env);
        assert_eq!(mods.serotonin, 0.6);
    }

    #[test]
    fn td_critic_rejects_non_positive_or_above_one_alpha() {
        assert!(matches!(TDCritic::new(0.0), Err(InvalidAlpha)));
        assert!(matches!(TDCritic::new(-0.1), Err(InvalidAlpha)));
        assert!(matches!(TDCritic::new(-0.0), Err(InvalidAlpha)));
        assert!(matches!(TDCritic::new(1.0001), Err(InvalidAlpha)));
        assert!(matches!(TDCritic::new(2.0), Err(InvalidAlpha)));
    }

    #[test]
    fn td_critic_rejects_nonfinite_alpha() {
        assert!(matches!(TDCritic::new(f32::NAN), Err(InvalidAlpha)));
        assert!(matches!(TDCritic::new(f32::INFINITY), Err(InvalidAlpha)));
        assert!(matches!(
            TDCritic::new(f32::NEG_INFINITY),
            Err(InvalidAlpha)
        ));
    }

    #[test]
    fn td_critic_accepts_unit_interval_alpha() {
        assert!(TDCritic::new(1.0).is_ok());
        assert!(TDCritic::new(f32::EPSILON).is_ok());
        assert!(TDCritic::new(0.5).is_ok());
    }

    struct AuxEnv {
        objective: f32,
        surprise: f32,
        volatility: f32,
        stress: f32,
    }

    impl Environment for AuxEnv {
        fn objective(&self) -> f32 {
            self.objective
        }
        fn surprise(&self) -> f32 {
            self.surprise
        }
        fn volatility(&self) -> f32 {
            self.volatility
        }
        fn stress(&self) -> f32 {
            self.stress
        }
    }

    #[test]
    fn simple_critic_nonfinite_and_negative_objectives() {
        // NaN / non-positive → dopamine 0 (the `objective > 0` branch).
        assert_eq!(SimpleCritic::assess(&ConstEnv(f32::NAN)).dopamine, 0.0);
        assert_eq!(
            SimpleCritic::assess(&ConstEnv(f32::NEG_INFINITY)).dopamine,
            0.0
        );
        assert_eq!(SimpleCritic::assess(&ConstEnv(-1.5)).dopamine, 0.0);
        assert_eq!(SimpleCritic::assess(&ConstEnv(0.0)).dopamine, 0.0);
        // +∞ clamps to 1.0
        assert_eq!(SimpleCritic::assess(&ConstEnv(f32::INFINITY)).dopamine, 1.0);
    }

    #[test]
    fn simple_critic_nonfinite_aux_signals_follow_clamp() {
        let nan = SimpleCritic::assess(&AuxEnv {
            objective: 0.2,
            surprise: f32::NAN,
            volatility: f32::NAN,
            stress: f32::NAN,
        });
        assert!(nan.acetylcholine.is_nan());
        assert!(nan.serotonin.is_nan());
        assert!(nan.norepinephrine.is_nan());

        let inf = SimpleCritic::assess(&AuxEnv {
            objective: 0.2,
            surprise: f32::INFINITY,
            volatility: f32::NEG_INFINITY,
            stress: f32::INFINITY,
        });
        assert_eq!(inf.acetylcholine, 1.0);
        assert_eq!(inf.serotonin, 0.0);
        assert_eq!(inf.norepinephrine, 1.0);
    }

    #[test]
    fn td_critic_nan_objective_propagates_through_td() {
        let mut td = TDCritic::new(0.1).expect("alpha in (0, 1]");
        let mods = td.assess(&ConstEnv(f32::NAN));
        assert!(mods.dopamine.is_nan());
        assert!(mods.acetylcholine.is_nan());
    }

    #[test]
    fn td_critic_infinite_objective_saturates_tanh() {
        let mut td = TDCritic::new(1.0).expect("alpha in (0, 1]");
        let pos = td.assess(&ConstEnv(f32::INFINITY));
        assert_eq!(pos.dopamine, 1.0);
        assert_eq!(pos.acetylcholine, 1.0);

        let mut td = TDCritic::new(1.0).expect("alpha in (0, 1]");
        let neg = td.assess(&ConstEnv(f32::NEG_INFINITY));
        assert_eq!(neg.dopamine, -1.0);
        assert_eq!(neg.acetylcholine, 1.0);
    }
}
