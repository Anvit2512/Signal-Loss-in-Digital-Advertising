"""
Layer 3 -- Privacy Engine

Tracks the differential privacy epsilon budget and applies Laplace noise
to conversion counts. This is the core mechanic that forces the agent to
trade signal quality against feature usage.

Rules:
  - Each feature in the action's feature_mask costs FEATURE_COST epsilon
  - Probabilistic attribution costs ATTRIBUTION_COST epsilon extra
  - Laplace noise scale = sensitivity / epsilon_remaining  (grows as budget depletes)
  - Regime transitions are driven by epsilon level (or forced by task config)

Noise model:
  A true conversion count of N becomes N + Laplace(0, 1/epsilon_remaining).
  When epsilon is high (e.g. 3.0), noise scale = 0.33 -- barely perceptible.
  When epsilon is low  (e.g. 0.1), noise scale = 10.0 -- signal is buried.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from app.models import AttributionMethod, PrivacyRegime
from app.data_loader import ALL_FEATURES, INTEGER_FEATURES, CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Cost constants
# ---------------------------------------------------------------------------

FEATURE_COST      = 0.05   # epsilon per feature in feature_mask
ATTRIBUTION_COST  = 0.20   # extra epsilon for probabilistic attribution
SENSITIVITY       = 1.0    # L1 sensitivity of a conversion count query

# ---------------------------------------------------------------------------
# Regime thresholds (epsilon_remaining)
# ---------------------------------------------------------------------------

THRESHOLD_HIGH_NOISE = 0.5   # below this -> HIGH_NOISE
THRESHOLD_EXHAUSTED  = 0.1   # below this -> EXHAUSTED

# Regulatory feature limit enforced by env.py (not privacy.py)
# privacy.py only enforces budget; compliance is the env's job.
MINIMAL_DATA_MAX_FEATURES = 1


class PrivacyEngine:
    """
    Stateful epsilon budget tracker and Laplace noise generator.

    Usage per episode:
        engine = PrivacyEngine(initial_epsilon=3.0, seed=42)
        cost   = engine.consume(feature_mask, attribution)   # deducts from budget
        noisy  = engine.add_noise(true_count)                # uses current budget
        regime = engine.regime                               # current PrivacyRegime
    """

    def __init__(
        self,
        initial_epsilon: float = 3.0,
        seed: Optional[int] = None,
        forced_regime: Optional[PrivacyRegime] = None,
    ) -> None:
        """
        Args:
            initial_epsilon:  Starting budget. Task 1/2 use 3.0, Task 3 uses 2.0.
            seed:             RNG seed for reproducible noise draws.
            forced_regime:    If set, overrides the epsilon-derived regime.
                              Used by Task 3 to pin MINIMAL_DATA from the start.
        """
        if initial_epsilon <= 0:
            raise ValueError(f"initial_epsilon must be > 0, got {initial_epsilon}")

        self._initial_epsilon  = initial_epsilon
        self._epsilon          = initial_epsilon
        self._forced_regime    = forced_regime
        self._rng              = np.random.default_rng(seed)

        # Audit trail -- list of (feature_count, attribution, cost) per step
        self._audit: List[Tuple[int, str, float]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def epsilon_remaining(self) -> float:
        return max(self._epsilon, 0.0)

    @property
    def epsilon_used(self) -> float:
        return self._initial_epsilon - self.epsilon_remaining

    @property
    def budget_fraction_remaining(self) -> float:
        return self.epsilon_remaining / self._initial_epsilon

    @property
    def regime(self) -> PrivacyRegime:
        """Current privacy regime based on epsilon level (or forced by task)."""
        if self._forced_regime is not None:
            return self._forced_regime
        if self._epsilon < THRESHOLD_EXHAUSTED:
            return PrivacyRegime.EXHAUSTED
        if self._epsilon < THRESHOLD_HIGH_NOISE:
            return PrivacyRegime.HIGH_NOISE
        return PrivacyRegime.STANDARD

    @property
    def noise_scale(self) -> float:
        """
        Laplace noise scale = sensitivity / epsilon_remaining.
        Clipped at a maximum of 50 to prevent inf when epsilon -> 0.
        """
        eps = max(self._epsilon, 1e-6)
        return min(SENSITIVITY / eps, 50.0)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def consume(
        self,
        feature_mask: List[str],
        attribution: AttributionMethod = AttributionMethod.LAST_CLICK,
    ) -> float:
        """
        Deduct epsilon for a single step's feature usage and attribution choice.

        Returns the total epsilon cost for this step.
        Raises ValueError if any feature name is not in ALL_FEATURES.
        """
        invalid = [f for f in feature_mask if f not in ALL_FEATURES]
        if invalid:
            raise ValueError(
                f"Unknown features in mask: {invalid}. "
                f"Valid: I1-I13, C1-C26."
            )

        feature_cost      = len(feature_mask) * FEATURE_COST
        attribution_cost  = ATTRIBUTION_COST if attribution == AttributionMethod.PROBABILISTIC else 0.0
        total_cost        = feature_cost + attribution_cost

        self._epsilon -= total_cost
        self._audit.append((len(feature_mask), attribution.value, total_cost))

        return total_cost

    def add_noise(self, true_count: float) -> float:
        """
        Add Laplace noise to a true conversion count.

        The noise scale grows as epsilon depletes -- this is the core
        signal-degradation mechanic the agent must reason about.

        Returns the noisy count (may be negative; env.py clips to 0).
        """
        noise = self._rng.laplace(loc=0.0, scale=self.noise_scale)
        return true_count + noise

    def force_high_noise(self) -> None:
        """
        Triggered by Task 2 at step 3 -- jumps regime to HIGH_NOISE
        by draining epsilon to just below the threshold.
        Used only by the environment, not by the agent.
        """
        if self._epsilon >= THRESHOLD_HIGH_NOISE:
            self._epsilon = THRESHOLD_HIGH_NOISE - 0.01

    def available_features(self) -> List[str]:
        """
        Features the agent is allowed to use given the current regime.

        MINIMAL_DATA (Task 3): only integer features, max 1 per step.
        All other regimes: full feature set.

        Note: the agent is told available_features in every Observation
        so it never has to guess what is allowed.
        """
        if self.regime == PrivacyRegime.MINIMAL_DATA:
            return INTEGER_FEATURES  # agent picks at most 1 from this list
        return ALL_FEATURES

    # ------------------------------------------------------------------
    # Diagnostic helpers (used by graders)
    # ------------------------------------------------------------------

    def steps_without_violation(self, max_features: int) -> int:
        """Count steps where feature_count <= max_features."""
        return sum(1 for (fc, _, _) in self._audit if fc <= max_features)

    def total_steps(self) -> int:
        return len(self._audit)

    def compliance_rate(self, max_features: int) -> float:
        """Fraction of steps that were compliant (0.0-1.0)."""
        if not self._audit:
            return 1.0
        return self.steps_without_violation(max_features) / self.total_steps()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset budget and RNG -- called by env.reset(), not by the agent."""
        self._epsilon = self._initial_epsilon
        self._audit.clear()
        self._rng = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Regulatory penalty calculator (used by env.py to build the reward signal)
# ---------------------------------------------------------------------------

def regulatory_penalty(feature_mask: List[str], max_features: int) -> float:
    """
    Quadratic penalty for exceeding the regulatory feature limit.

    penalty = (excess_features) ** 2

    Returns 0.0 when compliant. Grows quadratically so the agent learns
    there is a smooth cost curve, not a sudden death cliff.
    """
    excess = max(0, len(feature_mask) - max_features)
    return float(excess ** 2)


# ---------------------------------------------------------------------------
# Standalone test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PrivacyEngine standalone test ===\n")

    engine = PrivacyEngine(initial_epsilon=3.0, seed=42)

    print(f"Initial:  epsilon={engine.epsilon_remaining:.2f}  "
          f"regime={engine.regime.value}  noise_scale={engine.noise_scale:.4f}")

    # Step 1 -- use 3 features + last_click
    cost = engine.consume(["I1", "I2", "C1"], AttributionMethod.LAST_CLICK)
    noisy = max(0.0, engine.add_noise(80.0))
    print(f"\nStep 1:  features=3  attribution=last_click  cost={cost:.2f}")
    print(f"         true=80  noisy={noisy:.2f}  epsilon_left={engine.epsilon_remaining:.2f}  regime={engine.regime.value}")

    # Step 2 -- use 5 features + probabilistic
    cost = engine.consume(["I1","I2","I3","C1","C2"], AttributionMethod.PROBABILISTIC)
    noisy = max(0.0, engine.add_noise(80.0))
    print(f"\nStep 2:  features=5  attribution=probabilistic  cost={cost:.2f}")
    print(f"         true=80  noisy={noisy:.2f}  epsilon_left={engine.epsilon_remaining:.2f}  regime={engine.regime.value}")

    # Simulate Task 2 privacy update at step 3
    engine.force_high_noise()
    cost = engine.consume(["I1"], AttributionMethod.LAST_CLICK)
    noisy = max(0.0, engine.add_noise(80.0))
    print(f"\nStep 3:  force_high_noise() triggered")
    print(f"         features=1  cost={cost:.2f}")
    print(f"         true=80  noisy={noisy:.2f}  epsilon_left={engine.epsilon_remaining:.2f}  regime={engine.regime.value}")

    # Drain to exhaustion
    for i in range(4, 9):
        cost = engine.consume(["I1", "I2", "I3"], AttributionMethod.LAST_CLICK)
        noisy = max(0.0, engine.add_noise(80.0))
        print(f"\nStep {i}:  features=3  cost={cost:.2f}  "
              f"epsilon_left={engine.epsilon_remaining:.2f}  "
              f"regime={engine.regime.value}  noisy={noisy:.2f}")

    print(f"\nCompliance rate (max 1 feature): {engine.compliance_rate(1):.2f}")
    print(f"Budget fraction remaining: {engine.budget_fraction_remaining:.4f}")

    # Regulatory penalty examples
    print(f"\nregulatory_penalty(2 features, max=1) = {regulatory_penalty(['I1','I2'], 1)}")
    print(f"regulatory_penalty(4 features, max=1) = {regulatory_penalty(['I1','I2','I3','C1'], 1)}")
    print(f"regulatory_penalty(1 feature,  max=1) = {regulatory_penalty(['I1'], 1)}")

    # Test invalid feature rejection
    try:
        engine.consume(["I1", "INVALID_FEAT"])
    except ValueError as e:
        print(f"\nInvalid feature guard OK: {e}")
