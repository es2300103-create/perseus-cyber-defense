import math
from numbers import Integral, Real

import numpy as np

from config import (
    ATTACK_TACTICS,
    CUSUM_PARAMS,
    DEFENSE_WEAPONS,
    EXP3_GAMMA,
    NETWORK_ASSETS,
    RISK_APPETITE_THRESHOLD,
    SPRT_PARAMS,
)


def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    return (
        1.0
        / (sigma * math.sqrt(2.0 * math.pi))
        * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )


def _poisson_pmf(k: int, lam: float) -> float:
    k = int(k)
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    try:
        log_probability = k * math.log(lam) - lam - math.lgamma(k + 1)
    except OverflowError:
        return 0.0
    if log_probability < -745.0:
        return 0.0
    return math.exp(log_probability)


class BayesianUpdater:
    def __init__(self, initial_prior: float = 0.15):
        if isinstance(initial_prior, (bool, np.bool_)) or not isinstance(initial_prior, Real):
            raise TypeError("initial_prior must be a real number")
        initial_prior = float(initial_prior)
        if not np.isfinite(initial_prior) or not 0.0 <= initial_prior <= 1.0:
            raise ValueError("initial_prior must be finite and between 0.0 and 1.0")
        self.initial_prior = initial_prior
        self.current_prior = initial_prior
        self.posterior_history = [initial_prior]

    def compute_likelihoods(
        self,
        entropy: float,
        targets_vault: bool,
        suspicious_keywords: int,
    ) -> tuple[float, float]:
        if isinstance(entropy, (bool, np.bool_)) or not isinstance(entropy, Real):
            raise TypeError("entropy must be a real number")
        entropy = float(entropy)
        if not np.isfinite(entropy) or not 0.0 <= entropy <= 8.0:
            raise ValueError("entropy must be finite and between 0.0 and 8.0")
        if not isinstance(targets_vault, (bool, np.bool_)):
            raise TypeError("targets_vault must be a boolean")
        if isinstance(suspicious_keywords, (bool, np.bool_)) or not isinstance(
            suspicious_keywords, Integral
        ):
            raise TypeError("suspicious_keywords must be an integer")
        if suspicious_keywords < 0:
            raise ValueError("suspicious_keywords must be non-negative")

        likelihood_entropy_attack = _normal_pdf(entropy, 6.5, 1.2)
        likelihood_entropy_safe = _normal_pdf(entropy, 3.5, 1.0)
        likelihood_keywords_attack = _poisson_pmf(suspicious_keywords, 3.5)
        likelihood_keywords_safe = _poisson_pmf(suspicious_keywords, 0.3)
        vault_multiplier_attack = 2.5 if targets_vault else 1.0
        vault_multiplier_safe = 0.4 if targets_vault else 1.0

        likelihood_attack = float(
            likelihood_entropy_attack
            * likelihood_keywords_attack
            * vault_multiplier_attack
            + 1e-9
        )
        likelihood_safe = float(
            likelihood_entropy_safe
            * likelihood_keywords_safe
            * vault_multiplier_safe
            + 1e-9
        )
        return likelihood_attack, likelihood_safe

    def update(
        self,
        entropy: float,
        targets_vault: bool,
        suspicious_keywords: int,
    ) -> dict[str, float]:
        prior = self.current_prior
        likelihood_attack, likelihood_safe = self.compute_likelihoods(
            entropy, targets_vault, suspicious_keywords
        )
        posterior = (likelihood_attack * prior) / (
            likelihood_attack * prior + likelihood_safe * (1.0 - prior)
        )
        posterior = float(np.clip(posterior, 0.001, 0.999))

        self.current_prior = posterior
        self.posterior_history.append(posterior)
        return {
            "prior": prior,
            "likelihood_attack": likelihood_attack,
            "likelihood_safe": likelihood_safe,
            "posterior": posterior,
            "bayes_factor": likelihood_attack / likelihood_safe,
        }

    def reset(self) -> None:
        self.current_prior = self.initial_prior
        self.posterior_history.clear()


class WaldSPRT:
    def __init__(
        self,
        alpha: float = SPRT_PARAMS["alpha"],
        beta: float = SPRT_PARAMS["beta"],
        p0: float = SPRT_PARAMS["p0"],
        p1: float = SPRT_PARAMS["p1"],
    ):
        for name, value in (
            ("alpha", alpha),
            ("beta", beta),
            ("p0", p0),
            ("p1", p1),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(
                    f"{name} must be finite and strictly between 0.0 and 1.0"
                )

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.p0 = float(p0)
        self.p1 = float(p1)
        if self.p1 <= self.p0:
            raise ValueError("p1 must be greater than p0")

        self.A = math.log1p(-self.beta) - math.log(self.alpha)
        self.B = math.log(self.beta) - math.log1p(-self.alpha)
        self.llr = 0.0
        self.steps_taken = 0
        self.llr_history = [0.0]

    def evaluate(self, observation_score: float) -> dict:
        if isinstance(observation_score, (bool, np.bool_)) or not isinstance(
            observation_score, Real
        ):
            raise TypeError("observation_score must be a real number")
        observation_score = float(observation_score)
        if not np.isfinite(observation_score) or not 0.0 <= observation_score <= 1.0:
            raise ValueError("observation_score must be finite and between 0.0 and 1.0")

        x = float(np.clip(observation_score, 0.001, 0.999))
        self.steps_taken += 1
        log_density_attack = x * math.log(self.p1) + (1.0 - x) * math.log1p(
            -self.p1
        )
        log_density_safe = x * math.log(self.p0) + (1.0 - x) * math.log1p(
            -self.p0
        )
        delta_llr = log_density_attack - log_density_safe
        self.llr += delta_llr
        self.llr_history.append(self.llr)

        if self.llr >= self.A:
            decision = "ATTACK_CONFIRMED"
        elif self.llr <= self.B:
            decision = "SAFE_CONFIRMED"
        else:
            decision = "CONTINUE_SAMPLING"

        return {
            "decision": decision,
            "llr": self.llr,
            "delta_llr": delta_llr,
            "upper_bound": self.A,
            "lower_bound": self.B,
            "steps_taken": self.steps_taken,
        }

    def reset(self) -> None:
        self.llr = 0.0
        self.steps_taken = 0
        self.llr_history.clear()


class MarkovChainPredictor:
    def __init__(self, tactics: list[str] = ATTACK_TACTICS):
        if not isinstance(tactics, list) or not all(
            isinstance(tactic, str) for tactic in tactics
        ):
            raise TypeError("tactics must be a list of strings")
        if (
            len(tactics) != len(ATTACK_TACTICS)
            or len(set(tactics)) != len(tactics)
            or set(tactics) != set(ATTACK_TACTICS)
        ):
            raise ValueError("tactics must contain each configured attack tactic exactly once")

        self.tactics = tactics
        self.n_tactics = len(tactics)
        self.tactic_to_idx = {
            tactic: idx for idx, tactic in enumerate(self.tactics)
        }
        self.idx_to_tactic = {
            idx: tactic for idx, tactic in enumerate(self.tactics)
        }

        base_matrix = np.array(
            [
                [0.15, 0.45, 0.30, 0.07, 0.03],
                [0.00, 0.25, 0.50, 0.25, 0.00],
                [0.00, 0.00, 0.20, 0.55, 0.25],
                [0.00, 0.00, 0.00, 0.35, 0.65],
                [0.40, 0.30, 0.00, 0.00, 0.30],
            ],
            dtype=float,
        )
        order = [ATTACK_TACTICS.index(tactic) for tactic in self.tactics]
        self.transition_matrix = base_matrix[np.ix_(order, order)].copy()
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0):
            raise RuntimeError("Each transition matrix row must sum to 1.0")

    def _get_tactic_index(self, tactic: str) -> int:
        if not isinstance(tactic, str) or tactic not in self.tactic_to_idx:
            raise ValueError(f"Unknown tactic: {tactic!r}")
        return self.tactic_to_idx[tactic]

    def predict_next(self, current_tactic: str) -> dict:
        current_idx = self._get_tactic_index(current_tactic)
        probabilities = self.transition_matrix[current_idx]
        predicted_idx = int(np.argmax(probabilities))
        shannon_uncertainty = max(
            0.0,
            float(-np.sum(probabilities * np.log2(probabilities + 1e-12))),
        )
        return {
            "current_tactic": current_tactic,
            "predicted_next_tactic": self.idx_to_tactic[predicted_idx],
            "prediction_confidence": float(probabilities[predicted_idx]),
            "shannon_uncertainty": shannon_uncertainty,
            "full_probability_distribution": {
                tactic: float(probabilities[self.tactic_to_idx[tactic]])
                for tactic in self.tactics
            },
        }

    def update_transition(
        self,
        prev_tactic: str,
        observed_tactic: str,
        learning_rate: float = 0.15,
    ) -> np.ndarray:
        prev_idx = self._get_tactic_index(prev_tactic)
        observed_idx = self._get_tactic_index(observed_tactic)
        if isinstance(learning_rate, (bool, np.bool_)) or not isinstance(
            learning_rate, Real
        ):
            raise TypeError("learning_rate must be a real number")
        learning_rate = float(learning_rate)
        if not np.isfinite(learning_rate) or not 0.0 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must be finite and between 0.0 and 1.0")

        self.transition_matrix[prev_idx] *= 1.0 - learning_rate
        self.transition_matrix[prev_idx, observed_idx] += learning_rate
        row_sum = self.transition_matrix[prev_idx].sum()
        self.transition_matrix[prev_idx] /= row_sum
        return self.transition_matrix

    def predict_n_steps(self, current_tactic: str, n_steps: int = 3) -> dict:
        current_idx = self._get_tactic_index(current_tactic)
        if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, Integral):
            raise TypeError("n_steps must be an integer")
        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        transition_power = np.linalg.matrix_power(self.transition_matrix, n_steps)
        probabilities = transition_power[current_idx]
        predicted_idx = int(np.argmax(probabilities))
        shannon_uncertainty = max(
            0.0,
            float(-np.sum(probabilities * np.log2(probabilities + 1e-12))),
        )
        exfiltration_idx = self.tactic_to_idx["T1041_Data_Exfiltration"]
        return {
            "current_tactic": current_tactic,
            "n_steps": n_steps,
            "predicted_tactic": self.idx_to_tactic[predicted_idx],
            "prediction_confidence": float(probabilities[predicted_idx]),
            "shannon_uncertainty": shannon_uncertainty,
            "exfiltration_probability": float(probabilities[exfiltration_idx]),
            "full_probability_distribution": {
                tactic: float(probabilities[self.tactic_to_idx[tactic]])
                for tactic in self.tactics
            },
        }


class ThompsonSamplingDefender:
    def __init__(
        self,
        weapons: list[str] = DEFENSE_WEAPONS,
        prior_alpha: float = 2.0,
        prior_beta: float = 1.0,
    ):
        if not isinstance(weapons, list) or not all(
            isinstance(weapon, str) for weapon in weapons
        ):
            raise TypeError("weapons must be a list of strings")
        if not weapons or len(set(weapons)) != len(weapons):
            raise ValueError("weapons must be non-empty and contain unique names")
        for name, value in (("prior_alpha", prior_alpha), ("prior_beta", prior_beta)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than 0.0")

        self.weapons = weapons
        self.params = {
            weapon: {"alpha": float(prior_alpha), "beta": float(prior_beta)}
            for weapon in weapons
        }
        self.selection_counts = {weapon: 0 for weapon in weapons}

    def select_weapon(self) -> dict:
        sampled_thetas = {
            weapon: float(
                np.random.beta(params["alpha"], params["beta"])
            )
            for weapon, params in self.params.items()
        }
        expected_win_rates = {}
        for weapon, params in self.params.items():
            alpha = params["alpha"]
            beta = params["beta"]
            if alpha >= beta:
                ratio = beta / alpha
                expected_win_rates[weapon] = float(1.0 / (1.0 + ratio))
            else:
                ratio = alpha / beta
                expected_win_rates[weapon] = float(ratio / (1.0 + ratio))
        selected_weapon = max(sampled_thetas, key=sampled_thetas.get)
        self.selection_counts[selected_weapon] += 1
        return {
            "selected_weapon": selected_weapon,
            "sampled_thetas": sampled_thetas,
            "expected_win_rates": expected_win_rates,
            "beta_params": {
                weapon: dict(params) for weapon, params in self.params.items()
            },
        }

    def update_posterior(
        self,
        weapon: str,
        success: bool,
        weight: float = 1.0,
    ) -> dict:
        if weapon not in self.params:
            raise ValueError(f"Unknown weapon: {weapon!r}")
        if not isinstance(success, (bool, np.bool_)):
            raise TypeError("success must be a boolean")
        if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, Real):
            raise TypeError("weight must be a real number")
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be finite and non-negative")

        params = self.params[weapon]
        parameter_name = "alpha" if success else "beta"
        updated_parameter = params[parameter_name] + weight
        if not math.isfinite(updated_parameter):
            raise ValueError("posterior parameter update would overflow")
        params[parameter_name] = updated_parameter
        return dict(params)


class EXP3Attacker:
    def __init__(
        self,
        tactics: list[str] = ATTACK_TACTICS,
        gamma: float = EXP3_GAMMA,
    ):
        if not isinstance(tactics, list) or not all(
            isinstance(tactic, str) for tactic in tactics
        ):
            raise TypeError("tactics must be a list of strings")
        if not tactics or len(set(tactics)) != len(tactics):
            raise ValueError("tactics must be non-empty and contain unique names")
        if isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, Real):
            raise TypeError("gamma must be a real number")
        gamma = float(gamma)
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and between 0.0 and 1.0")

        self.tactics = tactics
        self.gamma = gamma
        self.K = len(tactics)
        self.weights = np.ones(self.K, dtype=np.float64)

    def select_tactic(self) -> dict:
        weight_sum = self.weights.sum()
        with np.errstate(under="ignore"):
            probabilities = (
                (1.0 - self.gamma) * self.weights / weight_sum
                + self.gamma / self.K
            )
            probabilities /= probabilities.sum()
        tactic_index = int(np.random.choice(self.K, p=probabilities))
        return {
            "tactic_index": tactic_index,
            "selected_tactic": self.tactics[tactic_index],
            "probabilities": probabilities,
        }

    def update_weights(
        self,
        tactic_idx: int,
        reward: float,
        last_probs: np.ndarray,
    ) -> np.ndarray:
        if isinstance(tactic_idx, (bool, np.bool_)) or not isinstance(
            tactic_idx, Integral
        ):
            raise TypeError("tactic_idx must be an integer")
        tactic_idx = int(tactic_idx)
        if not 0 <= tactic_idx < self.K:
            raise ValueError("tactic_idx is out of range")
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, Real):
            raise TypeError("reward must be a real number")
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")

        probabilities = np.asarray(last_probs, dtype=np.float64)
        if probabilities.shape != (self.K,):
            raise ValueError("last_probs must have one probability per tactic")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("last_probs must contain finite non-negative values")
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("last_probs must sum to 1.0")
        if probabilities[tactic_idx] <= 0.0:
            raise ValueError("the selected tactic must have positive probability")

        clipped_reward = float(np.clip(reward, 0.0, 1.0))
        log_multipliers = np.zeros(self.K, dtype=np.float64)
        if clipped_reward > 0.0 and self.gamma > 0.0:
            log_estimated_reward = math.log(clipped_reward) - math.log(
                float(probabilities[tactic_idx])
            )
            log_multipliers[tactic_idx] = (
                self.gamma * log_estimated_reward / self.K
            )
        log_multipliers -= np.max(log_multipliers)
        tiny_weight = np.finfo(np.float64).tiny
        self.weights = np.maximum(self.weights, tiny_weight)
        with np.errstate(under="ignore"):
            self.weights *= np.exp(log_multipliers)
        self.weights = np.maximum(self.weights, tiny_weight)
        self.weights /= np.max(self.weights)
        return self.weights


class NashMixedEquilibrium:
    def __init__(self, assets: dict = NETWORK_ASSETS):
        if not isinstance(assets, dict) or not assets:
            raise ValueError("assets must be a non-empty dictionary")

        self.assets = assets
        self.asset_values = {}
        self.attack_costs = {}
        for asset, data in assets.items():
            if not isinstance(asset, str) or not isinstance(data, dict):
                raise TypeError("each asset must have a string name and a dictionary")
            value = data.get("value")
            attack_cost = data.get("attack_cost", 0.0)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"value for {asset!r} must be a real number")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"value for {asset!r} must be finite and non-negative")
            if isinstance(attack_cost, (bool, np.bool_)) or not isinstance(
                attack_cost, Real
            ):
                raise TypeError(f"attack_cost for {asset!r} must be a real number")
            if not np.isfinite(attack_cost) or attack_cost < 0.0:
                raise ValueError(
                    f"attack_cost for {asset!r} must be finite and non-negative"
                )
            self.asset_values[asset] = float(value)
            self.attack_costs[asset] = float(attack_cost)

    def compute_attacker_strategy(self, defense_coverage: dict) -> dict:
        if not isinstance(defense_coverage, dict):
            raise TypeError("defense_coverage must be a dictionary")
        unknown_assets = set(defense_coverage) - set(self.asset_values)
        if unknown_assets:
            raise ValueError(f"Unknown assets in defense coverage: {unknown_assets}")

        asset_names = list(self.asset_values)
        expected_payoffs = {}
        for asset in asset_names:
            coverage = defense_coverage.get(asset, 0.0)
            if isinstance(coverage, (bool, np.bool_)) or not isinstance(
                coverage, Real
            ):
                raise TypeError(f"coverage for {asset!r} must be a real number")
            if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
                raise ValueError(f"coverage for {asset!r} must be between 0.0 and 1.0")
            expected_payoffs[asset] = (
                self.asset_values[asset] * (1.0 - float(coverage))
                - self.attack_costs[asset]
            )

        payoff_array = np.array(
            [expected_payoffs[asset] for asset in asset_names], dtype=np.float64
        )
        with np.errstate(over="ignore", under="ignore"):
            shifted_payoffs = payoff_array - np.max(payoff_array)
            exp_payoffs = np.exp(shifted_payoffs)
        probabilities = exp_payoffs / exp_payoffs.sum()
        selected_idx = int(np.random.choice(len(asset_names), p=probabilities))
        return {
            "target_asset": asset_names[selected_idx],
            "attack_probabilities": {
                asset: float(probabilities[idx])
                for idx, asset in enumerate(asset_names)
            },
            "expected_payoffs": expected_payoffs,
        }


class StackelbergSecurityGame:
    def __init__(
        self,
        assets: dict = NETWORK_ASSETS,
        total_budget: float = 1.0,
    ):
        if not isinstance(assets, dict) or not assets:
            raise ValueError("assets must be a non-empty dictionary")
        if isinstance(total_budget, (bool, np.bool_)) or not isinstance(
            total_budget, Real
        ):
            raise TypeError("total_budget must be a real number")
        total_budget = float(total_budget)
        if not np.isfinite(total_budget) or not 0.0 <= total_budget <= len(assets):
            raise ValueError("total_budget must be finite and between 0 and asset count")

        self.assets = assets
        self.total_budget = total_budget
        self.n_assets = len(assets)
        self.asset_values = {}
        for asset, data in assets.items():
            if not isinstance(asset, str) or not isinstance(data, dict):
                raise TypeError("each asset must have a string name and a dictionary")
            value = data.get("value")
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"value for {asset!r} must be a real number")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"value for {asset!r} must be finite and non-negative")
            self.asset_values[asset] = float(value)

    def optimize_defense_allocation(self, threat_priors: dict) -> dict:
        if not isinstance(threat_priors, dict):
            raise TypeError("threat_priors must be a dictionary")
        unknown_assets = set(threat_priors) - set(self.asset_values)
        if unknown_assets:
            raise ValueError(f"Unknown assets in threat priors: {unknown_assets}")

        asset_names = list(self.asset_values)
        threat_array = np.zeros(self.n_assets, dtype=np.float64)
        for idx, asset in enumerate(asset_names):
            threat = threat_priors.get(asset, 0.0)
            if isinstance(threat, (bool, np.bool_)) or not isinstance(threat, Real):
                raise TypeError(f"threat prior for {asset!r} must be a real number")
            if not np.isfinite(threat) or not 0.0 <= threat <= 1.0:
                raise ValueError(
                    f"threat prior for {asset!r} must be between 0.0 and 1.0"
                )
            threat_array[idx] = float(threat)

        value_array = np.array(
            [self.asset_values[asset] for asset in asset_names], dtype=np.float64
        )
        critical_weights = threat_array * value_array
        basic_coverage = min(
            0.05, self.total_budget / (2.0 * self.n_assets)
        )
        lower_bounds = np.full(self.n_assets, basic_coverage, dtype=np.float64)

        def required_allocation(loss_bound: float) -> np.ndarray:
            allocation = lower_bounds.copy()
            positive_weights = critical_weights > 0.0
            with np.errstate(over="ignore", divide="ignore"):
                allocation[positive_weights] = np.maximum(
                    lower_bounds[positive_weights],
                    1.0 - loss_bound / critical_weights[positive_weights],
                )
            return np.minimum(allocation, 1.0)

        if (
            self.total_budget <= lower_bounds.sum()
            or not np.any(critical_weights > 0.0)
        ):
            allocation = lower_bounds.copy()
        else:
            lower_loss = 0.0
            upper_loss = float(np.max(critical_weights))
            for _ in range(200):
                midpoint = (lower_loss + upper_loss) / 2.0
                if midpoint == lower_loss or midpoint == upper_loss:
                    break
                if required_allocation(midpoint).sum() > self.total_budget:
                    lower_loss = midpoint
                else:
                    upper_loss = midpoint
            allocation = required_allocation(upper_loss)

        remaining_budget = self.total_budget - float(allocation.sum())
        if remaining_budget > 0.0:
            for idx in np.argsort(-critical_weights):
                addition = min(remaining_budget, 1.0 - allocation[idx])
                allocation[idx] += addition
                remaining_budget -= addition
                if remaining_budget <= 0.0:
                    break
        elif remaining_budget < 0.0:
            for idx in np.argsort(critical_weights):
                reduction = min(-remaining_budget, allocation[idx] - lower_bounds[idx])
                allocation[idx] -= reduction
                remaining_budget += reduction
                if remaining_budget >= 0.0:
                    break

        allocation = np.clip(allocation, 0.0, 1.0)
        if not np.allclose(allocation.sum(), self.total_budget):
            raise RuntimeError("Defense allocation does not satisfy the budget")
        expected_losses = critical_weights * (1.0 - allocation)
        return {
            "allocation": {
                asset: float(allocation[idx])
                for idx, asset in enumerate(asset_names)
            },
            "minimax_expected_loss": float(np.max(expected_losses)),
        }


class CyberRiskEngine:
    def __init__(self, risk_appetite: float = RISK_APPETITE_THRESHOLD):
        if isinstance(risk_appetite, (bool, np.bool_)) or not isinstance(
            risk_appetite, Real
        ):
            raise TypeError("risk_appetite must be a real number")
        risk_appetite = float(risk_appetite)
        if not np.isfinite(risk_appetite) or risk_appetite < 0.0:
            raise ValueError("risk_appetite must be finite and non-negative")
        self.risk_appetite = risk_appetite

    def evaluate_round_risk(
        self,
        asset_value: float,
        attack_prob: float,
        defense_coverage: float,
        shield_active: bool,
        immunity_level: float,
    ) -> dict:
        if isinstance(asset_value, (bool, np.bool_)) or not isinstance(
            asset_value, Real
        ):
            raise TypeError("asset_value must be a real number")
        if not np.isfinite(asset_value) or asset_value < 0.0:
            raise ValueError("asset_value must be finite and non-negative")
        if not isinstance(shield_active, (bool, np.bool_)):
            raise TypeError("shield_active must be a boolean")
        for name, value in (
            ("attack_prob", attack_prob),
            ("defense_coverage", defense_coverage),
            ("immunity_level", immunity_level),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

        asset_value = float(asset_value)
        attack_prob = float(attack_prob)
        defense_coverage = float(defense_coverage)
        immunity_level = float(immunity_level)
        shield_reduction = 0.5 if shield_active else 0.0
        survival_factor = (
            (1.0 - defense_coverage)
            * (1.0 - shield_reduction)
            * (1.0 - immunity_level)
        )
        reduction_factor = 1.0 - survival_factor
        inherent_risk = asset_value * attack_prob
        residual_risk = inherent_risk * (1.0 - reduction_factor)
        loss_avoided = inherent_risk - residual_risk
        return {
            "inherent_risk": inherent_risk,
            "defense_reduction": defense_coverage,
            "shield_reduction": shield_reduction,
            "immunity_reduction": immunity_level,
            "reduction_factor": reduction_factor,
            "residual_risk": residual_risk,
            "loss_avoided": loss_avoided,
            "risk_appetite": self.risk_appetite,
            "within_risk_appetite": bool(residual_risk <= self.risk_appetite),
        }


class CUSUMDetector:
    def __init__(
        self,
        mu_0: float = CUSUM_PARAMS["mu_0"],
        k: float = CUSUM_PARAMS["k"],
        threshold_h: float = CUSUM_PARAMS["threshold_h"],
    ):
        for name, value in (("mu_0", mu_0), ("k", k), ("threshold_h", threshold_h)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if mu_0 < 0.0:
            raise ValueError("mu_0 must be non-negative")
        if k < 0.0:
            raise ValueError("k must be non-negative")
        if threshold_h <= 0.0:
            raise ValueError("threshold_h must be greater than 0.0")

        self.mu_0 = float(mu_0)
        self.k = float(k)
        self.threshold_h = float(threshold_h)
        self.S_t = 0.0
        self.consecutive_drift_steps = 0
        self.cusum_history = [0.0]

    def update(self, risk_score: float) -> dict:
        if isinstance(risk_score, (bool, np.bool_)) or not isinstance(
            risk_score, Real
        ):
            raise TypeError("risk_score must be a real number")
        risk_score = float(risk_score)
        if not np.isfinite(risk_score) or not 0.0 <= risk_score <= 1.0:
            raise ValueError("risk_score must be finite and between 0.0 and 1.0")

        instant_deviation = risk_score - self.mu_0 - self.k
        self.S_t = max(0.0, self.S_t + instant_deviation)
        self.cusum_history.append(self.S_t)
        if instant_deviation > 0.0:
            self.consecutive_drift_steps += 1
        if self.S_t == 0.0:
            self.consecutive_drift_steps = 0

        drift_detected = bool(self.S_t >= self.threshold_h)
        drift_ratio = min(1.0, self.S_t / self.threshold_h)
        return {
            "cusum_value": self.S_t,
            "instant_deviation": instant_deviation,
            "threshold": self.threshold_h,
            "drift_detected": drift_detected,
            "drift_ratio": drift_ratio,
            "consecutive_drift_steps": self.consecutive_drift_steps,
        }

    def reset_after_mitigation(self, partial_decay: float = 0.0) -> None:
        if isinstance(partial_decay, (bool, np.bool_)) or not isinstance(
            partial_decay, Real
        ):
            raise TypeError("partial_decay must be a real number")
        partial_decay = float(partial_decay)
        if not np.isfinite(partial_decay) or not 0.0 <= partial_decay <= 1.0:
            raise ValueError("partial_decay must be finite and between 0.0 and 1.0")

        if partial_decay == 0.0:
            self.S_t = 0.0
            self.consecutive_drift_steps = 0
        else:
            self.S_t *= 1.0 - partial_decay
