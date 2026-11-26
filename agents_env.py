import numpy as np
import pandas as pd

from config import ATTACK_TACTICS, DEFENSE_WEAPONS, NETWORK_ASSETS
from math_engine import (
    BayesianUpdater,
    CUSUMDetector,
    CyberRiskEngine,
    EXP3Attacker,
    MarkovChainPredictor,
    NashMixedEquilibrium,
    StackelbergSecurityGame,
    ThompsonSamplingDefender,
    WaldSPRT,
)


class PerseusDefenderAgent:
    def __init__(self):
        self.stackelberg = StackelbergSecurityGame()
        self.bayesian = BayesianUpdater()
        self.wald = WaldSPRT()
        self.markov = MarkovChainPredictor()
        self.cusum = CUSUMDetector()
        self.contextual_thompson = {
            tactic: ThompsonSamplingDefender() for tactic in ATTACK_TACTICS
        }
        self.mithridatic_immunity = {tactic: 0.0 for tactic in ATTACK_TACTICS}
        self.last_predicted_tactic = None
        self.last_prediction_prob = 0.0
        self.asset_attack_pressure = {asset: 0.0 for asset in NETWORK_ASSETS}
        self.defense_mode = "Balanced Posture"

    def allocate_preemptive_defense(self) -> dict:
        self.defense_mode = "Balanced Posture"
        dynamic_priors = {
            asset: min(
                0.95,
                float(data["base_vuln"]) + self.asset_attack_pressure[asset],
            )
            for asset, data in NETWORK_ASSETS.items()
        }
        allocation = dict(
            self.stackelberg.optimize_defense_allocation(dynamic_priors)["allocation"]
        )
        exfiltration_tactics = {
            "T1021_Lateral_Movement",
            "T1041_Data_Exfiltration",
            "T1041_Exfiltration",
        }
        if (
            self.last_predicted_tactic in exfiltration_tactics
            and self.last_prediction_prob > 0.40
        ):
            allocation["Host_03_Vault"] += 0.15
            total_allocation = sum(allocation.values())
            allocation = {
                asset: coverage / total_allocation
                for asset, coverage in allocation.items()
            }
            self.defense_mode = "Preemptive Vault Guard"
        return allocation

    def detect_and_respond(self, attack_packet: dict, shield_enabled: bool = True) -> dict:
        if not isinstance(attack_packet, dict):
            raise TypeError("attack_packet must be a dictionary")
        required_fields = {"target_asset", "tactic", "entropy", "anomaly", "is_stealth"}
        missing_fields = required_fields - set(attack_packet)
        if missing_fields:
            raise ValueError(f"Missing attack packet fields: {missing_fields}")
        if not isinstance(shield_enabled, bool):
            raise TypeError("shield_enabled must be a boolean")

        target_asset = attack_packet["target_asset"]
        tactic = attack_packet["tactic"]
        entropy = attack_packet["entropy"]
        anomaly = attack_packet["anomaly"]
        is_stealth = attack_packet["is_stealth"]
        if target_asset not in NETWORK_ASSETS:
            raise ValueError(f"Unknown target asset: {target_asset!r}")
        if tactic not in ATTACK_TACTICS:
            raise ValueError(f"Unknown tactic: {tactic!r}")
        if not isinstance(is_stealth, bool):
            raise TypeError("is_stealth must be a boolean")

        for asset, pressure in self.asset_attack_pressure.items():
            if asset == target_asset:
                self.asset_attack_pressure[asset] = min(0.5, pressure + 0.12)
            else:
                self.asset_attack_pressure[asset] = pressure * 0.85

        suspicious_keywords = attack_packet.get(
            "suspicious_keywords", 3 if is_stealth else 1
        )
        cusum_reading = self.cusum.update(anomaly)
        bayesian_reading = self.bayesian.update(
            entropy, target_asset == "Host_03_Vault", suspicious_keywords
        )
        wald_reading = self.wald.evaluate(anomaly)
        markov_reading = self.markov.predict_next(tactic)
        self.last_predicted_tactic = markov_reading["predicted_next_tactic"]
        self.last_prediction_prob = markov_reading["prediction_confidence"]

        overdrive_counter_shield = (
            attack_packet.get("attack_mode") == "Zero-Day Burst"
            or bayesian_reading["posterior"] > 0.88
        )
        stealth_lockdown = bool(cusum_reading["drift_detected"] and is_stealth)
        alert_triggered = bool(
            wald_reading["decision"] == "ATTACK_CONFIRMED"
            or cusum_reading["drift_detected"]
            or bayesian_reading["posterior"] > 0.70
            or overdrive_counter_shield
        )
        threat_detected = alert_triggered
        shield_bonus = 0.35
        if overdrive_counter_shield:
            self.defense_mode = "Overdrive Counter-Shield"
            shield_bonus = 0.52
        elif stealth_lockdown:
            self.defense_mode = "Stealth Lockdown"
            shield_bonus = 0.45

        selected_weapon = None
        weapon_selection = None
        if not shield_enabled:
            response = "None (Passive)"
        elif overdrive_counter_shield:
            selected_weapon = "Honeypot_Redirect"
            response = selected_weapon
        elif stealth_lockdown:
            selected_weapon = "Context_Poison_CounterPrompt"
            response = selected_weapon
        elif threat_detected:
            weapon_selection = self.contextual_thompson[tactic].select_weapon()
            selected_weapon = weapon_selection["selected_weapon"]
            if target_asset == "Host_03_Vault" or is_stealth:
                selected_weapon = "Honeypot_Redirect"
            response = selected_weapon
        else:
            response = "None (Monitoring)"

        return {
            "target_asset": target_asset,
            "tactic": tactic,
            "cusum_reading": cusum_reading,
            "bayesian_reading": bayesian_reading,
            "wald_reading": wald_reading,
            "markov_reading": markov_reading,
            "predicted_next_tactic": self.last_predicted_tactic,
            "prediction_probability": self.last_prediction_prob,
            "threat_detected": threat_detected,
            "alert_triggered": alert_triggered,
            "selected_weapon": selected_weapon,
            "weapon_selection": weapon_selection,
            "response": response,
            "defense_mode": self.defense_mode,
            "shield_bonus": shield_bonus,
            "mithridatic_immunity": self.mithridatic_immunity[tactic],
        }

    def learn_and_immunize(
        self,
        tactic: str,
        weapon: str | None,
        defense_succeeded: bool,
    ) -> float:
        if tactic not in self.contextual_thompson:
            raise ValueError(f"Unknown tactic: {tactic!r}")
        if weapon is not None and weapon not in DEFENSE_WEAPONS:
            raise ValueError(f"Unknown defense weapon: {weapon!r}")
        if not isinstance(defense_succeeded, bool):
            raise TypeError("defense_succeeded must be a boolean")

        for params in self.contextual_thompson[tactic].params.values():
            params["alpha"] *= 0.98
            params["beta"] *= 0.98
        if weapon in DEFENSE_WEAPONS:
            self.contextual_thompson[tactic].update_posterior(weapon, defense_succeeded)
        if defense_succeeded:
            self.mithridatic_immunity[tactic] = min(
                0.85, self.mithridatic_immunity[tactic] + 0.08
            )
            if self.last_predicted_tactic in self.mithridatic_immunity:
                self.mithridatic_immunity[self.last_predicted_tactic] = min(
                    0.85,
                    self.mithridatic_immunity[self.last_predicted_tactic] + 0.02,
                )
        return float(self.mithridatic_immunity[tactic])


class AdversarialAttackerAgent:
    def __init__(self):
        self.exp3 = EXP3Attacker()
        self.nash = NashMixedEquilibrium()
        self.total_attacks = 0
        self.successful_breaches = 0
        self.last_probs = None
        self.kill_chain_stage = 0
        self.consecutive_failures = 0
        self.locked_target = None
        self.attack_mode = "Kill-Chain"
        self.last_target_asset = None

    def act(self, defense_coverage: dict) -> dict:
        self.attack_mode = "Kill-Chain"
        is_zero_day_burst = (self.total_attacks + 1) % 10 == 0
        attack_power_bonus = 0.22 if is_zero_day_burst else 0.0

        if is_zero_day_burst:
            target_asset = "Host_03_Vault"
            target_probs = {
                asset: float(asset == target_asset) for asset in NETWORK_ASSETS
            }
            self.attack_mode = "Zero-Day Burst"
        elif self.locked_target is not None:
            target_asset = self.locked_target
            target_probs = {
                asset: float(asset == target_asset) for asset in NETWORK_ASSETS
            }
            self.attack_mode = "Target Lock"
        else:
            target_strategy = self.nash.compute_attacker_strategy(defense_coverage)
            target_asset = target_strategy["target_asset"]
            target_probs = target_strategy["attack_probabilities"]
        self.last_target_asset = target_asset

        tactic_choice = self.exp3.select_tactic()
        tactic_probs = np.asarray(tactic_choice["probabilities"], dtype=float).copy()
        tactic_probs[self.kill_chain_stage] += 0.25
        tactic_probs /= tactic_probs.sum()
        tactic_idx = int(np.random.choice(len(tactic_probs), p=tactic_probs))
        tactic = ATTACK_TACTICS[tactic_idx]

        if self.consecutive_failures >= 3 or tactic == "T1595_Gradual_Recon":
            is_stealth = True
            self.attack_mode = "Low-and-Slow Stealth"
            entropy = np.random.uniform(0.30, 0.48)
            anomaly = np.random.uniform(0.22, 0.34)
        else:
            is_stealth = False
            entropy = np.random.uniform(0.65, 0.95)
            anomaly = np.random.uniform(0.60, 0.92)

        self.last_probs = tactic_probs
        self.total_attacks += 1
        return {
            "target_asset": target_asset,
            "tactic": tactic,
            "tactic_idx": tactic_idx,
            "entropy": entropy,
            "anomaly": anomaly,
            "is_stealth": is_stealth,
            "target_probs": target_probs,
            "tactic_probs": tactic_probs.copy(),
            "exp3_weights": self.exp3.weights.copy(),
            "attack_mode": self.attack_mode,
            "attack_power_bonus": attack_power_bonus,
        }

    def learn(self, tactic_idx: int, breach_succeeded: bool, reward: float) -> dict:
        if not isinstance(breach_succeeded, bool):
            raise TypeError("breach_succeeded must be a boolean")
        if self.last_probs is None:
            raise RuntimeError("act must be called before learn")
        if breach_succeeded:
            self.successful_breaches += 1
            self.consecutive_failures = 0
            if self.last_target_asset is not None:
                self.locked_target = self.last_target_asset
            self.kill_chain_stage = min(
                len(ATTACK_TACTICS) - 1, self.kill_chain_stage + 1
            )
        else:
            self.consecutive_failures += 1
            self.locked_target = None

        self.exp3.update_weights(tactic_idx, reward, self.last_probs)
        breach_rate = (
            self.successful_breaches / self.total_attacks
            if self.total_attacks > 0
            else 0.0
        )
        learning_state = {
            "breach_rate": breach_rate,
            "updated_weights": self.exp3.weights.copy(),
        }
        if not breach_succeeded and self.consecutive_failures >= 4:
            learning_state["polymorphic_decay"] = 0.15
        return learning_state


class CyberSimulationEnv:
    def __init__(self):
        self.defender = PerseusDefenderAgent()
        self.history = []
        self.network_assets = {
            asset: {**data, "status": "operational"}
            for asset, data in NETWORK_ASSETS.items()
        }

    def step(
        self,
        round_num: int,
        attacker_agent,
        shield_enabled: bool = True,
    ) -> dict:
        if isinstance(round_num, bool) or not isinstance(round_num, int):
            raise TypeError("round_num must be an integer")
        if not isinstance(shield_enabled, bool):
            raise TypeError("shield_enabled must be a boolean")

        defense_coverage = self.defender.allocate_preemptive_defense()
        attack_packet = attacker_agent.act(defense_coverage)
        defense_response = self.defender.detect_and_respond(
            attack_packet, shield_enabled=shield_enabled
        )

        target_asset = attack_packet["target_asset"]
        tactic = attack_packet["tactic"]
        tactic_idx = attack_packet["tactic_idx"]
        attack_power_bonus = float(attack_packet.get("attack_power_bonus", 0.0))
        weapon = defense_response["selected_weapon"]
        immunity_level = float(defense_response["mithridatic_immunity"])
        shield_contribution = (
            float(defense_response.get("shield_bonus", 0.35))
            if shield_enabled
            and weapon is not None
            and defense_response["alert_triggered"]
            else 0.0
        )
        total_defense_power = float(
            0.35 * defense_coverage[target_asset]
            + shield_contribution
            + 0.30 * immunity_level
            - attack_power_bonus
            + np.random.uniform(-0.08, 0.08)
        )
        defense_succeeded = bool(total_defense_power >= 0.48)
        breach_succeeded = not defense_succeeded

        new_immunity = self.defender.learn_and_immunize(
            tactic, weapon, defense_succeeded
        )
        attacker_reward = 1.0 if breach_succeeded else 0.1
        attacker_learning = attacker_agent.learn(
            tactic_idx, breach_succeeded, attacker_reward
        )
        polymorphic_decay = float(attacker_learning.get("polymorphic_decay", 0.0))
        if polymorphic_decay:
            new_immunity = max(0.0, float(new_immunity) - polymorphic_decay)
            self.defender.mithridatic_immunity[tactic] = new_immunity
        risk_metrics = CyberRiskEngine().evaluate_round_risk(
            asset_value=self.network_assets[target_asset]["value"],
            attack_prob=attack_packet["target_probs"][target_asset],
            defense_coverage=defense_coverage[target_asset],
            shield_active=bool(shield_contribution > 0.0),
            immunity_level=new_immunity,
        )

        record = {
            "round_num": round_num,
            "target_asset": target_asset,
            "asset_value": float(self.network_assets[target_asset]["value"]),
            "attack_mode": attack_packet.get("attack_mode", "Kill-Chain"),
            "attack_power_bonus": attack_power_bonus,
            "defense_mode": defense_response.get("defense_mode", "Balanced Posture"),
            "tactic": tactic,
            "tactic_idx": tactic_idx,
            "entropy": attack_packet["entropy"],
            "anomaly": attack_packet["anomaly"],
            "is_stealth": attack_packet["is_stealth"],
            "shield_enabled": shield_enabled,
            "defense_coverage": float(defense_coverage[target_asset]),
            "defense_coverage_allocation": dict(defense_coverage),
            "predicted_next_tactic": defense_response["predicted_next_tactic"],
            "prediction_probability": defense_response["prediction_probability"],
            "alert_triggered": defense_response["alert_triggered"],
            "selected_weapon": weapon,
            "total_defense_power": total_defense_power,
            "defense_succeeded": defense_succeeded,
            "breach_succeeded": breach_succeeded,
            "attacker_reward": attacker_reward,
            "new_immunity": float(new_immunity),
            "bayesian_posterior": defense_response["bayesian_reading"]["posterior"],
            "wald_decision": defense_response["wald_reading"]["decision"],
            "cusum_value": defense_response["cusum_reading"]["cusum_value"],
            "cusum_drift_detected": defense_response["cusum_reading"]["drift_detected"],
            "attacker_breach_rate": attacker_learning["breach_rate"],
            "attacker_learning": attacker_learning,
            "inherent_risk": risk_metrics["inherent_risk"],
            "residual_risk": risk_metrics["residual_risk"],
            "loss_avoided": risk_metrics["loss_avoided"],
            "within_risk_appetite": risk_metrics["within_risk_appetite"],
            "attack_packet": attack_packet,
            "defense_response": defense_response,
            "risk_metrics": risk_metrics,
        }
        self.history.append(record)
        return record

    def run_simulation(
        self,
        num_rounds: int = 50,
        shield_enabled: bool = True,
    ) -> pd.DataFrame:
        if isinstance(num_rounds, bool) or not isinstance(num_rounds, int):
            raise TypeError("num_rounds must be an integer")
        if num_rounds < 0:
            raise ValueError("num_rounds must be non-negative")
        attacker_agent = AdversarialAttackerAgent()
        records = [
            self.step(
                round_num,
                attacker_agent,
                shield_enabled=shield_enabled,
            )
            for round_num in range(1, num_rounds + 1)
        ]
        return pd.DataFrame(records)


if __name__ == "__main__":
    environment = CyberSimulationEnv()
    simulation_results = environment.run_simulation(num_rounds=25, shield_enabled=True)
    defense_success_rate = float(simulation_results["defense_succeeded"].mean())
    total_loss_avoided = float(simulation_results["loss_avoided"].sum())
    print(f"Perseus defense success rate: {defense_success_rate:.2%}")
    print(f"Total loss avoided: ${total_loss_avoided:,.2f}")
