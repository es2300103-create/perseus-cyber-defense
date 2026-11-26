ASSETS = {
    'Host_01_Web': {'value': 50000.0, 'criticality': 0.3, 'base_vuln': 0.6},
    'Host_02_DB': {'value': 150000.0, 'criticality': 0.6, 'base_vuln': 0.4},
    'Host_03_Vault': {'value': 500000.0, 'criticality': 1.0, 'base_vuln': 0.2}
}
NETWORK_ASSETS = ASSETS
RISK_APPETITE_THRESHOLD = 25000.0
ATTACK_TACTICS = [
    'T1595_Gradual_Recon',
    'T1190_Injection_Attack',
    'T1068_Privilege_Escalation',
    'T1021_Lateral_Movement',
    'T1041_Data_Exfiltration'
]
DEFENSE_WEAPONS = [
    'Canary_Trap_Token',
    'Context_Poison_CounterPrompt',
    'Honeypot_Redirect'
]
CUSUM_PARAMS = {'mu_0': 0.25, 'k': 0.10, 'threshold_h': 1.50}
SPRT_PARAMS = {'alpha': 0.05, 'beta': 0.05, 'p0': 0.20, 'p1': 0.75}
EXP3_GAMMA = 0.15
DEFENSE_BUDGET = 1.0
