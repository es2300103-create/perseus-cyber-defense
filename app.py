import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents_env import AdversarialAttackerAgent, CyberSimulationEnv
from config import ATTACK_TACTICS, DEFENSE_WEAPONS, NETWORK_ASSETS


APP_TITLE = "🛡️ Mithridates & Perseus Shield: Multi-Agent Cyber Defense Simulator"
ASSET_LABELS = {
    "Host_01_Web": "Web",
    "Host_02_DB": "DB",
    "Host_03_Vault": "Vault",
}
TACTIC_LABELS = {
    "T1595_Gradual_Recon": "Gradual Recon",
    "T1190_Injection_Attack": "Injection Attack",
    "T1068_Privilege_Escalation": "Privilege Escalation",
    "T1021_Lateral_Movement": "Lateral Movement",
    "T1041_Data_Exfiltration": "Data Exfiltration",
}
WEAPON_LABELS = {
    "Canary_Trap_Token": "Canary Trap",
    "Context_Poison_CounterPrompt": "Counterprompt",
    "Honeypot_Redirect": "Honeypot Redirect",
}
TACTIC_COLORS = ["#f59e0b", "#ef4444", "#a855f7", "#38bdf8", "#22c55e"]
ASSET_COLORS = ["#38bdf8", "#a855f7", "#f59e0b"]


def reset_simulation() -> None:
    st.session_state.environment = CyberSimulationEnv()
    st.session_state.attacker = AdversarialAttackerAgent()


def execute_rounds(num_rounds: int, shield_enabled: bool) -> None:
    environment = st.session_state.environment
    attacker = st.session_state.attacker
    first_round = len(environment.history) + 1
    for round_num in range(first_round, first_round + num_rounds):
        environment.step(round_num, attacker, shield_enabled=shield_enabled)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 0%, rgba(34, 197, 94, 0.12), transparent 32%),
                radial-gradient(circle at 92% 8%, rgba(56, 189, 248, 0.12), transparent 30%),
                #070b16;
            color: #e5e7eb;
        }
        section[data-testid="stSidebar"] {
            background: rgba(10, 16, 30, 0.94);
            border-left: 1px solid rgba(148, 163, 184, 0.18);
        }
        .info-card {
            direction: rtl;
            text-align: right;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.72));
            border: 1px solid var(--card-accent, #38bdf8);
            border-radius: 16px;
            padding: 18px 22px;
            margin: 8px 0 20px 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
        }
        .info-card h4 {
            color: var(--card-accent, #38bdf8);
            margin: 0 0 8px 0;
            font-size: 1.05rem;
        }
        .info-card p {
            margin: 0;
            color: #cbd5e1;
            line-height: 1.85;
            font-size: 0.92rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 16px 18px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 10px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="info-card" style="--card-accent: {accent}">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 19}},
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb", "family": "Arial"},
        margin={"l": 25, "r": 25, "t": 85, "b": 45},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=False)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.13)",
        zeroline=False,
        automargin=False,
    )
    return fig


def render_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def render_metric_columns(metrics: list[tuple[str, object, str | None]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics):
        with column:
            st.metric(label, value, help=help_text)


def render_attacker_tab(df: pd.DataFrame) -> None:
    info_card(
        "عقل مهاجم يتكيّف باستمرار",
        "يختار المهاجم هدفه عبر <b>توازن ناش المختلط (Nash Equilibrium)</b> بالاستناد إلى "
        "توزيع الحماية والقيمة المالية لكل أصل. وبالتوازي، تتعلّم خوارزمية <b>EXP3</b> "
        "أي التكتيكات تحقق اختراقات ناجحة، فتزيد أوزانها وتعيد توزيع الهجمات كل جولة.",
        "#f59e0b",
    )

    if df.empty:
        render_metric_columns(
            [
                ("إجمالي الهجمات", "0", None),
                ("نسبة الاختراق الناجح", "0.0%", None),
                ("التكتيك الأكثر استخداماً", "—", None),
                ("الهدف الأكثر استهدافاً", "—", None),
            ]
        )
        st.info("شغّل محاكاة كاملة أو نفّذ جولة واحدة لعرض سلوك المهاجم وأوزان EXP3.")
        return

    breach_rate = float(df["breach_succeeded"].mean())
    most_used_tactic = str(df["tactic"].value_counts().idxmax())
    most_targeted_asset = str(df["target_asset"].value_counts().idxmax())
    render_metric_columns(
        [
            ("إجمالي الهجمات (Total Attacks)", f"{len(df):,}", None),
            ("نسبة الاختراق الناجح (Breach Rate)", f"{breach_rate:.1%}", None),
            (
                "التكتيك الأكثر استخداماً",
                TACTIC_LABELS.get(most_used_tactic, most_used_tactic),
                most_used_tactic,
            ),
            (
                "الهدف الأكثر استهدافاً",
                ASSET_LABELS.get(most_targeted_asset, most_targeted_asset),
                most_targeted_asset,
            ),
        ]
    )

    weights = np.vstack(
        df["attacker_learning"].map(lambda item: item["updated_weights"]).to_numpy()
    )
    weights_fig = go.Figure()
    for tactic_idx, tactic in enumerate(ATTACK_TACTICS):
        weights_fig.add_trace(
            go.Bar(
                x=df["round_num"],
                y=weights[:, tactic_idx],
                name=TACTIC_LABELS.get(tactic, tactic),
                marker_color=TACTIC_COLORS[tactic_idx],
                customdata=np.full(len(df), tactic),
                hovertemplate="الجولة %{x}<br>%{customdata}<br>الوزن %{y:.4f}<extra></extra>",
            )
        )
    style_chart(weights_fig, "تطوّر أوزان تكتيكات المهاجم (EXP3 Weights)", 430)
    weights_fig.update_layout(barmode="group", bargap=0.15)
    weights_fig.update_yaxes(title="وزن التكتيك")

    asset_order = list(NETWORK_ASSETS)
    attack_counts = df["target_asset"].value_counts().reindex(asset_order, fill_value=0)
    targets_fig = go.Figure(
        go.Pie(
            labels=[ASSET_LABELS.get(asset, asset) for asset in asset_order],
            values=attack_counts.to_numpy(),
            hole=0.58,
            marker={"colors": ASSET_COLORS},
            textinfo="label+percent",
            textfont={"size": 14},
            hovertemplate="%{label}<br>الهجمات %{value}<br>النسبة %{percent}<extra></extra>",
        )
    )
    style_chart(targets_fig, "توزيع الهجمات على الأصول", 430)
    targets_fig.update_layout(
        hovermode="closest",
        annotations=[
            {
                "text": f"<b>{len(df)}</b><br>هجمة",
                "showarrow": False,
                "font": {"size": 19, "color": "#e5e7eb"},
            }
        ],
        showlegend=False,
    )

    chart_columns = st.columns(2, gap="large")
    with chart_columns[0]:
        render_chart(weights_fig)
    with chart_columns[1]:
        render_chart(targets_fig)


def get_weapon_effectiveness(defender) -> tuple[dict[str, float], dict[str, int]]:
    effectiveness = {}
    selections = {}
    for weapon in DEFENSE_WEAPONS:
        alpha = sum(
            bandit.params[weapon]["alpha"]
            for bandit in defender.contextual_thompson.values()
        )
        beta = sum(
            bandit.params[weapon]["beta"]
            for bandit in defender.contextual_thompson.values()
        )
        effectiveness[weapon] = float(alpha / (alpha + beta))
        selections[weapon] = sum(
            bandit.selection_counts[weapon]
            for bandit in defender.contextual_thompson.values()
        )
    return effectiveness, selections


def render_defender_tab(df: pd.DataFrame) -> None:
    info_card(
        "دفاع متعدد الطبقات يتعلّم من كل مواجهة",
        "تبدأ الحراسة الاستباقية بلعبة <b>ستاكلبرغ</b> ويتنبأ <b>ماركوف</b> بالخطوة التالية. "
        "ترصد أجهزة <b>Bayesian وWald SPRT وCUSUM وMarkov</b> الإشارات مجتمعةً، "
        "ثم يختار <b>Contextual Thompson Sampling</b> السلاح الأنسب. وعند نجاح الصد، "
        "تُبنى <b>لقاحات ميثريداتية</b> ترفع مناعة التكتيك تدريجياً.",
        "#22c55e",
    )

    if df.empty:
        render_metric_columns(
            [
                ("نسبة صد الهجمات", "0.0%", None),
                ("دقة الإنذار المبكر", "0.0%", None),
                ("متوسط المناعة المكتسبة", "0.0%", None),
                ("وضع الدفاع الحالي (Current Defense Mode)", "—", None),
            ]
        )
        st.info("لم تُنفّذ جولات بعد؛ شغّل المحاكاة لعرض المناعة وكفاءة أسلحة الدرع.")
        return

    defense_success_rate = float(df["defense_succeeded"].mean())
    early_alert_accuracy = float(df["alert_triggered"].mean())
    average_immunity = float(df["new_immunity"].mean())
    render_metric_columns(
        [
            (
                "نسبة صد الهجمات (Defense Success Rate)",
                f"{defense_success_rate:.1%}",
                None,
            ),
            (
                "دقة الإنذار المبكر (Early Alert Accuracy)",
                f"{early_alert_accuracy:.1%}",
                "نسبة الهجمات الفعلية التي فعّلت إنذاراً مبكراً؛ جميع عينات المحاكاة هجومية.",
            ),
            (
                "متوسط المناعة الميثريداتية",
                f"{average_immunity:.1%}",
                None,
            ),
            (
                "وضع الدفاع الحالي (Current Defense Mode)",
                str(df.iloc[-1]["defense_mode"]),
                None,
            ),
        ]
    )

    immunity_fig = go.Figure()
    for tactic_idx, tactic in enumerate(ATTACK_TACTICS):
        tactic_rounds = df.loc[df["tactic"] == tactic]
        immunity_fig.add_trace(
            go.Scatter(
                x=tactic_rounds["round_num"],
                y=tactic_rounds["new_immunity"],
                mode="lines+markers",
                name=TACTIC_LABELS.get(tactic, tactic),
                line={"color": TACTIC_COLORS[tactic_idx], "width": 3},
                marker={"size": 7},
                hovertemplate="الجولة %{x}<br>المناعة %{y:.1%}<extra></extra>",
            )
        )
    style_chart(immunity_fig, "تصاعد المناعة الميثريداتية حسب التكتيك", 430)
    immunity_fig.update_yaxes(title="المناعة المكتسبة", range=[0, 0.9], tickformat=".0%")

    defender = st.session_state.environment.defender
    allocation = df.iloc[-1]["defense_coverage_allocation"]
    effectiveness, selections = get_weapon_effectiveness(defender)
    asset_order = list(NETWORK_ASSETS)
    categories = [ASSET_LABELS.get(asset, asset) for asset in asset_order] + [
        WEAPON_LABELS.get(weapon, weapon) for weapon in DEFENSE_WEAPONS
    ]
    coverage_values = [float(allocation[asset]) * 100 for asset in asset_order]
    weapon_values = [effectiveness[weapon] * 100 for weapon in DEFENSE_WEAPONS]
    weapon_selection_values = [selections[weapon] for weapon in DEFENSE_WEAPONS]
    resources_fig = go.Figure()
    resources_fig.add_trace(
        go.Bar(
            x=categories,
            y=coverage_values + [None] * len(DEFENSE_WEAPONS),
            name="توزيع حراسة ستاكلبرغ",
            marker_color="#38bdf8",
            text=[f"{value:.1f}%" for value in coverage_values] + [None] * len(DEFENSE_WEAPONS),
            textposition="outside",
            hovertemplate="%{x}<br>تغطية الحراسة %{y:.1f}%<extra></extra>",
        )
    )
    resources_fig.add_trace(
        go.Bar(
            x=categories,
            y=[None] * len(asset_order) + weapon_values,
            name="فعالية سلاح بيرسيوس",
            marker_color="#22c55e",
            customdata=[None] * len(asset_order) + weapon_selection_values,
            text=[None] * len(asset_order)
            + [f"{value:.1f}%" for value in weapon_values],
            textposition="outside",
            hovertemplate=(
                "%{x}<br>الفعالية المتوقعة %{y:.1f}%"
                "<br>مرات الاختيار %{customdata}<extra></extra>"
            ),
        )
    )
    style_chart(resources_fig, "ميزانية الحراسة وفعالية أسلحة الدرع", 430)
    resources_fig.update_layout(barmode="group", bargap=0.28)
    resources_fig.update_yaxes(title="النسبة", range=[0, 110], ticksuffix="%")

    chart_columns = st.columns(2, gap="large")
    with chart_columns[0]:
        render_chart(immunity_fig)
    with chart_columns[1]:
        render_chart(resources_fig)
    st.caption(
        "فعالية السلاح هي متوسط التوزيع اللاحق لتومبسون عبر جميع التكتيكات؛ "
        "أما توزيع الحراسة فهو تخصيص آخر جولة."
    )


def render_risk_tab(df: pd.DataFrame) -> None:
    info_card(
        "البيئة حَكَم الطرفين ومقياس الجدوى المالية",
        "تستقبل البيئة هجوم المهاجم واستجابة المدافع، ثم تحسم نتيجة الجولة وفق قوة الصد. "
        "يحسب <b>محرك المخاطر (CyberRiskEngine)</b> الخطر الكامن من قيمة الأصل واحتمال استهدافه، "
        "ثم يخفضه وفق الحماية والدرع والمناعة ليظهر الأثر المالي الحقيقي بالدولار.",
        "#a855f7",
    )

    if df.empty:
        render_metric_columns(
            [
                ("إجمالي الخطر الكامن (Inherent Risk $)", "$0", None),
                ("إجمالي الخطر المتبقي (Residual Risk $)", "$0", None),
                ("صافي الخسائر المتفاداة (Total Loss Avoided $)", "$0", None),
            ]
        )
        st.info("نفّذ جولة واحدة على الأقل لعرض المقارنة المالية وسجل المعارك.")
        return

    inherent_risk = float(df["inherent_risk"].sum())
    residual_risk = float(df["residual_risk"].sum())
    loss_avoided = float(df["loss_avoided"].sum())
    render_metric_columns(
        [
            ("إجمالي الخطر الكامن (Inherent Risk $)", f"${inherent_risk:,.0f}", None),
            ("إجمالي الخطر المتبقي (Residual Risk $)", f"${residual_risk:,.0f}", None),
            (
                "صافي الخسائر المتفاداة (Total Loss Avoided $)",
                f"${loss_avoided:,.0f}",
                None,
            ),
        ]
    )

    risk_fig = go.Figure()
    risk_fig.add_trace(
        go.Scatter(
            x=df["round_num"],
            y=df["inherent_risk"],
            mode="lines+markers",
            name="الخطر الكامن بدون حماية",
            line={"color": "#ef4444", "width": 3},
            marker={"size": 6},
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.10)",
            hovertemplate="الجولة %{x}<br>الخطر الكامن $%{y:,.2f}<extra></extra>",
        )
    )
    risk_fig.add_trace(
        go.Scatter(
            x=df["round_num"],
            y=df["residual_risk"],
            mode="lines+markers",
            name="الخطر المتبقي مع درع بيرسيوس",
            line={"color": "#22c55e", "width": 3},
            marker={"size": 6},
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.12)",
            hovertemplate="الجولة %{x}<br>الخطر المتبقي $%{y:,.2f}<extra></extra>",
        )
    )
    style_chart(risk_fig, "الخطر الكامن مقابل الخطر المتبقي عبر الجولات", 430)
    risk_fig.update_yaxes(title="القيمة بالدولار", tickprefix="$", rangemode="tozero")

    st.subheader("سجل المعارك الحي")
    battle_log = pd.DataFrame(
        {
            "رقم الجولة": df["round_num"],
            "الهدف": df["target_asset"],
            "التكتيك": df["tactic"],
            "وضع الهجوم": df["attack_mode"],
            "وضع الدفاع": df["defense_mode"],
            "حالة الإنذار": np.where(df["alert_triggered"], "إنذار مبكر", "لا يوجد إنذار"),
            "السلاح المستخدم": df["selected_weapon"].fillna("بدون سلاح"),
            "نتيجة الصد": np.where(df["defense_succeeded"], "تم الصد", "نجح الاختراق"),
            "الخسارة المتفاداة ($)": df["loss_avoided"],
        }
    )
    render_chart(risk_fig)
    st.dataframe(
        battle_log,
        width="stretch",
        hide_index=True,
        column_config={
            "رقم الجولة": st.column_config.NumberColumn("رقم الجولة", format="%d"),
            "الخسارة المتفاداة ($)": st.column_config.NumberColumn(
                "الخسارة المتفاداة ($)", format="$%.2f"
            ),
        },
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide")
    inject_styles()

    if "environment" not in st.session_state or "attacker" not in st.session_state:
        reset_simulation()

    st.title(APP_TITLE)
    st.caption(
        "مختبر تفاعلي لمحاكاة التعلم التنافسي بين المهاجم والمدافع، وقياس الأثر المالي "
        "للحماية السيبرانية جولةً بعد جولة."
    )

    with st.sidebar:
        st.header("⚙️ مركز التحكم بالمحاكاة")
        shield_enabled = st.toggle(
            "تفعيل درع بيرسيوس واللقاحات الميثريداتية (Perseus + Mithridatic Shield)",
            value=True,
        )
        num_rounds = st.slider(
            "عدد جولات المحاكاة (Simulation Rounds)",
            min_value=10,
            max_value=150,
            value=50,
            step=1,
        )
        run_clicked = st.button(
            "تشغيل محاكاة كاملة (Run Simulation)",
            type="primary",
            width="stretch",
        )
        step_clicked = st.button(
            "تنفيذ جولة واحدة حية (Step +1 Round)",
            width="stretch",
        )
        reset_clicked = st.button(
            "إعادة ضبط البيئة (Reset)",
            width="stretch",
        )
        st.divider()
        st.caption(
            "المحاكاة الكاملة تبدأ بيئة جديدة. أما Step +1 فيواصل التعلم من آخر جولة "
            "مع الاحتفاظ بجميع النتائج."
        )
        st.caption("تُطبّق حالة الدرع الحالية على الجولات المنفّذة بعد تغييرها.")

    if reset_clicked:
        reset_simulation()
    elif run_clicked:
        reset_simulation()
        execute_rounds(num_rounds, shield_enabled)
    elif step_clicked:
        execute_rounds(1, shield_enabled)

    records = st.session_state.environment.history
    df = pd.DataFrame(records) if records else pd.DataFrame()
    completed_rounds = len(records)
    status_columns = st.columns(3)
    with status_columns[0]:
        st.metric("الجولات المنفّذة", completed_rounds)
    with status_columns[1]:
        st.metric("حالة الدرع", "مفعّل" if shield_enabled else "معطّل")
    with status_columns[2]:
        latest_shield = bool(df.iloc[-1]["shield_enabled"]) if not df.empty else None
        latest_status = "—" if latest_shield is None else ("مفعّل" if latest_shield else "معطّل")
        st.metric("درع آخر جولة", latest_status)

    tabs = st.tabs(
        [
            "⚔️ 1. عقل المهاجم (Adversarial Attacker Arena)",
            "🛡️ 2. غرفة عمليات المدافع (Perseus Defender & Mithridatic Shield)",
            "🌐 3. ساحة المعركة والمخاطر المالية (Cyber Environment & Financial Risk)",
        ]
    )
    with tabs[0]:
        render_attacker_tab(df)
    with tabs[1]:
        render_defender_tab(df)
    with tabs[2]:
        render_risk_tab(df)


if __name__ == "__main__":
    main()
