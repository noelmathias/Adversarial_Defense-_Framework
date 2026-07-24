"""
demo/app.py

Agentic Defense Framework — Streamlit interactive demo.

Detection signal change (noise-entropy update)
----------------------------------------------
Section 3 (Detection) previously displayed "Grad Mag" as the fourth signal.
It now displays "Noise Entropy" — same UI position, same column, updated
label and caption.  All other sections are unchanged.  The underlying
compute_risk_score() in utils.py returns the key "grad_magnitude" as a
backward-compat alias of "noise_entropy", so Section 6 (comparison table)
still works without modification.
"""

import os
import sys
import json
import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image as PILImage
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.utils import (
    preprocess_image,
    tensor_to_display,
    predict,
    run_attack,
    confidence_delta_color,
    compute_risk_score,
    get_risk_level,
    select_defense,
    apply_selected_defense,
    compute_recovery_status,
    DEFENSE_CONFIG,
    CIFAR10_CLASSES,
    RISK_LOW_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
)
# OLD import
from models.cnn import DefenseCNN

# NEW import
from models.resnet import ResNet18 as DefenseCNN


# NEW


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Defense Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
    .stMetric label { font-size: 11px !important; color: #888 !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 18px !important; }
    div[data-testid="stHorizontalBlock"] { align-items: flex-start; }
    .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Model (cached) ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DefenseCNN(num_classes=10).to(device)
    ckpt   = "checkpoints/cnn_best.pth"
    if not os.path.exists(ckpt):
        st.error(f"Checkpoint not found: {ckpt}")
        st.stop()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛡️ Agentic Defense Framework")
st.caption(
    "Adversarial Attack Detection & Adaptive Defense · Phase 2 Demo · "
    f"Device: `{device}` · Model: `ResNet-18 (CIFAR-10)`"
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Upload + Clean Prediction
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("① Input & Clean Prediction")
col_upload, col_clean = st.columns([1, 1], gap="large")

with col_upload:
    uploaded = st.file_uploader(
        "Upload any image",
        type=["png", "jpg", "jpeg"],
        help="Resized to 32×32 to match CIFAR-10 input format.",
    )
    if uploaded is not None:
        pil_image = PILImage.open(uploaded)
        tensor    = preprocess_image(pil_image)
        st.image(tensor_to_display(tensor),
                 caption="Clean input (32×32, upscaled)", width=220)
        st.session_state["input_tensor"] = tensor
        st.session_state["pil_image"]    = pil_image

        if st.session_state.get("last_uploaded") != uploaded.name:
            for k in [
                "adv_tensor", "adv_label", "adv_confidence", "adv_probs",
                "attack_type", "epsilon_used",
                "detection", "clean_detection", "risk_score", "risk_level",
                "decision", "recovery",
            ]:
                st.session_state.pop(k, None)
            st.session_state["last_uploaded"] = uploaded.name

with col_clean:
    if "input_tensor" in st.session_state:
        t = st.session_state["input_tensor"]
        label, confidence, probs = predict(model, t, device)
        st.session_state.update({
            "clean_label":      label,
            "clean_confidence": confidence,
            "clean_probs":      probs,
        })
        st.metric("Predicted Class", label.upper())
        st.metric("Confidence",      f"{confidence*100:.1f}%")
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.barh(CIFAR10_CLASSES, probs,
                       color="steelblue", edgecolor="black", linewidth=0.4)
        bars[int(probs.argmax())].set_color("crimson")
        ax.set_xlim(0, 1); ax.set_xlabel("Probability")
        ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")
        ax.set_title("Clean Probabilities", fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Upload an image to see the clean prediction.")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Attack
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("② Adversarial Attack")

if "input_tensor" not in st.session_state:
    st.warning("Upload an image in Section ① first.")
else:
    ctrl_col, _, _ = st.columns([1, 1, 1])
    with ctrl_col:
        attack_type = st.radio(
            "Attack Method",
            options=["fgsm", "pgd"],
            format_func=lambda x: {
                "fgsm": "⚡ FGSM  (fast, single-step)",
                "pgd":  "🔁 PGD   (iterative, stronger)",
            }[x],
            horizontal=True,
            help="PGD runs 20 iterations — stronger but slightly slower.",
        )
        epsilon = st.slider(
            "Perturbation Budget (ε)",
            min_value=0.01, max_value=0.10, value=0.03, step=0.01,
        )
        run_btn = st.button(
            f"{'⚡' if attack_type=='fgsm' else '🔁'} "
            f"Run {attack_type.upper()} Attack",
            use_container_width=True,
        )

    if run_btn:
        msg = ("Generating adversarial example..."
               if attack_type == "fgsm" else "Running PGD (20 iterations)...")
        with st.spinner(msg):
            adv_tensor, adv_label, adv_conf, adv_probs = run_attack(
                model, st.session_state["input_tensor"],
                device, epsilon, attack_type=attack_type,
            )
        st.session_state.update({
            "adv_tensor":     adv_tensor,
            "adv_label":      adv_label,
            "adv_confidence": adv_conf,
            "adv_probs":      adv_probs,
            "epsilon_used":   epsilon,
            "attack_type":    attack_type,
        })
        for k in ["detection", "clean_detection",
                  "risk_score", "risk_level", "decision", "recovery"]:
            st.session_state.pop(k, None)

    if "adv_tensor" in st.session_state:
        eps_used    = st.session_state["epsilon_used"]
        atk_used    = st.session_state.get("attack_type", "fgsm").upper()
        adv_tensor  = st.session_state["adv_tensor"]
        adv_label   = st.session_state["adv_label"]
        adv_conf    = st.session_state["adv_confidence"]
        adv_probs   = st.session_state["adv_probs"]
        clean_conf  = st.session_state["clean_confidence"]
        clean_label = st.session_state["clean_label"]
        clean_probs = st.session_state["clean_probs"]
        delta       = adv_conf - clean_conf

        col_img, col_metrics, col_chart = st.columns([1, 1, 1], gap="large")

        with col_img:
            st.markdown("**Visual Comparison**")
            c1, c2 = st.columns(2)
            with c1:
                st.image(tensor_to_display(st.session_state["input_tensor"]),
                         caption="Clean", width=130)
            with c2:
                st.image(tensor_to_display(adv_tensor),
                         caption=f"{atk_used} (ε={eps_used})", width=130)
            diff = adv_tensor - st.session_state["input_tensor"].cpu()
            diff_amp  = torch.clamp(diff * 10 + 0.5, 0, 1)
            diff_disp = (diff_amp.squeeze(0).permute(1,2,0).numpy()*255).astype(np.uint8)
            diff_up   = np.array(
                PILImage.fromarray(diff_disp).resize((130,130), PILImage.NEAREST))
            st.image(diff_up, caption="Perturbation ×10", width=130)

        with col_metrics:
            st.markdown("**Prediction Change**")
            success = adv_label != clean_label
            st.markdown(
                f"**Attack:** `{atk_used}` at `ε={eps_used}`\n\n"
                f"**Status:** :{'error' if success else 'warning'}["
                f"{'✅ Flip succeeded' if success else '⚠️ No flip'}]"
            )
            st.metric("Clean Prediction",       clean_label.upper())
            st.metric("Adversarial Prediction", adv_label.upper(),
                      delta=f"{delta*100:.1f}% confidence",
                      delta_color="inverse")
            icon = confidence_delta_color(delta)
            st.markdown(
                f"**Confidence:** {icon} "
                f"`{clean_conf*100:.1f}% → {adv_conf*100:.1f}%`"
            )
            if abs(delta) > 0.4:    st.error("Threat Level: HIGH")
            elif abs(delta) > 0.15: st.warning("Threat Level: MEDIUM")
            else:                   st.success("Threat Level: LOW")

        with col_chart:
            st.markdown("**Probability Shift**")
            x, w = np.arange(len(CIFAR10_CLASSES)), 0.38
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(x-w/2, clean_probs, w, label="Clean",
                   color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.4)
            ax.bar(x+w/2, adv_probs,   w, label=atk_used,
                   color="crimson",   alpha=0.85, edgecolor="black", linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Probability"); ax.set_ylim(0, 1)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
            ax.set_title("Clean vs. Adversarial", fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Detection  (UPDATED: shows noise_entropy instead of grad_mag)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("③ Adversarial Detection")

if "adv_tensor" not in st.session_state:
    st.warning("Run an attack in Section ② first.")
else:
    if st.button("🔎 Analyze Input", use_container_width=False):
        with st.spinner(
            "Computing detection signals "
            "(noise entropy: 50 forward passes × 2 inputs)…"
        ):
            clean_sig = compute_risk_score(
                model, st.session_state["input_tensor"], device,
            )
            adv_sig = compute_risk_score(
                model, st.session_state["adv_tensor"], device,
                clean_label=st.session_state["clean_label"],
                clean_confidence=st.session_state["clean_confidence"],
            )
        st.session_state["detection"]       = adv_sig
        st.session_state["clean_detection"] = clean_sig
        st.session_state["risk_score"]      = adv_sig["risk_score"]
        st.session_state["risk_level"]      = get_risk_level(adv_sig["risk_score"])[0]
        for k in ["decision", "recovery"]:
            st.session_state.pop(k, None)

    if "detection" in st.session_state:
        adv_sig   = st.session_state["detection"]
        clean_sig = st.session_state["clean_detection"]
        risk      = adv_sig["risk_score"]
        level, _, icon = get_risk_level(risk)

        # Risk banner
        st.markdown("---")
        banner_col, _ = st.columns([2, 1])
        with banner_col:
            clr = {"HIGH":"#e05252","MEDIUM":"#e0b852","LOW":"#52e07a"}[level]
            bg  = {"HIGH":"#2d0f0f","MEDIUM":"#2d2200","LOW":"#0f2d12"}[level]
            st.markdown(
                f"""
                <div style="padding:16px 24px;border-radius:10px;
                    background:{bg};border-left:6px solid {clr};
                    margin-bottom:12px;">
                    <div style="font-size:13px;color:#aaa;">Risk Assessment</div>
                    <div style="font-size:32px;font-weight:700;color:{clr};">
                        {icon} {level} RISK &nbsp;
                        <span style="font-size:20px;">(score: {risk:.3f})</span>
                    </div>
                    <div style="font-size:12px;color:#888;margin-top:4px;">
                        Low &lt; {RISK_LOW_THRESHOLD} ·
                        Medium {RISK_LOW_THRESHOLD}–{RISK_MEDIUM_THRESHOLD} ·
                        High ≥ {RISK_MEDIUM_THRESHOLD}
                    </div>
                </div>
                """, unsafe_allow_html=True,
            )

        # ── Five signal columns ───────────────────────────────────────────
        st.markdown("**Detection Signal Breakdown**")
        s1, s2, s3, s4, s5 = st.columns(5)

        with s1:
            st.markdown("##### Confidence")
            st.metric(
                "Adversarial",
                f"{adv_sig['confidence']*100:.1f}%",
                delta=f"{(adv_sig['confidence']-clean_sig['confidence'])*100:.1f}% vs clean",
                delta_color="inverse",
            )
            st.caption("Low = model uncertain.")

        with s2:
            st.markdown("##### Entropy")
            st.metric(
                "Norm.",
                f"{adv_sig['entropy_norm']:.3f}",
                delta=f"{adv_sig['entropy_norm']-clean_sig['entropy_norm']:+.3f}",
                delta_color="inverse",
            )
            st.caption("High = spread across classes.")

        with s3:
            st.markdown("##### Conf Drop")
            st.metric("vs Clean", f"{adv_sig['conf_drop']:.3f}")
            st.caption("Relative confidence loss.")

        with s4:
            # ── UPDATED SIGNAL LABEL ─────────────────────────────────────
            st.markdown("##### Noise Entropy")
            ne_adv   = adv_sig.get("noise_entropy", adv_sig.get("grad_magnitude", 0.0))
            ne_clean = clean_sig.get("noise_entropy", clean_sig.get("grad_magnitude", 0.0))
            st.metric(
                "MC (50 passes)",
                f"{ne_adv:.3f}",
                delta=f"{ne_adv - ne_clean:+.3f} vs clean",
                delta_color="inverse",
            )
            st.caption(
                "High = predictions flip under noise → adversarial. "
                "Works even when confidence is 99% (confident-wrong PGD)."
            )

        with s5:
            st.markdown("##### Risk Score")
            mismatch_note = " (+flip)" if adv_sig.get("pred_mismatch") else ""
            st.metric(
                "Final",
                f"{risk:.3f}",
                delta=f"{risk - clean_sig['risk_score']:+.3f} vs clean",
                delta_color="inverse",
            )
            st.caption(f"5-signal formula{mismatch_note}.")

        # ── Grouped bar chart ─────────────────────────────────────────────
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            # Use noise_entropy with backward-compat fallback
            ne_adv_v   = adv_sig.get("noise_entropy", adv_sig.get("grad_magnitude", 0.0))
            ne_clean_v = clean_sig.get("noise_entropy", clean_sig.get("grad_magnitude", 0.0))

            sig_labels = ["1−Conf", "Entropy", "Conf\nDrop", "Noise\nEntropy", "Risk"]
            clean_vals = [
                1 - clean_sig["confidence"],
                clean_sig["entropy_norm"],
                clean_sig.get("conf_drop", 0.0),
                ne_clean_v,
                clean_sig["risk_score"],
            ]
            adv_vals = [
                1 - adv_sig["confidence"],
                adv_sig["entropy_norm"],
                adv_sig.get("conf_drop", 0.0),
                ne_adv_v,
                adv_sig["risk_score"],
            ]
            x, w = np.arange(5), 0.35
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(x-w/2, clean_vals, w, label="Clean",
                   color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.bar(x+w/2, adv_vals,   w, label="Adversarial",
                   color="crimson",   alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x); ax.set_xticklabels(sig_labels, fontsize=8)
            ax.set_ylim(0, 1); ax.set_ylabel("Score")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(RISK_LOW_THRESHOLD,    linestyle="--",
                       color="orange", linewidth=1, alpha=0.7)
            ax.axhline(RISK_MEDIUM_THRESHOLD, linestyle="--",
                       color="red",    linewidth=1, alpha=0.7)
            ax.set_title("Detection Signals: Clean vs. Adversarial", fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with chart_col2:
            # Risk gauge
            fig, ax = plt.subplots(figsize=(5, 2.0))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
            ax.barh(0.5, 1.0, height=0.35, color="#333333", left=0.0, align="center")
            ax.barh(0.5, RISK_LOW_THRESHOLD, height=0.35, color="#52e07a",
                    left=0.0, alpha=0.5, align="center")
            ax.barh(0.5, RISK_MEDIUM_THRESHOLD-RISK_LOW_THRESHOLD, height=0.35,
                    color="#e0b852", left=RISK_LOW_THRESHOLD, alpha=0.5, align="center")
            ax.barh(0.5, 1.0-RISK_MEDIUM_THRESHOLD, height=0.35, color="#e05252",
                    left=RISK_MEDIUM_THRESHOLD, alpha=0.5, align="center")
            ax.plot([risk, risk], [0.25, 0.75], color="white", linewidth=3, zorder=5)
            ax.plot(risk, 0.75, "v", color="white", markersize=8, zorder=6)
            ax.text(RISK_LOW_THRESHOLD/2, 0.08, "LOW",
                    ha="center", fontsize=8, color="#52e07a")
            ax.text((RISK_LOW_THRESHOLD+RISK_MEDIUM_THRESHOLD)/2, 0.08, "MEDIUM",
                    ha="center", fontsize=8, color="#e0b852")
            ax.text((RISK_MEDIUM_THRESHOLD+1.0)/2, 0.08, "HIGH",
                    ha="center", fontsize=8, color="#e05252")
            ax.text(risk, 0.92, f"{risk:.3f}",
                    ha="center", fontsize=10, fontweight="bold", color="white")
            ax.set_title("Risk Score Gauge", fontsize=10, color="white", pad=4)
            fig.patch.set_facecolor("#1a1a1a")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.session_state["risk_score"] = risk
        st.session_state["risk_level"] = level

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Decision
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("④ Defense Decision")

if "risk_score" not in st.session_state:
    st.warning("Run detection in Section ③ first.")
else:
    risk          = st.session_state["risk_score"]
    det           = st.session_state.get("detection", {})
    pred_mismatch = det.get("pred_mismatch", 0)
    unstable      = det.get("unstable", 0)
    entropy_norm  = det.get("entropy_norm", 0.0)

    tier, config, reason = select_defense(
        risk,
        pred_mismatch=pred_mismatch,
        unstable=unstable,
        entropy_norm=entropy_norm,
    )
    st.session_state["decision"] = {"tier": tier, "config": config, "reason": reason}

    tier_style = {
        "light":  ("#0f2d12", "#52e07a"),
        "medium": ("#2d2200", "#e0b852"),
        "strong": ("#2d0f0f", "#e05252"),
    }
    bg, fg = tier_style[tier]

    st.markdown(
        f"""
        <div style="padding:18px 28px;border-radius:10px;
            background:{bg};border-left:6px solid {fg};margin-bottom:16px;">
            <div style="font-size:12px;color:#aaa;margin-bottom:4px;">
                Decision Engine → Rule-Based Selector
            </div>
            <div style="font-size:28px;font-weight:700;color:{fg};">
                {config['icon']} {config['label']} Selected
            </div>
            <div style="font-size:13px;color:#ccc;margin-top:8px;">{reason}</div>
        </div>
        """, unsafe_allow_html=True,
    )

    st.markdown("**All Defense Options**")
    card_cols = st.columns(3)
    for col, (t, cfg) in zip(card_cols, DEFENSE_CONFIG.items()):
        is_sel  = t == tier
        b_color = fg if is_sel else "#444"
        b_bg    = bg if is_sel else "#1a1a1a"
        with col:
            st.markdown(
                f"""
                <div style="padding:14px;border-radius:8px;
                    background:{b_bg};border:2px solid {b_color};min-height:180px;">
                    <div style="font-size:16px;font-weight:600;
                        color:{'white' if is_sel else '#aaa'};">
                        {cfg['icon']} {cfg['label']}
                    </div>
                    <div style="font-size:11px;color:#888;margin:6px 0 8px 0;">
                        {cfg['risk_range']}
                    </div>
                    <div style="font-size:12px;color:#ccc;margin-bottom:8px;">
                        {cfg['description']}
                    </div>
                    <div style="font-size:11px;color:#777;">
                        💻 Cost: {cfg['cost']}
                    </div>
                </div>
                """, unsafe_allow_html=True,
            )
            if is_sel:
                st.markdown("**← SELECTED**")

    st.markdown("**Parameters passed to defense:**")
    pcols = st.columns(len(config["params"]) + 1)
    with pcols[0]:
        st.metric("Defense Tier", tier.upper())
    for i, (k, v) in enumerate(config["params"].items()):
        with pcols[i+1]:
            st.metric(k, str(v))

    with st.expander("🔬 Decision Trace (Research View)"):
        st.markdown(
            f"""
            | Field             | Value                                       |
            |-------------------|---------------------------------------------|
            | Risk Score        | `{risk:.4f}`                                |
            | Low threshold     | `{RISK_LOW_THRESHOLD}`                      |
            | Medium threshold  | `{RISK_MEDIUM_THRESHOLD}`                   |
            | Pred mismatch     | `{bool(pred_mismatch)}`                     |
            | Unstable flag     | `{bool(unstable)}`                          |
            | Entropy norm      | `{entropy_norm:.4f}`                        |
            | Selected tier     | `{tier}`                                    |
            | Defense params    | `{config['params']}`                        |
            | Compute cost      | {config['cost']}                            |
            | Rule engine       | Threshold-based (Phase 1)                   |
            | Phase 2 upgrade   | Contextual Bandit (Thompson Sampling)       |
            """
        )
        st.info(
            "In Phase 2 Stage 5, the rule-based selector is replaced by a "
            "contextual bandit that learns optimal defense selection "
            "from reward signals over time."
        )

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Defense Application & Recovery
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("⑤ Defense Application & Recovery")

if "decision" not in st.session_state:
    st.warning("Complete detection and decision in Sections ③–④ first.")
else:
    tier   = st.session_state["decision"]["tier"]
    config = st.session_state["decision"]["config"]

    if st.button(f"🛡️ Apply {config['label']}", use_container_width=False):
        with st.spinner(f"Applying {config['label']}…"):
            defended_img, def_label, def_conf, def_probs = apply_selected_defense(
                model,
                st.session_state["adv_tensor"],
                device, tier, config,
            )
        rec_status = compute_recovery_status(
            clean_label=st.session_state["clean_label"],
            clean_conf=st.session_state["clean_confidence"],
            adv_conf=st.session_state["adv_confidence"],
            def_label=def_label,
            def_conf=def_conf,
            pred_mismatch=st.session_state.get("detection", {}).get("pred_mismatch", 0),
        )
        st.session_state["recovery"] = {
            "defended_img":   defended_img,
            "defended_label": def_label,
            "defended_conf":  def_conf,
            "defended_probs": def_probs,
            "rec_status":     rec_status,
        }

    if "recovery" in st.session_state:
        rec         = st.session_state["recovery"]
        rec_status  = rec["rec_status"]
        clean_label = st.session_state["clean_label"]
        clean_conf  = st.session_state["clean_confidence"]
        clean_probs = st.session_state["clean_probs"]
        adv_label   = st.session_state["adv_label"]
        adv_conf    = st.session_state["adv_confidence"]
        adv_probs   = st.session_state["adv_probs"]
        def_label   = rec["defended_label"]
        def_conf    = rec["defended_conf"]
        def_probs   = rec["defended_probs"]
        recovered   = rec_status["recovered"]

        if recovered:
            st.success(
                f"✅ **Recovery Successful** — Prediction restored to "
                f"**{def_label.upper()}** (matches clean label). "
                f"Confidence recovery: {rec_status['conf_recovery_pct']:.1f}% of gap closed."
            )
        elif rec_status["failure_explanation"]:
            st.error(
                f"❌ **Recovery Failed** — Prediction remains "
                f"**{def_label.upper()}**, clean was **{clean_label.upper()}**. "
                f"{rec_status['failure_explanation']}"
            )
        else:
            st.warning(
                f"⚠️ **Partial Recovery** — Prediction is "
                f"**{def_label.upper()}**, clean was **{clean_label.upper()}**. "
                "Defense reduced adversarial effect but label not restored."
            )

        st.markdown("**Input Progression: Clean → Attack → Defense**")
        img_c1, arr1, img_c2, arr2, img_c3 = st.columns([1, 0.1, 1, 0.1, 1])
        atk_used = st.session_state.get("attack_type","fgsm").upper()
        eps_used = st.session_state.get("epsilon_used","?")
        with img_c1:
            st.image(tensor_to_display(st.session_state["input_tensor"]),
                     caption=f"① Clean\n{clean_label.upper()} ({clean_conf*100:.0f}%)", width=160)
        with arr1:
            st.markdown("<div style='font-size:26px;margin-top:65px;text-align:center;color:#666;'>→</div>",
                        unsafe_allow_html=True)
        with img_c2:
            st.image(tensor_to_display(st.session_state["adv_tensor"]),
                     caption=f"② {atk_used} (ε={eps_used})\n{adv_label.upper()} ({adv_conf*100:.0f}%)",
                     width=160)
        with arr2:
            st.markdown("<div style='font-size:26px;margin-top:65px;text-align:center;color:#666;'>→</div>",
                        unsafe_allow_html=True)
        with img_c3:
            st.image(tensor_to_display(rec["defended_img"]),
                     caption=f"③ Defended ({tier.capitalize()})\n{def_label.upper()} ({def_conf*100:.0f}%)",
                     width=160)

        st.divider()
        st.markdown("**Confidence Recovery**")
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Clean",        f"{clean_conf*100:.1f}%")
        with m2: st.metric("After Attack",  f"{adv_conf*100:.1f}%",
                            delta=f"{(adv_conf-clean_conf)*100:.1f}%", delta_color="inverse")
        with m3: st.metric("After Defense", f"{def_conf*100:.1f}%",
                            delta=f"{(def_conf-adv_conf)*100:.1f}% recovered", delta_color="normal")
        with m4:
            net = def_conf - clean_conf
            st.metric("Net vs Clean", f"{def_conf*100:.1f}%",
                      delta=f"{net*100:.1f}%",
                      delta_color="normal" if net >= -0.1 else "inverse")

        st.markdown("**Probability Distribution: Clean → Attack → Defense**")
        x, w = np.arange(len(CIFAR10_CLASSES)), 0.26
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(x-w, clean_probs, w, label="① Clean",
               color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.bar(x,   adv_probs,   w, label=f"② {atk_used}",
               color="crimson", alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.bar(x+w, def_probs,   w, label=f"③ Defended ({tier})",
               color="mediumseagreen", alpha=0.85, edgecolor="black", linewidth=0.3)
        true_idx = CIFAR10_CLASSES.index(clean_label)
        ax.axvline(true_idx, color="gold", linewidth=2, linestyle="--", alpha=0.6, label="True class")
        ax.set_xticks(x)
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Probability", fontsize=10); ax.set_ylim(0, 1)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
        ax.set_title("Prediction Distribution: Clean vs. Adversarial vs. Defended", fontsize=11)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        with st.expander("🔬 Recovery Trace (Research View)"):
            rd = (f"{rec_status['conf_recovery_pct']:.1f}% of gap closed"
                  if recovered else "No recovery (label not restored)")
            st.markdown(
                f"**Defense applied:** `{tier}` — `{config['params']}`\n\n"
                f"**Label recovered:** `{'Yes ✓' if recovered else 'No ✗'}`\n\n"
                f"**Confidence recovery:** `{rd}`"
            )

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Full Pipeline Comparison
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("⑥ Full Pipeline Comparison")

if "recovery" not in st.session_state:
    st.warning("Complete the full pipeline (Sections ①–⑤) to see comparison.")
else:
    clean_label = st.session_state["clean_label"]
    clean_conf  = st.session_state["clean_confidence"]
    clean_probs = st.session_state["clean_probs"]
    adv_label   = st.session_state["adv_label"]
    adv_conf    = st.session_state["adv_confidence"]
    adv_probs   = st.session_state["adv_probs"]
    eps_used    = st.session_state["epsilon_used"]
    atk_used    = st.session_state.get("attack_type","fgsm").upper()
    rec         = st.session_state["recovery"]
    rec_status  = rec["rec_status"]
    def_label   = rec["defended_label"]
    def_conf    = rec["defended_conf"]
    def_probs   = rec["defended_probs"]
    det         = st.session_state["detection"]
    risk        = st.session_state["risk_score"]
    tier        = st.session_state["decision"]["tier"]
    config      = st.session_state["decision"]["config"]
    recovered   = rec_status["recovered"]
    attack_ok   = adv_label != clean_label

    level, _, risk_icon = get_risk_level(risk)
    tier_icons = {"light":"🟢","medium":"🟡","strong":"🔴"}
    result_icon = "✅" if recovered else "⚠️"
    clr_map  = {"HIGH":"#e05252","MEDIUM":"#e0b852","LOW":"#52e07a"}
    tier_clr = {"light":"#52e07a","medium":"#e0b852","strong":"#e05252"}

    # Five-stage pipeline bar
    st.markdown(
        f"""
        <div style="display:flex;gap:0;margin-bottom:20px;border-radius:10px;
            overflow:hidden;border:1px solid #333;">
            <div style="flex:1;padding:14px 18px;background:#1a2a3a;text-align:center;">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">① CLEAN</div>
                <div style="font-size:15px;font-weight:600;color:#5aabff;">{clean_label.upper()}</div>
                <div style="font-size:12px;color:#aaa;">{clean_conf*100:.1f}%</div>
            </div>
            <div style="width:1px;background:#333;"></div>
            <div style="flex:1;padding:14px 18px;background:#3a1a1a;text-align:center;">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">② {atk_used} (ε={eps_used})</div>
                <div style="font-size:15px;font-weight:600;color:#ff6b6b;">{adv_label.upper()}</div>
                <div style="font-size:12px;color:#aaa;">{adv_conf*100:.1f}%</div>
            </div>
            <div style="width:1px;background:#333;"></div>
            <div style="flex:1;padding:14px 18px;background:#2a2a1a;text-align:center;">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">③ DETECTION</div>
                <div style="font-size:15px;font-weight:600;color:{clr_map[level]};">
                    {risk_icon} {level}</div>
                <div style="font-size:12px;color:#aaa;">risk={risk:.3f}</div>
            </div>
            <div style="width:1px;background:#333;"></div>
            <div style="flex:1;padding:14px 18px;background:#1a2a1a;text-align:center;">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">④ DECISION</div>
                <div style="font-size:15px;font-weight:600;color:{tier_clr[tier]};">
                    {tier_icons[tier]} {tier.upper()}</div>
                <div style="font-size:12px;color:#aaa;">{config['params']}</div>
            </div>
            <div style="width:1px;background:#333;"></div>
            <div style="flex:1;padding:14px 18px;
                background:{'#0f2d12' if recovered else '#2d1a0a'};text-align:center;">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">⑤ DEFENSE</div>
                <div style="font-size:15px;font-weight:600;
                    color:{'#52e07a' if recovered else '#e09052'};">
                    {result_icon} {def_label.upper()}</div>
                <div style="font-size:12px;color:#aaa;">{def_conf*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True,
    )

    # Three-column image + probability cards
    card_data = [
        (st.session_state["input_tensor"], "① Clean Input",
         clean_label, clean_conf, clean_probs,
         "steelblue", "#1a2a3a", "#5aabff", None),
        (st.session_state["adv_tensor"],
         f"② {atk_used} (ε={eps_used})",
         adv_label, adv_conf, adv_probs,
         "crimson", "#3a1a1a", "#ff6b6b",
         f"{'✅ Flip' if attack_ok else '⚠️ No flip'}"),
        (rec["defended_img"], f"③ Defended ({tier.capitalize()})",
         def_label, def_conf, def_probs,
         "mediumseagreen",
         "#0f2d12" if recovered else "#2d1a0a",
         "#52e07a" if recovered else "#e09052",
         f"{'✅ Recovered' if recovered else '⚠️ Partial'}"),
    ]
    col_a, col_b, col_c = st.columns(3, gap="large")
    for col, (tensor, title, lbl, conf, probs, bar_c, bg, fg, badge) in zip(
        [col_a, col_b, col_c], card_data
    ):
        with col:
            st.markdown(
                f"""
                <div style="background:{bg};border-radius:10px;padding:14px;
                    margin-bottom:10px;border:1px solid #333;">
                    <div style="font-size:12px;color:#888;margin-bottom:6px;">{title}</div>
                    <div style="font-size:22px;font-weight:700;color:{fg};">{lbl.upper()}</div>
                    <div style="font-size:14px;color:#ccc;">Confidence: {conf*100:.1f}%</div>
                    {'<div style="font-size:11px;color:#aaa;margin-top:4px;">' + badge + '</div>'
                     if badge else ''}
                </div>
                """, unsafe_allow_html=True,
            )
            st.image(tensor_to_display(tensor), width=200)
            fig, ax = plt.subplots(figsize=(3.8, 2.8))
            ax.barh(CIFAR10_CLASSES, probs, color=bar_c, alpha=0.85,
                    edgecolor="black", linewidth=0.3)
            ax.barh([CIFAR10_CLASSES[int(probs.argmax())]], [probs.max()],
                    color=fg, alpha=1.0, edgecolor="black", linewidth=0.3)
            true_idx2 = CIFAR10_CLASSES.index(clean_label)
            ax.barh([CIFAR10_CLASSES[true_idx2]], [probs[true_idx2]],
                    color="gold", alpha=0.6, edgecolor="black", linewidth=0.3)
            ax.set_xlim(0, 1); ax.set_xlabel("Probability", fontsize=8)
            ax.invert_yaxis(); ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, axis="x"); ax.set_title(title, fontsize=8, pad=3)
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
            ax.title.set_color("white"); ax.spines[:].set_color("#444")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()

    # Summary tables
    st.markdown("**Pipeline Summary**")
    tbl_c1, tbl_c2 = st.columns([1, 1])
    rd = (f"{rec_status['conf_recovery_pct']:.1f}% of gap closed"
          if recovered else "No recovery (label not restored)")

    # Noise entropy with legacy fallback
    ne_det   = det.get("noise_entropy", det.get("grad_magnitude", 0.0))
    ne_clean = st.session_state.get("clean_detection", {}).get(
        "noise_entropy", st.session_state.get("clean_detection", {}).get("grad_magnitude", 0.0)
    )

    with tbl_c1:
        st.markdown(
            f"""
            | Stage         | Label                 | Confidence   |
            |---------------|-----------------------|--------------|
            | ① Clean       | {clean_label.upper()} | {clean_conf*100:.1f}% |
            | ② Adversarial | {adv_label.upper()}   | {adv_conf*100:.1f}%  |
            | ③ Defended    | {def_label.upper()}   | {def_conf*100:.1f}%  |

            **Attack:** `{atk_used}` at `ε={eps_used}`
            **Attack succeeded:** `{'Yes' if attack_ok else 'No'}`
            **Label recovered:** `{'Yes ✓' if recovered else 'No ✗'}`
            **Confidence recovery:** `{rd}`
            **Defense tier:** `{tier}` — `{config['params']}`
            **Risk score:** `{risk:.4f}` → `{level}`
            """
        )
    with tbl_c2:
        st.markdown(
            f"""
            | Detection Signal  | Clean      | Adversarial  |
            |-------------------|------------|--------------|
            | Confidence        | {clean_conf*100:.1f}%   | {det['confidence']*100:.1f}% |
            | Entropy (norm)    | —          | {det['entropy_norm']:.3f}    |
            | Conf drop         | —          | {det['conf_drop']:.3f}       |
            | Noise Entropy     | {ne_clean:.3f}  | {ne_det:.3f}       |
            | Pred mismatch     | —          | {'Yes' if det['pred_mismatch'] else 'No'} |
            | Unstable flag     | —          | {'Yes' if det['unstable'] else 'No'}      |
            | **Risk Score**    | —          | **{risk:.4f}**               |
            """
        )

    # JSON Export
    with st.expander("📋 Export Pipeline Result"):
        export = {
            "attack":      {"type": st.session_state.get("attack_type","fgsm"), "epsilon": eps_used},
            "clean":       {"label": clean_label, "confidence": round(clean_conf, 4)},
            "adversarial": {"label": adv_label, "confidence": round(adv_conf, 4),
                            "attack_succeeded": attack_ok},
            "detection":   {
                "risk_score":    round(risk, 4),
                "risk_level":    level,
                "entropy_norm":  round(det["entropy_norm"], 4),
                "conf_drop":     round(det["conf_drop"], 4),
                "noise_entropy": round(ne_det, 4),
                "pred_mismatch": det["pred_mismatch"],
                "unstable":      det["unstable"],
            },
            "decision":    {"tier": tier, "params": config["params"]},
            "defended":    {
                "label":               def_label,
                "confidence":          round(def_conf, 4),
                "recovered":           recovered,
                "conf_recovery_pct":   rec_status["conf_recovery_pct"],
                "recovery_note": (
                    rec_status["failure_explanation"]
                    or ("Successful" if recovered else "Partial — label not restored")
                ),
            },
        }
        st.code(json.dumps(export, indent=2), language="json")
        st.download_button(
            label="⬇️ Download as JSON",
            data=json.dumps(export, indent=2),
            file_name=f"pipeline_{atk_used.lower()}_eps{eps_used}.json",
            mime="application/json",
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Model: ResNet-18 · Device: `{device}` · "
    "Attacks: FGSM / PGD · "
    "Detection: 5-signal (conf, entropy, conf_drop, **noise_entropy**, mismatch) · "
    "Defense: Light / Medium / Strong (randomized smoothing) · "
    "Phase 2 in progress"
)