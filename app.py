# -*- coding: utf-8 -*-
"""
PushoverBilin v3.0 — Streamlit Web Edition
Seismic Pushover, SDOF Idealization & Target Displacement Analysis (EN 1998-1 / RPA)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Page Configuration ──
st.set_page_config(
    page_title="PushoverBilin Web",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Styling ──
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e1e24;
        border: 1px solid #3a3a45;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Helper Functions ──
def get_spectral_acceleration(T, A, I, S, eta, t1, t2, t3):
    """Calculates Se(T) in units of 'g' (Handwritten Page 3)"""
    if 0 <= T < t1:
        return A * I * S * (1.0 + (T / t1) * (2.5 * eta - 1.0))
    elif t1 <= T < t2:
        return A * I * S * (2.5 * eta)
    elif t2 <= T < t3:
        return A * I * S * (t2 / T)
    else: # T >= t3 up to 4.0s
        return A * I * S * (2.5 * eta) * (t2 * t3 / (T ** 2))

# ── Sidebar Inputs ──
st.sidebar.title("⬡ PushoverBilin Web")
st.sidebar.caption("Seismic SDOF & Target Displacement Analysis")

# 1. Curve Source Selection
st.sidebar.header("📁 1. MDOF Capacity Curve")
uploaded_file = st.sidebar.file_uploader("Upload Pushover Excel File (.xlsx)", type=["xlsx", "xls"])

# Generate synthetic/default capacity curve data if no file is uploaded
if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        num_cols = df_raw.select_dtypes(include=[np.number]).columns
        if len(num_cols) < 2:
            st.sidebar.error("Excel must contain at least 2 numerical columns (Disp, Force).")
            st.stop()
        raw_disp = df_raw[num_cols[0]].values
        raw_force = df_raw[num_cols[1]].values
        st.sidebar.success("Excel parsed successfully.")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.stop()
else:
    # Default synthetic curve data (yield near 35mm, max force near 350kN)
    raw_disp = np.linspace(0, 120, 100)
    raw_force = 350.0 * (1 - np.exp(-raw_disp / 20.0))
    st.sidebar.info("Using default capacity curve data (No file uploaded).")

# 2. Dynamic Building Dynamics (Editable Table)
st.sidebar.header("🏢 2. Story Properties")
st.sidebar.write("Add, delete, or edit story properties below:")

# Setup default DataFrame for the editor
default_stories = pd.DataFrame({
    "Story": [1, 2, 3],
    "Mass mi (t)": [150.0, 150.0, 120.0],
    "Mode Shape φi": [0.33, 0.66, 1.00]
})

edited_df = st.sidebar.data_editor(
    default_stories,
    num_rows="dynamic",
    column_config={
        "Story": st.column_config.NumberColumn("Story ID", disabled=True),
        "Mass mi (t)": st.column_config.NumberColumn("Mass mi (t)", min_value=0.1, step=1.0),
        "Mode Shape φi": st.column_config.NumberColumn("φi", min_value=0.01, max_value=10.0, step=0.05)
    },
    hide_index=True
)

# Extract inputs from edited table
m = edited_df["Mass mi (t)"].values
phi = edited_df["Mode Shape φi"].values

if len(m) == 0 or len(phi) == 0:
    st.error("Please specify at least one story in the table.")
    st.stop()

# 3. Seismic Parameter Inputs
st.sidebar.header("📈 3. Seismic Demand Spectra")
A = st.sidebar.number_input("Acceleration Coeff (A)", value=0.25, step=0.05)
damping = st.sidebar.number_input("Damping ratio (ξ %)", value=5.0, step=0.5)
S = st.sidebar.number_input("Soil Factor (S)", value=1.2, step=0.1)
t1 = st.sidebar.number_input("Period T1 (s)", value=0.15, step=0.05)
t2 = st.sidebar.number_input("Period T2 (Tc) (s)", value=0.50, step=0.05)
t3 = st.sidebar.number_input("Period T3 (s)", value=2.00, step=0.1)

if t1 >= t2 or t2 >= t3:
    st.sidebar.error("Error: Periods must satisfy T1 < T2 < T3.")
    st.stop()

# ── Main Calculation Engine ──

# SDOF Participation Factor calculations (Handwritten Page 1)
m_star = np.sum(m * phi)
M_1 = np.sum(m * phi**2)
gamma = m_star / M_1

# Transform raw curve to SDOF equivalent
sdof_disp = raw_disp / gamma
sdof_force = raw_force / gamma

# Bilinearization calculations (Handwritten Page 2)
idx_max = np.argmax(sdof_force)
F_max_star = sdof_force[idx_max]
d_max_star = sdof_disp[idx_max]

F_y_star = F_max_star  # Equal to SDOF peak capacity (Vy = Vmax)

# Compute area under curve using trapezoidal rule
d_s = sdof_disp[:idx_max+1]
f_s = sdof_force[:idx_max+1]
area_raw = np.trapz(f_s, d_s)

# Solve for yield displacement (Sdy*) via equal energy balance
d_y_star = 2.0 * (d_max_star - (area_raw / F_y_star))

# Physics verification and fallback limits
if d_y_star <= 0 or d_y_star >= d_max_star:
    d_y_star = 0.6 * d_max_star

# Effective SDOF properties (converted displacement to meters in the period root)
T_star = 2.0 * np.pi * np.sqrt(((d_y_star / 1000.0) * m_star) / F_y_star)
S_ay = (F_y_star / m_star) / 9.81

# ── Main Panel Layout ──
st.title("Seismic Pushover SDOF & Target Displacement Analysis")
st.write("Calculations follow Eurocode 8 Annex B and RPA formulations.")

# Summary Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Modal Factor (Γ)", f"{gamma:.4f}")
with col2:
    st.metric("Modal Mass (m*)", f"{m_star:.2f} t")
with col3:
    st.metric("Yield Capacity (Fy*)", f"{F_y_star:.1f} kN")
with col4:
    st.metric("Yield Disp (Sdy*)", f"{d_y_star:.2f} mm")
with col5:
    st.metric("Effective Period (T*)", f"{T_star:.3f} s")

# Tab views
tab_curves, tab_spectra, tab_results = st.tabs([
    "📊 Capacity Curves (MDOF & SDOF)",
    "📈 Response Spectra",
    "📋 Target Displacement Summary"
])

# ── Tab 1: Curves Drawing ──
with tab_curves:
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_m, ax_m = plt.subplots(figsize=(6, 4))
        ax_m.plot(raw_disp, raw_force, color="#58a6ff", linewidth=2.5, label="MDOF Capacity Curve")
        ax_m.set_title("Multi-Degree-of-Freedom Base Curve")
        ax_m.set_xlabel("Control Displacement d (mm)")
        ax_m.set_ylabel("Base Shear V (kN)")
        ax_m.grid(True, linestyle=":")
        ax_m.legend()
        st.pyplot(fig_m)
        
    with col_b:
        fig_s, ax_s = plt.subplots(figsize=(6, 4))
        ax_s.plot(sdof_disp, sdof_force, color="#ffa657", linewidth=2.5, label="SDOF Transformed")
        
        # Plot Bilinear representation
        d_bilin = [0.0, d_y_star, d_max_star]
        f_bilin = [0.0, F_y_star, F_y_star]
        ax_s.plot(d_bilin, f_bilin, color="#ff7b72", linestyle="--", linewidth=2.5, label="Bilinear Idealized")
        ax_s.scatter([d_y_star], [F_y_star], color="#ff7b72", zorder=5, s=80, label=f"Yield Point: {d_y_star:.2f} mm")
        
        ax_s.set_title("SDOF Curve & Bilinearization")
        ax_s.set_xlabel("SDOF Displacement d* (mm)")
        ax_s.set_ylabel("SDOF Shear F* (kN)")
        ax_s.grid(True, linestyle=":")
        ax_s.legend()
        st.pyplot(fig_s)

# ── Tab 2 & 3: Spectra calculations & Target Results ──
limit_states = [
    ("LS of Near Collapse", 2475, "#ff7b72"),
    ("LS of Significant Damage", 475, "#ffa657"),
    ("LS of Damage Limitation", 225, "#58a6ff"),
    ("LS of Rpor 2024", 95, "#bc8cff"),
    ("LS of FEMA 356", 72, "#7ee787")
]

eta = np.sqrt(10.0 / (5.0 + damping))
if eta < 0.55:
    eta = 0.55

summary_data = []
demands_to_plot = []

for name, tr_star, color_hex in limit_states:
    # Importance Scale factor calculations (Handwritten Page 3)
    I = (475.0 / tr_star) ** (-1.0 / 2.7)
    
    # Peak spectral demand
    sa_val = get_spectral_acceleration(T_star, A, I, S, eta, t1, t2, t3)
    
    # Elastic demand calculation target
    d_et_star = (sa_val * 9.81) * ((T_star / (2.0 * np.pi)) ** 2) * 1000.0
    
    # Base evaluations of targets (Handwritten Page 4)
    if T_star < t2:
        q_u = (sa_val * 9.81 * m_star) / F_y_star
        if S_ay >= sa_val:
            d_t_star = d_et_star
        else:
            d_t_star = (d_et_star / q_u) * (1.0 + (q_u - 1.0) * (t2 / T_star))
            if d_t_star < d_et_star:
                d_t_star = d_et_star
    else:
        q_u = 1.0
        d_t_star = d_et_star

    d_t_mdof = gamma * d_t_star
    
    summary_data.append({
        "Limit State": name,
        "Return Period (Tr)": tr_star,
        "Scale Factor (I)": round(I, 3),
        "Demand Sae(T*) [g]": round(sa_val, 3),
        "Ductility demand (qu)": round(q_u, 3),
        "SDOF Disp dt* (mm)": round(d_t_star, 2),
        "MDOF Disp dt (mm)": round(d_t_mdof, 2)
    })
    demands_to_plot.append((name, sa_val, color_hex))

# Summary Tab display table
with tab_results:
    st.subheader("📋 Performance Evaluation Summary Table")
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Educational Verification Check block
    st.info(
        "**Verification Check:** "
        "The calculated target displacements (dt) represent the maximum estimated roof displacements "
        "under structural seismic conditions corresponding to each limit state return period."
    )

# Spectra Curve Drawing Tab
with tab_spectra:
    st.subheader("📈 Multi-Limit State Elastic Response Spectra")
    fig_spec, ax_spec = plt.subplots(figsize=(10, 5.5))
    t_arr = np.linspace(0.01, 3.5, 200)

    # Plot response spectrum lines
    for name, tr_star, color_hex in limit_states:
        I_val = (475.0 / tr_star) ** (-1.0 / 2.7)
        sa_arr = [get_spectral_acceleration(t, A, I_val, S, eta, t1, t2, t3) for t in t_arr]
        ax_spec.plot(t_arr, sa_arr, label=f"{name} (Tr={tr_star} yr)", color=color_hex, linewidth=1.8)

    # Annotate target effective period performance points
    for idx, (name, sa_val, color_hex) in enumerate(demands_to_plot):
        ax_spec.scatter([T_star], [sa_val], color=color_hex, s=100, zorder=6, edgecolors="black")
        ax_spec.annotate(f"{sa_val:.2f}g", (T_star, sa_val), textcoords="offset points", xytext=(10,3), fontsize=9)

    ax_spec.axvline(T_star, color="gray", linestyle=":", alpha=0.8, label=f"T* = {T_star:.3f}s")
    ax_spec.set_xlabel("Period T (seconds)")
    ax_spec.set_ylabel("Spectral Acceleration Sae (g)")
    ax_spec.grid(True, linestyle=":")
    ax_spec.legend()
    st.pyplot(fig_spec)
