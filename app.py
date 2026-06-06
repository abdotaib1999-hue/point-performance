import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. MATHEMATICAL ENGINE
# ==========================================
class N2Engine:
    @staticmethod
    def calculate_sdof_properties(masses, phis):
        m_star = np.sum(masses * phis)
        M1 = np.sum(masses * (phis**2))
        gamma = m_star / M1 if M1 != 0 else 0
        return m_star, gamma

    @staticmethod
    def bilinearize_curve(d_mdof, V_mdof, gamma):
        # 1. Convert to SDOF
        d_star = np.array(d_mdof) / gamma
        F_star = np.array(V_mdof) / gamma
        
        # 2. Calculate Total Energy (Area under curve)
        E_m = np.trapz(F_star, d_star)
        
        # 3. Find Fy* and maximum displacement
        Fy_star = np.max(F_star)
        d_m_star = np.max(d_star)
        
        # 4. Calculate Sdy* via equal energy formula
        Sdy_star = 2 * (d_m_star - (E_m / Fy_star)) if Fy_star != 0 else 0
        
        return d_star, F_star, Fy_star, Sdy_star

    @staticmethod
    def evaluate_limit_states(m_star, gamma, Fy_star, Sdy_star, seismic_params):
        A, S, T1, T2, T3, xi, eta, k = seismic_params
        
        Say = (Fy_star / m_star) / 9.81 if m_star != 0 else 0
        T_star = 2 * math.pi * math.sqrt((Sdy_star * m_star) / Fy_star) if Fy_star != 0 else 0

        limit_states = [
            {"name": "Near Collapse (NC)", "Tr_star": 2475},
            {"name": "Significant Damage (SD)", "Tr_star": 475},
            {"name": "Damage Limitation (DL)", "Tr_star": 225},
            {"name": "RPA99 (Zone/Site)", "Tr_star": 95},
            {"name": "FEMA 356", "Tr_star": 72}
        ]

        def calc_Sae(T, I):
            if T < T1: return A * I * S * (1 + (T / T1) * (2.5 * eta - 1))
            elif T1 <= T < T2: return A * I * S * (2.5 * eta)
            elif T2 <= T < T3: return A * I * S * (2.5 * eta) * (T2 / T)
            else: return A * I * S * (2.5 * eta) * ((T2 * T3) / (T**2))

        results = []
        for ls in limit_states:
            Tr_star = ls["Tr_star"]
            I = (475 / Tr_star) ** (-1 / k)
            Sae_g = calc_Sae(T_star, I)
            
            d_el_star = Sae_g * 9.81 * ((T_star / (2 * math.pi))**2)
            dt_star = 0
            
            if T_star < T2:
                if Say >= Sae_g:
                    dt_star = d_el_star
                else:
                    qu = (Sae_g * m_star * 9.81) / Fy_star if Fy_star != 0 else 0
                    if qu > 0:
                        dt_star = (1 + (qu - 1) * (T2 / T_star)) * (d_el_star / qu) if T_star != 0 else 0
                    else:
                        dt_star = d_el_star
            else:
                dt
