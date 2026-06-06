import sys
import math
import numpy as np
import pandas as pd
import csv

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QLabel, QScrollArea, QSplitter, QFileDialog, QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==========================================
# 1. MATHEMATICAL ENGINE (Model)
# ==========================================
class N2Engine:
    @staticmethod
    def calculate_sdof_properties(n_etages, masses, phis):
        """Calculates basic MDOF to SDOF conversion parameters."""
        m_star = np.sum(masses * phis)
        M1 = np.sum(masses * (phis**2))
        gamma = m_star / M1 if M1 != 0 else 0
        return m_star, gamma

    @staticmethod
    def bilinearize_curve(d_mdof, V_mdof, gamma):
        """
        Converts raw MDOF curve to SDOF and applies the Equal Energy Rule 
        to automatically find Fy* and Sdy*.
        """
        # 1. Convert to SDOF
        d_star = np.array(d_mdof) / gamma
        F_star = np.array(V_mdof) / gamma
        
        # 2. Calculate Total Energy (Area under curve using trapezoidal rule)
        E_m = np.trapz(F_star, d_star)
        
        # 3. Find Fy* (Assumed as max force for standard N2)
        Fy_star = np.max(F_star)
        d_m_star = np.max(d_star)
        
        # 4. Calculate Sdy* via equal energy formula
        Sdy_star = 2 * (d_m_star - (E_m / Fy_star))
        
        return d_star, F_star, Fy_star, Sdy_star

    @staticmethod
    def evaluate_limit_states(m_star, gamma, Fy_star, Sdy_star, seismic_params):
        A, S, T1, T2, T3, xi, eta, k = seismic_params
        
        # Calculate Effective Period and Say
        Say = (Fy_star / m_star) / 9.81 if m_star != 0 else 0
        K_eff = Say / Sdy_star if Sdy_star != 0 else 0
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
                dt_star = d_el_star
                
            dt_mdof = gamma * dt_star
            elastic_limit_mdof = gamma * Sdy_star
            status = "Plastique" if dt_mdof > elastic_limit_mdof else "Élastique"
            
            results.append({
                "État Limite": ls["name"],
                "Tr* (ans)": Tr_star,
                "Facteur I": round(I, 3),
                "Sae(T*) [g]": round(Sae_g, 3),
                "dt* SDOF (m)": round(dt_star, 4),
                "dt MDOF (m)": round(dt_mdof, 4),
                "État": status
            })
            
        return pd.DataFrame(results), T_star, Say, K_eff

# ==========================================
# 2. CUSTOM UI WIDGETS
# ==========================================
class MetricCard(QGroupBox):
    def __init__(self, title, unit=""):
        super().__init__(title)
        self.unit = unit
        layout = QVBoxLayout()
        self.value_label = QLabel("0.000")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.value_label.setFont(font)
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        self.setLayout(layout)

    def set_value(self, value, decimals=3):
        self.value_label.setText(f"{value:.{decimals}f} {self.unit}")

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.tight_layout()

    def plot_curves(self, d_star_arr, F_star_arr, Sdy_star, Fy_star, df_results):
        self.axes.clear()
        
        # 1. Plot raw SDOF curve
        self.axes.plot(d_star_arr, F_star_arr, color='gray', linestyle='-', alpha=0.5, label='Courbe SDOF Nette')
        
        # 2. Plot Bilinear Curve
        max_dt = max(df_results["dt* SDOF (m)"].max(), Sdy_star * 2)
        plot_max = max(d_star_arr[-1], max_dt)
        x_curve = [0, Sdy_star, plot_max]
        y_curve = [0, Fy_star, Fy_star]
        
        self.axes.plot(x_curve, y_curve, 'b-', linewidth=2, label='Bilinéarisation (Aires Égales)')
        self.axes.plot(Sdy_star, Fy_star, 'go', markersize=6, label=f'Fluage (d={Sdy_star:.3f}m, F={Fy_star:.1f}kN)')
        
        # 3. Plot target displacements
        colors = ['red', 'orange', 'purple', 'magenta', 'cyan']
        for i, row in df_results.iterrows():
            dt = row["dt* SDOF (m)"]
            y_val = (Fy_star / Sdy_star) * dt if dt < Sdy_star else Fy_star
            self.axes.plot(dt, y_val, marker='x', color=colors[i%len(colors)], 
                           markersize=8, markeredgewidth=2, label=row["État Limite"])

        self.axes.set_title("Bilinéarisation & Points de Performance")
        self.axes.set_xlabel("Déplacement d* (m)")
        self.axes.set_ylabel("Effort Tranchant F* (kN)")
        self.axes.grid(True, linestyle='--', alpha=0.6)
        self.axes.legend(fontsize=8)
        self.draw()

# ==========================================
# 3. MAIN WINDOW
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Détermination de la Performance Sismique - Automatisée")
        self.resize(1300, 850)
        
        self.d_mdof_raw = None
        self.V_mdof_raw = None
        self.current_results_df = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        self.left_panel = self._build_input_panel()
        self.right_panel = self._build_results_panel()
        
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([450, 850])
        self._update_table_rows()

    def _build_input_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        # 1. Structure
        group_struct = QGroupBox("1. Paramètres de la Structure")
        form_struct = QFormLayout()
        
        self.spin_n_etages = QSpinBox()
        self.spin_n_etages.setRange(1, 50)
        self.spin_n_etages.setValue(5)
        self.spin_n_etages.valueChanged.connect(self._update_table_rows)
        form_struct.addRow("Nombre d'étages:", self.spin_n_etages)
        
        self.table_etages = QTableWidget(0, 2)
        self.table_etages.setHorizontalHeaderLabels(["Masse (t)", "Mode propre (φ)"])
        self.table_etages.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        form_struct.addRow(self.table_etages)
        group_struct.setLayout(form_struct)
        layout.addWidget(group_struct)

        # 2. Pushover Data Loader
        group_data = QGroupBox("2. Données Pushover (MDOF)")
        form_data = QVBoxLayout()
        
        self.btn_load_csv = QPushButton("Charger Courbe (CSV)")
        self.btn_load_csv.clicked.connect(self.load_csv_data)
        form_data.addWidget(self.btn_load_csv)
        
        self.label_data_status = QLabel("Aucun fichier chargé.")
        self.label_data_status.setStyleSheet("color: orange;")
        form_data.addWidget(self.label_data_status)
        
        group_data.setLayout(form_data)
        layout.addWidget(group_data)

        # 3. Seismic Parameters
        group_seismic = QGroupBox("3. Paramètres Sismiques & Site")
        form_seismic = QFormLayout()
        
        self.spin_A = QDoubleSpinBox(); self.spin_A.setValue(0.30)
        self.spin_S = QDoubleSpinBox(); self.spin_S.setValue(1.30)
        self.spin_T1 = QDoubleSpinBox(); self.spin_T1.setValue(0.15); self.spin_T1.setSingleStep(0.05)
        self.spin_T2 = QDoubleSpinBox(); self.spin_T2.setValue(0.60); self.spin_T2.setSingleStep(0.05)
        self.spin_T3 = QDoubleSpinBox(); self.spin_T3.setValue(2.00); self.spin_T3.setSingleStep(0.1)
        self.spin_xi = QDoubleSpinBox(); self.spin_xi.setValue(5.0)
        self.spin_eta = QDoubleSpinBox(); self.spin_eta.setValue(1.0)
        self.spin_k = QDoubleSpinBox(); self.spin_k.setValue(2.7)

        form_seismic.addRow("Accélération (A):", self.spin_A)
        form_seismic.addRow("Paramètre site (S):", self.spin_S)
        form_seismic.addRow("T1 (s):", self.spin_T1)
        form_seismic.addRow("T2 (Tc) (s):", self.spin_T2)
        form_seismic.addRow("T3 (Td) (s):", self.spin_T3)
        form_seismic.addRow("Amortissement ξ (%):", self.spin_xi)
        form_seismic.addRow("Facteur correction (η):", self.spin_eta)
        form_seismic.addRow("Exposant (k):", self.spin_k)
        group_seismic.setLayout(form_seismic)
        layout.addWidget(group_seismic)

        # Actions
        btn_layout = QHBoxLayout()
        self.btn_calc = QPushButton("Analyser & Bilinéariser")
        self.btn_calc.setMinimumHeight(40)
        self.btn_calc.setStyleSheet("font-weight: bold; background-color: #0d6efd; color: white;")
        self.btn_calc.clicked.connect(self.run_analysis)
        self.btn_calc.setEnabled(False) # Disabled until data is loaded
        
        btn_layout.addWidget(self.btn_calc)
        layout.addLayout(btn_layout)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_results_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        # Auto-Calculated Bilinear Properties
        bilin_layout = QHBoxLayout()
        self.card_fy = MetricCard("Force Fluage (Fy*)", "kN")
        self.card_sdy = MetricCard("Dép. Fluage (Sdy*)", "m")
        self.card_t_star = MetricCard("Période Eff. (T*)", "s")
        self.card_gamma = MetricCard("Participation (Γ)")
        
        bilin_layout.addWidget(self.card_fy)
        bilin_layout.addWidget(self.card_sdy)
        bilin_layout.addWidget(self.card_t_star)
        bilin_layout.addWidget(self.card_gamma)
        layout.addLayout(bilin_layout)

        # Plot
        self.plot_canvas = PlotCanvas(self, width=6, height=4)
        layout.addWidget(self.plot_canvas)

        # Results Table
        self.table_results = QTableWidget(0, 7)
        self.table_results.setHorizontalHeaderLabels(
            ["État Limite", "Tr* (ans)", "Facteur I", "Sae(T*) [g]", "dt* SDOF (m)", "dt MDOF (m)", "État"]
        )
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_results.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table_results)

        return container

    def _update_table_rows(self):
        n = self.spin_n_etages.value()
        current_rows = self.table_etages.rowCount()
        self.table_etages.setRowCount(n)
        for i in range(current_rows, n):
            self.table_etages.setItem(i, 0, QTableWidgetItem("200.0"))
            self.table_etages.setItem(i, 1, QTableWidgetItem(f"{(i + 1) / n:.3f}"))

    def load_csv_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger CSV Pushover", "", "CSV Files (*.csv)")
        if not path: return
        
        try:
            d_list, V_list = [], []
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            # Parse floats, replacing commas with dots if necessary
                            d = float(row[0].replace(',', '.'))
                            V = float(row[1].replace(',', '.'))
                            d_list.append(d)
                            V_list.append(V)
                        except ValueError:
                            continue # Skip headers or text
            
            if len(d_list) < 2:
                raise ValueError("Le fichier doit contenir au moins 2 points numériques.")
                
            self.d_mdof_raw = np.array(d_list)
            self.V_mdof_raw = np.array(V_list)
            
            self.label_data_status.setText(f"✓ Fichier chargé : {len(self.d_mdof_raw)} points.")
            self.label_data_status.setStyleSheet("color: lightgreen;")
            self.btn_calc.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Format invalide.\nAssurez-vous d'avoir 2 colonnes numériques (Déplacement, Force).\nErreur: {str(e)}")

    def run_analysis(self):
        if self.d_mdof_raw is None: return
        
        try:
            # 1. Structure arrays
            n = self.spin_n_etages.value()
            masses = np.zeros(n)
            phis = np.zeros(n)
            for i in range(n):
                masses[i] = float(self.table_etages.item(i, 0).text())
                phis[i] = float(self.table_etages.item(i, 1).text())

            seismic_params = (
                self.spin_A.value(), self.spin_S.value(), self.spin_T1.value(),
                self.spin_T2.value(), self.spin_T3.value(), self.spin_xi.value(),
                self.spin_eta.value(), self.spin_k.value()
            )

            # 2. Calculations
            m_star, gamma = N2Engine.calculate_sdof_properties(n, masses, phis)
            d_star_arr, F_star_arr, Fy_star, Sdy_star = N2Engine.bilinearize_curve(self.d_mdof_raw, self.V_mdof_raw, gamma)
            
            df_results, T_star, Say, K_eff = N2Engine.evaluate_limit_states(
                m_star, gamma, Fy_star, Sdy_star, seismic_params
            )

            # 3. Update UI
            self.card_fy.set_value(Fy_star, 1)
            self.card_sdy.set_value(Sdy_star, 4)
            self.card_t_star.set_value(T_star, 3)
            self.card_gamma.set_value(gamma, 3)

            # Update Table
            self.table_results.setRowCount(len(df_results))
            for row_idx, row_data in df_results.iterrows():
                for col_idx, value in enumerate(row_data):
                    item = QTableWidgetItem(str(value))
                    if col_idx == 6: 
                        item.setForeground(Qt.red if value == "Plastique" else Qt.green)
                    self.table_results.setItem(row_idx, col_idx, item)

            # Plot
            self.plot_canvas.plot_curves(d_star_arr, F_star_arr, Sdy_star, Fy_star, df_results)

        except Exception as e:
            QMessageBox.critical(self, "Erreur de Calcul", f"Une erreur est survenue :\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
