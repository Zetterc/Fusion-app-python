import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import warnings
import os
import io
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

# Système d'authentification
def check_password():
    """Retourne `True` si l'utilisateur a entré le bon mot de passe."""
    def password_entered():
        if st.session_state["password"] == "RRNPAMDP":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Veuillez entrer le mot de passe", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Mot de passe incorrect. Réessayez:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    else:
        return True

def load_data(uploaded_file):
    """Charge les données depuis un fichier Excel et les groupe par préfixe"""
    if uploaded_file is not None:
        excel = pd.ExcelFile(uploaded_file)
        sheets = excel.sheet_names
        
        # Grouper les feuilles par préfixe
        prefix_groups = {}
        for sheet in sheets:
            prefix = sheet.split('_')[0]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(sheet)
            
        # Charger les données pour chaque feuille
        data_dict = {}
        for sheet in sheets:
            df = pd.read_excel(uploaded_file, sheet_name=sheet, index_col=0)
            # S'assurer que l'index est au format datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    st.error(f"Erreur: L'index de la feuille {sheet} n'a pas pu être converti en dates.")
                    return None, None
            data_dict[sheet] = df
                    
        return data_dict, prefix_groups
    return None, None

def analyze_series(series, name):
    """Analyse une série temporelle et retourne ses caractéristiques"""
    series = series.dropna()
    if len(series) < 4:
        return None
    
    stats_dict = {}
    
    try:
        adf_result = adfuller(series, regression='ct')
        stats_dict['stationary'] = adf_result[1] < 0.05
    except:
        stats_dict['stationary'] = None
    
    try:
        decomposition = seasonal_decompose(series, period=min(len(series)//2, 12), extrapolate_trend='freq')
        trend = decomposition.trend
        trend_clean = trend.dropna()
        if len(trend_clean) > 1:
            x = np.arange(len(trend_clean))
            z = np.polyfit(x, trend_clean, 1)
            stats_dict['trend_slope'] = z[0]
        else:
            stats_dict['trend_slope'] = 0
            
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        seasonal_strength = 1 - np.nanvar(residual) / np.nanvar(seasonal + residual)
        stats_dict['seasonal_strength'] = seasonal_strength
        
    except:
        stats_dict['trend_slope'] = 0
        stats_dict['seasonal_strength'] = 0
    
    try:
        stats_dict['skewness'] = stats.skew(series)
        stats_dict['kurtosis'] = stats.kurtosis(series)
        returns = series.pct_change().dropna()
        stats_dict['volatility'] = returns.std()
        stats_dict['magnitude'] = np.log10(abs(series.mean())) if series.mean() != 0 else 0
    except:
        stats_dict['skewness'] = 0
        stats_dict['kurtosis'] = 0
        stats_dict['volatility'] = 0
        stats_dict['magnitude'] = 0
    
    return stats_dict

def get_nice_scale(min_val, max_val, include_zero=False):
    """Calcule une échelle 'propre' pour les valeurs données."""
    if min_val == max_val:
        if max_val == 0:
            return -1, 1
        return max_val * 0.9, max_val * 1.1

    if include_zero:
        min_val = min(0, min_val)
        max_val = max(0, max_val)
    
    range_val = max_val - min_val
    min_val = min_val - (range_val * 0.1)
    max_val = max_val + (range_val * 0.1)
    
    def round_to_nice(x, round_up=True):
        abs_x = abs(x)
        sign = 1 if x >= 0 else -1
        
        if abs_x <= 5:
            step = 1
        elif abs_x <= 10:
            step = 2
        elif abs_x <= 100:
            step = 10
        else:
            step = 50
        
        if round_up:
            return sign * (((abs_x // step) + 1) * step)
        else:
            return sign * ((abs_x // step) * step)
    
    return round_to_nice(min_val, False), round_to_nice(max_val, True)

def calculate_scale_ratio(series):
    """Calcule le ratio d'échelle d'une série."""
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 1
    
    abs_max = abs(clean_series).max()
    abs_min = abs(clean_series[clean_series != 0]).min() if any(clean_series != 0) else abs_max
    
    return abs_max / abs_min if abs_min > 0 else 1

def get_column_axis(column_name):
    """Détermine l'axe d'une colonne basé sur son nom."""
    if '(L)' in column_name:
        return 'left'
    elif '(R)' in column_name:
        return 'right'
    return None

def assign_axis(data):
    """Assigne les colonnes aux axes."""
    primary_cols = []
    secondary_cols = []
    
    # Première passe : utiliser les marqueurs explicites
    for col in data.columns:
        axis = get_column_axis(col)
        if axis == 'left':
            primary_cols.append(col)
        elif axis == 'right':
            secondary_cols.append(col)
    
    # Deuxième passe : colonnes non marquées
    unmarked_cols = [col for col in data.columns if col not in primary_cols + secondary_cols]
    
    if unmarked_cols:
        scale_ratios = {col: calculate_scale_ratio(data[col]) for col in unmarked_cols}
        max_ratio = max(scale_ratios.values())
        
        for col in unmarked_cols:
            if scale_ratios[col] < max_ratio / 10:
                secondary_cols.append(col)
            else:
                primary_cols.append(col)
    
    return primary_cols, secondary_cols

def is_bar_column(column_name):
    """Vérifie si une colonne doit être affichée en barres."""
    return '[BAR]' in column_name

def create_multi_axis_plot(data, primary_cols, secondary_cols, title=None, 
                         primary_bars=None, secondary_bars=None,
                         primary_stack=False, secondary_stack=False):
    """Crée un graphique avec deux axes Y."""
    if not primary_cols and not secondary_cols:
        return None
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Fonction pour créer les traces
    def create_traces(columns, is_secondary=False, bars=None, stack=False):
        traces = []
        if not columns:
            return traces
        
        for col in columns:
            is_bar = bars is not None and col in bars
            
            if is_bar:
                trace = go.Bar(
                    name=col,
                    x=data.index,
                    y=data[col],
                    offsetgroup=1 if is_secondary else 0
                )
            else:
                trace = go.Scatter(
                    name=col,
                    x=data.index,
                    y=data[col],
                    mode='lines'
                )
            
            traces.append(trace)
        
        return traces
    
    # Ajouter les traces
    primary_traces = create_traces(primary_cols, False, primary_bars, primary_stack)
    secondary_traces = create_traces(secondary_cols, True, secondary_bars, secondary_stack)
    
    for trace in primary_traces:
        fig.add_trace(trace, secondary_y=False)
    
    for trace in secondary_traces:
        fig.add_trace(trace, secondary_y=True)
    
    # Mise à jour du layout
    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode='x unified',
        barmode='stack' if primary_stack or secondary_stack else 'group'
    )
    
    return fig

def calculate_trend(series):
    """Calcule la tendance d'une série."""
    if len(series) < 2:
        return "Stable"
    
    try:
        x = np.arange(len(series))
        z = np.polyfit(x, series, 1)
        slope = z[0]
        
        if abs(slope) < 0.001:
            return "Stable"
        elif slope > 0:
            return "Croissante"
        else:
            return "Décroissante"
    except:
        return "Stable"

def display_statistics(data, sheet_name):
    """Affiche les statistiques descriptives des séries."""
    st.subheader(f"Statistiques pour {sheet_name}")
    
    stats_data = []
    for column in data.columns:
        series = data[column].dropna()
        if len(series) > 0:
            stats_dict = {
                'Colonne': column,
                'Moyenne': series.mean(),
                'Médiane': series.median(),
                'Écart-type': series.std(),
                'Min': series.min(),
                'Max': series.max(),
                'Tendance': calculate_trend(series)
            }
            stats_data.append(stats_dict)
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df)

def export_to_pdf(data, primary_cols, secondary_cols, title=None, 
                 primary_bars=None, secondary_bars=None,
                 primary_stack=False, secondary_stack=False,
                 comment=None):
    """Export le graphique en PDF."""
    fig = create_multi_axis_plot(
        data, primary_cols, secondary_cols,
        title=title,
        primary_bars=primary_bars,
        secondary_bars=secondary_bars,
        primary_stack=primary_stack,
        secondary_stack=secondary_stack
    )
    
    if fig is None:
        st.error("Impossible de créer le graphique pour l'export PDF")
        return
    
    # Créer le PDF
    pdf_filename = f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        # Configuration de la figure matplotlib
        plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        
        # Convertir la figure Plotly en image
        img_bytes = fig.to_image(format="png")
        plt.imshow(plt.imread(io.BytesIO(img_bytes)))
        plt.axis('off')
        
        # Ajouter le commentaire si présent
        if comment:
            plt.figtext(0.1, 0.02, comment, wrap=True)
        
        pdf.savefig()
        plt.close()
    
    # Téléchargement du PDF
    with open(pdf_filename, "rb") as f:
        st.download_button(
            label="Télécharger le PDF",
            data=f,
            file_name=pdf_filename,
            mime="application/pdf"
        )

def export_all_to_pdf(data_dict, prefix_groups):
    """Export tous les graphiques en PDF."""
    if not data_dict:
        st.error("Aucune donnée à exporter")
        return
    
    pdf_filename = f"all_graphs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        for prefix, sheets in prefix_groups.items():
            for sheet in sheets:
                data = data_dict[sheet]
                primary_cols, secondary_cols = assign_axis(data)
                
                if not primary_cols and not secondary_cols:
                    continue
                
                fig = create_multi_axis_plot(
                    data, primary_cols, secondary_cols,
                    title=sheet
                )
                
                if fig is not None:
                    # Configuration de la figure matplotlib
                    plt.figure(figsize=(11.69, 8.27))  # A4 landscape
                    
                    # Convertir la figure Plotly en image
                    img_bytes = fig.to_image(format="png")
                    plt.imshow(plt.imread(io.BytesIO(img_bytes)))
                    plt.axis('off')
                    
                    pdf.savefig()
                    plt.close()
    
    # Téléchargement du PDF
    with open(pdf_filename, "rb") as f:
        st.download_button(
            label="Télécharger le PDF complet",
            data=f,
            file_name=pdf_filename,
            mime="application/pdf"
        )

def export_selected_to_pdf(data_dict, selected_sheets):
    """Export les graphiques sélectionnés en PDF."""
    if not data_dict or not selected_sheets:
        st.error("Aucune donnée à exporter")
        return
    
    pdf_filename = f"selected_graphs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        for sheet in selected_sheets:
            if sheet in data_dict:
                data = data_dict[sheet]
                primary_cols, secondary_cols = assign_axis(data)
                
                if not primary_cols and not secondary_cols:
                    continue
                
                fig = create_multi_axis_plot(
                    data, primary_cols, secondary_cols,
                    title=sheet
                )
                
                if fig is not None:
                    # Configuration de la figure matplotlib
                    plt.figure(figsize=(11.69, 8.27))  # A4 landscape
                    
                    # Convertir la figure Plotly en image
                    img_bytes = fig.to_image(format="png")
                    plt.imshow(plt.imread(io.BytesIO(img_bytes)))
                    plt.axis('off')
                    
                    pdf.savefig()
                    plt.close()
    
    # Téléchargement du PDF
    with open(pdf_filename, "rb") as f:
        st.download_button(
            label="Télécharger le PDF des graphiques sélectionnés",
            data=f,
            file_name=pdf_filename,
            mime="application/pdf"
        )
