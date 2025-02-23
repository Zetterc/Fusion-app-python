import streamlit as st

# Doit être la première commande Streamlit
st.set_page_config(page_title="Analyse Économique", layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import warnings
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

# Réutilisation des fonctions existantes
from app_argus import (
    load_data, analyze_series, get_nice_scale,
    calculate_scale_ratio, get_column_axis, assign_axis,
    is_bar_column, create_multi_axis_plot, calculate_trend,
    display_statistics, export_to_pdf, export_all_to_pdf,
    export_selected_to_pdf, check_password
)

def main():
    st.title("Analyse Économique")
    
    if not check_password():
        st.warning("Veuillez entrer un mot de passe valide pour accéder à l'application.")
        return

    # Réinitialiser la session state pour les nouvelles assignations
    if 'reset_done' not in st.session_state:
        for key in list(st.session_state.keys()):
            if key.endswith('_last'):
                del st.session_state[key]
        st.session_state['reset_done'] = True

    # Interface utilisateur
    uploaded_file = st.sidebar.file_uploader("Choisir un fichier Excel", type="xlsx")
    
    if uploaded_file is not None:
        data_dict, prefix_groups = load_data(uploaded_file)
        
        if data_dict is not None:
            # Sélection du groupe de données
            selected_prefix = st.sidebar.selectbox(
                "Sélectionner un groupe de données",
                options=list(prefix_groups.keys())
            )
            
            # Afficher les options pour le groupe sélectionné
            if selected_prefix:
                sheets = prefix_groups[selected_prefix]
                selected_sheet = st.sidebar.selectbox(
                    "Sélectionner une feuille de données",
                    options=sheets
                )
                
                if selected_sheet:
                    data = data_dict[selected_sheet]
                    
                    # Assignation des axes
                    primary_cols, secondary_cols = assign_axis(data)
                    
                    # Interface pour la sélection des colonnes
                    with st.expander("Options d'affichage", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            selected_primary = st.multiselect(
                                "Sélectionner les colonnes (axe gauche)",
                                primary_cols,
                                default=primary_cols[:2] if len(primary_cols) > 1 else primary_cols
                            )
                            
                            primary_stack = st.checkbox(
                                "Empiler les colonnes (axe gauche)",
                                key=f"primary_stack_{selected_sheet}"
                            )
                        
                        with col2:
                            selected_secondary = st.multiselect(
                                "Sélectionner les colonnes (axe droit)",
                                secondary_cols,
                                default=secondary_cols[:2] if len(secondary_cols) > 1 else secondary_cols
                            )
                            
                            secondary_stack = st.checkbox(
                                "Empiler les colonnes (axe droit)",
                                key=f"secondary_stack_{selected_sheet}"
                            )
                    
                    # Déterminer quelles colonnes doivent être en barres
                    primary_bars = [col for col in selected_primary if is_bar_column(col)]
                    secondary_bars = [col for col in selected_secondary if is_bar_column(col)]
                    
                    # Créer et afficher le graphique
                    fig = create_multi_axis_plot(
                        data, selected_primary, selected_secondary,
                        title=selected_sheet,
                        primary_bars=primary_bars,
                        secondary_bars=secondary_bars,
                        primary_stack=primary_stack,
                        secondary_stack=secondary_stack
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Afficher les statistiques
                    if st.checkbox("Afficher les statistiques"):
                        display_statistics(data, selected_sheet)
                    
                    # Options d'export PDF
                    st.subheader("Export PDF")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Exporter ce graphique en PDF"):
                            comment = st.text_area("Ajouter un commentaire (optionnel)")
                            export_to_pdf(
                                data, selected_primary, secondary_cols,
                                title=selected_sheet,
                                primary_bars=primary_bars,
                                secondary_bars=secondary_bars,
                                primary_stack=primary_stack,
                                secondary_stack=secondary_stack,
                                comment=comment
                            )
                    
                    with col2:
                        if st.button("Exporter tous les graphiques en PDF"):
                            export_all_to_pdf(data_dict, prefix_groups)

if __name__ == "__main__":
    main()
