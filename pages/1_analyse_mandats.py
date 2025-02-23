import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Doit être la première commande Streamlit
st.set_page_config(page_title="Analyse des Mandats", layout="wide")

# Réutilisation des fonctions existantes
from app_tcd_mdp import (
    load_data, calculate_global_metrics, get_safe_value,
    calculate_safe_weighted_average, weighted_median,
    show_global_view, show_detailed_view, show_pivot_table,
    show_documentation, show_formulas
)

# Système d'authentification
def check_access_level():
    """Retourne `True` si l'utilisateur a entré un mot de passe valide."""
    if 'access_level' not in st.session_state:
        st.session_state.access_level = None
    
    password = st.text_input("Mot de passe", type="password")
    if password == "RRNPA":
        st.session_state.access_level = "basic"
        return True
    elif password == "RRNPAMDP":
        st.session_state.access_level = "full"
        return True
    return False

def get_menu_options():
    """Retourne les options de menu disponibles selon le niveau d'accès."""
    if st.session_state.access_level == "full":
        return ["Vue Globale", "Vue Détaillée", "Tableau Croisé", "Documentation", "Formules"]
    else:
        return ["Vue Globale", "Vue Détaillée"]

def get_mandat_display_name(mandat, nom_mandat):
    """Retourne le nom du mandat selon le niveau d'accès."""
    if st.session_state.access_level == "full":
        return f"{mandat} - {nom_mandat}"
    return f"Mandat {mandat}"

# Documentation cache
DOCUMENTATION_FR = """
# Application d'Analyse de Portefeuille Obligataire

## Fonctionnalités
[... Documentation existante ...]
"""

def main():
    st.title("Analyse des Mandats")
    
    if not check_access_level():
        st.warning("Veuillez entrer un mot de passe valide pour accéder à l'application.")
        return
    
    # Chargement des données
    df = load_data()
    if df is not None:
        menu_options = get_menu_options()
        
        # Utilisation d'onglets au lieu d'un menu déroulant
        tab_titles = menu_options
        tabs = st.tabs(tab_titles)
        
        # Vue Globale
        with tabs[0]:
            show_global_view(df)
        
        # Vue Détaillée
        with tabs[1]:
            show_detailed_view(df)
        
        # Tableau Croisé (accès complet uniquement)
        if st.session_state.access_level == "full" and len(tabs) > 2:
            with tabs[2]:
                show_pivot_table(df)
        
        # Documentation (accès complet uniquement)
        if st.session_state.access_level == "full" and len(tabs) > 3:
            with tabs[3]:
                show_documentation()
        
        # Formules (accès complet uniquement)
        if st.session_state.access_level == "full" and len(tabs) > 4:
            with tabs[4]:
                show_formulas()

if __name__ == "__main__":
    main()
