import streamlit as st

st.set_page_config(
    page_title="Application d'Analyse Financière",
    page_icon="📊",
    layout="wide"
)

st.title("Application d'Analyse Financière")
st.sidebar.success("Sélectionnez une page ci-dessus.")

st.markdown("""
## Bienvenue dans l'Application d'Analyse Financière

Cette application combine deux outils puissants :

### 1. Analyse des Mandats 📈
- Analyse détaillée des portefeuilles obligataires
- Visualisation des métriques clés
- Tableaux croisés dynamiques
- Accès sécurisé à plusieurs niveaux

### 2. Analyse Économique 📊
- Visualisation de données économiques
- Analyse de séries temporelles
- Export de graphiques en PDF
- Analyses statistiques détaillées

Sélectionnez une page dans le menu de gauche pour commencer.
""")
