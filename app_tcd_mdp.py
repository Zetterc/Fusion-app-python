import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

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

### 1. Analyse Globale du Portefeuille
- Allocation Investment Grade, High Yield et Cash
- Visualisation des poids sectoriels
- Distribution des ratings
- Analyse des rangs de paiement

### 2. Métriques de Performance
- Yield to Maturity
- Duration Modifiée
- Contribution à la Duration
- Contribution au Yield
- RPAP (Ratio Contribution Duration/Yield)

### 3. Visualisations
- Graphiques d'allocation sectorielle
- Distribution des ratings
- Box plots des ratings
- Analyse des flux futurs
- Matrices croisées (Secteur x Duration, Pays x Duration)

### 4. Analyse RPAP
L'application inclut une analyse détaillée du RPAP (Ratio Contribution Duration/Yield) :
- Distribution des RPAP
- Analyse par rating
- Statistiques descriptives
- Interprétation :
  * RPAP = 1 : contribution égale à la duration et au yield
  * RPAP < 1 : meilleure efficience (plus de contribution au yield qu'à la duration)
  * RPAP > 1 : efficience moindre (plus de contribution à la duration qu'au yield)
"""

DOCUMENTATION_EN = """
# Bond Portfolio Analysis Application

## Features

### 1. Portfolio Global Analysis
- Investment Grade, High Yield, and Cash allocation
- Sector weight visualization
- Rating distribution
- Payment rank analysis

### 2. Performance Metrics
- Yield to Maturity
- Modified Duration
- Duration Contribution
- Yield Contribution
- RPAP (Duration/Yield Contribution Ratio)

### 3. Visualizations
- Sector allocation charts
- Rating distribution
- Rating box plots
- Future cash flow analysis
- Cross matrices (Sector x Duration, Country x Duration)

### 4. RPAP Analysis
The application includes detailed RPAP (Duration/Yield Contribution Ratio) analysis:
- RPAP distribution
- Analysis by rating
- Descriptive statistics
- Interpretation:
  * RPAP = 1: equal contribution to duration and yield
  * RPAP < 1: better efficiency (more yield contribution than duration)
  * RPAP > 1: lower efficiency (more duration contribution than yield)
"""

# Fonction pour charger les données
def load_data():
    uploaded_file = st.sidebar.file_uploader("Choisir un fichier Excel", type="xlsx")
    if uploaded_file is not None:
        # Lire les noms des feuilles du fichier Excel
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        
        # Créer un sélecteur pour choisir la feuille
        selected_sheet = st.sidebar.selectbox(
            "Sélectionner la feuille Excel",
            options=sheet_names,
            index=0
        )
        
        # Lire uniquement la feuille sélectionnée
        return pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    return None

# Fonction pour calculer les métriques globales
def calculate_global_metrics(df):
    # Filtrer les obligations et calculer leurs poids
    all_bonds = df[df['ISIN'].notna()].copy()
    all_bonds['Montant Valorise'] = all_bonds['QUANTITE'] * all_bonds['Px_Last'] / 100
    total_bonds = all_bonds['Montant Valorise'].sum()
    all_bonds['Poids en %'] = all_bonds['Montant Valorise'] / total_bonds * 100 if total_bonds > 0 else 0
    
    # Calculer les contributions et les métriques du portefeuille
    all_bonds['Contribution Yield'] = all_bonds['YAS_BOND_YLD'] * all_bonds['Poids en %'] / 100
    all_bonds['Contribution Duration'] = all_bonds['YAS_MOD_DUR'] * all_bonds['Poids en %'] / 100
    
    # Vérifier les données de spread
    has_valid_spreads = not all_bonds['YAS_ASW_SPREAD'].isna().all()
    if has_valid_spreads:
        all_bonds['Contribution Spread'] = all_bonds['YAS_ASW_SPREAD'] * all_bonds['Poids en %'] / 100
        spread_portfolio = all_bonds['Contribution Spread'].sum()
    else:
        spread_portfolio = 0
    
    # Calculer les métriques du portefeuille
    metrics = {
        'Yield Portfolio': all_bonds['Contribution Yield'].sum(),
        'Duration Portfolio': all_bonds['Contribution Duration'].sum(),
        'AUM Total': df['QUANTITE'].sum(),
        'Spread Portfolio': spread_portfolio
    }
    
    return metrics

# Fonction pour la page globale
def show_global_view(df):
    st.title("Vue Globale des Mandats")
    
    # Métriques globales
    metrics = calculate_global_metrics(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Yield du Portefeuille", f"{metrics['Yield Portfolio']:.2f}%")
    with col2:
        st.metric("Duration du Portefeuille", f"{metrics['Duration Portfolio']:.2f}")
    with col3:
        st.metric("AUM Total", f"{metrics['AUM Total']:,.0f}")
    with col4:
        st.metric("Spread du Portefeuille", f"{metrics['Spread Portfolio']:.0f} bps")
    
    # Répartition des actifs
    st.subheader("Répartition des Actifs par Mandat")
    
    # Préparer les données pour la répartition des actifs
    asset_allocation = []
    for mandat in df['PORTEFEUILLE'].unique():
        mandat_data = df[df['PORTEFEUILLE'] == mandat].copy()
        nom_mandat = mandat_data['Nom du Mandats'].iloc[0]  # Récupérer le nom du mandat
        
        # Calculer les montants pour chaque type d'actif
        bonds = mandat_data[mandat_data['ISIN'].notna() & (mandat_data['GENRE DU TITRE'] == 'Obligation')]['QUANTITE'].sum()
        monetary = mandat_data[mandat_data['GENRE DU TITRE'] == 'Monétaire']['QUANTITE'].sum()
        cash = mandat_data[mandat_data['GENRE DU TITRE'] == 'Cash']['QUANTITE'].sum()
        
        total = bonds + monetary + cash
        
        asset_allocation.append({
            'Mandat': get_mandat_display_name(mandat, nom_mandat),
            'Type': 'Obligations',
            'Montant': bonds,
            'Pourcentage': bonds/total*100 if total > 0 else 0
        })
        asset_allocation.append({
            'Mandat': get_mandat_display_name(mandat, nom_mandat),
            'Type': 'Monétaire',
            'Montant': monetary,
            'Pourcentage': monetary/total*100 if total > 0 else 0
        })
        asset_allocation.append({
            'Mandat': get_mandat_display_name(mandat, nom_mandat),
            'Type': 'Cash',
            'Montant': cash,
            'Pourcentage': cash/total*100 if total > 0 else 0
        })
    
    asset_df = pd.DataFrame(asset_allocation)
    
    # Créer le graphique
    fig = px.bar(
        asset_df,
        x='Mandat',
        y='Pourcentage',
        color='Type',
        title='Répartition des Actifs par Mandat',
        labels={'Pourcentage': '% du Portefeuille'},
        height=400,
        text='Pourcentage'  # Afficher les valeurs sur les barres
    )
    # Formater l'affichage des valeurs
    fig.update_traces(
        texttemplate='%{text:.1f}%',  # Afficher 1 décimale et le symbole %
        textposition='inside'          # Positionner le texte à l'intérieur des barres
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif
    st.subheader("Récapitulatif par Mandat")
    
    # Préparation des données pour le graphique Yield/Duration
    yield_duration_data = []
    for mandat in df['PORTEFEUILLE'].unique():
        mandat_data = df[df['PORTEFEUILLE'] == mandat].copy()
        nom_mandat = mandat_data['Nom du Mandats'].iloc[0]
        
        # Calculer le yield et la duration pour ce mandat
        bonds_data = mandat_data[mandat_data['ISIN'].notna()].copy()
        if not bonds_data.empty:
            bonds_data['Montant Valorise'] = bonds_data['QUANTITE'] * bonds_data['Px_Last'] / 100
            total_bonds = bonds_data['Montant Valorise'].sum()
            bonds_data['Poids en %'] = bonds_data['Montant Valorise'] / total_bonds * 100 if total_bonds > 0 else 0
            
            yield_contribution = (bonds_data['YAS_BOND_YLD'] * bonds_data['Poids en %'] / 100).sum()
            duration_contribution = (bonds_data['YAS_MOD_DUR'] * bonds_data['Poids en %'] / 100).sum()
            
            yield_duration_data.extend([
                {
                    'Mandat': get_mandat_display_name(mandat, nom_mandat),
                    'Valeur': yield_contribution,
                    'Métrique': 'Yield (%)'
                },
                {
                    'Mandat': get_mandat_display_name(mandat, nom_mandat),
                    'Valeur': duration_contribution,
                    'Métrique': 'Duration'
                }
            ])
    
    # Créer le graphique Yield/Duration
    if yield_duration_data:
        st.subheader("Yield et Duration par Mandat")
        yield_duration_df = pd.DataFrame(yield_duration_data)
        fig_yd = px.bar(
            yield_duration_df,
            x='Mandat',
            y='Valeur',
            color='Métrique',
            barmode='group',
            title='Yield et Duration par Mandat',
            labels={'Valeur': 'Valeur', 'Mandat': 'Mandat'},
            height=400,
            color_discrete_map={
                'Duration': 'red',
                'Yield (%)': '#1f77b4'  # Couleur bleue par défaut de Plotly
            },
            text='Valeur'  # Afficher les valeurs sur les barres
        )
        # Personnaliser le graphique
        fig_yd.update_layout(
            legend_title_text='Métrique',
            xaxis_tickangle=-45
        )
        # Formater l'affichage des valeurs
        fig_yd.update_traces(
            texttemplate='%{text:.2f}',  # Afficher 2 décimales
            textposition='outside'        # Positionner le texte au-dessus des barres
        )
        st.plotly_chart(fig_yd, use_container_width=True)
    
    recap_data = []
    for mandat in df['PORTEFEUILLE'].unique():
        mandat_data = df[df['PORTEFEUILLE'] == mandat]
        nom_mandat = mandat_data['Nom du Mandats'].iloc[0]
        bonds_only = mandat_data[mandat_data['GENRE DU TITRE'] == 'Obligation']
        
        # Calculer le rating moyen
        if not bonds_only.empty:
            avg_rating = bonds_only['worst rating'].mode().iloc[0] if not bonds_only['worst rating'].empty else "N/A"
        else:
            avg_rating = "N/A"
        
        # Calculer l'AUM
        aum = mandat_data['QUANTITE'].sum()
        
        recap_data.append({
            'Mandat': get_mandat_display_name(mandat, nom_mandat),
            'Duration': bonds_only['YAS_MOD_DUR'].mean(),
            'Yield': bonds_only['YAS_BOND_YLD'].mean(),
            'Note Crédit': avg_rating,
            'AUM': aum,
            'Spread': bonds_only['YAS_ASW_SPREAD'].mean()
        })
    
    recap_df = pd.DataFrame(recap_data)
    # Formater les colonnes
    recap_df['Duration'] = recap_df['Duration'].round(2)
    recap_df['Yield'] = recap_df['Yield'].round(2).map('{:.2f}%'.format)
    recap_df['AUM'] = recap_df['AUM'].map('{:,.0f}'.format)
    recap_df['Spread'] = recap_df['Spread'].round(0).map('{:.0f} bps'.format)
    
    st.dataframe(
        recap_df.set_index('Mandat'),
        column_config={
            "Duration": "Duration",
            "Yield": "Yield",
            "Note Crédit": "Note Crédit",
            "AUM": "AUM",
            "Spread": "Spread"
        }
    )
    
    # Afficher les obligations sans spread en bas de page
    st.subheader("Obligations sans ASW Spread")
    
    # Filtrer les obligations et calculer leurs poids
    all_bonds = df[df['ISIN'].notna()].copy()
    all_bonds['Montant Valorise'] = all_bonds['QUANTITE'] * all_bonds['Px_Last'] / 100
    total_bonds = all_bonds['Montant Valorise'].sum()
    all_bonds['Poids en %'] = all_bonds['Montant Valorise'] / total_bonds * 100 if total_bonds > 0 else 0
    
    # Filtrer les obligations sans spread
    bonds_without_spread = all_bonds[
        all_bonds['YAS_ASW_SPREAD'].isna()
    ][['PORTEFEUILLE', 'Nom du Mandats', 'ISIN', 'name', 'worst rating', 'YAS_BOND_YLD', 'YAS_MOD_DUR', 'Poids en %']]
    
    if not bonds_without_spread.empty:
        # Créer une colonne avec le nom du mandat formaté selon le niveau d'accès
        bonds_without_spread['Mandat'] = bonds_without_spread.apply(
            lambda x: get_mandat_display_name(x['PORTEFEUILLE'], x['Nom du Mandats']), 
            axis=1
        )
        
        # Afficher le DataFrame avec la nouvelle colonne, en excluant les colonnes originales de portefeuille et nom
        display_df = bonds_without_spread.drop(['PORTEFEUILLE', 'Nom du Mandats'], axis=1)
        display_df = display_df.rename(columns={'Mandat': 'Portefeuille'})
        
        st.dataframe(
            display_df,
            column_config={
                'Portefeuille': 'Mandat',
                'ISIN': 'ISIN',
                'name': 'Nom',
                'worst rating': 'Rating',
                'YAS_BOND_YLD': 'Yield',
                'YAS_MOD_DUR': 'Duration',
                'Poids en %': 'Poids'
            },
            hide_index=True,
            height=min(35 * (len(bonds_without_spread) + 1), 500)  # Hauteur dynamique basée sur le nombre de lignes
        )
        st.write(f"Nombre d'obligations sans spread : {len(bonds_without_spread)}")
    else:
        st.write("Toutes les obligations ont un spread ASW")

def get_safe_value(df, column, default_text="-", default_num=0):
    """
    Récupère une valeur de manière sécurisée avec des valeurs par défaut appropriées.
    Si la colonne n'existe pas ou si la valeur est NaN, retourne la valeur par défaut.
    """
    try:
        if column not in df.columns:
            return default_text if isinstance(default_text, str) else default_num
        
        value = df[column].iloc[0]
        
        if pd.isna(value):
            return default_text if isinstance(default_text, str) else default_num
            
        return value
    except:
        return default_text if isinstance(default_text, str) else default_num

def calculate_safe_weighted_average(df, value_col, weight_col, default=0):
    """
    Calcule une moyenne pondérée de manière sécurisée.
    Retourne la valeur par défaut si le calcul est impossible.
    """
    try:
        if value_col not in df.columns or weight_col not in df.columns:
            return default
        
        values = df[value_col].fillna(0)
        weights = df[weight_col].fillna(0)
        
        if len(values) == 0 or weights.sum() == 0:
            return default
            
        return np.average(values, weights=weights)
    except:
        return default

def weighted_median(data, weights):
    """
    Calcule la médiane pondérée d'un ensemble de données.
    
    Args:
        data: Les valeurs
        weights: Les poids correspondants
    
    Returns:
        La médiane pondérée
    """
    # Trier les données et les poids correspondants
    sorted_idx = np.argsort(data)
    sorted_data = data[sorted_idx]
    sorted_weights = weights[sorted_idx]
    
    # Calculer les poids cumulés
    cumsum = np.cumsum(sorted_weights)
    # Normaliser les poids cumulés
    cumsum = cumsum / cumsum[-1]
    
    # Trouver l'index où les poids cumulés dépassent 0.5
    median_idx = np.searchsorted(cumsum, 0.5)
    
    return sorted_data[median_idx]

def show_detailed_view(df):
    st.title("Analyse Détaillée par Mandat")

    # Créer un dictionnaire de correspondance entre numéro et nom de mandat
    mandats_dict = dict(zip(df['PORTEFEUILLE'], df['Nom du Mandats']))
    reverse_dict = {v: k for k, v in mandats_dict.items()}
    
    # Initialiser le mandat sélectionné si nécessaire
    if 'selected_mandat' not in st.session_state:
        st.session_state.selected_mandat = list(mandats_dict.keys())[0]
        st.session_state.selected_name = mandats_dict[st.session_state.selected_mandat]

    if st.session_state.access_level == "full":
        # Version complète avec deux sélecteurs
        col1, col2 = st.columns(2)
        
        with col1:
            selected_num = st.selectbox(
                "Sélectionner par numéro de mandat",
                options=list(mandats_dict.keys()),
                key='mandat_num',
                index=list(mandats_dict.keys()).index(st.session_state.selected_mandat)
            )
        
        with col2:
            selected_name = st.selectbox(
                "Sélectionner par nom de mandat",
                options=list(mandats_dict.values()),
                key='mandat_name',
                index=list(mandats_dict.values()).index(mandats_dict[st.session_state.selected_mandat])
            )
        
        # Mettre à jour la sélection en fonction du dernier changement
        if selected_num != st.session_state.selected_mandat:
            st.session_state.selected_mandat = selected_num
            st.session_state.selected_name = mandats_dict[selected_num]
            st.rerun()
        elif selected_name != mandats_dict[st.session_state.selected_mandat]:
            st.session_state.selected_mandat = reverse_dict[selected_name]
            st.session_state.selected_name = selected_name
            st.rerun()
    else:
        # Version basique avec un seul sélecteur
        selected_mandat = st.selectbox(
            "Sélectionner un mandat",
            options=list(mandats_dict.keys()),
            format_func=lambda x: f"Mandat {x}"
        )
        st.session_state.selected_mandat = selected_mandat
        st.session_state.selected_name = mandats_dict[selected_mandat]

    # Filtrer les données pour le mandat sélectionné
    mandat_data = df[df['PORTEFEUILLE'] == st.session_state.selected_mandat].copy()
    
    # Définir les listes de ratings pour le tri
    notes_moodys = ["Aaa", "Aa1", "Aa2", "Aa3", "A1", "A2", "A3", "Baa1", "Baa2", "Baa3", 
                    "Ba1", "Ba2", "Ba3", "B1", "B2", "B3", "Caa1", "Caa2", "Caa3"]

    notes_fitch = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
                   "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-"]

    notes_lmdg = ["A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-"]
    
    # Combiner toutes les notes dans un seul dictionnaire pour le tri
    all_ratings = {}
    for i, rating in enumerate(notes_moodys + notes_fitch + notes_lmdg):
        if rating not in all_ratings:
            all_ratings[rating] = i
    
    # Calculer les montants valorisés
    mandat_data['Montant Valorise'] = mandat_data.apply(
        lambda row: row['QUANTITE'] * row['Px_Last'] / 100 if row['GENRE DU TITRE'] == 'Obligation' else row['QUANTITE'],
        axis=1
    )
    
    # Calculer les poids en pourcentage
    total_amount = mandat_data['Montant Valorise'].sum()
    mandat_data['Poids en %'] = (mandat_data['Montant Valorise'] / total_amount * 100) if total_amount > 0 else 0
    
    bonds_only = mandat_data[mandat_data['GENRE DU TITRE'] == 'Obligation'].copy()
    
    # Recalculer les poids pour que bonds_only totalise 100%
    bonds_total = bonds_only['Montant Valorise'].sum()
    bonds_only['Poids en %'] = bonds_only['Montant Valorise'] / bonds_total * 100 if bonds_total > 0 else 0
    
    # Calculer les métriques clés
    total_poids = mandat_data['Poids en %'].sum()
    
    # Fonction pour déterminer si un rating est IG
    def is_investment_grade(rating):
        if pd.isna(rating) or rating == 'Cash':
            return False
        return (rating.startswith(('AAA', 'AA', 'A', 'BBB')) or 
                rating.startswith(('Aaa', 'Aa', 'A', 'Baa')))
    
    # Afficher les métriques principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Yield moyen
        avg_yield = calculate_safe_weighted_average(bonds_only, 'YAS_BOND_YLD', 'Poids en %')
        st.metric("Yield Moyen", f"{avg_yield:.2f}%")
        
        # Duration moyenne
        avg_duration = calculate_safe_weighted_average(bonds_only, 'YAS_MOD_DUR', 'Poids en %')
        st.metric("Duration Moyenne", f"{avg_duration:.2f}")
        
        # Montant monétaire
        monetary_data = mandat_data[mandat_data['GENRE DU TITRE'] == 'Monétaire']
        monetary_amount = monetary_data['QUANTITE'].sum() if not monetary_data.empty else 0
        monetary_weight = monetary_data['Poids en %'].sum() if not monetary_data.empty else 0
        st.metric("Monétaire", f"{monetary_amount:,.0f}€ ({monetary_weight:.1f}%)")

        # Condition du mandat
        condition = get_safe_value(mandat_data, 'Condition', default_text="-")
        st.metric("Condition du Mandat", condition)
    
    with col2:
        # Ratio IG en utilisant les poids rebasés
        ig_weight = bonds_only[bonds_only['worst rating'].apply(is_investment_grade)]['Poids en %'].sum() if not bonds_only.empty else 0
        st.metric("Investment Grade", f"{ig_weight:.1f}%")
        
        # Rating moyen
        rating_map = {
            'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
            'A+': 5, 'A': 6, 'A-': 7,
            'BBB+': 8, 'BBB': 9, 'BBB-': 10,
            'BB+': 11, 'BB': 12, 'BB-': 13,
            'B+': 14, 'B': 15, 'B-': 16,
            'CCC+': 17, 'CCC': 18, 'CCC-': 19
        }
        reverse_rating_map = {v: k for k, v in rating_map.items()}
        
        bonds_data = bonds_only.copy()
        bonds_data['Rating_Value'] = bonds_data['worst rating'].map(rating_map)
        
        if not bonds_data.empty and not bonds_data['Rating_Value'].isna().all():
            avg_rating_value = calculate_safe_weighted_average(
                bonds_data,
                'Rating_Value',
                'Poids en %'
            )
            avg_rating = reverse_rating_map.get(round(avg_rating_value), "-")
        else:
            avg_rating = "-"
        
        st.metric("Rating Moyen", avg_rating)
        
        # Montant cash
        cash_data = mandat_data[mandat_data['GENRE DU TITRE'] == 'Cash']
        cash_amount = cash_data['QUANTITE'].sum() if not cash_data.empty else 0
        cash_weight = cash_data['Poids en %'].sum() if not cash_data.empty else 0
        st.metric("Cash", f"{cash_amount:,.0f}€ ({cash_weight:.1f}%)")

        ca = get_safe_value(mandat_data, 'Ca', default_text="-")
        st.metric("Client Account", str(ca))
    
    with col3:
        # Ratio HY en utilisant les poids rebasés
        hy_weight = bonds_only[(~bonds_only['worst rating'].apply(is_investment_grade)) & (bonds_only['worst rating'] != 'Cash')]['Poids en %'].sum() if not bonds_only.empty else 0
        st.metric("High Yield", f"{hy_weight:.1f}%")
        
        # Nombre de lignes
        num_bonds = len(bonds_only)
        st.metric("Nombre de Lignes", num_bonds)
        
        # Total AUM
        total_aum = mandat_data['Montant Valorise'].sum() if 'Montant Valorise' in mandat_data.columns else 0
        st.metric("AUM Total", f"{total_aum:,.0f}€")
    
    # Analyses graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en barres horizontales des secteurs
        sector_data = mandat_data[mandat_data['INDUSTRY_SECTOR'].notna()].groupby('INDUSTRY_SECTOR')['Poids en %'].sum()
        if not sector_data.empty:
            sector_data = sector_data.sort_values(ascending=True)
            fig_sectors = px.bar(
                y=sector_data.index,
                x=sector_data.values,
                orientation='h',
                title='Répartition par Secteur'
            )
            
            fig_sectors.update_layout(
                xaxis_title="Poids (%)",
                yaxis_title="Secteur",
                showlegend=False,
                height=400
            )
            
            # Formater les valeurs en pourcentage
            fig_sectors.update_traces(
                texttemplate='%{x:.1f}%',
                textposition='outside'
            )
            
            st.plotly_chart(fig_sectors)
    
    with col2:
        # Graphique des payment ranks
        rank_data = mandat_data[mandat_data['payment rank'].notna()].groupby('payment rank')['Poids en %'].sum()
        if not rank_data.empty:
            fig_ranks = px.pie(
                values=rank_data.values,
                names=rank_data.index,
                title='Répartition par Rang de Payment'
            )
            st.plotly_chart(fig_ranks)
    
    # Graphique en barres des notes de crédit
    st.subheader("Répartition par Rating")
    rating_data = bonds_only[bonds_only['worst rating'].notna()].groupby('worst rating')['Poids en %'].sum()
    
    # Trier les ratings selon l'ordre défini
    rating_order = sorted(rating_data.index, key=lambda x: all_ratings.get(x, float('inf')) if x != 'Cash' else float('inf'))
    rating_data = rating_data.reindex(rating_order)
    
    if not rating_data.empty:
        # Créer un DataFrame avec les catégories IG/HY
        ratings_df = pd.DataFrame({
            'Rating': rating_data.index,
            'Poids': rating_data.values,
            'Catégorie': ['IG' if is_investment_grade(r) else 'HY' if r != 'Cash' else 'Cash' for r in rating_data.index]
        })
        
        fig_ratings = px.bar(
            ratings_df,
            x='Rating',
            y='Poids',
            color='Catégorie',
            title='Répartition par Rating',
            color_discrete_map={'IG': '#1a5f7a', 'HY': '#3498db', 'Cash': '#2ecc71'}
        )
        
        fig_ratings.update_layout(
            xaxis_title="Rating",
            yaxis_title="Poids (%)",
            showlegend=True,
            height=400,
            xaxis={'tickangle': 45}
        )
        
        # Formater les valeurs en pourcentage
        fig_ratings.update_traces(
            texttemplate='%{y:.1f}%',
            textposition='outside'
        )
        
        st.plotly_chart(fig_ratings)

    # Ajouter des métriques supplémentaires
    st.subheader("Métriques de Contribution")
    contrib_col1, contrib_col2 = st.columns(2)
    
    with contrib_col1:
        avg_yield_contrib = bonds_only['Contribution To Yield'].mean()
        st.metric("Contribution Moyenne au Yield", f"{avg_yield_contrib:.2f}%")
        
        # Afficher le nombre de CoCo
        coco_count = bonds_only[bonds_only['CoCo'] == 'Yes'].shape[0]
        coco_weight = bonds_only[bonds_only['CoCo'] == 'Yes']['Poids'].sum()
        st.metric("CoCo", f"{coco_count} titres ({coco_weight:.1f}%)")
    
    with contrib_col2:
        avg_dur_contrib = bonds_only['Contribution to duration'].mean()
        st.metric("Contribution Moyenne à la Duration", f"{avg_dur_contrib:.2f}")
        
        # Afficher le spread moyen
        avg_spread = bonds_only['YAS_ASW_SPREAD'].mean()
        st.metric("Spread Moyen", f"{avg_spread:.0f} bps")
    
    # Ajouter un graphique comparatif des ratings
    st.subheader("Comparaison des Ratings")
    ratings_col1, ratings_col2 = st.columns(2)
    
    with ratings_col1:
        # Graphique des ratings S&P vs Moody's
        ratings_comparison = bonds_only[['RTG_SP_NO_WATCH', 'RTG_MOODY_NO_WATCH', 'Poids']].copy()
        ratings_comparison = ratings_comparison.dropna()
        
        if not ratings_comparison.empty:
            fig_ratings = px.scatter(
                ratings_comparison,
                x='RTG_SP_NO_WATCH',
                y='RTG_MOODY_NO_WATCH',
                size='Poids',
                title='S&P vs Moody\'s',
                labels={'RTG_SP_NO_WATCH': 'S&P', 'RTG_MOODY_NO_WATCH': 'Moody\'s'},
                height=400
            )
            st.plotly_chart(fig_ratings)
    
    with ratings_col2:
        # Distribution des ratings LMDG
        lmdg_ratings = bonds_only['LMDG'].value_counts()
        if not lmdg_ratings.empty:
            fig_lmdg = px.pie(
                values=lmdg_ratings.values,
                names=lmdg_ratings.index,
                title='Distribution des Ratings LMDG'
            )
            st.plotly_chart(fig_lmdg)
    
    # Graphique des cash flows futurs
    st.subheader("Cash Flows Futurs")
    
    # Filtrer les obligations avec des dates de maturité
    bonds_with_dates = mandat_data[mandat_data['maturity'].notna()].copy()
    bonds_with_dates['Date Finale'] = bonds_with_dates.apply(
        lambda x: x['NXT_CALL_DT'] if pd.notna(x['NXT_CALL_DT']) else x['maturity'],
        axis=1
    )
    
    # Convertir les dates en datetime
    bonds_with_dates['Date Finale'] = pd.to_datetime(bonds_with_dates['Date Finale'])
    
    # Créer un DataFrame pour les cash flows
    cash_flows = []
    today = pd.Timestamp.now()
    
    for _, bond in bonds_with_dates.iterrows():
        # Calculer les dates de paiement des coupons
        if pd.notna(bond['cpn']):
            date_range = pd.date_range(
                start=today,
                end=bond['Date Finale'],
                freq='Y'  # Fréquence annuelle pour les coupons
            )
            
            # Ajouter les flux de coupons
            coupon_amount = (bond['cpn'] / 100) * bond['QUANTITE']
            for date in date_range:
                cash_flows.append({
                    'Date': date,
                    'Montant': coupon_amount,
                    'Type': 'Coupon',
                    'Bond': bond['name']
                })
        
        # Ajouter le remboursement du nominal
        cash_flows.append({
            'Date': bond['Date Finale'],
            'Montant': bond['QUANTITE'],
            'Type': 'Remboursement',
            'Bond': bond['name']
        })
    
    if cash_flows:
        cf_df = pd.DataFrame(cash_flows)
        
        # Extraire l'année de la date et regrouper par année et type
        cf_df['Année'] = cf_df['Date'].dt.year
        cf_agg = cf_df.groupby(['Année', 'Type'])['Montant'].sum().reset_index()
        
        # Créer le graphique
        fig_cf = px.bar(
            cf_agg,
            x='Année',
            y='Montant',
            color='Type',
            title=f'Cash Flows Futurs - {st.session_state.selected_mandat}',
            barmode='stack',
            color_discrete_map={
                'Coupon': '#3498db',
                'Remboursement': '#1a5f7a'
            },
            category_orders={"Type": ["Remboursement", "Coupon"]}  # Définir l'ordre des catégories
        )
        
        # Formater l'axe des années
        fig_cf.update_xaxes(
            tickangle=0,
            type='category',
            dtick=1
        )
        
        fig_cf.update_layout(
            xaxis_title="Année",
            yaxis_title="Montant (€)",
            showlegend=True,
            height=400,
            bargap=0
        )
        
        # Formater les montants en euros
        fig_cf.update_traces(
            texttemplate='%{y:,.0f}€',
            textposition='outside',
            width=0.8,  # Largeur des barres
            marker_line_color='rgb(8,48,107)',  # Bordure des barres
            marker_line_width=1.5  # Épaisseur de la bordure
        )
        
        # Ajouter une grille horizontale
        fig_cf.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(189,189,189,0.2)'
        )
        
        st.plotly_chart(fig_cf)
    else:
        st.info("Aucun cash flow futur à afficher pour ce mandat.")
    
    # Matrices dans un expander
    with st.expander("Voir les Matrices"):
        # Matrice Secteur x Duration
        st.subheader("Matrice Secteur x Duration")
        mandat_data['Duration_Range'] = pd.cut(
            pd.to_numeric(mandat_data['YAS_MOD_DUR'], errors='coerce'),
            bins=[0, 2, 4, 6, 8, 10, float('inf')],
            labels=['0-2', '2-4', '4-6', '6-8', '8-10', '10+']
        )
        
        matrix_data = mandat_data[
            (mandat_data['INDUSTRY_SECTOR'].notna()) & 
            (mandat_data['Duration_Range'].notna())
        ].pivot_table(
            values='Poids en %',
            index='INDUSTRY_SECTOR',
            columns='Duration_Range',
            aggfunc='sum',
            fill_value=0
        ).round(2)
        
        if not matrix_data.empty:
            fig_matrix = px.imshow(
                matrix_data,
                labels=dict(x="Duration", y="Secteur", color="Poids (%)"),
                title=f"Matrice Secteur x Duration - {st.session_state.selected_mandat}",
                aspect="auto",
                color_continuous_scale=[[0, 'rgb(220,240,255)'], [1, 'rgb(8,48,107)']]  # Bleu clair à bleu foncé
            )
            fig_matrix.update_traces(text=matrix_data.values, texttemplate="%{text:.1f}%")
            st.plotly_chart(fig_matrix)
        
        # Matrice Country x Duration
        st.subheader("Matrice Country x Duration")
        matrix_data = mandat_data[
            (mandat_data['country'].notna()) & 
            (mandat_data['Duration_Range'].notna())
        ].pivot_table(
            values='Poids en %',
            index='country',
            columns='Duration_Range',
            aggfunc='sum',
            fill_value=0
        ).round(2)
        
        if not matrix_data.empty:
            fig_matrix = px.imshow(
                matrix_data,
                labels=dict(x="Duration", y="Country", color="Poids (%)"),
                title=f"Matrice Country x Duration - {st.session_state.selected_mandat}",
                aspect="auto",
                color_continuous_scale=[[0, 'rgb(220,240,255)'], [1, 'rgb(8,48,107)']]  # Bleu clair à bleu foncé
            )
            fig_matrix.update_traces(text=matrix_data.values, texttemplate="%{text:.1f}%")
            st.plotly_chart(fig_matrix)
    
    st.write("")  # Ajouter un espacement
    
    # Liste des titres à la fin
    st.subheader("Liste des Titres")
    
    # Préparer les données des titres
    bonds_df = mandat_data[[
        'PORTEFEUILLE', 'ISIN', 'QUANTITE', 'Date', 'name', 'security name', 'cpn', 'INDUSTRY_SECTOR', 'currency', 'maturity', 'is perpetual', 'payment rank', 'TICKER', 'country', 'Contribution To Yield', 'Contribution to duration', 'YAS_MOD_DUR', 'YAS_ASW_SPREAD', 'NXT_CALL_DT', 'Final Maturity', 'YAS_BOND_YLD', 'Yield', 'Moody\'s', 'S&P', 'Fitch', 'LMDG', 'rating class', 'worst rating', 'Worst Raitng F', 'Px_Last', 'Poids', 'CouponContribution', 'CoCo', 'Avg Rating Portefeuille', 'Référencement', 'Univers Investissable'
    ]].copy()
    
    # Calculer les contributions
    bonds_df['Contrib_Duration'] = bonds_df['YAS_MOD_DUR'] * bonds_df['Poids'] / 100
    bonds_df['Contrib_Yield'] = bonds_df['YAS_BOND_YLD'] * bonds_df['Poids'] / 100
    
    # Calculer le RPAP (Ratio Pondéré Apport Performance)
    bonds_df['RPAP'] = bonds_df.apply(
        lambda row: row['Contrib_Duration'] / row['Contrib_Yield'] if row['Contrib_Yield'] != 0 else 0,
        axis=1
    )
    
    # Ajouter les lignes de cash
    cash_df = mandat_data[mandat_data['name'] == 'Cash'][[
        'PORTEFEUILLE', 'ISIN', 'QUANTITE', 'Date', 'name', 'security name', 'cpn', 'INDUSTRY_SECTOR', 
        'currency', 'maturity', 'is perpetual', 'payment rank', 'TICKER', 'country', 'Contribution To Yield', 
        'Contribution to duration', 'YAS_MOD_DUR', 'YAS_ASW_SPREAD', 'NXT_CALL_DT', 'Final Maturity', 
        'YAS_BOND_YLD', 'Yield', 'Moody\'s', 'S&P', 'Fitch', 'LMDG', 'rating class', 'worst rating', 
        'Worst Raitng F', 'Px_Last', 'Poids', 'CouponContribution', 'CoCo', 'Avg Rating Portefeuille', 
        'Référencement', 'Univers Investissable'
    ]].copy()
    
    cash_df['worst rating'] = 'Cash'
    cash_df['YAS_BOND_YLD'] = 0
    cash_df['YAS_MOD_DUR'] = 0
    cash_df['cpn'] = ''
    cash_df['Px_Last'] = 100
    cash_df['Contrib_Duration'] = 0
    cash_df['Contrib_Yield'] = 0
    cash_df['RPAP'] = 0
    
    # Combiner les données
    bonds_df = pd.concat([bonds_df, cash_df])
    
    # Filtrer pour n'avoir que les obligations avec RPAP valide
    rpap_data = bonds_df[
        (bonds_df['RPAP'] > 0) & 
        (bonds_df['Contrib_Duration'] != 0) & 
        (bonds_df['Contrib_Yield'] != 0)
    ].copy()
    
    # Fonction de formatage sécurisée pour les valeurs numériques
    def format_numeric(x, format_str, suffix=''):
        try:
            return format_str.format(float(x)) + suffix
        except (ValueError, TypeError):
            return ""
    
    # Formater les colonnes
    bonds_df['YAS_BOND_YLD'] = bonds_df['YAS_BOND_YLD'].round(2).map('{:.2f}%'.format)
    bonds_df['YAS_MOD_DUR'] = bonds_df['YAS_MOD_DUR'].round(2)
    bonds_df['Poids'] = bonds_df['Poids'].round(2).map('{:.2f}%'.format)
    bonds_df['QUANTITE'] = bonds_df['QUANTITE'].map('{:,.0f}'.format)
    bonds_df['Px_Last'] = bonds_df['Px_Last'].round(3).map('{:.3f}'.format)
    bonds_df['cpn'] = bonds_df['cpn'].apply(lambda x: format_numeric(x, '{:.2f}', '%'))
    bonds_df['Contrib_Duration'] = bonds_df['Contrib_Duration'].round(3)
    bonds_df['Contrib_Yield'] = bonds_df['Contrib_Yield'].round(3).map('{:.3f}%'.format)
    bonds_df['RPAP'] = bonds_df['RPAP'].round(2)
    
    # Afficher le tableau des titres
    st.dataframe(
        bonds_df,
        column_config={
            "PORTEFEUILLE": "Mandat",
            "ISIN": "ISIN",
            "QUANTITE": "Quantité",
            "Date": "Date",
            "name": "Nom",
            "security name": "Nom Complet",
            "cpn": "Coupon",
            "INDUSTRY_SECTOR": "Secteur",
            "currency": "Devise",
            "maturity": "Maturité",
            "is perpetual": "Perpétuelle",
            "payment rank": "Rang",
            "TICKER": "Ticker",
            "country": "Pays",
            "Contribution To Yield": "Contrib. Yield",
            "Contribution to duration": "Contrib. Duration",
            "YAS_MOD_DUR": "Duration",
            "YAS_ASW_SPREAD": "ASW Spread",
            "NXT_CALL_DT": "Prochaine Call",
            "Final Maturity": "Maturité Finale",
            "YAS_BOND_YLD": "YTM",
            "Yield": "Yield",
            "Moody's": "Moody's",
            "S&P": "S&P",
            "Fitch": "Fitch",
            "LMDG": "LMDG",
            "rating class": "Classe Rating",
            "worst rating": "Rating",
            "Worst Raitng F": "Pire Rating",
            "Px_Last": "Prix",
            "Poids": "Poids",
            "CouponContribution": "Contrib. Coupon",
            "CoCo": "CoCo",
            "Avg Rating Portefeuille": "Rating Moyen",
            "Référencement": "Référencement",
            "Univers Investissable": "Univers Invest."
        },
        hide_index=True
    )
    
    # Analyse du RPAP dans un expander
    with st.expander("Analyse du RPAP (Rapport Contributions Duration/Yield)", expanded=False):
        # Ajouter l'explication du RPAP
        st.info("""
        **Interprétation du RPAP :**
        - Un RPAP de 1 signifie que la position contribue autant à la duration qu'au yield
        - Un RPAP < 1 signifie que la position contribue plus au yield qu'à la duration (plus efficient)
        - Un RPAP > 1 signifie que la position contribue plus à la duration qu'au yield (moins efficient)
        """)
        
        if not rpap_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # S'assurer que les colonnes sont numériques
                rpap_data['RPAP'] = pd.to_numeric(rpap_data['RPAP'], errors='coerce')
                rpap_data['Poids'] = pd.to_numeric(rpap_data['Poids'].str.rstrip('%'), errors='coerce') if rpap_data['Poids'].dtype == 'object' else rpap_data['Poids']
                
                # Filtrer les valeurs non-numériques
                rpap_data = rpap_data.dropna(subset=['RPAP', 'Poids'])
                
                # Recalculer les poids pour avoir un total de 100%
                total_weight = rpap_data['Poids'].sum()
                rpap_data['Poids'] = rpap_data['Poids'] / total_weight * 100
                
                # Calculer les bins pour l'histogramme
                n_bins = 20
                hist_range = [rpap_data['RPAP'].min(), rpap_data['RPAP'].max()]
                bins = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
                
                # Calculer l'histogramme pondéré
                hist_weights, bin_edges = np.histogram(
                    rpap_data['RPAP'],
                    bins=bins,
                    weights=rpap_data['Poids']
                )
                
                # Créer un DataFrame pour le graphique
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                hist_df = pd.DataFrame({
                    'RPAP': bin_centers,
                    'Poids': hist_weights
                })
                
                # Créer le graphique en barres
                fig_rpap = px.bar(
                    hist_df,
                    x='RPAP',
                    y='Poids',
                    title='Distribution des RPAP (% du portefeuille obligataire)',
                    color_discrete_sequence=['#3498db']
                )
                
                fig_rpap.update_layout(
                    xaxis_title="RPAP (contribution duration / contribution yield)",
                    yaxis_title="Poids des Obligations (%)",
                    showlegend=False,
                    height=400,
                    bargap=0
                )
                
                # Calculer la moyenne et la médiane pondérées
                mean_rpap = np.average(rpap_data['RPAP'], weights=rpap_data['Poids'])
                median_rpap = weighted_median(rpap_data['RPAP'].values, rpap_data['Poids'].values)
                
                # Ajouter des lignes verticales pour la moyenne et la médiane
                fig_rpap.add_vline(
                    x=mean_rpap,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Moyenne: {mean_rpap:.2f}"
                )
                
                fig_rpap.add_vline(
                    x=median_rpap,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Médiane: {median_rpap:.2f}"
                )
                
                st.plotly_chart(fig_rpap)
            
            with col2:
                # Box plot par rating
                # Trier les ratings selon l'ordre défini
                rating_order = sorted(rpap_data['worst rating'].unique(), 
                                   key=lambda x: all_ratings.get(x, float('inf')) if x != 'Cash' else float('inf'))
                
                rpap_data['worst rating'] = pd.Categorical(
                    rpap_data['worst rating'],
                    categories=rating_order,
                    ordered=True
                )
                
                fig_box = px.box(
                    rpap_data,
                    x='worst rating',
                    y='RPAP',
                    title='Distribution des RPAP par Rating',
                    category_orders={'worst rating': rating_order}
                )
                
                fig_box.update_layout(
                    xaxis_title="Rating",
                    yaxis_title="RPAP",
                    showlegend=False,
                    xaxis={'tickangle': 45}
                )
                
                st.plotly_chart(fig_box)
            
            # Statistiques descriptives
            st.write("Statistiques descriptives du RPAP :")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Minimum", f"{rpap_data['RPAP'].min():.2f}")
            with col2:
                st.metric("Maximum", f"{rpap_data['RPAP'].max():.2f}")
            with col3:
                st.metric("Moyenne", f"{mean_rpap:.2f}")
            with col4:
                st.metric("Médiane", f"{median_rpap:.2f}")

def show_pivot_table(df):
    """Affiche un tableau croisé dynamique interactif."""
    st.title("Tableau Croisé Dynamique")
    
    if df is None:
        st.warning("Veuillez d'abord charger un fichier Excel.")
        return
    
    # Filtres
    st.subheader("Filtres")
    col1, col2 = st.columns(2)
    
    with col1:
        # Filtre ISIN
        selected_isins = st.multiselect(
            "Filtrer par ISIN",
            options=sorted(df['ISIN'].dropna().unique()),
            placeholder="Sélectionner un ou plusieurs ISIN"
        )
    
    with col2:
        # Filtre Mandat
        selected_mandats = st.multiselect(
            "Filtrer par Mandat",
            options=sorted(df['PORTEFEUILLE'].unique()),
            placeholder="Sélectionner un ou plusieurs mandats"
        )
    
    # Appliquer les filtres
    filtered_df = df.copy()
    if selected_isins:
        filtered_df = filtered_df[filtered_df['ISIN'].isin(selected_isins)]
    if selected_mandats:
        filtered_df = filtered_df[filtered_df['PORTEFEUILLE'].isin(selected_mandats)]
        
    # Obtenir toutes les colonnes numériques
    numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    # Obtenir toutes les colonnes catégorielles
    categorical_columns = filtered_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Sélecteurs pour les dimensions du tableau croisé
    st.subheader("Configuration du tableau")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        index = st.multiselect(
            "Lignes",
            options=categorical_columns,
            default=categorical_columns[0] if categorical_columns else None
        )
    
    with col2:
        columns = st.multiselect(
            "Colonnes",
            options=categorical_columns,
            default=categorical_columns[1] if len(categorical_columns) > 1 else None
        )
    
    with col3:
        values = st.multiselect(
            "Valeurs à sommer",
            options=numeric_columns,
            default=numeric_columns[0] if numeric_columns else None
        )
    
    # Créer le tableau croisé si tous les paramètres sont sélectionnés
    if index and values:
        try:
            pivot_table = pd.pivot_table(
                filtered_df,
                values=values,
                index=index,
                columns=columns if columns else None,
                aggfunc='sum',  # Toujours utiliser la somme
                fill_value=0
            )
            
            # Afficher le tableau
            st.subheader("Résultats")
            st.dataframe(
                pivot_table,
                use_container_width=True,
                height=500
            )
            
            # Option pour télécharger le tableau
            csv = pivot_table.to_csv()
            st.download_button(
                label="Télécharger le tableau (CSV)",
                data=csv,
                file_name="tableau_croise.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Erreur lors de la création du tableau croisé : {str(e)}")
    else:
        st.info("Veuillez sélectionner au moins une ligne et une valeur pour créer le tableau croisé.")

def show_documentation():
    st.title("Documentation")
    
    # Initialiser l'état de la langue si nécessaire
    if 'doc_language' not in st.session_state:
        st.session_state.doc_language = "Français"
    
    # Créer deux colonnes pour les boutons
    col1, col2 = st.columns(2)
    
    # Boutons pour changer de langue
    with col1:
        if st.button("🇫🇷 Français"):
            st.session_state.doc_language = "Français"
    
    with col2:
        if st.button("🇬🇧 English"):
            st.session_state.doc_language = "English"
    
    # Afficher la documentation depuis le cache
    st.markdown(DOCUMENTATION_FR if st.session_state.doc_language == "Français" else DOCUMENTATION_EN)

def show_formulas():
    st.title("Formules de Calcul / Calculation Formulas")
    
    with st.expander("🔢 Notations Moyennes / Average Ratings", expanded=True):
        st.markdown("""
        ### Moyenne des Notations / Average Ratings
        
        La moyenne des notations est calculée en :
        1. Convertissant chaque notation en valeur numérique (AAA=1, AA+=2, etc.)
        2. Calculant la moyenne pondérée par le poids de chaque obligation
        3. Reconvertissant le résultat en notation

        ```python
        rating_numeric = weighted_average(ratings_converted, weights)
        rating_final = convert_numeric_to_rating(round(rating_numeric))
        ```

        ### Rating Médian / Median Rating
        
        Le rating médian est calculé en utilisant la médiane pondérée :
        ```python
        rating_median = weighted_median(ratings_converted, weights)
        ```
        """)

    with st.expander("📊 Yield et Duration / Yield and Duration", expanded=True):
        st.markdown("""
        ### Yield du Portefeuille / Portfolio Yield
        
        Le yield du portefeuille est la somme des contributions au yield :
        ```python
        # Contribution au yield pour chaque obligation
        yield_contrib = yield * weight
        
        # Yield du portefeuille
        portfolio_yield = sum(yield_contrib)  # équivalent à sum(yields * weights)
        ```

        ### Duration du Portefeuille / Portfolio Duration
        
        La duration du portefeuille est la somme des contributions à la duration :
        ```python
        # Contribution à la duration pour chaque obligation
        duration_contrib = duration * weight
        
        # Duration du portefeuille
        portfolio_duration = sum(duration_contrib)  # équivalent à sum(durations * weights)
        ```

        ### Contribution au Yield / Yield Contribution
        
        La contribution au yield pour chaque obligation :
        ```python
        yield_contrib = yield * weight
        ```

        ### Contribution à la Duration / Duration Contribution
        
        La contribution à la duration pour chaque obligation :
        ```python
        duration_contrib = duration * weight
        ```
        """)

    with st.expander("📈 RPAP (Ratio Performance/Risque)", expanded=True):
        st.markdown("""
        ### RPAP (Ratio Duration/Yield)
        
        Le RPAP mesure l'efficience entre la contribution à la duration et au yield :
        ```python
        rpap = duration_contribution / yield_contribution
        ```

        Interprétation :
        - RPAP = 1 : contribution égale à la duration et au yield
        - RPAP < 1 : meilleure efficience (plus de contribution au yield qu'à la duration)
        - RPAP > 1 : efficience moindre (plus de contribution à la duration qu'au yield)
        """)

    with st.expander("🎯 Statistiques Sectorielles / Sector Statistics", expanded=True):
        st.markdown("""
        ### Poids Sectoriel / Sector Weight
        
        Le poids de chaque secteur est calculé comme :
        ```python
        sector_weight = sum(weights_in_sector) / total_portfolio_weight
        ```

        ### Yield Moyen par Secteur / Average Yield by Sector
        
        Le yield moyen pour chaque secteur est une moyenne pondérée :
        ```python
        sector_yield = sum(yields_in_sector * weights_in_sector) / sum(weights_in_sector)
        ```

        ### Duration Moyenne par Secteur / Average Duration by Sector
        
        La duration moyenne pour chaque secteur est une moyenne pondérée :
        ```python
        sector_duration = sum(durations_in_sector * weights_in_sector) / sum(weights_in_sector)
        ```
        """)

def login_page():
    """Affiche la page de connexion."""
    st.title("Connexion")
    
    # Centrer le formulaire de connexion
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### Veuillez vous connecter pour accéder à l'application")
        if check_access_level():
            st.session_state["authenticated"] = True
            st.success("Connexion réussie!")
            st.rerun()
        
def main_page(df):
    """Affiche la page principale de l'application."""
    # Création des onglets en haut de la page
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Vue Globale",
        "Vue Détaillée",
        "Tableau Croisé",
        "Formules de Calcul",
        "Documentation"
    ])
    
    with tab1:
        show_global_view(df)
    
    with tab2:
        show_detailed_view(df)
        
    with tab3:
        if st.session_state.access_level == "full":
            show_pivot_table(df)
        else:
            st.info("Cette fonctionnalité nécessite un accès complet.")
        
    with tab4:
        if st.session_state.access_level == "full":
            show_formulas()
        else:
            st.info("Cette fonctionnalité nécessite un accès complet.")
        
    with tab5:
        show_documentation()

def main():
    if not check_access_level():
        st.error("Veuillez entrer un mot de passe valide")
        return

    df = load_data()
    if df is not None:
        tabs = st.tabs(get_menu_options())
        
        with tabs[0]:  # Vue Globale
            show_global_view(df)
            
        with tabs[1]:  # Vue Détaillée
            show_detailed_view(df)
            
        if st.session_state.access_level == "full":
            with tabs[2]:  # Tableau Croisé
                show_pivot_table(df)
                
            with tabs[3]:  # Documentation
                show_documentation()
                
            with tabs[4]:  # Formules
                show_formulas()

if __name__ == "__main__":
    main()