import ast
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

st.set_page_config(page_title="Gastronomie", layout="wide")

# Clean les listes
def clean_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except Exception:
        return []

# Convertit les dates en début de mois
def convert_to_month(dt_series):
    dt = pd.to_datetime(dt_series, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()

# Pour la normalisation
def min_max(s):
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

# Calcule la pente linéaire de tendance
def linear_trend(y):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3 or np.all(np.isnan(y)):
        return 0.0
    t = np.arange(n).reshape(-1, 1)
    y2 = np.nan_to_num(y, nan=np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0)
    model = LinearRegression()
    model.fit(t, y2)
    return float(model.coef_[0])

# Graphique
def plot_line(df, x, y, title):
    fig = plt.figure()
    plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Affiche les données avec cache (@st.cache_data)
@st.cache_data
def load_data(interactions_path, recipes_path):
    interactions = pd.read_csv(interactions_path)
    recipes = pd.read_csv(recipes_path)

    interactions["date"] = pd.to_datetime(interactions["date"], errors="coerce")
    interactions = interactions.dropna(subset=["date"])

    recipes["ingredients_list"] = recipes["ingredients"].apply(clean_list)
    recipes["tags_list"] = recipes["tags"].apply(clean_list)
    recipes["submitted"] = pd.to_datetime(recipes["submitted"], errors="coerce")

    # Jointure avec le csv des recettes
    merged_csv = interactions.merge(
        recipes,
        left_on="recipe_id",
        right_on="id",
        how="left",
        suffixes=("_i", "_r")
    )

    merged_csv["month"] = convert_to_month(merged_csv["date"])
    return interactions, recipes, merged_csv

############################ Barre menu de l'application ############################
st.sidebar.title("Gastronomie")
page = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Tendances recettes", "Tendances ingrédients", "Prévisions", "Attentes consommateurs"]
)

data_mode = st.sidebar.radio("Source des données", ["Dossier data/", "Ajouter des fichiers"])

if data_mode == "Dossier data/":
    interactions_path = "data/RAW_interactions.csv"
    recipes_path = "data/RAW_recipes.csv"
    try:
        interactions, recipes, merged_csv = load_data(interactions_path, recipes_path)
    except Exception as e:
        st.error("Impossible de charger les fichiers depuis data/. Vérifiez qu'ils existent.")
        st.exception(e)
        st.stop()
else:
    up_i = st.sidebar.file_uploader("RAW_interactions.csv", type=["csv"])
    up_r = st.sidebar.file_uploader("RAW_recipes.csv", type=["csv"])
    if not up_i or not up_r:
        st.info("Ajoute les deux fichiers CSV pour continuer.")
        st.stop()
    interactions = pd.read_csv(up_i)
    recipes = pd.read_csv(up_r)

    interactions["date"] = pd.to_datetime(interactions["date"], errors="coerce")
    interactions = interactions.dropna(subset=["date"])
    recipes["ingredients_list"] = recipes["ingredients"].apply(clean_list)
    recipes["tags_list"] = recipes["tags"].apply(clean_list)
    recipes["submitted"] = pd.to_datetime(recipes["submitted"], errors="coerce")

    merged_csv = interactions.merge(recipes, left_on="recipe_id", right_on="id", how="left")
    merged_csv["month"] = convert_to_month(merged_csv["date"])

# Filtre
st.sidebar.subheader("Filtres")
min_month = merged_csv["month"].min()
max_month = merged_csv["month"].max()
start, end = st.sidebar.slider(
    "Période",
    min_value=min_month.to_pydatetime(),
    max_value=max_month.to_pydatetime(),
    value=(min_month.to_pydatetime(), max_month.to_pydatetime())
)
mask = (merged_csv["month"] >= pd.to_datetime(start)) & (merged_csv["month"] <= pd.to_datetime(end))
df = merged_csv.loc[mask].copy()

min_inter = st.sidebar.number_input("Min interactions par recette", min_value=1, value=30, step=10)

recipe_stats = (
    df.groupby(["recipe_id", "name"], dropna=False)
      .agg(
          n_interactions=("rating", "size"),
          avg_rating=("rating", "mean"),
          n_users=("user_id", "nunique"),
      )
      .reset_index()
)

recipe_stats = recipe_stats[recipe_stats["n_interactions"] >= min_inter].copy()

recipe_stats["score_volume"] = min_max(recipe_stats["n_interactions"])
recipe_stats["score_rating"] = min_max(recipe_stats["avg_rating"].fillna(recipe_stats["avg_rating"].mean()))
recipe_stats["popularity_score"] = 0.7 * recipe_stats["score_volume"] + 0.3 * recipe_stats["score_rating"]

recipe_month = (
    df.groupby(["recipe_id", "name", "month"])
      .agg(n=("rating", "size"), r=("rating", "mean"))
      .reset_index()
)
recipe_month["score"] = 0.7 * min_max(recipe_month["n"]) + 0.3 * min_max(recipe_month["r"].fillna(recipe_month["r"].mean()))

############################ Pages de l'application ############################
if page == "Accueil":
    st.title("Accueil")

    col1, col2, col3 = st.columns(3)
    col1.metric("Interactions", f"{len(df):,}".replace(",", " "))
    col2.metric("Recettes", f"{df['recipe_id'].nunique():,}".replace(",", " "))
    col3.metric("Utilisateurs", f"{df['user_id'].nunique():,}".replace(",", " "))

    # Tendance générale
    global_month = df.groupby("month").agg(
        interactions=("rating", "size"),
        avg_rating=("rating", "mean")
    ).reset_index()

    st.subheader("Tendance générale")
    c1, c2 = st.columns(2)
    with c1:
        plot_line(global_month, "month", "interactions", "Volume d’interactions par mois")
    with c2:
        plot_line(global_month, "month", "avg_rating", "Rating moyen par mois")

    st.subheader("Top 10 recettes populaires")
    top = recipe_stats.sort_values("popularity_score", ascending=False).head(10)
    st.dataframe(top[["recipe_id", "name", "n_interactions", "avg_rating", "popularity_score"]], width="stretch")

elif page == "Tendances recettes":
    st.title("Tendances recettes")

    st.subheader("Recettes les plus populaires")
    t = recipe_stats.sort_values("n_interactions", ascending=False).head(20)
    st.dataframe(t[["recipe_id", "name", "n_interactions", "avg_rating"]], width="stretch")

    st.subheader("Recettes avec les meilleures notes")
    t = recipe_stats.sort_values("avg_rating", ascending=False).head(20)
    st.dataframe(t[["recipe_id", "name", "avg_rating", "n_interactions"]], width="stretch")

elif page == "Tendances ingrédients":
    st.title("Tendances ingrédients")

    ing_df = df[["month", "ingredients_list"]].dropna()
    exploded = ing_df.explode("ingredients_list").rename(columns={"ingredients_list": "ingredient"})
    exploded["ingredient"] = exploded["ingredient"].astype(str).str.strip().str.lower()
    exploded = exploded[exploded["ingredient"].str.len() > 0]

    ing_month = exploded.groupby(["ingredient", "month"]).size().reset_index(name="count")

    totals = ing_month.groupby("ingredient")["count"].sum().reset_index()
    keep = totals[totals["count"] >= 200]["ingredient"]  # ajuste le seuil si besoin
    ing_month2 = ing_month[ing_month["ingredient"].isin(keep)].copy()

    months_sorted = sorted(ing_month2["month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}
    ing_month2["t"] = ing_month2["month"].map(month_to_idx)

    slopes = []
    for ing, g in ing_month2.groupby("ingredient"):
        series = g.set_index("t")["count"]
        full = np.zeros(len(months_sorted))
        full[series.index.values] = series.values
        slopes.append((ing, linear_trend(full), full.sum()))

    slopes_df = pd.DataFrame(slopes, columns=["ingredient", "slope", "total_count"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ingrédients en hausse")
        up = slopes_df.sort_values("slope", ascending=False).head(20)
        st.dataframe(up, width="stretch")

    with col2:
        st.subheader("Ingrédients en baisse")
        down = slopes_df.sort_values("slope", ascending=True).head(20)
        st.dataframe(down, width="stretch")

    st.subheader("Visualiser un ingrédient")
    picked = st.selectbox("Choisir un ingrédient", options=sorted(slopes_df["ingredient"].unique()))
    g = ing_month[ing_month["ingredient"] == picked].sort_values("month")
    plot_line(g, "month", "count", f"Occurrences : {picked}")

elif page == "Prévisions":
    st.title("Prévisions : prochain plat et ingrédients à la mode")

    # Predictions recettes
    candidates = []
    for (rid, name), g in recipe_month.groupby(["recipe_id", "name"]):
        g2 = g.sort_values("month")
        if len(g2) < 6:
            continue
        t = np.arange(len(g2)).reshape(-1, 1)
        y = g2["score"].values.astype(float)
        if np.isnan(y).all():
            continue
        model = LinearRegression()
        model.fit(t, np.nan_to_num(y, nan=np.nanmean(y)))
        next_score = float(model.predict([[len(g2)]])[0])
        slope = float(model.coef_[0])
        last_score = float(np.nanmean(y[-3:])) 
        candidates.append((rid, name, slope, last_score, next_score, len(g2), g2["n"].sum()))

    cand = pd.DataFrame(candidates, columns=[
        "recipe_id", "name", "trend_slope", "recent_score", "pred_next_score", "n_months", "total_interactions"
    ])

    if cand.empty:
        st.warning("Pas assez de données (recettes avec au moins 6 mois dans la période). Élargis la période.")
        st.stop()

    cand = cand[cand["total_interactions"] >= min_inter].copy()
    cand["rank"] = 0.6 * min_max(cand["pred_next_score"]) + 0.4 * min_max(cand["trend_slope"])
    cand = cand.sort_values("rank", ascending=False)

    st.subheader("Prédiction du top 20 recettes")
    st.dataframe(cand.head(20), width="stretch")

    best = cand.iloc[0]
    st.subheader("Recette la plus probable d’être à la mode")
    st.write(f"**{best['name']}** (recipe_id={int(best['recipe_id'])})")
    st.write(f"- Pente (croissance): `{best['trend_slope']:.4f}`")
    st.write(f"- Score récent: `{best['recent_score']:.3f}`")
    st.write(f"- Score prédit prochain mois: `{best['pred_next_score']:.3f}`")

    gbest = recipe_month[recipe_month["recipe_id"] == best["recipe_id"]].sort_values("month")
    plot_line(gbest, "month", "score", f"{best['name']}")

    rec_row = recipes[recipes["id"] == best["recipe_id"]]
    if len(rec_row) > 0:
        ing = rec_row.iloc[0]["ingredients_list"][:25]
        tags = rec_row.iloc[0]["tags_list"][:25]
        st.write("**Ingrédients** :", ", ".join(map(str, ing)) if ing else "N/A")

    # Predictions ingrédients
    st.subheader("Prédiction des ingrédients en hausse et en baisse")
    ing_df = df[["month", "ingredients_list"]].dropna()
    exploded = ing_df.explode("ingredients_list").rename(columns={"ingredients_list": "ingredient"})
    exploded["ingredient"] = exploded["ingredient"].astype(str).str.strip().str.lower()
    exploded = exploded[exploded["ingredient"].str.len() > 0]
    ing_month = exploded.groupby(["ingredient", "month"]).size().reset_index(name="count")

    totals = ing_month.groupby("ingredient")["count"].sum().reset_index()
    keep = totals[totals["count"] >= 200]["ingredient"]
    ing_month = ing_month[ing_month["ingredient"].isin(keep)].copy()

    months_sorted = sorted(ing_month["month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}

    preds = []
    for ing, g in ing_month.groupby("ingredient"):
        g = g.sort_values("month")
        full = np.zeros(len(months_sorted))
        idxs = g["month"].map(month_to_idx).values
        full[idxs] = g["count"].values
        t = np.arange(len(full)).reshape(-1, 1)
        model = LinearRegression().fit(t, full)
        next_count = float(model.predict([[len(full)]])[0])
        preds.append((ing, float(model.coef_[0]), full.sum(), next_count))

    preds_df = pd.DataFrame(preds, columns=["ingredient", "slope", "total", "pred_next_count"])
    preds_df["rank_up"] = 0.5 * min_max(preds_df["pred_next_count"]) + 0.5 * min_max(preds_df["slope"])
    preds_df["rank_down"] = 0.5 * min_max(-preds_df["pred_next_count"]) + 0.5 * min_max(-preds_df["slope"])

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Prévision des ingrédients en hausse**")
        st.dataframe(preds_df.sort_values("rank_up", ascending=False).head(15), width="stretch")
    with c2:
        st.write("**Prévision des ingrédients en baisse**")
        st.dataframe(preds_df.sort_values("rank_down", ascending=False).head(15), width="stretch")

elif page == "Attentes consommateurs":
    st.title("Attentes consommateurs à venir (Topic Modeling sur les reviews)")

    df_reviews = df.dropna(subset=["review"]).copy()
    if df_reviews.empty:
        st.warning("Aucune review textuelle sur la période sélectionnée.")
        st.stop()

    # configurable
    n_topics = st.slider("Nombre de topics", 3, 12, 6)
    
    max_reviews = st.slider("Max reviews à analyser", 5000, 50000, 10000, step=5000)
    
    if len(df_reviews) > max_reviews:
        st.info(f"Échantillonnage de {max_reviews:,} reviews sur {len(df_reviews):,} pour la performance.")
        df_reviews = df_reviews.sample(n=max_reviews, random_state=42)

    # Extraire les topics avec LDA
    @st.cache_data
    def topic_modeling(texts_tuple, n_topics, max_features=5000):
        texts = list(texts_tuple)
        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=max_features,
            min_df=5,
            max_df=0.95
        )
        X = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method="online",
            max_iter=5,
            n_jobs=-1
        )
        doc_topics = lda.fit_transform(X)
        vocab = vectorizer.get_feature_names_out()
        return doc_topics, lda.components_, vocab

    texts = df_reviews["review"].astype(str).str.lower().tolist()
    
    with st.spinner("Analyse des topics en cours (peut prendre 10-30 sec)..."):
        doc_topics, lda_components, vocab = topic_modeling(tuple(texts), n_topics)

    vocab = np.array(vocab)

    st.subheader("Topics principaux (mots-clés)")
    
    topic_labels = {}
    for k in range(n_topics):
        top_idx = np.argsort(lda_components[k])[::-1][:10]
        words = vocab[top_idx]
        topic_labels[k] = ", ".join(words)
        st.write(f"**Topic {k+1}:** {topic_labels[k]}")

    st.subheader("Mots-clés par topic")
    selected_topic = st.selectbox(
        "Choisir un topic à visualiser",
        options=list(range(n_topics)),
        format_func=lambda x: f"Topic {x+1}"
    )
    
    top_n = 12
    top_idx = np.argsort(lda_components[selected_topic])[::-1][:top_n]
    top_words = vocab[top_idx]
    top_weights = lda_components[selected_topic][top_idx]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(top_n), top_weights[::-1], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_words[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Topic {selected_topic+1} - mots-clés les plus importants")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Évolution des topics")
    
    df_reviews["month"] = convert_to_month(df_reviews["date"])
    topic_cols = [f"topic_{i+1}" for i in range(n_topics)]
    topic_df = pd.DataFrame(doc_topics, columns=topic_cols)
    topic_df["month"] = df_reviews["month"].values

    monthly = topic_df.groupby("month")[topic_cols].mean().reset_index()

    # Calcul des tendances topics
    trends = []
    for col in topic_cols:
        slope = linear_trend(monthly[col].values)
        trends.append((col, slope))
    trends_sorted = sorted(trends, key=lambda x: x[1], reverse=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Topics en hausse**")
        for col, slope in trends_sorted[:3]:
            st.write(f"{col}: `{slope:.5f}`")
    with col2:
        st.write("**Topics en baisse**")
        for col, slope in trends_sorted[-3:]:
            st.write(f"{col}: `{slope:.5f}`")

    chosen = st.multiselect("Sélectionne les topics", topic_cols, default=topic_cols[:min(3, len(topic_cols))])
    if chosen:
        fig = plt.figure(figsize=(10, 5))
        for c in chosen:
            plt.plot(monthly["month"], monthly[c], label=c)
        plt.title("Topics")
        plt.xlabel("month")
        plt.ylabel("importance")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Sélectionne au moins un topic.")

    st.subheader("Exemples de reviews par topic")
    
    example_topic = st.selectbox(
        "Voir des exemples pour le topic",
        options=list(range(n_topics)),
        format_func=lambda x: f"Topic {x+1}",
        key="example_topic_select"
    )
    
    topic_scores = doc_topics[:, example_topic]
    top_indices = np.argsort(topic_scores)[::-1][:5]
    
    for i, idx in enumerate(top_indices, 1):
        review_text = str(df_reviews.iloc[idx]["review"])
        review_text = review_text[:300] + "..." if len(review_text) > 300 else review_text
        rating = df_reviews.iloc[idx]["rating"]
        st.write(f"**Review {i}** (rating: {rating:.0f}/5)")
        st.caption(review_text)
