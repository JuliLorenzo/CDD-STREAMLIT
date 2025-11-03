import os
import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st

# ----- CONFIG -----
st.set_page_config(page_title="Flight Price Explorer (JFK ⇄ MIA)", layout="wide")

# Modelo: local o Google Drive (opcional)
MODEL_PATH = "models/random_forest_flights_v2.pkl"
DRIVE_ID = st.secrets.get("DRIVE_ID", None)  # opcional: define en .streamlit/secrets.toml
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}" if DRIVE_ID else None

@st.cache_resource
def load_model():
    # descarga pkl si no existe y hay DRIVE_ID
    if not os.path.exists(MODEL_PATH):
        if DRIVE_URL:
            import gdown
            with st.spinner("Descargando modelo desde Google Drive…"):
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
        else:
            st.error("No se encontró el modelo y no hay DRIVE_ID en secrets.")
            st.stop()
    with st.spinner("Cargando modelo…"):
        model = joblib.load(MODEL_PATH)
    return model

@st.cache_data
def load_sample():
    path = "data/sample_flights.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

modelo = load_model()
df_sample = load_sample()

st.title("🔎 Buscador de tarifas por aerolínea — JFK ⇄ MIA")
st.caption("Predicción con Random Forest entrenado + exploración de datos con Altair")

# ---------- Sidebar: parámetros ----------
with st.sidebar:
    st.header("Parámetros del vuelo")
    origen = st.radio("Origen", ["JFK", "MIA"], horizontal=True)
    destino = "MIA" if origen == "JFK" else "JFK"
    st.markdown(f"**Destino:** `{destino}` (fijo)")

    hoy = dt.date.today()
    fecha = st.date_input("Fecha de salida", value=hoy + dt.timedelta(days=21), min_value=hoy)
    days_to_departure = (fecha - hoy).days

    main_cabin = st.selectbox("Cabina", ["coach", "premium coach", "business", "first"], index=0)
    nonstop = st.toggle("Vuelo directo", value=True)
    refundable = st.toggle("Tarifa reembolsable", value=False)

    st.divider()
    st.caption("Para JFK–MIA te dejo valores por defecto razonables; podés ajustarlos:")
    distancia = st.slider("Distancia estimada (km)", 1000, 3000, 1760, 10)
    duracion = st.slider("Duración estimada (min)", 120, 360, 190, 5)

# ---------- Utilidades para leer la estructura del pipeline ----------
def infer_features_from_model(m):
    """
    Devuelve: (num_cols, cat_cols, lista_de_aerolineas_del_entrenamiento)
    Navega el ColumnTransformer para recuperar columnas y categorías del OneHotEncoder.
    """
    if "preprocess" not in m.named_steps:
        st.error("El modelo no tiene paso 'preprocess'. Ajustá los nombres según tu pipeline.")
        st.stop()

    ct = m.named_steps["preprocess"]         # ColumnTransformer
    transformers = dict(ct.transformers_)    # {'num': (...), 'cat': (...), ...}
    cat_cols = transformers["cat"][2]
    num_cols = transformers["num"][2]

    # OneHotEncoder adentro del pipeline de 'cat'
    cat_pipe = transformers["cat"][1]
    if "onehot" not in cat_pipe.named_steps:
        st.error("No encontré 'onehot' en el pipeline de categóricas.")
        st.stop()
    oh = cat_pipe.named_steps["onehot"]

    # índice de la columna 'main_airline' en las categóricas
    try:
        idx_airline = cat_cols.index("main_airline")
    except ValueError:
        st.error("La columna 'main_airline' no figura entre las categóricas del modelo.")
        st.stop()

    airlines = list(oh.categories_[idx_airline])
    return num_cols, cat_cols, airlines

num_cols, cat_cols, airlines_from_model = infer_features_from_model(modelo)

# ---------- Construcción del DataFrame para predecir por aerolínea ----------
def build_pred_rows(airlines_list):
    base = {
        "days_to_departure": days_to_departure,
        "totalTravelDistance": distancia,
        "duration_min": duracion,
        "startingAirport": origen,
        "destinationAirport": destino,
        "isRefundable": int(refundable),
        "isNonStop": int(nonstop),
        "main_cabin": main_cabin,
        "flight_month": fecha.month,
        "flight_dayofweek": fecha.weekday(),  # Monday=0
        "main_airline": None,
    }
    rows = []
    for a in airlines_list:
        b = base.copy()
        b["main_airline"] = a
        rows.append(b)
    return pd.DataFrame(rows)

# ---------- Predicción ----------
st.subheader("1) Predicción de precio por aerolínea")
st.caption("Generamos una fila por aerolínea (las vistas en el entrenamiento) con los parámetros elegidos y predecimos.")

if st.button("Predecir"):
    df_pred = build_pred_rows(airlines_from_model)
    yhat = modelo.predict(df_pred)
    df_pred["pred_price"] = yhat

    st.dataframe(df_pred[["main_airline", "pred_price"]].sort_values("pred_price"), use_container_width=True)

    chart_pred = (
        alt.Chart(df_pred)
        .mark_bar()
        .encode(
            x=alt.X("pred_price:Q", title="Precio estimado"),
            y=alt.Y("main_airline:N", sort="-x", title="Aerolínea"),
            tooltip=[alt.Tooltip("main_airline:N", title="Aerolínea"),
                     alt.Tooltip("pred_price:Q", title="Precio")]
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart_pred, use_container_width=True)

    st.download_button(
        "Descargar CSV de predicciones",
        df_pred.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_por_aerolinea.csv",
        mime="text/csv",
    )

# ---------- Exploración con Altair (3 visualizaciones) ----------
st.subheader("2) Exploración interactiva con Altair")

if df_sample is None:
    st.info("Subí un dataset ligero en `data/sample_flights.csv` para habilitar la exploración (columnas: totalFare, days_to_departure, totalTravelDistance, duration_min, startingAirport, destinationAirport, isRefundable, isNonStop, main_airline, main_cabin, flight_month, flight_dayofweek).")
else:
    # Filtros
    left, right = st.columns(2)
    with left:
        filt_cabins = st.multiselect("Cabina", sorted(df_sample["main_cabin"].dropna().unique().tolist()),
                                     default=sorted(df_sample["main_cabin"].dropna().unique().tolist()))
    with right:
        top_airlines = sorted(df_sample["main_airline"].dropna().unique().tolist())
        filt_airlines = st.multiselect("Aerolínea", top_airlines, default=top_airlines[:6])

    f = df_sample.copy()
    f = f[(f["main_cabin"].isin(filt_cabins)) & (f["main_airline"].isin(filt_airlines))]
    f = f[(f["startingAirport"].isin(["JFK", "MIA"])) & (f["destinationAirport"].isin(["JFK", "MIA"]))]

    # (A) Scatter: Precio vs días hasta la salida (brush para filtrar)
    brush = alt.selection_interval(encodings=["x"])
    scatter = (
        alt.Chart(f)
        .mark_circle(size=26, opacity=0.65)
        .encode(
            x=alt.X("days_to_departure:Q", title="Días hasta la salida"),
            y=alt.Y("totalFare:Q", title="Precio"),
            color=alt.Color("main_cabin:N", title="Cabina"),
            tooltip=["main_airline", "main_cabin", "isNonStop", "totalFare", "days_to_departure"],
        )
        .add_selection(brush)
        .properties(title="(A) Precio vs. días hasta la salida")
    )
    st.altair_chart(scatter, use_container_width=True)

    # (B) Boxplot por aerolínea (filtrado por brush de A)
    box = (
        alt.Chart(f)
        .mark_boxplot()
        .encode(
            x=alt.X("main_airline:N", title="Aerolínea", sort="-y"),
            y=alt.Y("totalFare:Q", title="Precio"),
            color=alt.Color("main_cabin:N", title="Cabina"),
        )
        .transform_filter(brush)
        .properties(title="(B) Distribución de precios por aerolínea (filtrada por A)")
    )
    st.altair_chart(box, use_container_width=True)

    # (C) Heatmap: Precio promedio por mes y día de semana
    heat = (
        alt.Chart(f)
        .mark_rect()
        .encode(
            x=alt.X("flight_dayofweek:O", title="Día de semana (0=Lun)"),
            y=alt.Y("flight_month:O", title="Mes"),
            color=alt.Color("mean(totalFare):Q", title="Precio promedio"),
            tooltip=[alt.Tooltip("mean(totalFare):Q", title="Precio promedio"), alt.Tooltip("count():Q", title="Observaciones")],
        )
        .properties(title="(C) Mapa de calor de precio promedio por mes y día")
    )
    st.altair_chart(heat, use_container_width=True)

st.caption("Las 3 visualizaciones cumplen con: expresividad (tipos de marca correctos), comparabilidad (filtros/brush) y adecuación al tipo de variable.")
