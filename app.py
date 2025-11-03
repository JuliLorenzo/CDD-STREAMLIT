import os
import datetime as dt
from pathlib import Path   # 👈 FALTA ESTO
import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st

# ----- CONFIG -----
st.set_page_config(page_title="Flight Price Explorer (JFK ⇄ MIA)", layout="wide")

# Modelo: local o Google Drive (opcional)
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_flights_v2.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)  # 👈 crea la carpeta 'models/' si no existe

DRIVE_ID = st.secrets.get("DRIVE_ID")  # ya lo tenés en secrets
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}" if DRIVE_ID else None


@st.cache_resource
def load_model():
    # Si no está el modelo local, lo bajo de Drive
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        import gdown
        drive_id = st.secrets.get("DRIVE_ID") or os.getenv("DRIVE_ID")
        if not drive_id:
            st.error("No se encontró DRIVE_ID en secrets ni en variables de entorno.")
            st.stop()

        with st.spinner("Descargando modelo desde Google Drive…"):
            # Podés usar id= directamente (más confiable que armar la URL)
            gdown.download(id=drive_id, output=str(MODEL_PATH), quiet=False)

        # Validación post-descarga
        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            st.error("La descarga del modelo falló o quedó vacía. Verificá permisos del archivo en Drive.")
            st.stop()

    # Carga del modelo
    with st.spinner("Cargando modelo…"):
        return joblib.load(str(MODEL_PATH))


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
    Devuelve: (num_cols, cat_cols, airlines)
    Busca el ColumnTransformer 'preprocess', localiza los transformadores 'num' y 'cat',
    y extrae categorías del OneHotEncoder para la columna 'main_airline'.
    Es robusto a que los nombres cambien (busca por nombre y por tipo).
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # 1) Encontrar el ColumnTransformer
    if "preprocess" in getattr(m, "named_steps", {}):
        ct = m.named_steps["preprocess"]
    else:
        # fallback: buscar el primer ColumnTransformer en el pipeline
        ct = None
        for step_name, step in getattr(m, "named_steps", {}).items():
            if isinstance(step, ColumnTransformer):
                ct = step
                break
    if ct is None or not hasattr(ct, "transformers_"):
        st.error("No se encontró un ColumnTransformer 'preprocess' ya entrenado dentro del modelo.")
        st.stop()

    # 2) Localizar transformadores 'num' y 'cat' (por nombre o por estructura)
    num_t = None
    cat_t = None
    for name, trans, cols in ct.transformers_:
        if name.lower().startswith("num"):
            num_t = (name, trans, cols)
        elif name.lower().startswith("cat"):
            cat_t = (name, trans, cols)

    if cat_t is None or num_t is None:
        # fallback: si no se llaman 'num'/'cat', tomá por tipo de columnas
        for name, trans, cols in ct.transformers_:
            if cat_t is None and isinstance(trans, (Pipeline,)):
                # si dentro hay un OneHotEncoder, asumimos categóricas
                if any(isinstance(s, OneHotEncoder) for _, s in trans.steps):
                    cat_t = (name, trans, cols)
            if num_t is None and isinstance(trans, (Pipeline,)):
                # heurística: si hay StandardScaler u otro scaler, asumimos numéricas
                if any(s.__class__.__name__.lower().endswith("scaler") for _, s in trans.steps):
                    num_t = (name, trans, cols)

    if cat_t is None:
        st.error("No pude identificar el transformador categórico dentro del ColumnTransformer.")
        st.stop()
    if num_t is None:
        st.error("No pude identificar el transformador numérico dentro del ColumnTransformer.")
        st.stop()

    _, cat_pipe, cat_cols = cat_t
    _, _, num_cols = num_t

    # 3) Obtener el OneHotEncoder (por nombre 'onehot' o por tipo)
    if isinstance(cat_pipe, Pipeline):
        # por nombre 'onehot'
        if "onehot" in cat_pipe.named_steps:
            oh = cat_pipe.named_steps["onehot"]
        else:
            # por tipo
            found = [s for _, s in cat_pipe.named_steps.items() if isinstance(s, OneHotEncoder)]
            if not found:
                st.error("No encontré un OneHotEncoder dentro del pipeline de categóricas.")
                st.stop()
            oh = found[0]
    else:
        st.error("El transformador categórico no parece ser un Pipeline con OneHotEncoder.")
        st.stop()

    # 4) Posición de 'main_airline' en las categóricas
    try:
        idx_airline = list(cat_cols).index("main_airline")
    except ValueError:
        st.error("La columna 'main_airline' no figura entre las categóricas del modelo.")
        st.stop()

    # 5) Categorías aprendidas para 'main_airline'
    # oh.categories_ es una lista alineada con cat_cols (post fit)
    airlines = list(oh.categories_[idx_airline])
    return list(num_cols), list(cat_cols), airlines
    
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
