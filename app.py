# app.py
import os
import math
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st

# =======================
# CONFIG
# =======================
st.set_page_config(page_title="Flight Price Explorer (JFK ⇄ MIA)", layout="wide")

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
MODEL_PATH = BASE_DIR / "models" / "random_forest_flights_v2.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

DRIVE_ID = st.secrets.get("DRIVE_ID", None)  # definido en .streamlit/secrets.toml
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}" if DRIVE_ID else None

# Orígenes permitidos en la UI (en tu caso trabajás con JFK/MIA)
ORIGINS = ["JFK", "MIA"]

# Destinos permitidos (incluye MIA/JFK para ida y vuelta)
DESTS = ["ATL","BOS","CLT","DEN","DFW","DTW","IAD","LAX","MIA","OAK","ORD","PHL","SFO","EWR","JFK","LGA"]

# Etiquetas de cabina para la UI (tu modelo v2 se entrenó sólo con coach; no afecta la predicción)
CABINS = {
    "Turista (Economy)": "coach",
    "Premium Economy": "premium coach",
    "Business": "business",
    "First": "first",
}

# Coordenadas aproximadas (lat, lon) para cálculo de distancia (haversine)
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277),
    "BOS": (42.3656, -71.0096),
    "CLT": (35.2140, -80.9431),
    "DEN": (39.8561, -104.6737),
    "DFW": (32.8998, -97.0403),
    "DTW": (42.2124, -83.3534),
    "IAD": (38.9531, -77.4565),
    "LAX": (33.9416, -118.4085),
    "MIA": (25.7959, -80.2871),
    "OAK": (37.7126, -122.2197),
    "ORD": (41.9742, -87.9073),
    "PHL": (39.8744, -75.2424),
    "SFO": (37.6213, -122.3790),
    "EWR": (40.6895, -74.1745),
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
}

# =======================
# Fallbacks para versiones de Streamlit
# =======================
def segmented_or_radio(label, options, default):
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options=options, default=default)
    idx = options.index(default) if default in options else 0
    return st.radio(label, options, index=idx, horizontal=True)

def toggle_or_checkbox(label, value=False):
    if hasattr(st, "toggle"):
        return st.toggle(label, value=value)
    return st.checkbox(label, value=value)

def primary_button(label):
    try:
        return st.button(label, type="primary")
    except TypeError:
        return st.button(label)

# =======================
# Helpers
# =======================
@st.cache_resource
def load_model():
    """Carga el modelo local; si no existe y hay DRIVE_ID, lo descarga de Google Drive."""
    if (not MODEL_PATH.exists()) or MODEL_PATH.stat().st_size == 0:
        if DRIVE_URL:
            try:
                import gdown
            except Exception:
                st.error(
                    "No se encontró 'gdown'. Agregá 'gdown' a requirements.txt para descargar el modelo desde Drive."
                )
                raise
            with st.spinner("Descargando modelo desde Google Drive..."):
                gdown.download(DRIVE_URL, str(MODEL_PATH), quiet=False)
        else:
            st.warning("No hay modelo local y no se definió DRIVE_ID en secrets. "
                       "Cargá el pkl en /models o seteá DRIVE_ID.")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo desde {MODEL_PATH}.\nDetalle: {e}")
        raise

def haversine_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def route_distance_km(origin: str, dest: str) -> float:
    if origin not in AIRPORT_COORDS or dest not in AIRPORT_COORDS:
        return np.nan
    (lat1, lon1) = AIRPORT_COORDS[origin]
    (lat2, lon2) = AIRPORT_COORDS[dest]
    return haversine_distance_km(lat1, lon1, lat2, lon2)

def estimate_duration_min(distance_km: float, nonstop: bool = True) -> int:
    if np.isnan(distance_km):
        return 120
    hours = distance_km / 800.0        # ~800 km/h
    base_min = hours * 60 + 40         # taxi + seguridad
    penalty = 0 if nonstop else 60     # escala
    return int(round(base_min + penalty))

# === introspección del modelo para saber columnas y categorías ===
def infer_features_from_model(m):
    """
    Devuelve: (num_cols, cat_cols, airlines, dow_categories)
    """
    if "preprocess" not in m.named_steps:
        st.error("El modelo no tiene paso 'preprocess'.")
        st.stop()

    ct = m.named_steps["preprocess"]
    transformers = dict(ct.transformers_)
    cat_cols = transformers["cat"][2]
    num_cols = transformers["num"][2]

    cat_pipe = transformers["cat"][1]
    oh = cat_pipe.named_steps["onehot"]

    # categorías de aerolíneas
    idx_airline = cat_cols.index("main_airline")
    airlines = list(oh.categories_[idx_airline])

    # categorías de flight_dayofweek (¿'Mon'..'Sun' o 0..6?)
    idx_dow = cat_cols.index("flight_dayofweek")
    dow_categories = list(oh.categories_[idx_dow])

    return num_cols, cat_cols, airlines, dow_categories

def normalize_dow_value(date: dt.date, dow_categories):
    """
    Devuelve flight_dayofweek en el mismo formato que el entrenamiento.
    Si fue strings ('Mon'), devolvemos eso; si fue 0..6, devolvemos int.
    """
    sample = dow_categories[0]
    if isinstance(sample, str):
        return date.strftime("%a")  # 'Mon', 'Tue', ...
    return date.weekday()           # 0..6

# === construir exactamente la fila de features que el modelo espera ===
def build_features(origin, dest, flight_date, cabin_value, airline_code, is_refundable, nonstop,
                   num_cols, cat_cols, dow_categories):
    # Nota: cabin_value NO se usa porque el modelo v2 se entrenó sólo con coach
    today = dt.date.today()
    days_to_departure = max((flight_date - today).days, 0)

    distance = route_distance_km(origin, dest)
    duration = estimate_duration_min(distance, nonstop=nonstop)

    flight_month = flight_date.month
    flight_day = flight_date.day
    flight_dayofweek = normalize_dow_value(flight_date, dow_categories)
    is_weekend = int(flight_date.weekday() in (5, 6))
    is_holiday_season = int(flight_month in (6, 7))  # regla usada en el training

    row = {
        "days_to_departure": days_to_departure,
        "totalTravelDistance": distance,
        "duration_min": duration,
        "startingAirport": origin,
        "destinationAirport": dest,
        "isRefundable": int(is_refundable),
        "isNonStop": int(nonstop),
        "main_airline": airline_code,
        "flight_month": flight_month,
        "flight_dayofweek": flight_dayofweek,
        "flight_day": flight_day,
        "is_weekend": is_weekend,
        "is_holiday_season": is_holiday_season,
    }
    X = pd.DataFrame([row])

    # Orden exacto de columnas que espera el ColumnTransformer
    expected = list(num_cols) + list(cat_cols)
    missing = [c for c in expected if c not in X.columns]
    if missing:
        st.error(f"Faltan columnas para el modelo: {missing}")
        st.stop()
    return X[expected]

def predict(model, X: pd.DataFrame) -> float:
    y = model.predict(X)
    return float(y[0])

def format_currency(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

# =======================
# Cargar modelo y leer su “firma”
# =======================
model = load_model()
num_cols, cat_cols, airlines_from_model, dow_categories = infer_features_from_model(model)

# =======================
# UI
# =======================
st.title("Vuelos")

trip_type = segmented_or_radio(
    "Tipo de viaje",
    options=["Ida y vuelta", "Solo ida"],
    default="Ida y vuelta",
)

c1, c2, c3, c4 = st.columns([1, 2, 1.2, 1.2])

with c1:
    origin = st.selectbox("Desde", ORIGINS, index=0)

with c2:
    dest_options = [d for d in DESTS if d != origin]
    default_dest_index = dest_options.index("MIA") if "MIA" in dest_options else 0
    dest = st.selectbox("¿A dónde quieres ir?", dest_options, index=default_dest_index)

with c3:
    dep_date = st.date_input("Salida", value=dt.date.today() + dt.timedelta(days=14), min_value=dt.date.today())

with c4:
    if trip_type == "Ida y vuelta":
        ret_date = st.date_input("Regreso", value=dt.date.today() + dt.timedelta(days=21), min_value=dep_date)
    else:
        ret_date = None
        st.write(" ")

c5, c6, c7, c8 = st.columns([1.2, 1.2, 1, 1])
with c5:
    cabin_label = st.selectbox("Cabina (visual)", list(CABINS.keys()), index=0)
    cabin_value = CABINS[cabin_label]  # no afecta la predicción en v2
with c6:
    airline_code = st.selectbox("Aerolínea (según entrenamiento)", options=sorted(airlines_from_model))
with c7:
    nonstop = toggle_or_checkbox("Solo vuelos directos", value=True)
with c8:
    is_refundable = toggle_or_checkbox("Reembolsable", value=False)

st.caption("⚠️ En este modelo v2, la cabina no participa de la predicción (se entrenó con 'coach').")

cta = primary_button("Explorar")

if cta:
    try:
        # Ida
        X_out = build_features(
            origin=origin,
            dest=dest,
            flight_date=dep_date,
            cabin_value=cabin_value,
            airline_code=airline_code,
            is_refundable=is_refundable,
            nonstop=nonstop,
            num_cols=num_cols,
            cat_cols=cat_cols,
            dow_categories=dow_categories,
        )
        price_out = predict(model, X_out)

        # Vuelta (opcional)
        total_price = price_out
        if trip_type == "Ida y vuelta" and ret_date is not None:
            X_back = build_features(
                origin=dest,
                dest=origin,
                flight_date=ret_date,
                cabin_value=cabin_value,
                airline_code=airline_code,
                is_refundable=is_refundable,
                nonstop=nonstop,
                num_cols=num_cols,
                cat_cols=cat_cols,
                dow_categories=dow_categories,
            )
            price_back = predict(model, X_back)
            total_price += price_back
        else:
            price_back = None

        # Métricas
        st.subheader("Precio estimado")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Ida", format_currency(price_out))
        with mc2:
            st.metric("Vuelta", format_currency(price_back) if price_back is not None else "—")
        with mc3:
            st.metric("Total", format_currency(total_price))

        st.divider()

        # Viz 1: Sensibilidad a la anticipación (días)
        st.markdown("#### Cómo cambia el precio según la anticipación")
        days_range = np.arange(1, 181)  # 1..180 días
        X_sens = pd.concat([
            build_features(
                origin=origin,
                dest=dest,
                flight_date=dt.date.today() + dt.timedelta(days=int(d)),
                cabin_value=cabin_value,
                airline_code=airline_code,
                is_refundable=is_refundable,
                nonstop=nonstop,
                num_cols=num_cols,
                cat_cols=cat_cols,
                dow_categories=dow_categories,
            )
            for d in days_range
        ], ignore_index=True)
        y_sens = model.predict(X_sens)
        df_sens = pd.DataFrame({"days_to_departure": days_range, "pred_price": y_sens})
        chart1 = (
            alt.Chart(df_sens)
            .mark_line()
            .encode(
                x=alt.X("days_to_departure:Q", title="Días hasta la salida"),
                y=alt.Y("pred_price:Q", title="Precio estimado (USD)"),
                tooltip=["days_to_departure", alt.Tooltip("pred_price:Q", format="$.0f")],
            )
            .properties(height=280)
            .interactive()
        )
        st.altair_chart(chart1, use_container_width=True)

        st.divider()

        # Viz 2: Comparativa por aerolínea (misma ruta/fecha)
        st.markdown("#### Comparar por aerolínea (misma ruta y fecha)")
        airlines_to_compare = airlines_from_model  # podés limitar si son muchas
        X_air = pd.concat([
            build_features(
                origin=origin,
                dest=dest,
                flight_date=dep_date,
                cabin_value=cabin_value,
                airline_code=a,
                is_refundable=is_refundable,
                nonstop=nonstop,
                num_cols=num_cols,
                cat_cols=cat_cols,
                dow_categories=dow_categories,
            ).assign(main_airline=a)
            for a in airlines_to_compare
        ], ignore_index=True)
        y_air = model.predict(X_air.drop(columns=["main_airline"]))
        df_air = pd.DataFrame({"Aerolínea": X_air["main_airline"], "Precio": y_air})
        chart2 = (
            alt.Chart(df_air)
            .mark_bar()
            .encode(
                x=alt.X("Precio:Q", title="Precio estimado (USD)"),
                y=alt.Y("Aerolínea:N", sort="-x"),
                tooltip=[alt.Tooltip("Precio:Q", format="$.0f")],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(chart2, use_container_width=True)

        st.info("ℹ️ Estimaciones basadas en un modelo entrenado. Distancia y duración son aproximadas (demo).")
    except Exception as e:
        st.exception(e)
else:
    st.caption("Elegí ruta, fechas y opciones. Luego presioná **Explorar** para estimar precios.")
