# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import importlib
import pickle
import math
from pathlib import Path


st.set_page_config(
    page_title="ALDIMI Predict - Dashboard Pro",
    layout="wide",
    page_icon="🏥"
)


st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: #1f2937;
        }

        .stMetric {
            background-color: white;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.06);
        }

        div[data-testid="stMetricValue"] {
            font-size: 28px;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] {
            background-color: #0f172a;
        }

        section[data-testid="stSidebar"] * {
            color: white;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: white;
            border-radius: 10px 10px 0px 0px;
            padding: 10px 18px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #dbeafe;
            color: #1e40af;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# MANEJO DE RUTAS
# =====================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(".").resolve()

PROJECT_DIR = BASE_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


import database
importlib.reload(database)

from database import (
    cargar_tabla,
    cargar_pacientes,
    cargar_inventario,
    cargar_inventario_completo,
    cargar_features_logisticas,
    insertar_registro,
    actualizar_registro,
    eliminar_registro,
    agregar_csv_a_tabla,
    inicializar_bd_si_no_existe,
    contar_registros,
    contar_por_valor
)


# =====================================================
# MODELO PREDICTIVO
# =====================================================

MODEL_PATH = DATA_DIR / "modelo_clinico.pkl"
LOGISTIC_MODEL_PATH = DATA_DIR / "modelo_logistico.pkl"
LOGISTIC_DATA_PATH = DATA_DIR / "dataset_logistico_1.csv"


@st.cache_resource
def cargar_modelo_clinico():
    """
    Carga el modelo clinico entrenado.
    Puede ser:
    - modelo directo
    - diccionario con modelo y features
    - pipeline de sklearn
    """
    if not MODEL_PATH.exists():
        return None

    with open(MODEL_PATH, "rb") as file:
        payload = pickle.load(file)

    return payload


@st.cache_resource
def cargar_modelo_logistico():
    if not LOGISTIC_MODEL_PATH.exists():
        return None

    with open(LOGISTIC_MODEL_PATH, "rb") as file:
        payload = pickle.load(file)

    return payload


@st.cache_data(ttl=300)
def cargar_historico_logistico():
    df_features_bd = cargar_features_logisticas(limit=50000)

    if not df_features_bd.empty:
        return df_features_bd

    return pd.DataFrame()


def extraer_modelo_y_features(payload):
    """
    Intenta reconocer el formato del .pkl.
    """
    modelo = payload
    features = None

    if isinstance(payload, dict):
        for key in ["model", "modelo", "pipeline", "classifier", "clf", "estimator"]:
            if key in payload:
                modelo = payload[key]
                break

        for key in ["features", "feature_names", "columnas", "feature_columns", "columns"]:
            if key in payload:
                features = payload[key]
                break

    if features is None and hasattr(modelo, "feature_names_in_"):
        features = list(modelo.feature_names_in_)

    if features is not None:
        features = list(features)

    return modelo, features


def preparar_features_logisticas(df_historico, features):
    df_features = df_historico.copy()

    equivalencias = {
        "product_id_cod": ["product_id", "ID del producto"],
        "store_id_cod": ["store_id", "ID de tienda"],
        "first_category_id_cod": ["first_category_id", "primer_id_de_categoria"],
        "second_category_id_cod": ["second_category_id", "segundo_id_categoria"],
        "third_category_id_cod": ["third_category_id", "tercer_id_categoria"],
        "discount": ["discount", "descuento"],
        "holiday_flag_cod": ["holiday_flag", "bandera de vacaciones"],
        "activity_flag_cod": ["activity_flag", "bandera de actividad"],
    }

    for feature, origenes in equivalencias.items():
        if feature not in features or feature in df_features.columns:
            continue

        for origen in origenes:
            if origen in df_features.columns:
                df_features[feature] = df_features[origen]
                break

    for feature in features:
        if feature not in df_features.columns:
            df_features[feature] = 0

    X = df_features[features].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        mediana = X[col].median()
        if pd.isna(mediana):
            mediana = 0
        X[col] = X[col].fillna(mediana)

    return X


def normalizar_texto_producto(valor):
    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()
    reemplazos = {
        "\u00e1": "a",
        "\u00e9": "e",
        "\u00ed": "i",
        "\u00f3": "o",
        "\u00fa": "u",
        "\u00f1": "n",
    }

    for origen, destino in reemplazos.items():
        texto = texto.replace(origen, destino)

    texto = " ".join(texto.split())
    return texto


def predecir_demanda_logistica_por_producto(df_inventario):
    payload = cargar_modelo_logistico()
    historico = cargar_historico_logistico()

    if payload is None or historico.empty:
        return pd.DataFrame()

    modelo, features = extraer_modelo_y_features(payload)

    if features is None:
        return pd.DataFrame()

    if "dt" in historico.columns:
        historico = historico.sort_values("dt")

    if "name_products" not in historico.columns and "nombre_productos" in historico.columns:
        historico = historico.rename(columns={"nombre_productos": "name_products"})

    if "name_products" not in historico.columns:
        return pd.DataFrame()

    ultimos = historico.dropna(subset=["name_products"]).groupby("name_products", as_index=False).tail(1)
    X = preparar_features_logisticas(ultimos, features)
    ultimos = ultimos.copy()
    ultimos["Demanda_Predicha"] = modelo.predict(X)
    ultimos["Demanda_Predicha"] = ultimos["Demanda_Predicha"].clip(lower=0).round().astype(int)

    predicciones = ultimos[["name_products", "Demanda_Predicha"]].rename(
        columns={"name_products": "Producto"}
    )

    if "Producto" not in df_inventario.columns:
        return pd.DataFrame()

    inventario = df_inventario.copy()
    inventario["_producto_key"] = inventario["Producto"].apply(normalizar_texto_producto)
    predicciones["_producto_key"] = predicciones["Producto"].apply(normalizar_texto_producto)
    predicciones = predicciones.drop(columns=["Producto"])

    resultado = inventario.merge(predicciones, on="_producto_key", how="left")
    resultado = resultado.drop(columns=["_producto_key"])
    return resultado


def normalizar_prioridad(prediccion):
    """
    Convierte la salida del modelo a Alta / Media / Baja.
    """
    valor = prediccion

    try:
        valor = valor.item()
    except Exception:
        pass

    texto = str(valor).strip()

    mapa = {
        "0": "Baja",
        "1": "Media",
        "2": "Alta",
        "baja": "Baja",
        "media": "Media",
        "alta": "Alta",
        "low": "Baja",
        "medium": "Media",
        "high": "Alta"
    }

    return mapa.get(texto.lower(), texto)


def predecir_prioridad_paciente(data_paciente, df_clinico):
    """
    Predice la prioridad del paciente usando modelo_clinico.pkl.
    El usuario NO escribe la prioridad.
    """
    payload = cargar_modelo_clinico()

    if payload is None:
        raise ValueError(
            "No se encontro modelo_clinico.pkl en la carpeta data. "
            "Coloca el archivo en data/modelo_clinico.pkl."
        )

    modelo, features = extraer_modelo_y_features(payload)

    fila = pd.DataFrame([data_paciente])

    # Si el modelo tiene columnas esperadas, se respetan.
    if features is not None:
        for col in features:
            if col not in fila.columns:
                fila[col] = 0

        fila = fila[features]

    # Si no tiene columnas esperadas, se usan las columnas de la BD sin id ni prioridad.
    else:
        columnas_modelo = [
            col for col in df_clinico.columns
            if col not in ["id_registro", "Prioridad"]
            and "prioridad" not in col.lower()
        ]

        for col in columnas_modelo:
            if col not in fila.columns:
                fila[col] = 0

        fila = fila[columnas_modelo]

    prediccion = modelo.predict(fila)[0]

    return normalizar_prioridad(prediccion)


RECURSOS_PRIORIZADOS = [
    {
        "Producto": "Metotrexato 500mg",
        "Tipo": "Clinico",
        "Baja": 0,
        "Media": 1,
        "Alta": 2,
    },
    {
        "Producto": "Suero Fisiologico 500ml",
        "Tipo": "Clinico",
        "Baja": 1,
        "Media": 2,
        "Alta": 4,
    },
    {
        "Producto": "Cateter Endovenoso 24G",
        "Tipo": "Clinico",
        "Baja": 0,
        "Media": 1,
        "Alta": 2,
    },
    {
        "Producto": "Jeringa 5ml",
        "Tipo": "Clinico",
        "Baja": 1,
        "Media": 2,
        "Alta": 3,
    },
    {
        "Producto": "Ondansetron 8mg",
        "Tipo": "Clinico",
        "Baja": 0,
        "Media": 1,
        "Alta": 2,
    },
    {
        "Producto": "Dexametasona 4mg",
        "Tipo": "Clinico",
        "Baja": 0,
        "Media": 1,
        "Alta": 2,
    },
    {
        "Producto": "Pediasure Plus 200ml",
        "Tipo": "Clinico",
        "Baja": 1,
        "Media": 2,
        "Alta": 3,
    },
    {
        "Producto": "Huevo de gallina",
        "Tipo": "Alimento",
        "Baja": 2,
        "Media": 4,
        "Alta": 6,
    },
    {
        "Producto": "Leche evaporada",
        "Tipo": "Alimento",
        "Baja": 1,
        "Media": 3,
        "Alta": 5,
    },
    {
        "Producto": "Pechuga de pollo",
        "Tipo": "Alimento",
        "Baja": 1,
        "Media": 3,
        "Alta": 5,
    },
    {
        "Producto": "Arroz extra",
        "Tipo": "Alimento",
        "Baja": 2,
        "Media": 4,
        "Alta": 6,
    },
    {
        "Producto": "Avena",
        "Tipo": "Alimento",
        "Baja": 1,
        "Media": 3,
        "Alta": 5,
    },
    {
        "Producto": "Pescado bonito",
        "Tipo": "Alimento",
        "Baja": 1,
        "Media": 2,
        "Alta": 4,
    },
]


def obtener_valor_columna(fila, posibles, defecto="No disponible"):
    for col in posibles:
        if col in fila.index and not pd.isna(fila[col]):
            return fila[col]
    return defecto


def calcular_riesgo_integrado(demanda_total, stock_actual):
    if stock_actual <= 0:
        return "Alto"

    cobertura = stock_actual / demanda_total if demanda_total > 0 else 1

    if cobertura < 0.8:
        return "Alto"
    if cobertura < 1.15:
        return "Medio"
    return "Bajo"


def construir_recursos_integrados(prioridad, df_inventario):
    df_pred = predecir_demanda_logistica_por_producto(df_inventario)

    if df_pred.empty:
        return pd.DataFrame()

    df_pred = df_pred.copy()
    df_pred["_producto_key"] = df_pred["Producto"].apply(normalizar_texto_producto)

    filas = []

    for recurso in RECURSOS_PRIORIZADOS:
        demanda_adicional = recurso.get(prioridad, 0)

        if demanda_adicional <= 0:
            continue

        key = normalizar_texto_producto(recurso["Producto"])
        coincidencias = df_pred[df_pred["_producto_key"] == key]

        if coincidencias.empty:
            continue

        item = coincidencias.iloc[0]
        demanda_historica = convertir_a_numero(item.get("Demanda_Predicha", 0), 0)
        stock_actual = convertir_a_numero(item.get("Stock_Actual", 0), 0)
        demanda_total = demanda_historica + demanda_adicional

        filas.append({
            "Producto": item.get("Producto", recurso["Producto"]),
            "Tipo": recurso["Tipo"],
            "Demanda_Predicha": int(round(demanda_historica)),
            "Demanda_Adicional_Paciente": int(round(demanda_adicional)),
            "Demanda_Total_Esperada": int(round(demanda_total)),
            "Stock_Actual": int(round(stock_actual)),
            "Riesgo_Operativo": calcular_riesgo_integrado(demanda_total, stock_actual),
        })

    resultado = pd.DataFrame(filas)

    if resultado.empty:
        return resultado

    orden = {"Alto": 0, "Medio": 1, "Bajo": 2}
    resultado["_orden"] = resultado["Riesgo_Operativo"].map(orden)
    resultado = resultado.sort_values(
        ["_orden", "Demanda_Total_Esperada"],
        ascending=[True, False]
    )

    return resultado.drop(columns=["_orden"])


# =====================================================
# COLORES Y CONFIGURACIONES
# =====================================================

PRIORIDAD_COLORES = {
    "Alta": "#d32f2f",
    "Media": "#f9a825",
    "Baja": "#388e3c",
    "Alto": "#d32f2f",
    "Medio": "#f9a825",
    "Bajo": "#388e3c",
    "High": "#d32f2f",
    "Medium": "#f9a825",
    "Low": "#388e3c"
}

PRIORIDAD_ORDEN = ["Alta", "Media", "Baja"]

RANGOS_EDAD = {
    "0 - 17": 12,
    "18 - 30": 24,
    "31 - 45": 38,
    "46 - 60": 53,
    "61 - 75": 68,
    "76 a mas": 80
}

COLUMNAS_NUMERICAS_POR_NOMBRE = [
    "age",
    "edad",
    "income",
    "ingreso",
    "salary",
    "salario",
    "stock",
    "cantidad",
    "precio",
    "price",
    "costo",
    "cost",
    "volumen",
    "tasa",
    "rate",
    "rotacion",
    "months",
    "meses",
    "month",
    "year",
    "anio",
    "año",
    "survival",
    "reorden",
    "consumo",
    "actual",
    "nivel",
    "level",
    "score",
    "puntaje",
    "monto",
    "amount",
    "valor",
    "value",
    "days",
    "dias",
    "día"
]

COLUMNAS_TEXTO_LIBRE = [
    "producto",
    "descripcion",
    "description",
    "codigo",
    "code",
    "nombre",
    "name"
]


# =====================================================
# CARGA DE DATOS DESDE BD
# =====================================================

@st.cache_data(ttl=10)
def load_data():
    try:
        inicializar_bd_si_no_existe()

        df_c = cargar_pacientes()
        df_i = cargar_inventario_completo()

        return df_c, df_i

    except Exception as e:
        st.error(f"Error al cargar datos desde la base de datos: {e}")
        return None, None


# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

def limpiar_cache_y_recargar():
    load_data.clear()
    st.rerun()


def convertir_a_numero(valor, defecto=0.0):
    try:
        if pd.isna(valor):
            return defecto

        texto = str(valor)
        texto = texto.replace(",", "")
        texto = texto.replace("S/", "")
        texto = texto.replace("$", "")
        texto = texto.strip()

        if texto == "":
            return defecto

        return float(texto)

    except Exception:
        return defecto


def obtener_unicos_limpios(serie):
    valores = serie.dropna().astype(str).unique().tolist()
    valores = sorted(valores)

    if not valores:
        valores = ["Sin dato"]

    return valores


def es_columna_numerica_forzada(columna):
    nombre = columna.lower()

    for palabra in COLUMNAS_NUMERICAS_POR_NOMBRE:
        if palabra in nombre:
            return True

    return False


def es_columna_texto_libre(columna):
    nombre = columna.lower()

    for palabra in COLUMNAS_TEXTO_LIBRE:
        if palabra in nombre:
            return True

    return False


def es_columna_prioridad(columna):
    nombre = columna.lower()
    return columna == "Prioridad" or "prioridad" in nombre or "priority" in nombre


def es_columna_binaria(serie):
    try:
        valores = set(serie.dropna().unique().tolist())
        return valores.issubset({0, 1, 0.0, 1.0, True, False})
    except Exception:
        return False


def es_columna_binaria_flexible(serie):
    try:
        valores = serie.dropna().astype(str).str.strip().str.lower()
        if valores.empty:
            return False
        return set(valores.unique().tolist()).issubset({"0", "1", "0.0", "1.0", "true", "false"})
    except Exception:
        return False


def buscar_columna_edad(df):
    posibles = [
        "Age",
        "Edad",
        "edad",
        "AGE",
        "Patient_Age",
        "patient_age",
        "Edad_Paciente",
        "edad_paciente"
    ]

    for col in posibles:
        if col in df.columns:
            return col

    for col in df.columns:
        nombre = col.lower()
        if "edad" in nombre or "age" in nombre:
            return col

    return None


def buscar_columnas_edad_one_hot(df):
    columnas = []

    for col in df.columns:
        nombre = col.lower()

        if not (nombre.startswith("age_") or nombre.startswith("edad_")):
            continue

        if es_columna_binaria_flexible(df[col]):
            columnas.append(col)

    return columnas


def etiqueta_edad_one_hot(columna):
    texto = columna

    for prefijo in ["Age_", "age_", "Edad_", "edad_"]:
        if texto.startswith(prefijo):
            texto = texto.replace(prefijo, "", 1)

    texto = texto.replace("_", " ")
    texto = texto.replace("to", "a")
    texto = texto.replace("plus", "a mas")
    texto = texto.replace("mas", "a mas")
    texto = texto.replace("years", "")
    texto = texto.replace("year", "")

    return texto.strip()


def preparar_columna_edad_visual(df):
    df_vista = df.copy()
    columnas_edad = buscar_columnas_edad_one_hot(df_vista)

    if columnas_edad:
        col_visual = "Edad"
        df_vista[col_visual] = df_vista.apply(
            lambda fila: etiqueta_edad_one_hot(
                obtener_columna_activa_one_hot(fila, columnas_edad)
            ),
            axis=1
        )
        return df_vista, col_visual

    col_edad = buscar_columna_edad(df_vista)
    return df_vista, col_edad


def ordenar_edades_visual(valores):
    def clave(valor):
        texto = str(valor)
        numeros = []
        actual = ""

        for caracter in texto:
            if caracter.isdigit():
                actual += caracter
            elif actual:
                numeros.append(int(actual))
                actual = ""

        if actual:
            numeros.append(int(actual))

        if numeros:
            return numeros[0]

        return 999

    return sorted([valor for valor in valores if pd.notna(valor)], key=clave)


def obtener_rango_edad(valor):
    edad = convertir_a_numero(valor, None)

    if edad is None:
        return "Sin dato"

    if edad <= 17:
        return "0 - 17"
    elif edad <= 30:
        return "18 - 30"
    elif edad <= 45:
        return "31 - 45"
    elif edad <= 60:
        return "46 - 60"
    elif edad <= 75:
        return "61 - 75"
    else:
        return "76 a mas"


def detectar_grupos_one_hot(df):
    grupos = {}

    for col in df.columns:
        if col == "id_registro":
            continue

        if es_columna_prioridad(col):
            continue

        if "_" not in col:
            continue

        if not es_columna_binaria_flexible(df[col]):
            continue

        prefijo = col.split("_")[0]

        if prefijo not in grupos:
            grupos[prefijo] = []

        grupos[prefijo].append(col)

    grupos_finales = {}

    for prefijo, columnas in grupos.items():
        if len(columnas) > 1:
            grupos_finales[prefijo] = columnas

    return grupos_finales


def etiqueta_desde_columna_one_hot(prefijo, columna):
    if prefijo.lower() in ["age", "edad"]:
        return etiqueta_edad_one_hot(columna)

    texto = columna.replace(prefijo + "_", "")
    texto = texto.replace("_", " ")
    return texto


def obtener_columna_activa_one_hot(fila, columnas):
    for col in columnas:
        try:
            if int(float(fila[col])) == 1:
                return col
        except Exception:
            pass

    return columnas[0]


def obtener_opciones_columna(df, columna):
    opciones = df[columna].dropna().astype(str).unique().tolist()
    opciones = sorted(opciones)

    if not opciones:
        opciones = ["Sin dato"]

    return opciones


def nombre_visual_desde_prefijo(prefijo):
    nombres = {
        "Race": "Raza",
        "Gender": "Genero",
        "Marital": "Estado civil",
        "Grade": "Grado clinico",
        "Stage": "Estadio clinico",
        "T": "Categoria T",
        "N": "Categoria N",
        "M": "Categoria M",
        "Surgery": "Cirugia",
        "Radiation": "Radioterapia",
        "Chemotherapy": "Quimioterapia",
        "Age": "Rango de edad",
        "Edad": "Rango de edad",
    }

    return nombres.get(prefijo, prefijo.replace("_", " ").title())


def construir_vista_legible(df):
    df_vista = df.copy()
    grupos_one_hot = detectar_grupos_one_hot(df_vista)
    columnas_a_ocultar = []
    columnas_visuales_a_ocultar = {
        "Estado clinico",
        "Estadio clinico",
    }

    for prefijo, columnas_grupo in grupos_one_hot.items():
        columnas_validas = [col for col in columnas_grupo if col in df_vista.columns]

        if not columnas_validas:
            continue

        nombre_columna = nombre_visual_desde_prefijo(prefijo)
        df_vista[nombre_columna] = df_vista.apply(
            lambda fila: etiqueta_desde_columna_one_hot(
                prefijo,
                obtener_columna_activa_one_hot(fila, columnas_validas)
            ),
            axis=1
        )
        columnas_a_ocultar.extend(columnas_validas)

    columnas_a_ocultar = [
        col for col in columnas_a_ocultar
        if col in df_vista.columns and col not in ["id_registro", "Prioridad"]
    ]

    if columnas_a_ocultar:
        df_vista = df_vista.drop(columns=columnas_a_ocultar)

    columnas_visuales_presentes = [
        col for col in columnas_visuales_a_ocultar
        if col in df_vista.columns
    ]

    if columnas_visuales_presentes:
        df_vista = df_vista.drop(columns=columnas_visuales_presentes)

    for col in df_vista.columns:
        if col in ["id_registro", "Prioridad"]:
            continue

        if pd.api.types.is_numeric_dtype(df_vista[col]) and es_columna_binaria(df_vista[col]):
            df_vista[col] = df_vista[col].map({
                1: "Si",
                1.0: "Si",
                True: "Si",
                0: "No",
                0.0: "No",
                False: "No",
            }).fillna(df_vista[col])

    return df_vista


# =====================================================
# FORMULARIOS AMIGABLES
# =====================================================

def crear_input_amigable(df, columna, valor_actual=None, key=""):
    nombre = columna.lower()

    # EDAD COMO RANGO
    if "edad" in nombre or "age" in nombre:
        opciones = list(RANGOS_EDAD.keys())

        if valor_actual is not None and not pd.isna(valor_actual):
            edad_actual = convertir_a_numero(valor_actual, 24)
            rango_actual = obtener_rango_edad(edad_actual)
            index = opciones.index(rango_actual) if rango_actual in opciones else 1
        else:
            index = 1

        rango_elegido = st.selectbox(
            columna + " - rango",
            opciones,
            index=index,
            key=key
        )

        return RANGOS_EDAD[rango_elegido]

    # BINARIOS 0/1 COMO SI/NO
    if pd.api.types.is_numeric_dtype(df[columna]) and es_columna_binaria(df[columna]):
        if valor_actual is not None and not pd.isna(valor_actual):
            try:
                index = 0 if int(float(valor_actual)) == 1 else 1
            except Exception:
                index = 1
        else:
            index = 1

        opcion = st.selectbox(
            columna,
            ["Si", "No"],
            index=index,
            key=key
        )

        return 1 if opcion == "Si" else 0

    # NUMÉRICOS RESTRINGIDOS A SOLO NÚMEROS
    if pd.api.types.is_numeric_dtype(df[columna]) or es_columna_numerica_forzada(columna):
        valor_base = convertir_a_numero(valor_actual, 0.0)

        if (
            "year" in nombre or
            "anio" in nombre or
            "año" in nombre or
            "stock" in nombre or
            "cantidad" in nombre or
            "months" in nombre or
            "meses" in nombre or
            "month" in nombre or
            "reorden" in nombre or
            "days" in nombre or
            "dias" in nombre or
            "día" in nombre
        ):
            return int(st.number_input(
                columna,
                min_value=0,
                value=int(valor_base),
                step=1,
                key=key
            ))

        return float(st.number_input(
            columna,
            min_value=0.0,
            value=float(valor_base),
            step=1.0,
            key=key
        ))

    # TEXTO LIBRE SOLO DONDE TIENE SENTIDO
    if es_columna_texto_libre(columna):
        if valor_actual is None or pd.isna(valor_actual):
            valor_actual = ""

        return st.text_input(
            columna,
            value=str(valor_actual),
            key=key
        )

    # TODO LO DEMAS COMO SELECTOR
    opciones = obtener_opciones_columna(df, columna)

    if valor_actual is not None and not pd.isna(valor_actual):
        valor_actual = str(valor_actual)

        if valor_actual not in opciones:
            opciones.append(valor_actual)

        index = opciones.index(valor_actual)
    else:
        index = 0

    return st.selectbox(
        columna,
        opciones,
        index=index,
        key=key
    )


def construir_formulario_amigable(df, nombre_tabla, modo, fila=None):
    data = {}

    columnas = [
        col for col in df.columns
        if col != "id_registro"
    ]

    # En pacientes, Prioridad NO se ingresa.
    # La predice el modelo automáticamente.
    if nombre_tabla == "pacientes":
        columnas = [
            col for col in columnas
            if not es_columna_prioridad(col)
        ]

    grupos_one_hot = detectar_grupos_one_hot(df)

    columnas_usadas = set()

    # GRUPOS ONE-HOT COMO SELECTOR
    for prefijo, columnas_grupo in grupos_one_hot.items():
        columnas_grupo = [
            col for col in columnas_grupo
            if col in columnas
        ]

        if len(columnas_grupo) <= 1:
            continue

        columnas_usadas.update(columnas_grupo)

        opciones = [
            etiqueta_desde_columna_one_hot(prefijo, col)
            for col in columnas_grupo
        ]

        if fila is not None:
            col_activa = obtener_columna_activa_one_hot(fila, columnas_grupo)
            opcion_actual = etiqueta_desde_columna_one_hot(prefijo, col_activa)
            index = opciones.index(opcion_actual) if opcion_actual in opciones else 0
        else:
            index = 0

        opcion = st.selectbox(
            prefijo,
            opciones,
            index=index,
            key=f"{modo}_{nombre_tabla}_{prefijo}_onehot"
        )

        for col in columnas_grupo:
            etiqueta = etiqueta_desde_columna_one_hot(prefijo, col)
            data[col] = 1 if etiqueta == opcion else 0

    # COLUMNAS NORMALES
    for columna in columnas:
        if columna in columnas_usadas:
            continue

        if fila is not None:
            valor_actual = fila[columna]
        else:
            valor_actual = None

        data[columna] = crear_input_amigable(
            df=df,
            columna=columna,
            valor_actual=valor_actual,
            key=f"{modo}_{nombre_tabla}_{columna}"
        )

    return data


# =====================================================
# RESUMEN GENERAL
# =====================================================

def render_summary(df_c, df_i):
    st.title("🏥 ALDIMI Predict")
    st.caption("Dashboard predictivo para gestión clínica e inventario hospitalario")

    st.header("📊 Resumen de Operaciones")

    col1, col2, col3, col4 = st.columns(4)

    try:
        total_pacientes = contar_registros("pacientes")
        prioridad_alta = contar_por_valor("pacientes", "Prioridad", "Alta")
    except Exception:
        total_pacientes = len(df_c)

        if "Prioridad" in df_c.columns:
            prioridad_alta = len(df_c[df_c["Prioridad"].astype(str) == "Alta"])
        else:
            prioridad_alta = 0

    if "Stock_Actual" in df_i.columns and "Punto_Reorden" in df_i.columns:
        stock_critico = len(df_i[df_i["Stock_Actual"] <= df_i["Punto_Reorden"]])
    else:
        stock_critico = 0

    if "Estado_Vencimiento" in df_i.columns:
        vencimientos = len(df_i[df_i["Estado_Vencimiento"].astype(str) != "Seguro"])
    else:
        vencimientos = 0

    col1.metric("Total Pacientes", total_pacientes)

    if total_pacientes > 0:
        col2.metric(
            "Prioridad Alta 🚨",
            prioridad_alta,
            f"{prioridad_alta / total_pacientes:.1%}",
            delta_color="inverse"
        )
    else:
        col2.metric("Prioridad Alta 🚨", prioridad_alta)

    col3.metric("Stock en Reorden", stock_critico)
    col4.metric("Alertas Vencimiento", vencimientos)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Carga de Prioridad Clínica")

        if "Prioridad" in df_c.columns:
            try:
                prioridad_resumen = pd.DataFrame({
                    "Prioridad": PRIORIDAD_ORDEN,
                    "Cantidad": [
                        contar_por_valor("pacientes", "Prioridad", prioridad)
                        for prioridad in PRIORIDAD_ORDEN
                    ]
                })
            except Exception:
                prioridad_resumen = (
                    df_c["Prioridad"]
                    .astype(str)
                    .value_counts()
                    .reindex(PRIORIDAD_ORDEN, fill_value=0)
                    .reset_index()
                )
                prioridad_resumen.columns = ["Prioridad", "Cantidad"]

            fig_prio = px.pie(
                prioridad_resumen,
                names="Prioridad",
                values="Cantidad",
                title="Carga de Prioridad Clínica",
                color="Prioridad",
                color_discrete_map=PRIORIDAD_COLORES,
                category_orders={"Prioridad": PRIORIDAD_ORDEN}
            )

            fig_prio.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )

            fig_prio.update_layout(
                legend_title_text="Prioridad",
                height=430
            )

            st.plotly_chart(fig_prio, use_container_width=True)
        else:
            st.warning("No existe la columna 'Prioridad' en pacientes.")

    with c2:
        st.subheader("Distribución por Edad")

        df_edad_visual, col_edad = preparar_columna_edad_visual(df_c)
        fig_edad = None

        if col_edad is not None and "Prioridad" in df_edad_visual.columns:
            orden_edades = ordenar_edades_visual(df_edad_visual[col_edad].unique())
            fig_edad = px.histogram(
                df_edad_visual,
                x=col_edad,
                color="Prioridad",
                barmode="group",
                title="Pacientes por Edad y Prioridad",
                color_discrete_map=PRIORIDAD_COLORES,
                category_orders={
                    col_edad: orden_edades,
                    "Prioridad": PRIORIDAD_ORDEN
                }
            )
        elif col_edad is not None:
            orden_edades = ordenar_edades_visual(df_edad_visual[col_edad].unique())
            fig_edad = px.histogram(
                df_edad_visual,
                x=col_edad,
                title="Pacientes por Edad",
                category_orders={col_edad: orden_edades}
            )

        if fig_edad is not None:
            fig_edad.update_layout(
                xaxis_title="Edad",
                yaxis_title="Cantidad de pacientes",
                height=430
            )

            st.plotly_chart(fig_edad, use_container_width=True)
        else:
            st.warning("No existe una columna de edad para graficar.")

    st.divider()

    st.subheader("📦 Inventario por Categoría y Estado")

    if "Categoria" in df_i.columns and "Stock_Actual" in df_i.columns:
        if "Estado" in df_i.columns:
            fig_stock = px.bar(
                df_i,
                x="Categoria",
                y="Stock_Actual",
                color="Estado",
                title="Inventario por Categoría y Estado",
                barmode="group"
            )
        else:
            fig_stock = px.bar(
                df_i,
                x="Categoria",
                y="Stock_Actual",
                title="Inventario por Categoría"
            )

        fig_stock.update_layout(
            xaxis_title="Categoría",
            yaxis_title="Stock actual",
            height=430
        )

        st.plotly_chart(fig_stock, use_container_width=True)
    else:
        st.warning("Faltan columnas para graficar inventario.")


# =====================================================
# ANÁLISIS CLÍNICO
# =====================================================

def render_clinical_analysis(df):
    st.header("🩺 Gestión de Prioridad Oncológica")

    df_clinico = df.copy()
    df_edad_visual, col_edad = preparar_columna_edad_visual(df_clinico)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Pacientes por Edad")
        fig_edad = None

        if col_edad is not None and "Prioridad" in df_edad_visual.columns:
            orden_edades = ordenar_edades_visual(df_edad_visual[col_edad].unique())
            fig_edad = px.histogram(
                df_edad_visual,
                x=col_edad,
                color="Prioridad",
                barmode="group",
                title="Distribución por Edad y Prioridad",
                color_discrete_map=PRIORIDAD_COLORES,
                category_orders={
                    col_edad: orden_edades,
                    "Prioridad": PRIORIDAD_ORDEN
                }
            )
        elif col_edad is not None:
            orden_edades = ordenar_edades_visual(df_edad_visual[col_edad].unique())
            fig_edad = px.histogram(
                df_edad_visual,
                x=col_edad,
                title="Distribución por Edad",
            )

        if fig_edad is not None:
            fig_edad.update_layout(
                xaxis_title="Edad",
                yaxis_title="Cantidad",
                height=450
            )

            st.plotly_chart(fig_edad, use_container_width=True)
        else:
            st.warning("No existe una columna de edad para graficar.")

    with col2:
        st.subheader("Pacientes con Prioridad Alta")

        if "Prioridad" in df_clinico.columns:
            pacientes_altos = df_clinico[df_clinico["Prioridad"].astype(str) == "Alta"]

            columnas_mostrar = []

            for col in [
                "id_registro",
                "Age",
                "Edad",
                "Income",
                "income",
                "Year",
                "Survival_Months",
                "Prioridad"
            ]:
                if col in pacientes_altos.columns:
                    columnas_mostrar.append(col)

            if columnas_mostrar:
                st.dataframe(
                    construir_vista_legible(pacientes_altos[columnas_mostrar].head(30)),
                    use_container_width=True
                )
            else:
                st.dataframe(construir_vista_legible(pacientes_altos.head(30)), use_container_width=True)
        elif col_edad is not None:
            st.warning("No existe la columna 'Prioridad'.")
            st.dataframe(construir_vista_legible(df_clinico.head(30)), use_container_width=True)

    st.divider()

    st.subheader("Filtro clínico")

    if "Survival_Months" in df.columns:
        max_meses = int(df["Survival_Months"].max())

        meses = st.slider(
            "Meses de supervivencia",
            0,
            max_meses,
            (0, min(60, max_meses))
        )

        df_f = df_clinico[
            (df_clinico["Survival_Months"] >= meses[0]) &
            (df_clinico["Survival_Months"] <= meses[1])
        ]

        st.dataframe(construir_vista_legible(df_f.head(50)), use_container_width=True)
    else:
        st.info("No existe la columna 'Survival_Months'. Se muestra la tabla general.")
        st.dataframe(construir_vista_legible(df_clinico.head(50)), use_container_width=True)


# =====================================================
# CONTROL DE INVENTARIO
# =====================================================

def render_inventory_control(df):
    st.header("📦 Control de Suministros Limpio")

    if "Categoria" not in df.columns:
        st.warning("No existe la columna 'Categoria' en inventario.")
        st.dataframe(df, use_container_width=True)
        return

    categorias = st.multiselect(
        "Filtrar Categoría:",
        df["Categoria"].dropna().unique(),
        default=df["Categoria"].dropna().unique()
    )

    df_f = df[df["Categoria"].isin(categorias)]

    columnas_necesarias = [
        "Stock_Actual",
        "Volumen_Consumo",
        "Estado_Vencimiento",
        "Tasa_Rotacion",
        "Producto"
    ]

    faltantes = [col for col in columnas_necesarias if col not in df_f.columns]

    if not faltantes:
        fig = px.scatter(
            df_f,
            x="Stock_Actual",
            y="Volumen_Consumo",
            color="Estado_Vencimiento",
            size="Tasa_Rotacion",
            hover_name="Producto",
            title="Análisis de Rotación y Riesgo de Vencimiento"
        )

        fig.update_layout(
            xaxis_title="Stock actual",
            yaxis_title="Volumen de consumo",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No se puede crear el gráfico. Faltan columnas: {faltantes}")

    st.error("### ⚠️ Requerimientos de Compra Inmediata")

    columnas_requeridas = ["Producto", "Stock_Actual"]
    faltantes_requerimientos = [
        col for col in columnas_requeridas
        if col not in df_f.columns
    ]

    if faltantes_requerimientos:
        st.warning(f"No se puede calcular requerimientos. Faltan columnas: {faltantes_requerimientos}")
        return

    df_pred = predecir_demanda_logistica_por_producto(df_f)

    if df_pred.empty or "Demanda_Predicha" not in df_pred.columns:
        st.warning(
            "No se pudo generar la prediccion logistica. "
            "Verifica que existan data/modelo_logistico.pkl y data/dataset_logistico_1.csv."
        )
        return

    df_pred["Stock_Actual"] = pd.to_numeric(df_pred["Stock_Actual"], errors="coerce").fillna(0)
    df_pred["Demanda_Predicha"] = pd.to_numeric(df_pred["Demanda_Predicha"], errors="coerce")
    df_pred = df_pred.dropna(subset=["Demanda_Predicha"])
    df_pred["Demanda_Predicha"] = df_pred["Demanda_Predicha"].round().astype(int)

    margen_seguridad = 1.15
    df_pred["Stock_Sugerido"] = (df_pred["Demanda_Predicha"] * margen_seguridad).apply(math.ceil)
    df_pred["Cantidad_Requerida"] = (
        df_pred["Stock_Sugerido"] - df_pred["Stock_Actual"]
    ).clip(lower=0)
    df_pred["Brecha_Demanda_Stock"] = (
        df_pred["Demanda_Predicha"] - df_pred["Stock_Actual"]
    )

    df_pred["Riesgo_Operativo"] = "Bajo"
    df_pred.loc[df_pred["Brecha_Demanda_Stock"] > 0, "Riesgo_Operativo"] = "Medio"
    df_pred.loc[
        df_pred["Cantidad_Requerida"] >= df_pred["Demanda_Predicha"] * 0.5,
        "Riesgo_Operativo"
    ] = "Alto"

    orden_riesgo = {"Alto": 0, "Medio": 1, "Bajo": 2}
    urgentes = df_pred[df_pred["Cantidad_Requerida"] > 0].copy()
    urgentes["_orden_riesgo"] = urgentes["Riesgo_Operativo"].map(orden_riesgo)
    urgentes = urgentes.sort_values(
        ["_orden_riesgo", "Cantidad_Requerida"],
        ascending=[True, False]
    )

    columnas_tabla = [
        col for col in [
            "Producto",
            "Categoria",
            "Stock_Actual",
            "Demanda_Predicha",
            "Stock_Sugerido",
            "Cantidad_Requerida",
            "Riesgo_Operativo"
        ]
        if col in urgentes.columns
    ]

    if urgentes.empty:
        st.success("No hay requerimientos inmediatos segun la demanda predicha por el modelo.")
    else:
        st.caption(
            "La demanda esperada se calcula con el modelo logistico. "
            "El stock sugerido considera un margen operativo de 15%."
        )
        st.dataframe(
            urgentes[columnas_tabla],
            use_container_width=True
        )


# =====================================================
# GESTIÓN DE TABLAS
# =====================================================

def render_priorizacion_integrada(df_clinico, df_inventario):
    st.header("Priorizacion Integrada de Recursos")
    st.caption(
        "La prioridad clinica del paciente genera una demanda adicional. "
        "Esa demanda se cruza con la prediccion logistica y el stock actual para estimar riesgo operativo."
    )

    if df_clinico.empty:
        st.warning("No hay pacientes disponibles para evaluar.")
        return

    if df_inventario.empty:
        st.warning("No hay inventario disponible para cruzar con la prediccion logistica.")
        return

    col_id = "id_registro" if "id_registro" in df_clinico.columns else None
    df_opciones, col_edad = preparar_columna_edad_visual(df_clinico)

    if col_id is not None:
        df_opciones = df_opciones.sort_values(col_id, ascending=False)

    opciones = []
    for idx, fila in df_opciones.head(1000).iterrows():
        partes = []

        if col_id is not None:
            partes.append(f"Paciente {fila[col_id]}")
        else:
            partes.append(f"Paciente fila {idx}")

        if col_edad is not None and not pd.isna(fila[col_edad]):
            partes.append(f"Edad: {fila[col_edad]}")

        survival = obtener_valor_columna(fila, ["Survival_Months", "survival_months"], None)
        if survival is not None and not pd.isna(survival):
            partes.append(f"Supervivencia: {int(convertir_a_numero(survival, 0))} meses")

        opciones.append((" | ".join(partes), idx))

    if not opciones:
        st.warning("No se encontraron pacientes para seleccionar.")
        return

    etiqueta = st.selectbox(
        "Selecciona un paciente para evaluar:",
        [opcion[0] for opcion in opciones]
    )

    idx_seleccionado = dict(opciones)[etiqueta]
    paciente = df_opciones.loc[idx_seleccionado]
    data_paciente = paciente.to_dict()

    try:
        prioridad_predicha = predecir_prioridad_paciente(data_paciente, df_clinico)
    except Exception as e:
        st.error(f"No se pudo predecir la prioridad clinica: {e}")
        return

    recursos = construir_recursos_integrados(prioridad_predicha, df_inventario)

    c1, c2, c3 = st.columns(3)
    c1.metric("Prioridad clinica predicha", prioridad_predicha)
    c2.metric("Recursos priorizados", len(recursos))

    if not recursos.empty:
        recursos_alto = len(recursos[recursos["Riesgo_Operativo"] == "Alto"])
    else:
        recursos_alto = 0

    c3.metric("Recursos con riesgo alto", recursos_alto)

    st.subheader("Flujo de decision")
    st.info(
        "Paciente -> clasificacion clinica -> demanda adicional por prioridad -> "
        "demanda historica predicha -> demanda total esperada -> riesgo operativo."
    )

    if recursos.empty:
        st.warning(
            "No se encontraron recursos priorizados con coincidencia en inventario. "
            "Revisa que los nombres de productos existan en la tabla inventario y en las features logisticas."
        )
        return

    st.subheader("Recursos priorizados para el paciente")

    st.dataframe(
        recursos,
        use_container_width=True,
        hide_index=True
    )

    resumen = (
        recursos.groupby(["Tipo", "Riesgo_Operativo"])
        .size()
        .reset_index(name="Cantidad")
    )

    fig = px.bar(
        resumen,
        x="Tipo",
        y="Cantidad",
        color="Riesgo_Operativo",
        barmode="group",
        title="Riesgo operativo por tipo de recurso"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Las cantidades son unidades operativas de cobertura por paciente. "
        "No representan dosis medicas; sirven para estimar presion adicional sobre inventario."
    )


def render_gestion_tabla(df, nombre_tabla, titulo):
    st.header(titulo)
    flash_key = f"mensaje_gestion_{nombre_tabla}"
    pagina_key = f"pagina_{nombre_tabla}"
    page_size = 1000

    if pagina_key not in st.session_state:
        st.session_state[pagina_key] = 0

    if flash_key in st.session_state:
        st.success(st.session_state.pop(flash_key))

    pagina_actual = st.session_state[pagina_key]
    if pagina_actual == 0:
        df_pagina = df.copy()
    else:
        df_pagina = cargar_tabla(
            nombre_tabla,
            limit=page_size,
            offset=pagina_actual * page_size
        )

    tabs = st.tabs([
        "📋 Ver datos",
        "➕ Agregar",
        "✏️ Editar",
        "🗑️ Eliminar",
        "📥 Carga Masiva"
    ])

    # VER DATOS
    with tabs[0]:
        st.subheader("Datos registrados")

        texto_busqueda = st.text_input(
            "Buscar en la tabla:",
            key=f"buscar_{nombre_tabla}"
        )

        df_mostrar = construir_vista_legible(df_pagina)

        if texto_busqueda:
            mascara = df_mostrar.astype(str).apply(
                lambda fila: fila.str.contains(texto_busqueda, case=False, na=False).any(),
                axis=1
            )
            df_mostrar = df_mostrar[mascara]

        col_prev, col_info, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button(
                "Pagina anterior",
                key=f"btn_prev_{nombre_tabla}",
                disabled=pagina_actual == 0
            ):
                st.session_state[pagina_key] = max(pagina_actual - 1, 0)
                limpiar_cache_y_recargar()

        with col_info:
            inicio = pagina_actual * page_size + 1
            fin = pagina_actual * page_size + len(df_pagina)
            st.caption(f"Pagina {pagina_actual + 1} | Registros {inicio} - {fin}")

        with col_next:
            if st.button(
                "Pagina siguiente",
                key=f"btn_next_{nombre_tabla}",
                disabled=len(df_pagina) < page_size
            ):
                st.session_state[pagina_key] = pagina_actual + 1
                limpiar_cache_y_recargar()

        st.dataframe(df_mostrar, use_container_width=True)

    # AGREGAR
    with tabs[1]:
        st.subheader("Agregar nuevo registro")

        if nombre_tabla == "pacientes":
            st.info(
                "Completa los datos del paciente. La prioridad NO se ingresa manualmente: "
                "el modelo predictivo la clasificará automáticamente."
            )
        else:
            st.info(
                "Completa el formulario usando selectores y campos numéricos. "
                "Los campos como Stock, consumo o reorden solo aceptan numeros."
            )

        with st.form(f"form_agregar_{nombre_tabla}"):
            data = construir_formulario_amigable(
                df=df,
                nombre_tabla=nombre_tabla,
                modo="agregar",
                fila=None
            )

            guardar = st.form_submit_button("Guardar registro")

            if guardar:
                try:
                    if nombre_tabla == "pacientes" and "Prioridad" in df.columns:
                        prioridad_predicha = predecir_prioridad_paciente(data, df)
                        data["Prioridad"] = prioridad_predicha

                        registro_insertado = insertar_registro(nombre_tabla, data)
                        id_insertado = (
                            registro_insertado.get("id_registro")
                            if isinstance(registro_insertado, dict)
                            else None
                        )

                        mensaje = "Registro agregado correctamente. "
                        if id_insertado is not None:
                            mensaje += f"ID: {id_insertado}. "
                        mensaje += f"Prioridad predicha: {prioridad_predicha}"
                        st.session_state[flash_key] = mensaje
                        st.session_state[pagina_key] = 0
                    else:
                        registro_insertado = insertar_registro(nombre_tabla, data)
                        id_insertado = (
                            registro_insertado.get("id_registro")
                            if isinstance(registro_insertado, dict)
                            else None
                        )
                        mensaje = "Registro agregado correctamente."
                        if id_insertado is not None:
                            mensaje += f" ID: {id_insertado}."
                        st.session_state[flash_key] = mensaje
                        st.session_state[pagina_key] = 0

                    limpiar_cache_y_recargar()

                except Exception as e:
                    st.error(f"No se pudo agregar el registro: {e}")

    # EDITAR
    with tabs[2]:
        st.subheader("Editar registro existente")

        if df_pagina.empty:
            st.warning("No hay registros para editar.")
        else:
            id_seleccionado = st.selectbox(
                "Selecciona el ID del registro:",
                df_pagina["id_registro"].tolist(),
                key=f"select_editar_{nombre_tabla}"
            )

            fila = df_pagina[df_pagina["id_registro"] == id_seleccionado].iloc[0]

            with st.form(f"form_editar_{nombre_tabla}"):
                data = construir_formulario_amigable(
                    df=df,
                    nombre_tabla=nombre_tabla,
                    modo=f"editar_{id_seleccionado}",
                    fila=fila
                )

                actualizar = st.form_submit_button("Actualizar registro")

                if actualizar:
                    try:
                        if nombre_tabla == "pacientes" and "Prioridad" in df.columns:
                            prioridad_predicha = predecir_prioridad_paciente(data, df)
                            data["Prioridad"] = prioridad_predicha

                            actualizar_registro(nombre_tabla, id_seleccionado, data)

                            st.success(
                                f"Registro actualizado correctamente. "
                                f"Nueva prioridad predicha: {prioridad_predicha}"
                            )
                        else:
                            actualizar_registro(nombre_tabla, id_seleccionado, data)
                            st.success("Registro actualizado correctamente.")

                        limpiar_cache_y_recargar()

                    except Exception as e:
                        st.error(f"No se pudo actualizar el registro: {e}")

    # ELIMINAR
    with tabs[3]:
        st.subheader("Eliminar registro")

        if df_pagina.empty:
            st.warning("No hay registros para eliminar.")
        else:
            id_eliminar = st.selectbox(
                "Selecciona el ID a eliminar:",
                df_pagina["id_registro"].tolist(),
                key=f"select_eliminar_{nombre_tabla}"
            )

            fila_eliminar = df_pagina[df_pagina["id_registro"] == id_eliminar]

            st.write("Registro seleccionado:")
            st.dataframe(construir_vista_legible(fila_eliminar), use_container_width=True)

            st.warning("Esta acción eliminará el registro seleccionado.")

            confirmar = st.checkbox(
                "Confirmo que deseo eliminar este registro",
                key=f"confirmar_eliminar_{nombre_tabla}"
            )

            if st.button("Eliminar registro", key=f"btn_eliminar_{nombre_tabla}"):
                if confirmar:
                    try:
                        eliminar_registro(nombre_tabla, id_eliminar)
                        st.success("Registro eliminado correctamente.")
                        limpiar_cache_y_recargar()
                    except Exception as e:
                        st.error(f"No se pudo eliminar el registro: {e}")
                else:
                    st.error("Marca la confirmación antes de eliminar.")

    # CARGA MASIVA
    with tabs[4]:
        st.subheader("Carga masiva de registros")

        st.info("Carga administrativa para importar o actualizar registros por lote desde fuentes externas. Para registros individuales, usa las pestañas Agregar o Editar. El archivo CSV debe respetar la estructura de columnas de la tabla seleccionada.")

        archivo = st.file_uploader(
            "Sube un archivo CSV",
            type=["csv"],
            key=f"upload_{nombre_tabla}"
        )

        if archivo is not None:
            try:
                df_preview = pd.read_csv(archivo)

                st.write("Vista previa:")
                st.dataframe(df_preview.head(), use_container_width=True)

                archivo.seek(0)

                if st.button("Importar registros a la base de datos", key=f"btn_csv_{nombre_tabla}"):
                    agregar_csv_a_tabla(nombre_tabla, archivo)
                    st.success("Carga masiva completada correctamente.")
                    limpiar_cache_y_recargar()

            except Exception as e:
                st.error(f"No se pudo cargar el CSV: {e}")


# =====================================================
# =====================================================
# EJECUCIÓN PRINCIPAL

def main():
    df_clinico, df_inv = load_data()

    if df_clinico is not None and df_inv is not None:
        st.sidebar.title("ALDIMI Predict")
        st.sidebar.markdown("---")

        opcion = st.sidebar.radio(
            "Menú Principal",
            [
                "Resumen General",
                "Análisis Clínico",
                "Control de Inventario",
                "Priorizacion Integrada",
                "Gestionar Pacientes",
                "Gestionar Inventario"
            ]
        )

        if opcion == "Resumen General":
            render_summary(df_clinico, df_inv)

        elif opcion == "Análisis Clínico":
            render_clinical_analysis(df_clinico)

        elif opcion == "Control de Inventario":
            render_inventory_control(df_inv)

        elif opcion == "Priorizacion Integrada":
            render_priorizacion_integrada(df_clinico, df_inv)

        elif opcion == "Gestionar Pacientes":
            render_gestion_tabla(
                df=df_clinico,
                nombre_tabla="pacientes",
                titulo="🩺 Gestión de Pacientes Clínicos"
            )

        elif opcion == "Gestionar Inventario":
            render_gestion_tabla(
                df=df_inv,
                nombre_tabla="inventario",
                titulo="📦 Gestión de Inventario"
            )


if __name__ == "__main__":
    main()
