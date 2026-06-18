# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import pickle
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


from database import (
    cargar_pacientes,
    cargar_inventario,
    insertar_registro,
    actualizar_registro,
    eliminar_registro,
    agregar_csv_a_tabla,
    inicializar_bd_si_no_existe
)


# =====================================================
# MODELO PREDICTIVO
# =====================================================

MODEL_PATH = DATA_DIR / "modelo_clinico.pkl"


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

ORDEN_RANGOS_EDAD = list(RANGOS_EDAD.keys())


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
        df_i = cargar_inventario()

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


def preparar_df_con_rango_edad(df):
    df_temp = df.copy()
    col_edad = buscar_columna_edad(df_temp)

    if col_edad is not None:
        df_temp["Rango_Edad"] = df_temp[col_edad].apply(obtener_rango_edad)
    else:
        df_temp["Rango_Edad"] = "Sin columna de edad"

    return df_temp


def detectar_grupos_one_hot(df):
    grupos = {}

    for col in df.columns:
        if col == "id_registro":
            continue

        if es_columna_prioridad(col):
            continue

        if "_" not in col:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if not es_columna_binaria(df[col]):
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

    # NUMERICOS RESTRINGIDOS A SOLO NUMEROS
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
    # La predice el modelo automaticamente.
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
            fig_prio = px.pie(
                df_c,
                names="Prioridad",
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

        df_edad = preparar_df_con_rango_edad(df_c)

        if "Prioridad" in df_edad.columns:
            fig_edad = px.histogram(
                df_edad,
                x="Rango_Edad",
                color="Prioridad",
                barmode="group",
                title="Pacientes por Rango de Edad y Prioridad",
                color_discrete_map=PRIORIDAD_COLORES,
                category_orders={
                    "Rango_Edad": ORDEN_RANGOS_EDAD,
                    "Prioridad": PRIORIDAD_ORDEN
                }
            )
        else:
            fig_edad = px.histogram(
                df_edad,
                x="Rango_Edad",
                title="Pacientes por Rango de Edad",
                category_orders={"Rango_Edad": ORDEN_RANGOS_EDAD}
            )

        fig_edad.update_layout(
            xaxis_title="Rango de edad",
            yaxis_title="Cantidad de pacientes",
            height=430
        )

        st.plotly_chart(fig_edad, use_container_width=True)

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
# ANALISIS CLINICO
# =====================================================

def render_clinical_analysis(df):
    st.header("🩺 Gestión de Prioridad Oncológica")

    df_edad = preparar_df_con_rango_edad(df)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Pacientes por Rango de Edad")

        if "Prioridad" in df_edad.columns:
            fig_edad = px.histogram(
                df_edad,
                x="Rango_Edad",
                color="Prioridad",
                barmode="group",
                title="Distribución por Edad y Prioridad",
                color_discrete_map=PRIORIDAD_COLORES,
                category_orders={
                    "Rango_Edad": ORDEN_RANGOS_EDAD,
                    "Prioridad": PRIORIDAD_ORDEN
                }
            )
        else:
            fig_edad = px.histogram(
                df_edad,
                x="Rango_Edad",
                title="Distribución por Edad",
                category_orders={"Rango_Edad": ORDEN_RANGOS_EDAD}
            )

        fig_edad.update_layout(
            xaxis_title="Rango de edad",
            yaxis_title="Cantidad",
            height=450
        )

        st.plotly_chart(fig_edad, use_container_width=True)

    with col2:
        st.subheader("Pacientes con Prioridad Alta")

        if "Prioridad" in df_edad.columns:
            pacientes_altos = df_edad[df_edad["Prioridad"].astype(str) == "Alta"]

            columnas_mostrar = []

            for col in [
                "id_registro",
                "Rango_Edad",
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
                    pacientes_altos[columnas_mostrar].head(30),
                    use_container_width=True
                )
            else:
                st.dataframe(pacientes_altos.head(30), use_container_width=True)
        else:
            st.warning("No existe la columna 'Prioridad'.")
            st.dataframe(df_edad.head(30), use_container_width=True)

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

        df_f = df_edad[
            (df_edad["Survival_Months"] >= meses[0]) &
            (df_edad["Survival_Months"] <= meses[1])
        ]

        st.dataframe(df_f.head(50), use_container_width=True)
    else:
        st.info("No existe la columna 'Survival_Months'. Se muestra la tabla general.")
        st.dataframe(df_edad.head(50), use_container_width=True)


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

    if "Stock_Actual" in df_f.columns and "Punto_Reorden" in df_f.columns:
        urgentes = df_f[df_f["Stock_Actual"] <= df_f["Punto_Reorden"]]

        columnas_tabla = [
            col for col in [
                "id_registro",
                "Producto",
                "Categoria",
                "Stock_Actual",
                "Punto_Reorden",
                "Estado"
            ]
            if col in urgentes.columns
        ]

        st.dataframe(
            urgentes[columnas_tabla],
            use_container_width=True
        )
    else:
        st.warning("Faltan columnas 'Stock_Actual' o 'Punto_Reorden'.")


# =====================================================
# GESTION DE TABLAS
# =====================================================

def render_gestion_tabla(df, nombre_tabla, titulo):
    st.header(titulo)

    tabs = st.tabs([
        "📋 Ver datos",
        "➕ Agregar",
        "✏️ Editar",
        "🗑️ Eliminar",
        "📥 Cargar CSV"
    ])

    # VER DATOS
    with tabs[0]:
        st.subheader("Datos registrados")

        texto_busqueda = st.text_input(
            "Buscar en la tabla:",
            key=f"buscar_{nombre_tabla}"
        )

        df_mostrar = df.copy()

        if texto_busqueda:
            mascara = df_mostrar.astype(str).apply(
                lambda fila: fila.str.contains(texto_busqueda, case=False, na=False).any(),
                axis=1
            )
            df_mostrar = df_mostrar[mascara]

        st.dataframe(df_mostrar, use_container_width=True)

    # AGREGAR
    with tabs[1]:
        st.subheader("Agregar nuevo registro")

        if nombre_tabla == "pacientes":
            st.info(
                "Completa los datos del paciente. La prioridad NO se ingresa manualmente: "
                "el modelo predictivo la clasificara automaticamente."
            )
        else:
            st.info(
                "Completa el formulario usando selectores y campos numericos. "
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

                        insertar_registro(nombre_tabla, data)

                        st.success(
                            f"Registro agregado correctamente. "
                            f"Prioridad predicha: {prioridad_predicha}"
                        )
                    else:
                        insertar_registro(nombre_tabla, data)
                        st.success("Registro agregado correctamente.")

                    limpiar_cache_y_recargar()

                except Exception as e:
                    st.error(f"No se pudo agregar el registro: {e}")

    # EDITAR
    with tabs[2]:
        st.subheader("Editar registro existente")

        if df.empty:
            st.warning("No hay registros para editar.")
        else:
            id_seleccionado = st.selectbox(
                "Selecciona el ID del registro:",
                df["id_registro"].tolist(),
                key=f"select_editar_{nombre_tabla}"
            )

            fila = df[df["id_registro"] == id_seleccionado].iloc[0]

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

        if df.empty:
            st.warning("No hay registros para eliminar.")
        else:
            id_eliminar = st.selectbox(
                "Selecciona el ID a eliminar:",
                df["id_registro"].tolist(),
                key=f"select_eliminar_{nombre_tabla}"
            )

            fila_eliminar = df[df["id_registro"] == id_eliminar]

            st.write("Registro seleccionado:")
            st.dataframe(fila_eliminar, use_container_width=True)

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

    # CARGAR CSV
    with tabs[4]:
        st.subheader("Agregar datos desde CSV")

        st.info("El CSV debe tener las mismas columnas que la tabla original.")

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

                if st.button("Agregar CSV a la base de datos", key=f"btn_csv_{nombre_tabla}"):
                    agregar_csv_a_tabla(nombre_tabla, archivo)
                    st.success("CSV agregado correctamente.")
                    limpiar_cache_y_recargar()

            except Exception as e:
                st.error(f"No se pudo cargar el CSV: {e}")


# =====================================================
# EJECUCION PRINCIPAL
# =====================================================

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