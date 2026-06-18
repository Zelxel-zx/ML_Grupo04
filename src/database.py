# -*- coding: utf-8 -*-

import sqlite3
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

APP_DIR = PROJECT_DIR / "app"
DATA_DIR = PROJECT_DIR / "data"

DB_PATH = DATA_DIR / "aldimi_predict.db"


def validar_tabla(nombre_tabla):
    tablas_validas = ["pacientes", "inventario"]

    if nombre_tabla not in tablas_validas:
        raise ValueError("Tabla no permitida: " + str(nombre_tabla))


def quote_identifier(nombre):
    return '"' + str(nombre).replace('"', '""') + '"'


def get_connection():
    DATA_DIR.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def buscar_archivo(posibles_rutas):
    for ruta in posibles_rutas:
        if ruta.exists():
            return ruta

    rutas_texto = "\n".join(str(r) for r in posibles_rutas)
    raise FileNotFoundError("No se encontro ningun archivo en:\n" + rutas_texto)


def obtener_ruta_csv_clinico():
    return buscar_archivo([
        DATA_DIR / "data_modelo_clinico.csv",
        APP_DIR / "data_modelo_clinico.csv",
        DATA_DIR / "dataset_clinico_app.csv"
    ])


def obtener_ruta_csv_inventario():
    return buscar_archivo([
        APP_DIR / "inventario_limpio.csv",
        DATA_DIR / "inventario_limpio.csv",
        DATA_DIR / "dataset_logistico_1.csv"
    ])


def crear_bd_desde_csv():

    path_clinico = obtener_ruta_csv_clinico()
    path_inventario = obtener_ruta_csv_inventario()

    df_clinico = pd.read_csv(path_clinico)
    df_inventario = pd.read_csv(path_inventario)

    with get_connection() as conn:
        df_clinico.to_sql("pacientes", conn, if_exists="replace", index=False)
        df_inventario.to_sql("inventario", conn, if_exists="replace", index=False)

    print("Base de datos creada correctamente.")
    print("BD creada en:", DB_PATH)
    print("CSV clinico usado:", path_clinico)
    print("CSV inventario usado:", path_inventario)


def tabla_existe(nombre_tabla):
    validar_tabla(nombre_tabla)

    query = """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?
    """

    with get_connection() as conn:
        cursor = conn.execute(query, [nombre_tabla])
        resultado = cursor.fetchone()

    return resultado is not None


def inicializar_bd_si_no_existe():
    if not DB_PATH.exists():
        crear_bd_desde_csv()
        return

    if not tabla_existe("pacientes") or not tabla_existe("inventario"):
        crear_bd_desde_csv()


def cargar_pacientes():
    inicializar_bd_si_no_existe()

    with get_connection() as conn:
        df = pd.read_sql_query(
            'SELECT rowid AS id_registro, * FROM "pacientes"',
            conn
        )

    return df


def cargar_inventario():
    inicializar_bd_si_no_existe()

    with get_connection() as conn:
        df = pd.read_sql_query(
            'SELECT rowid AS id_registro, * FROM "inventario"',
            conn
        )

    return df


def obtener_columnas_tabla(nombre_tabla):
    validar_tabla(nombre_tabla)
    inicializar_bd_si_no_existe()

    with get_connection() as conn:
        cursor = conn.execute("PRAGMA table_info(" + quote_identifier(nombre_tabla) + ")")
        columnas = [fila[1] for fila in cursor.fetchall()]

    return columnas


def insertar_registro(nombre_tabla, data):
    validar_tabla(nombre_tabla)

    columnas_validas = obtener_columnas_tabla(nombre_tabla)

    data_limpia = {
        columna: valor
        for columna, valor in data.items()
        if columna in columnas_validas
    }

    if not data_limpia:
        raise ValueError("No hay datos validos para insertar.")

    columnas = list(data_limpia.keys())
    valores = list(data_limpia.values())

    columnas_sql = ", ".join([quote_identifier(col) for col in columnas])
    placeholders = ", ".join(["?"] * len(columnas))

    query = (
        "INSERT INTO " + quote_identifier(nombre_tabla) +
        " (" + columnas_sql + ") VALUES (" + placeholders + ")"
    )

    with get_connection() as conn:
        conn.execute(query, valores)
        conn.commit()

def actualizar_registro(nombre_tabla, id_registro, data):
    validar_tabla(nombre_tabla)

    columnas_validas = obtener_columnas_tabla(nombre_tabla)

    data_limpia = {
        columna: valor
        for columna, valor in data.items()
        if columna in columnas_validas
    }

    if not data_limpia:
        raise ValueError("No hay datos validos para actualizar.")

    columnas = list(data_limpia.keys())
    valores = list(data_limpia.values())

    set_sql = ", ".join([quote_identifier(columna) + " = ?" for columna in columnas])

    query = (
        "UPDATE " + quote_identifier(nombre_tabla) +
        " SET " + set_sql +
        " WHERE rowid = ?"
    )

    with get_connection() as conn:
        conn.execute(query, valores + [id_registro])
        conn.commit()


def eliminar_registro(nombre_tabla, id_registro):
    validar_tabla(nombre_tabla)

    query = (
        "DELETE FROM " + quote_identifier(nombre_tabla) +
        " WHERE rowid = ?"
    )

    with get_connection() as conn:
        conn.execute(query, [id_registro])
        conn.commit()


def agregar_csv_a_tabla(nombre_tabla, archivo_csv):
    validar_tabla(nombre_tabla)

    try:
        archivo_csv.seek(0)
    except Exception:
        pass

    df_nuevo = pd.read_csv(archivo_csv)

    columnas_tabla = obtener_columnas_tabla(nombre_tabla)
    columnas_csv = list(df_nuevo.columns)

    columnas_faltantes = [col for col in columnas_tabla if col not in columnas_csv]
    columnas_extra = [col for col in columnas_csv if col not in columnas_tabla]

    if columnas_faltantes:
        raise ValueError("Al CSV le faltan estas columnas: " + str(columnas_faltantes))

    if columnas_extra:
        raise ValueError("El CSV tiene columnas extra que no existen en la BD: " + str(columnas_extra))

    with get_connection() as conn:
        df_nuevo.to_sql(nombre_tabla, conn, if_exists="append", index=False)


if __name__ == "__main__":
    crear_bd_desde_csv()