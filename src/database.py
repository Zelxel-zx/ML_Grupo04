# -*- coding: utf-8 -*-

import os

import pandas as pd
import requests


TABLAS_VALIDAS = [
    "pacientes",
    "inventario",
    "movimientos_inventario",
    "inventario_features_modelo",
    "inventario_caracter\u00edsticas",
]


def validar_tabla(nombre_tabla):
    if nombre_tabla not in TABLAS_VALIDAS:
        raise ValueError("Tabla no permitida: " + str(nombre_tabla))


def get_supabase_config():
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")

    if not url or not key:
        raise RuntimeError(
            "Faltan SUPABASE_URL y SUPABASE_KEY. "
            "Definelas antes de ejecutar la app."
        )

    return url, key


def get_headers(prefer=None):
    _, key = get_supabase_config()
    headers = {
        "apikey": key,
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if prefer:
        headers["Prefer"] = prefer

    return headers


def get_table_url(nombre_tabla):
    validar_tabla(nombre_tabla)
    url, _ = get_supabase_config()
    return url + "/rest/v1/" + nombre_tabla


def request_supabase(method, nombre_tabla, params=None, json_data=None, prefer=None):
    response = requests.request(
        method=method,
        url=get_table_url(nombre_tabla),
        headers=get_headers(prefer=prefer),
        params=params,
        json=json_data,
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            "Error Supabase "
            + str(response.status_code)
            + ": "
            + response.text
        )

    if response.text:
        return response.json()

    return None


def count_supabase(nombre_tabla, params=None):
    validar_tabla(nombre_tabla)

    parametros = {"select": "id_registro"}
    if params:
        parametros.update(params)

    response = requests.get(
        get_table_url(nombre_tabla),
        headers={
            **get_headers(prefer="count=exact"),
            "Range": "0-0",
        },
        params=parametros,
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            "Error Supabase "
            + str(response.status_code)
            + ": "
            + response.text
        )

    content_range = response.headers.get("Content-Range", "")

    if "/" in content_range:
        total = content_range.split("/")[-1]
        if total.isdigit():
            return int(total)

    return 0


def inicializar_bd_si_no_existe():
    # En Supabase la base ya existe. Esta funcion queda como validacion ligera.
    get_supabase_config()


def limpiar_valor(valor):
    if pd.isna(valor):
        return None

    if hasattr(valor, "item"):
        valor = valor.item()

    return valor


def limpiar_payload(data):
    data_limpia = {}

    for columna, valor in data.items():
        if columna == "id_registro":
            continue

        data_limpia[columna] = limpiar_valor(valor)

    return data_limpia


def cargar_tabla(nombre_tabla, limit=1000, offset=0, order=None):
    validar_tabla(nombre_tabla)

    params = {
        "select": "*",
        "limit": str(limit),
        "offset": str(offset),
    }

    if order is not None:
        params["order"] = order
    elif nombre_tabla in ["pacientes", "inventario"]:
        params["order"] = "id_registro.desc"

    registros = request_supabase(
        "GET",
        nombre_tabla,
        params=params,
    )

    df = pd.DataFrame(registros)

    if df.empty:
        columnas = obtener_columnas_tabla(nombre_tabla)
        return pd.DataFrame(columns=columnas)

    return df


def cargar_tabla_por_paginas(nombre_tabla, max_registros=50000, page_size=1000, order=None):
    validar_tabla(nombre_tabla)

    registros = []
    offset = 0

    while offset < max_registros:
        pagina = request_supabase(
            "GET",
            nombre_tabla,
            params={
                "select": "*",
                "limit": str(page_size),
                "offset": str(offset),
                **({"order": order} if order else {}),
            },
        )

        if not pagina:
            break

        registros.extend(pagina)

        if len(pagina) < page_size:
            break

        offset += page_size

    return pd.DataFrame(registros)


def cargar_pacientes(limit=1000, offset=0):
    return cargar_tabla("pacientes", limit=limit, offset=offset)


def cargar_pacientes_completo(max_registros=50000):
    return cargar_tabla_por_paginas(
        "pacientes",
        max_registros=max_registros,
        page_size=1000,
        order="id_registro.desc",
    )


def cargar_inventario(limit=1000, offset=0):
    return cargar_tabla("inventario", limit=limit, offset=offset)


def cargar_inventario_completo(max_registros=10000):
    return cargar_tabla_por_paginas(
        "inventario",
        max_registros=max_registros,
        page_size=1000,
        order="id_registro.desc",
    )


def cargar_features_logisticas(limit=50000):
    tablas_candidatas = [
        "inventario_features_modelo",
        "inventario_caracter\u00edsticas",
    ]
    dataframes = []

    for tabla in tablas_candidatas:
        try:
            df_tabla = cargar_tabla_por_paginas(
                tabla,
                max_registros=limit,
                page_size=1000,
                order="dt.desc",
            )
        except Exception:
            try:
                df_tabla = cargar_tabla_por_paginas(
                    tabla,
                    max_registros=limit,
                    page_size=1000,
                )
            except Exception:
                df_tabla = pd.DataFrame()

        if not df_tabla.empty:
            dataframes.append(df_tabla)

    if dataframes:
        return pd.concat(dataframes, ignore_index=True, sort=False)

    return pd.DataFrame()


def contar_registros(nombre_tabla):
    return count_supabase(nombre_tabla)


def contar_por_valor(nombre_tabla, columna, valor):
    return count_supabase(nombre_tabla, params={columna: "eq." + str(valor)})


def obtener_columnas_tabla(nombre_tabla):
    df = cargar_tabla_sin_orden(nombre_tabla, limit=1)

    if not df.empty:
        return list(df.columns)

    # Respaldo para permitir formularios aunque una tabla este vacia.
    columnas_base = {
        "pacientes": [
            "id_registro",
            "Year",
            "Survival_Months",
            "Prioridad",
        ],
        "inventario": [
            "id_registro",
            "Producto",
            "Categoria",
            "Stock_Actual",
            "Punto_Reorden",
            "Cantidad_Pedido",
            "Fecha_Recepcion",
            "Ultimo_Pedido",
            "Fecha_Vencimiento",
            "Volumen_Consumo",
            "Tasa_Rotacion",
            "Estado",
            "Lead_Time_Check",
            "Dias_para_Vencimiento",
            "Estado_Vencimiento",
        ],
    }

    return columnas_base[nombre_tabla]


def cargar_tabla_sin_orden(nombre_tabla, limit=1):
    validar_tabla(nombre_tabla)

    registros = request_supabase(
        "GET",
        nombre_tabla,
        params={
            "select": "*",
            "limit": str(limit),
        },
    )

    return pd.DataFrame(registros)


def obtener_siguiente_id_registro(nombre_tabla):
    validar_tabla(nombre_tabla)

    registros = request_supabase(
        "GET",
        nombre_tabla,
        params={
            "select": "id_registro",
            "order": "id_registro.desc.nullslast",
            "limit": "1",
        },
    )

    if registros and registros[0].get("id_registro") is not None:
        return int(registros[0]["id_registro"]) + 1

    return 1


def insertar_registro(nombre_tabla, data):
    validar_tabla(nombre_tabla)
    columnas_validas = obtener_columnas_tabla(nombre_tabla)

    data_limpia = {
        columna: valor
        for columna, valor in limpiar_payload(data).items()
        if columna in columnas_validas
    }

    if "id_registro" in columnas_validas and not data_limpia.get("id_registro"):
        data_limpia["id_registro"] = obtener_siguiente_id_registro(nombre_tabla)

    if not data_limpia:
        raise ValueError("No hay datos validos para insertar.")

    registro_insertado = request_supabase(
        "POST",
        nombre_tabla,
        json_data=data_limpia,
        prefer="return=representation",
    )

    if not registro_insertado:
        raise RuntimeError("Supabase no devolvio el registro insertado.")

    return registro_insertado[0]


def actualizar_registro(nombre_tabla, id_registro, data):
    validar_tabla(nombre_tabla)
    columnas_validas = obtener_columnas_tabla(nombre_tabla)

    data_limpia = {
        columna: valor
        for columna, valor in limpiar_payload(data).items()
        if columna in columnas_validas and columna != "id_registro"
    }

    if not data_limpia:
        raise ValueError("No hay datos validos para actualizar.")

    request_supabase(
        "PATCH",
        nombre_tabla,
        params={"id_registro": "eq." + str(id_registro)},
        json_data=data_limpia,
        prefer="return=minimal",
    )


def eliminar_registro(nombre_tabla, id_registro):
    validar_tabla(nombre_tabla)

    request_supabase(
        "DELETE",
        nombre_tabla,
        params={"id_registro": "eq." + str(id_registro)},
        prefer="return=minimal",
    )


def agregar_csv_a_tabla(nombre_tabla, archivo_csv):
    validar_tabla(nombre_tabla)

    try:
        archivo_csv.seek(0)
    except Exception:
        pass

    df_nuevo = pd.read_csv(archivo_csv)

    columnas_tabla = obtener_columnas_tabla(nombre_tabla)
    columnas_csv = list(df_nuevo.columns)

    columnas_faltantes = [
        col
        for col in columnas_tabla
        if col != "id_registro" and col not in columnas_csv
    ]
    columnas_extra = [
        col
        for col in columnas_csv
        if col not in columnas_tabla
    ]

    if columnas_faltantes:
        raise ValueError("Al CSV le faltan estas columnas: " + str(columnas_faltantes))

    if columnas_extra:
        raise ValueError("El CSV tiene columnas extra que no existen en la BD: " + str(columnas_extra))

    registros = []
    for registro in df_nuevo.to_dict(orient="records"):
        registros.append(limpiar_payload(registro))

    if registros:
        request_supabase(
            "POST",
            nombre_tabla,
            json_data=registros,
            prefer="return=minimal",
        )


if __name__ == "__main__":
    inicializar_bd_si_no_existe()
    print("Conexion Supabase configurada correctamente.")
