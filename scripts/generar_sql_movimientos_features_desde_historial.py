from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOGISTICO_PATH = ROOT / "data" / "dataset_logistico_1.csv"
OUTPUT_SQL = ROOT / "data" / "insertar_movimientos_features_desde_historial.sql"


def sql_text(value):
    if pd.isna(value):
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def sql_num(value, default=0):
    if pd.isna(value):
        return str(default)
    try:
        return str(float(value))
    except Exception:
        return str(default)


def sql_int(value, default=0):
    if pd.isna(value):
        return str(default)
    try:
        return str(int(float(value)))
    except Exception:
        return str(default)


def cargar_historial():
    df = pd.read_csv(LOGISTICO_PATH)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df[~df["name_products"].astype(str).str.match(r"^Producto\s+\d+$", flags=re.IGNORECASE, na=False)]
    df = df.sort_values(["name_products", "dt"])
    df = df.drop_duplicates(subset=["name_products", "dt"], keep="last")
    return df.groupby("name_products", group_keys=False).tail(90).reset_index(drop=True)


def crear_sql(df):
    lineas = [
        'DROP TABLE IF EXISTS "movimientos_inventario" CASCADE;',
        'DROP TABLE IF EXISTS "inventario_features_modelo" CASCADE;',
        "",
        'CREATE TABLE IF NOT EXISTS "movimientos_inventario" (',
        "    id_movimiento BIGSERIAL PRIMARY KEY,",
        "    producto TEXT,",
        "    fecha DATE,",
        "    tipo_movimiento TEXT,",
        "    cantidad DOUBLE PRECISION,",
        "    stock_resultante DOUBLE PRECISION,",
        "    horas_con_stock DOUBLE PRECISION,",
        "    descuento DOUBLE PRECISION,",
        '    "bandera de vacaciones" INTEGER,',
        '    "bandera de actividad" INTEGER,',
        '    "ID del producto" TEXT,',
        '    "ID de la tienda" INTEGER,',
        "    primer_id_de_categoria INTEGER,",
        "    segundo_id_categoria INTEGER,",
        "    tercer_id_categoria INTEGER,",
        "    creado_en TIMESTAMP DEFAULT NOW()",
        ");",
        "",
        'CREATE TABLE IF NOT EXISTS "inventario_features_modelo" (',
        "    dt DATE,",
        "    sale_amount DOUBLE PRECISION,",
        "    name_products TEXT,",
        "    tipo_producto_app TEXT,",
        "    product_id_cod INTEGER,",
        "    store_id_cod INTEGER,",
        "    first_category_id_cod INTEGER,",
        "    second_category_id_cod INTEGER,",
        "    third_category_id_cod INTEGER,",
        "    discount DOUBLE PRECISION,",
        "    holiday_flag_cod INTEGER,",
        "    activity_flag_cod INTEGER,",
        "    mes INTEGER,",
        "    dia_mes INTEGER,",
        "    dia_semana INTEGER,",
        "    fin_semana INTEGER,",
        "    horas_con_stock DOUBLE PRECISION,",
        "    venta_lag_1 DOUBLE PRECISION,",
        "    venta_lag_7 DOUBLE PRECISION,",
        "    venta_promedio_7d DOUBLE PRECISION,",
        "    venta_promedio_14d DOUBLE PRECISION,",
        "    stock_lag_1 DOUBLE PRECISION",
        ");",
        "",
    ]

    for _, fila in df.iterrows():
        producto = fila["name_products"]
        fecha = pd.to_datetime(fila["dt"]).date().isoformat()

        lineas.append(
            'INSERT INTO "movimientos_inventario" '
            '(producto, fecha, tipo_movimiento, cantidad, stock_resultante, horas_con_stock, descuento, '
            '"bandera de vacaciones", "bandera de actividad", "ID del producto", "ID de la tienda", '
            "primer_id_de_categoria, segundo_id_categoria, tercer_id_categoria) "
            "SELECT "
            f"{sql_text(producto)}, {sql_text(fecha)}, 'venta', {sql_num(fila.get('sale_amount'))}, "
            f"{sql_num(fila.get('stock_lag_1'))}, {sql_num(fila.get('horas_con_stock'))}, "
            f"{sql_num(fila.get('discount'), 1)}, {sql_int(fila.get('holiday_flag'))}, "
            f"{sql_int(fila.get('activity_flag'))}, {sql_text(fila.get('product_id'))}, "
            f"{sql_int(fila.get('store_id'))}, {sql_int(fila.get('first_category_id'))}, "
            f"{sql_int(fila.get('second_category_id'))}, {sql_int(fila.get('third_category_id'))} "
            "WHERE NOT EXISTS ("
            'SELECT 1 FROM "movimientos_inventario" '
            f"WHERE producto = {sql_text(producto)} AND fecha = {sql_text(fecha)}"
            ");"
        )

        lineas.append(
            'INSERT INTO "inventario_features_modelo" '
            "(dt, sale_amount, name_products, tipo_producto_app, product_id_cod, store_id_cod, "
            "first_category_id_cod, second_category_id_cod, third_category_id_cod, discount, "
            "holiday_flag_cod, activity_flag_cod, mes, dia_mes, dia_semana, fin_semana, "
            "horas_con_stock, venta_lag_1, venta_lag_7, venta_promedio_7d, venta_promedio_14d, stock_lag_1) "
            "SELECT "
            f"{sql_text(fecha)}, {sql_num(fila.get('sale_amount'))}, {sql_text(producto)}, "
            f"{sql_text(fila.get('tipo_producto_app'))}, {sql_int(fila.get('product_id'))}, "
            f"{sql_int(fila.get('store_id'))}, {sql_int(fila.get('first_category_id'))}, "
            f"{sql_int(fila.get('second_category_id'))}, {sql_int(fila.get('third_category_id'))}, "
            f"{sql_num(fila.get('discount'), 1)}, {sql_int(fila.get('holiday_flag'))}, "
            f"{sql_int(fila.get('activity_flag'))}, {sql_int(fila.get('mes'))}, "
            f"{sql_int(fila.get('dia_mes'))}, {sql_int(fila.get('dia_semana'))}, "
            f"{sql_int(fila.get('fin_semana'))}, {sql_num(fila.get('horas_con_stock'))}, "
            f"{sql_num(fila.get('venta_lag_1'))}, {sql_num(fila.get('venta_lag_7'))}, "
            f"{sql_num(fila.get('venta_promedio_7d'))}, {sql_num(fila.get('venta_promedio_14d'))}, "
            f"{sql_num(fila.get('stock_lag_1'))} "
            "WHERE NOT EXISTS ("
            'SELECT 1 FROM "inventario_features_modelo" '
            f"WHERE name_products = {sql_text(producto)} AND dt = {sql_text(fecha)}"
            ");"
        )

    lineas.extend(
        [
            "",
            "SELECT setval(pg_get_serial_sequence('\"movimientos_inventario\"', 'id_movimiento'), "
            "COALESCE((SELECT MAX(id_movimiento) FROM \"movimientos_inventario\"), 1));",
            "",
        ]
    )
    return "\n".join(lineas)


def main():
    df = cargar_historial()
    OUTPUT_SQL.write_text(crear_sql(df), encoding="utf-8")
    print(f"Productos con historial: {df['name_products'].nunique()}")
    print(f"Filas historicas exportadas: {len(df)}")
    print(f"SQL generado: {OUTPUT_SQL}")


if __name__ == "__main__":
    main()
