from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOGISTICO_PATH = ROOT / "data" / "dataset_logistico_1.csv"
OUTPUT_SQL = ROOT / "data" / "reemplazar_inventario_desde_historial.sql"
OUTPUT_CSV = ROOT / "data" / "inventario_desde_historial.csv"


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
        return str(int(round(float(value))))
    except Exception:
        return str(default)


def categoria_alimento(nombre):
    texto = nombre.lower()

    if any(palabra in texto for palabra in ["leche", "queso", "huevo"]):
        return "Dairy"
    if any(palabra in texto for palabra in ["pan", "harina"]):
        return "Bakery"
    if any(palabra in texto for palabra in ["pescado", "jurel", "sardina", "atun"]):
        return "Seafood"
    if any(palabra in texto for palabra in ["pollo", "carne"]):
        return "Meat & Poultry"
    if any(
        palabra in texto
        for palabra in [
            "papa",
            "camote",
            "yuca",
            "zapallo",
            "platano",
            "naranja",
            "lechuga",
            "cebolla",
            "zanahoria",
            "limon",
            "manzana",
            "tomate",
        ]
    ):
        return "Fruits & Vegetables"
    if any(palabra in texto for palabra in ["arroz", "fideo", "avena", "lenteja", "frejol", "arveja"]):
        return "Grains & Pulses"
    if "aceite" in texto:
        return "Oils & Fats"
    if "agua" in texto:
        return "Beverages"
    if any(palabra in texto for palabra in ["azucar", "sal"]):
        return "Pantry"

    return "Alimentos"


def categoria_clinica(nombre):
    texto = nombre.lower()

    if any(palabra in texto for palabra in ["metotrexato", "ciclofosfamida", "vincristina", "doxorrubicina"]):
        return "Oncológicos"
    if any(palabra in texto for palabra in ["suero", "dextrosa", "cloruro"]):
        return "Insumos"
    if any(palabra in texto for palabra in ["cateter", "jeringa", "equipo", "llave"]):
        return "Soporte"
    if any(palabra in texto for palabra in ["pediasure", "formula", "espesante"]):
        return "Nutrición"
    if any(palabra in texto for palabra in ["mascarilla", "guantes", "alcohol", "jabon"]):
        return "Higiene"
    if "tubo" in texto:
        return "Laboratorio"
    if "rayos" in texto:
        return "Imágenes"
    if "hidrocortisona" in texto:
        return "Tópicos"
    if "paracetamol" in texto or "ibuprofeno" in texto:
        return "Analgesicos"
    if "ceftriaxona" in texto:
        return "Antibióticos"
    if "ondansetron" in texto:
        return "Antieméticos"

    return "Insumos"


def asignar_categoria(nombre, tipo):
    if tipo == "Alimento":
        return categoria_alimento(nombre)
    if tipo == "Clinico":
        return categoria_clinica(nombre)
    return "Producto general"


def estado_vencimiento(dias):
    if dias < 0:
        return "Vencido"
    if dias <= 15:
        return "Crítico (15 días)"
    if dias <= 30:
        return "Próximo a vencer"
    return "Seguro"


def estado_stock(stock, punto_reorden):
    if stock <= 0:
        return "Sin stock"
    if stock <= punto_reorden:
        return "Reorden"
    return "Activo"


def construir_inventario():
    df = pd.read_csv(LOGISTICO_PATH)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df[~df["name_products"].astype(str).str.match(r"^Producto\s+\d+$", flags=re.IGNORECASE, na=False)]
    df = df.sort_values(["name_products", "dt"])

    productos = []

    for idx, (nombre, grupo) in enumerate(df.groupby("name_products"), start=1):
        grupo = grupo.sort_values("dt")
        ultimo = grupo.iloc[-1]
        ultimos_30 = grupo.tail(30)

        tipo = str(ultimo.get("tipo_producto_app", "Producto general"))
        categoria = asignar_categoria(nombre, tipo)
        demanda_promedio = float(ultimos_30["sale_amount"].mean())
        demanda_max = float(ultimos_30["sale_amount"].max())

        factor_stock = 0.65 + ((idx % 7) * 0.18)
        stock_actual = max(0, round(demanda_promedio * factor_stock))
        punto_reorden = max(1, round(demanda_promedio * 1.10))
        cantidad_pedido = max(0, round((demanda_max * 1.20) - stock_actual))
        volumen_consumo = max(1, round(demanda_promedio))
        tasa_rotacion = round(volumen_consumo / max(stock_actual, 1), 2)

        dias_recepcion = 3 + (idx % 11)
        dias_vencimiento = 30 + ((idx * 7) % 240)
        ultimo_pedido = pd.Timestamp("2026-07-04") - pd.Timedelta(days=idx % 21)
        fecha_recepcion = ultimo_pedido + pd.Timedelta(days=dias_recepcion)
        fecha_vencimiento = pd.Timestamp("2026-07-04") + pd.Timedelta(days=dias_vencimiento)

        productos.append(
            {
                "Producto": nombre,
                "Categoria": categoria,
                "Stock_Actual": stock_actual,
                "Punto_Reorden": punto_reorden,
                "Cantidad_Pedido": cantidad_pedido,
                "Fecha_Recepcion": fecha_recepcion.date().isoformat(),
                "Ultimo_Pedido": ultimo_pedido.date().isoformat(),
                "Fecha_Vencimiento": fecha_vencimiento.date().isoformat(),
                "Volumen_Consumo": volumen_consumo,
                "Tasa_Rotacion": tasa_rotacion,
                "Estado": estado_stock(stock_actual, punto_reorden),
                "Lead_Time_Check": float(dias_recepcion),
                "Dias_para_Vencimiento": int(dias_vencimiento),
                "Estado_Vencimiento": estado_vencimiento(dias_vencimiento),
            }
        )

    return pd.DataFrame(productos)


def crear_sql(inventario):
    lineas = [
        'DROP TABLE IF EXISTS "inventario" CASCADE;',
        "",
        'CREATE TABLE IF NOT EXISTS "inventario" (',
        "    id_registro BIGSERIAL,",
        '    "Producto" TEXT,',
        '    "Categoria" TEXT,',
        '    "Stock_Actual" INTEGER,',
        '    "Punto_Reorden" INTEGER,',
        '    "Cantidad_Pedido" INTEGER,',
        '    "Fecha_Recepcion" TEXT,',
        '    "Ultimo_Pedido" TEXT,',
        '    "Fecha_Vencimiento" TEXT,',
        '    "Volumen_Consumo" INTEGER,',
        '    "Tasa_Rotacion" DOUBLE PRECISION,',
        '    "Estado" TEXT,',
        '    "Lead_Time_Check" DOUBLE PRECISION,',
        '    "Dias_para_Vencimiento" INTEGER,',
        '    "Estado_Vencimiento" TEXT',
        ");",
        "",
    ]

    columnas = [
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
    ]

    for idx, fila in inventario.reset_index(drop=True).iterrows():
        valores = [
            str(idx + 1),
            sql_text(fila["Producto"]),
            sql_text(fila["Categoria"]),
            sql_int(fila["Stock_Actual"]),
            sql_int(fila["Punto_Reorden"]),
            sql_int(fila["Cantidad_Pedido"]),
            sql_text(fila["Fecha_Recepcion"]),
            sql_text(fila["Ultimo_Pedido"]),
            sql_text(fila["Fecha_Vencimiento"]),
            sql_int(fila["Volumen_Consumo"]),
            sql_num(fila["Tasa_Rotacion"]),
            sql_text(fila["Estado"]),
            sql_num(fila["Lead_Time_Check"]),
            sql_int(fila["Dias_para_Vencimiento"]),
            sql_text(fila["Estado_Vencimiento"]),
        ]

        columnas_sql = ", ".join([f'"{col}"' if col != "id_registro" else col for col in columnas])
        lineas.append(
            f'INSERT INTO "inventario" ({columnas_sql}) VALUES ({", ".join(valores)});'
        )

    lineas.extend(
        [
            "",
            "SELECT setval(pg_get_serial_sequence('\"inventario\"', 'id_registro'), "
            "COALESCE((SELECT MAX(id_registro) FROM \"inventario\"), 1));",
            "",
        ]
    )

    return "\n".join(lineas)


def main():
    inventario = construir_inventario()
    inventario.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    OUTPUT_SQL.write_text(crear_sql(inventario), encoding="utf-8")

    print(f"Productos generados: {len(inventario)}")
    print("Categorias:")
    print(inventario["Categoria"].value_counts().to_string())
    print(f"CSV generado: {OUTPUT_CSV}")
    print(f"SQL generado: {OUTPUT_SQL}")


if __name__ == "__main__":
    main()
