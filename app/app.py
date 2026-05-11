import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="ALDIMI Predict - Dashboard Pro", layout="wide", page_icon="🏥")

# --- MANEJO DE RUTAS (Compatible con VS Code y Notebooks) ---
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path('.').resolve()

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    # Definir rutas a los archivos subidos
    path_clinico = BASE_DIR / "data_modelo_clinico.csv"
    path_inventario = BASE_DIR / "inventario_limpio.csv"
    
    try:
        df_c = pd.read_csv(path_clinico)
        df_i = pd.read_csv(path_inventario)
        return df_c, df_i
    except FileNotFoundError as e:
        st.error(f"Archivo no encontrado: {e.filename}. Asegúrate de que estén en la misma carpeta que este script.")
        return None, None

# --- COMPONENTES DEL DASHBOARD ---

def render_summary(df_c, df_i):
    st.header("📊 Resumen de Operaciones")
    
    # KPIs Principales
    col1, col2, col3, col4 = st.columns(4)
    
    total_pacientes = len(df_c)
    prioridad_alta = len(df_c[df_c['Prioridad'] == 'Alta'])
    # Productos donde el stock es menor o igual al punto de reorden
    stock_critico = len(df_i[df_i['Stock_Actual'] <= df_i['Punto_Reorden']])
    # Productos próximos a vencer (según tu columna Estado_Vencimiento)
    vencimientos = len(df_i[df_i['Estado_Vencimiento'] != 'Seguro'])

    col1.metric("Total Pacientes", total_pacientes)
    col2.metric("Prioridad Alta 🚨", prioridad_alta, f"{prioridad_alta/total_pacientes:.1%}", delta_color="inverse")
    col3.metric("Stock en Reorden", stock_critico)
    col4.metric("Alertas Vencimiento", vencimientos)

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        fig_prio = px.pie(df_c, names='Prioridad', title="Carga de Prioridad Clínica",
                          color_discrete_map={'Alta':'#d32f2f', 'Media':'#f9a825', 'Baja':'#388e3c'})
        st.plotly_chart(fig_prio, use_container_width=True)
    with c2:
        fig_stock = px.bar(df_i, x='Categoria', y='Stock_Actual', color='Estado',
                           title="Inventario por Categoría y Estado", barmode='group')
        st.plotly_chart(fig_stock, use_container_width=True)

def render_clinical_analysis(df):
    st.header("🩺 Gestión de Prioridad Oncológica")
    
    # Filtro dinámico
    meses = st.slider("Meses de Supervivencia", 0, int(df['Survival_Months'].max()), (0, 60))
    df_f = df[(df['Survival_Months'] >= meses[0]) & (df['Survival_Months'] <= meses[1])]

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Distribución por Etapa")
        # Identificar columnas de etapa (Stage_)
        stage_cols = [c for c in df.columns if 'Stage_' in c]
        etapas = df_f[stage_cols].sum().sort_values(ascending=False)
        st.bar_chart(etapas)

    with col2:
        st.subheader("Pacientes con Prioridad Alta")
        st.dataframe(df_f[df_f['Prioridad'] == 'Alta'][['Year', 'Survival_Months', 'Prioridad']].head(20), use_container_width=True)

def render_inventory_control(df):
    st.header("📦 Control de Suministros Limpio")
    
    # Filtro de categorías del nuevo archivo
    categorias = st.multiselect("Filtrar Categoría:", df['Categoria'].unique(), default=df['Categoria'].unique())
    df_f = df[df['Categoria'].isin(categorias)]

    # Gráfico de Consumo vs Stock
    fig = px.scatter(df_f, x="Stock_Actual", y="Volumen_Consumo", color="Estado_Vencimiento",
                     size="Tasa_Rotacion", hover_name="Producto",
                     title="Análisis de Rotación y Riesgo de Vencimiento")
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de Pedidos Urgentes
    st.error("### ⚠️ Requerimientos de Compra Inmediata")
    urgentes = df_f[df_f['Stock_Actual'] <= df_f['Punto_Reorden']]
    st.dataframe(urgentes[['Producto', 'Categoria', 'Stock_Actual', 'Punto_Reorden', 'Estado']], use_container_width=True)

# --- EJECUCIÓN ---
def main():
    df_clinico, df_inv = load_data()

    if df_clinico is not None and df_inv is not None:
        st.sidebar.title("ALDIMI Predict")
        st.sidebar.markdown("---")
        opcion = st.sidebar.radio("Menú Principal", ["Resumen General", "Análisis Clínico", "Control de Inventario"])
        
        if opcion == "Resumen General":
            render_summary(df_clinico, df_inv)
        elif opcion == "Análisis Clínico":
            render_clinical_analysis(df_clinico)
        elif opcion == "Control de Inventario":
            render_inventory_control(df_inv)

if __name__ == "__main__":
    main()