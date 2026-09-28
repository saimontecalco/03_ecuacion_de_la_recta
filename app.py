"""Ajuste de recta y predicción interactiva con Streamlit.

Asignatura: Ciencia de Datos
Cuatrimestre: 7mo
Alumno: [Saimon Josue Tecalco Martinez]
Matrícula: [2403230384]
Fecha: [27/09/2026]

Ejecución:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from logica_recta import (
    ErrorDeDatos,
    calcular_pendiente_y_ordenada,
    evaluar_recta,
    formatear_ecuacion,
    generar_tabla_valores,
    validar_puntos,
)


def crear_grafica(
    puntos: pd.DataFrame,
    tabla_valores: pd.DataFrame,
    pendiente: float,
    ordenada: float,
) -> plt.Figure:
    """Construye la figura con los puntos, la recta y los valores predichos.

    Args:
        puntos: DataFrame con los puntos P1 y P2.
        tabla_valores: Tabla de interpolación y predicción.
        pendiente: Pendiente m de la recta.
        ordenada: Ordenada al origen b.

    Returns:
        Figura de Matplotlib lista para mostrarse en Streamlit.
    """
    x_min = puntos["x"].min()
    x_max = tabla_valores["x"].max()
    x_recta = np.linspace(x_min, x_max, 200)
    y_recta = evaluar_recta(x_recta, pendiente, ordenada)

    figura, eje = plt.subplots(figsize=(9, 5))
    eje.plot(
        x_recta,
        y_recta,
        color="tab:blue",
        linewidth=2,
        label=f"Recta ajustada: {formatear_ecuacion(pendiente, ordenada)}",
    )

    predichos = tabla_valores[tabla_valores["tipo"] == "Predicción"]
    eje.scatter(
        predichos["x"],
        predichos["y"],
        color="tab:green",
        marker="D",
        s=45,
        zorder=3,
        label="Valores predichos (x > x2)",
    )

    colores = ["tab:red", "tab:orange"]
    for indice, fila in puntos.iterrows():
        eje.scatter(
            fila["x"],
            fila["y"],
            color=colores[indice],
            s=140,
            edgecolor="black",
            zorder=4,
            label=f"P{indice + 1}({fila['x']:g}, {fila['y']:g})",
        )

    eje.set_title("Ajuste lineal a partir de dos puntos")
    eje.set_xlabel("Eje X")
    eje.set_ylabel("Eje Y")
    eje.grid(True, linestyle="--", alpha=0.6)
    eje.legend()
    figura.tight_layout()
    return figura


def main() -> None:
    """Define la interfaz de Streamlit y coordina la lógica numérica."""
    st.set_page_config(page_title="Ajuste de recta", page_icon="📈")
    st.title("📈 Ajuste de recta y predicción")
    st.write(
        "Carga un archivo **.csv** con las columnas `x` e `y` y "
        "exactamente dos puntos: P1 y P2."
    )

    archivo = st.file_uploader("Archivo .csv", type="csv")
    cantidad_futuros = st.sidebar.slider(
        "Valores futuros a predecir (x > x2)", 1, 20, 5
    )

    if archivo is None:
        st.info("Esperando archivo. Puedes usar `datos/puntos.csv`.")
        return

    try:
        puntos = validar_puntos(pd.read_csv(archivo))
    except (
        ErrorDeDatos,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        st.error(f"No se pudo procesar el archivo: {error}")
        return

    pendiente, ordenada = calcular_pendiente_y_ordenada(puntos)

    st.subheader("Puntos de referencia")
    st.dataframe(
        puntos.rename(index={0: "P1", 1: "P2"}), width="stretch"
    )

    st.subheader("Parámetros de la recta")
    col_m, col_b = st.columns(2)
    col_m.metric("Pendiente (m)", f"{pendiente:.6f}")
    col_b.metric("Ordenada al origen (b)", f"{ordenada:.6f}")
    st.latex(formatear_ecuacion(pendiente, ordenada))

    tabla_valores = generar_tabla_valores(
        puntos, pendiente, ordenada, cantidad_futuros
    )

    st.subheader("Gráfica")
    st.pyplot(crear_grafica(puntos, tabla_valores, pendiente, ordenada))

    st.subheader("Tabla de interpolación y predicción")
    st.dataframe(
        tabla_valores.style.format({"x": "{:g}", "y": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()
