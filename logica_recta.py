"""Lógica numérica para el ajuste de una recta a partir de dos puntos.

Asignatura: Ciencia de Datos
Cuatrimestre: 7mo
Alumno: [Saimon Josue Tecalco Martinez]
Matrícula: [2403230384]
Fecha: [27/09/2026]

Este módulo no depende de Streamlit: contiene la validación de
datos, el cálculo de la pendiente y la ordenada al origen, y la generación
de la tabla de interpolación y predicción.
"""

import numpy as np
import pandas as pd


class ErrorDeDatos(ValueError):
    """Se lanza cuando el archivo de entrada no cumple los requisitos."""


def validar_puntos(datos_crudos: pd.DataFrame) -> pd.DataFrame:
    """Valida el DataFrame leído del .csv y lo devuelve ordenado por x.

    Reglas: columnas 'x' e 'y' (sin importar mayúsculas), exactamente dos
    registros, valores numéricos sin nulos y x1 distinto de x2.

    Args:
        datos_crudos: DataFrame tal como lo entrega ``pd.read_csv``.

    Returns:
        DataFrame con columnas ``x`` e ``y`` de tipo float, con dos filas
        ordenadas de menor a mayor x.

    Raises:
        ErrorDeDatos: si alguna de las reglas no se cumple.
    """
    puntos = datos_crudos.copy()
    puntos.columns = [str(col).strip().lower() for col in puntos.columns]

    if not {"x", "y"}.issubset(puntos.columns):
        raise ErrorDeDatos("El archivo debe tener las columnas 'x' e 'y'.")

    puntos = puntos[["x", "y"]]

    if len(puntos) != 2:
        raise ErrorDeDatos(
            f"El archivo debe tener exactamente 2 registros; "
            f"se encontraron {len(puntos)}."
        )

    puntos = puntos.apply(pd.to_numeric, errors="coerce")
    if puntos.isna().any().any():
        raise ErrorDeDatos("Los valores de x e y deben ser numéricos.")

    puntos = puntos.astype(float).sort_values("x").reset_index(drop=True)

    if puntos.loc[0, "x"] == puntos.loc[1, "x"]:
        raise ErrorDeDatos(
            "x1 y x2 deben ser distintos (la recta sería vertical)."
        )

    return puntos


def calcular_pendiente_y_ordenada(puntos: pd.DataFrame) -> tuple:
    """Calcula la pendiente m y la ordenada al origen b.

    Fórmulas:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

    Args:
        puntos: DataFrame validado con dos filas y columnas x e y.

    Returns:
        Tupla ``(m, b)`` de valores float.
    """
    delta_x, delta_y = np.diff(puntos["x"].to_numpy()), np.diff(
        puntos["y"].to_numpy()
    )
    pendiente = float(delta_y[0] / delta_x[0])
    ordenada = float(puntos.loc[0, "y"] - pendiente * puntos.loc[0, "x"])
    return pendiente, ordenada


def evaluar_recta(x: np.ndarray, pendiente: float, ordenada: float):
    """Evalúa y = m * x + b de forma vectorizada."""
    return pendiente * np.asarray(x, dtype=float) + ordenada


def generar_tabla_valores(
    puntos: pd.DataFrame,
    pendiente: float,
    ordenada: float,
    cantidad_futuros: int = 5,
) -> pd.DataFrame:
    """Genera la tabla de interpolación y de predicción (extrapolación).

    Incluye los enteros dentro de [x1, x2], los propios x1 y x2 y, a
    continuación, ``cantidad_futuros`` enteros consecutivos con x > x2.

    Args:
        puntos: DataFrame validado con los dos puntos de referencia.
        pendiente: Pendiente m de la recta.
        ordenada: Ordenada al origen b.
        cantidad_futuros: Número de valores de x posteriores a x2.

    Returns:
        DataFrame con columnas ``x``, ``y`` y ``tipo``.
    """
    x1, x2 = puntos["x"].to_numpy()

    x_intervalo = np.union1d(
        np.arange(np.ceil(x1), np.floor(x2) + 1), [x1, x2]
    )
    x_futuros = np.floor(x2) + 1 + np.arange(cantidad_futuros)

    x_todos = np.concatenate([x_intervalo, x_futuros])
    tipos = np.where(
        np.isin(x_todos, [x1, x2]),
        "Punto original",
        np.where(x_todos > x2, "Predicción", "Interpolación"),
    )

    return pd.DataFrame(
        {
            "x": x_todos,
            "y": evaluar_recta(x_todos, pendiente, ordenada),
            "tipo": tipos,
        }
    )


def formatear_ecuacion(pendiente: float, ordenada: float) -> str:
    """Devuelve la ecuación como texto, por ejemplo 'y = 1.5714x + 1.4286'."""
    signo = "+" if ordenada >= 0 else "-"
    return f"y = {pendiente:.4f}x {signo} {abs(ordenada):.4f}"
