"""
App de Streamlit: Interpolación de Lagrange
Caso aplicado: Estimación del tiempo de respuesta de un servidor
según el número de usuarios concurrentes.

Ejecutar con:
    streamlit run app_lagrange.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Interpolación de Lagrange - Ing. de Sistemas",
    page_icon="📈",
    layout="wide",
)

# ----------------------------------------------------------------------
# FUNCIONES DE CÁLCULO
# ----------------------------------------------------------------------
def lagrange_basis(x_vals, i, x):
    """Calcula L_i(x) para un valor escalar o array x."""
    xi = x_vals[i]
    result = np.ones_like(x, dtype=float)
    for j, xj in enumerate(x_vals):
        if j != i:
            result *= (x - xj) / (xi - xj)
    return result


def lagrange_interpolate(x_vals, y_vals, x):
    """Evalúa el polinomio interpolador de Lagrange en x (escalar o array)."""
    x = np.atleast_1d(np.array(x, dtype=float))
    total = np.zeros_like(x, dtype=float)
    for i, yi in enumerate(y_vals):
        total += yi * lagrange_basis(x_vals, i, x)
    return total


def build_polynomial_string(x_vals, y_vals):
    """Construye una representación simbólica aproximada del polinomio (coeficientes)."""
    n = len(x_vals)
    coeffs = np.zeros(n)
    for i in range(n):
        # Construye el polinomio L_i(x) como coeficientes usando numpy.poly
        roots = [x_vals[j] for j in range(n) if j != i]
        poly = np.poly(roots)  # coeficientes del polinomio (x - r1)(x - r2)...
        denom = 1.0
        for j in range(n):
            if j != i:
                denom *= (x_vals[i] - x_vals[j])
        coeffs += (y_vals[i] / denom) * poly
    return np.poly1d(coeffs)


# ----------------------------------------------------------------------
# BARRA LATERAL: CONTEXTO DEL EJERCICIO
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("📘 Contexto del problema")
    st.markdown(
        """
        Eres el encargado de **pruebas de rendimiento** de un servidor web.
        Durante pruebas de carga mediste el **tiempo de respuesta promedio
        (en milisegundos)** para distintos niveles de **usuarios concurrentes**.

        No probaste todos los niveles posibles de usuarios (sería muy caro
        y lento), así que solo tienes algunos puntos de referencia.

        **Objetivo:** usar **interpolación de Lagrange** para estimar el
        tiempo de respuesta en niveles de usuarios que *no* mediste
        directamente, sin necesidad de correr una nueva prueba de carga.
        """
    )
    st.divider()
    st.markdown(
        """
        **¿Por qué Lagrange y no una regresión lineal?**

        Porque no buscamos la "mejor recta promedio", sino un polinomio
        que **pase exactamente** por los datos medidos — útil cuando
        confiamos en que cada medición es precisa y queremos una
        estimación exacta en esos puntos.
        """
    )

# ----------------------------------------------------------------------
# ENCABEZADO PRINCIPAL
# ----------------------------------------------------------------------
st.title("📈 Interpolación de Lagrange")
st.subheader("Aplicación: tiempo de respuesta de un servidor vs. usuarios concurrentes")

st.markdown(
    """
    Ingresa los datos que obtuviste en tus pruebas de carga: para cada
    cantidad de **usuarios concurrentes** (x), el **tiempo de respuesta
    medido** en milisegundos (y).
    """
)

# ----------------------------------------------------------------------
# ENTRADA DE DATOS
# ----------------------------------------------------------------------
st.markdown("### 1️⃣ Ingresa tus datos medidos")

col_a, col_b = st.columns([1, 2])
with col_a:
    n_puntos = st.number_input(
        "¿Cuántos puntos de medición tienes?",
        min_value=2,
        max_value=15,
        value=3,
        step=1,
        help="Mínimo 2 puntos. Cada punto es una prueba de carga distinta.",
    )

# Valores por defecto de ejemplo (para que la interfaz no aparezca vacía)
default_x = [10, 50, 100, 200, 500]
default_y = [45, 80, 130, 310, 900]

st.caption(
    "Tip: puedes editar la tabla directamente (doble clic en una celda)."
)

data_default = pd.DataFrame(
    {
        "Usuarios concurrentes (x)": [
            default_x[i] if i < len(default_x) else 0 for i in range(n_puntos)
        ],
        "Tiempo de respuesta ms (y)": [
            default_y[i] if i < len(default_y) else 0 for i in range(n_puntos)
        ],
    }
)

edited_df = st.data_editor(
    data_default,
    num_rows="fixed",
    use_container_width=True,
    key="tabla_datos",
)

x_vals = edited_df["Usuarios concurrentes (x)"].to_numpy(dtype=float)
y_vals = edited_df["Tiempo de respuesta ms (y)"].to_numpy(dtype=float)

# ----------------------------------------------------------------------
# VALIDACIONES
# ----------------------------------------------------------------------
error = False
if len(set(x_vals)) != len(x_vals):
    st.error(
        "⚠️ Los valores de 'Usuarios concurrentes (x)' deben ser todos "
        "diferentes entre sí (no puede haber dos mediciones con el mismo x)."
    )
    error = True

if np.any(np.isnan(x_vals)) or np.any(np.isnan(y_vals)):
    st.error("⚠️ Completa todos los valores de la tabla antes de continuar.")
    error = True

# ----------------------------------------------------------------------
# CÁLCULO Y RESULTADOS
# ----------------------------------------------------------------------
if not error:
    st.markdown("### 2️⃣ Estima un valor intermedio")

    col1, col2 = st.columns([1, 1])
    with col1:
        x_min, x_max = float(min(x_vals)), float(max(x_vals))
        x_consulta = st.slider(
            "Selecciona el número de usuarios concurrentes a estimar",
            min_value=x_min,
            max_value=x_max,
            value=(x_min + x_max) / 2,
            help="Puedes elegir cualquier valor dentro del rango medido.",
        )
    with col2:
        y_estimado = lagrange_interpolate(x_vals, y_vals, x_consulta)[0]
        st.metric(
            label=f"Tiempo de respuesta estimado para {x_consulta:.0f} usuarios",
            value=f"{y_estimado:.2f} ms",
        )

    if x_consulta < x_min or x_consulta > x_max:
        st.warning(
            "⚠️ Estás **extrapolando** (fuera del rango medido). "
            "La interpolación de Lagrange puede dar resultados poco confiables aquí."
        )

    st.markdown("### 3️⃣ Polinomio interpolador y gráfico")

    poly = build_polynomial_string(x_vals, y_vals)
    st.code(f"P(x) = {poly}", language="text")

    # Gráfico interactivo (Plotly)
    x_plot = np.linspace(x_min, x_max, 300)
    y_plot = lagrange_interpolate(x_vals, y_vals, x_plot)

    fig = go.Figure()

    # Curva del polinomio interpolador
    fig.add_trace(
        go.Scatter(
            x=x_plot, y=y_plot,
            mode="lines",
            name="Polinomio de Lagrange",
            line=dict(color="#2563eb", width=3),
            hovertemplate="Usuarios: %{x:.0f}<br>Tiempo estimado: %{y:.2f} ms<extra></extra>",
        )
    )

    # Puntos medidos originales
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            name="Datos medidos",
            marker=dict(color="#dc2626", size=11, symbol="circle"),
            hovertemplate="Usuarios: %{x:.0f}<br>Tiempo medido: %{y:.2f} ms<extra></extra>",
        )
    )

    # Punto de la estimación seleccionada
    fig.add_trace(
        go.Scatter(
            x=[x_consulta], y=[y_estimado],
            mode="markers",
            name="Estimación seleccionada",
            marker=dict(color="#16a34a", size=14, symbol="diamond"),
            hovertemplate="Usuarios: %{x:.0f}<br>Estimado: %{y:.2f} ms<extra></extra>",
        )
    )

    fig.update_layout(
        title="Interpolación de Lagrange: tiempo de respuesta del servidor",
        xaxis_title="Usuarios concurrentes",
        yaxis_title="Tiempo de respuesta (ms)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=40, r=20),
        height=480,
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 ¿Cómo se calculó esto? (paso a paso)"):
        st.markdown(
            r"""
            El polinomio interpolador de Lagrange se construye como:

            $$P(x) = \sum_{i=0}^{n} y_i \cdot L_i(x)$$

            donde cada $L_i(x)$ vale 1 en su propio punto $x_i$ y 0 en
            todos los demás puntos medidos:

            $$L_i(x) = \prod_{j \ne i} \frac{x - x_j}{x_i - x_j}$$

            Con tus $n = %d$ puntos, se generan $n$ polinomios base
            $L_i(x)$ que luego se combinan (ponderados por cada $y_i$)
            para formar el polinomio final mostrado arriba.
            """ % len(x_vals)
        )

    st.info(
        """
        💡 **Nota de ingeniería:** en un caso real, si tuvieras más de
        8-10 puntos, el polinomio de Lagrange puede oscilar mucho
        (fenómeno de Runge) y sería mejor usar **splines cúbicos** o
        un modelo de regresión. Lagrange es ideal cuando tienes
        **pocos puntos de medición confiables**.
        """
    )
else:
    st.stop()
