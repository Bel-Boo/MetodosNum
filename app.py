"""
App de Streamlit: Métodos Numéricos aplicados
Incluye dos problemas, seleccionables desde un menú de navegación:

1) Interpolación de Lagrange
   Caso aplicado: tiempo de respuesta de un servidor según usuarios concurrentes.

2) Regresión lineal (mínimos cuadrados)
   Caso aplicado: temperatura de ebullición del agua según la altitud,
   aplicado a La Paz, El Alto y Zona Sur (Bolivia).

Ejecutar con:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA (una sola vez, para toda la app)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Métodos Numéricos - Ing. de Sistemas",
    page_icon="📈",
    layout="wide",
)

# ----------------------------------------------------------------------
# FUNCIONES DE CÁLCULO — LAGRANGE
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
        roots = [x_vals[j] for j in range(n) if j != i]
        poly = np.poly(roots)  # coeficientes del polinomio (x - r1)(x - r2)...
        denom = 1.0
        for j in range(n):
            if j != i:
                denom *= (x_vals[i] - x_vals[j])
        coeffs += (y_vals[i] / denom) * poly
    return np.poly1d(coeffs)


# ----------------------------------------------------------------------
# FUNCIONES DE CÁLCULO — REGRESIÓN LINEAL (MÍNIMOS CUADRADOS)
# ----------------------------------------------------------------------
FT_PER_M = 3.280839895  # 1 metro en pies


def linear_fit(x_vals, y_vals):
    """Devuelve (m, b) de la recta y = m*x + b que mejor ajusta los datos."""
    m, b = np.polyfit(x_vals, y_vals, 1)
    return m, b


def r_squared(x_vals, y_vals, m, b):
    y_pred = m * x_vals + b
    ss_res = np.sum((y_vals - y_pred) ** 2)
    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 1.0


# ----------------------------------------------------------------------
# BARRA LATERAL: MENÚ DE NAVEGACIÓN
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("🧭 Navegación")
    problema = st.radio(
        "Selecciona el problema:",
        [
            "1️⃣ Interpolación de Lagrange",
            "2️⃣ Regresión lineal: ebullición del agua",
        ],
    )
    st.divider()

# ========================================================================
# PÁGINA 1 — INTERPOLACIÓN DE LAGRANGE
# ========================================================================
def pagina_lagrange():
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

    st.title("📈 Interpolación de Lagrange")
    st.subheader("Aplicación: tiempo de respuesta de un servidor vs. usuarios concurrentes")

    st.markdown(
        """
        Ingresa los datos que obtuviste en tus pruebas de carga: para cada
        cantidad de **usuarios concurrentes** (x), el **tiempo de respuesta
        medido** en milisegundos (y).
        """
    )

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

    default_x = [10, 50, 100, 200, 500]
    default_y = [45, 80, 130, 310, 900]

    st.caption("Tip: puedes editar la tabla directamente (doble clic en una celda).")

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
        key="tabla_datos_lagrange",
    )

    x_vals = edited_df["Usuarios concurrentes (x)"].to_numpy(dtype=float)
    y_vals = edited_df["Tiempo de respuesta ms (y)"].to_numpy(dtype=float)

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

    if error:
        st.stop()

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

    x_plot = np.linspace(x_min, x_max, 300)
    y_plot = lagrange_interpolate(x_vals, y_vals, x_plot)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_plot, y=y_plot,
            mode="lines",
            name="Polinomio de Lagrange",
            line=dict(color="#2563eb", width=3),
            hovertemplate="Usuarios: %{x:.0f}<br>Tiempo estimado: %{y:.2f} ms<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            name="Datos medidos",
            marker=dict(color="#dc2626", size=11, symbol="circle"),
            hovertemplate="Usuarios: %{x:.0f}<br>Tiempo medido: %{y:.2f} ms<extra></extra>",
        )
    )

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


# ========================================================================
# PÁGINA 2 — REGRESIÓN LINEAL: EBULLICIÓN DEL AGUA
# ========================================================================
def pagina_regresion():
    with st.sidebar:
        st.header("📘 Contexto del problema")
        st.markdown(
            """
            La temperatura de ebullición del agua $T_B$ depende de la
            **altitud** $h$: a mayor altura, menor presión atmosférica,
            y el agua hierve a una temperatura más baja.

            Se midió $T_B$ a distintas altitudes $h$ (en pies). Como los
            puntos **no son exactos** (vienen de mediciones con algo de
            error), no tiene sentido forzar una curva que pase por todos
            ellos — en vez de eso buscamos la **recta que mejor se ajuste**
            en promedio.

            **Objetivo:** encontrar $T_B = mh + b$ por **mínimos cuadrados**
            y usarla para estimar la temperatura de ebullición en
            **La Paz, El Alto y Zona Sur** (Bolivia).
            """
        )
        st.divider()
        st.markdown(
            """
            **¿Por qué regresión y no Lagrange?**

            Porque aquí no buscamos un polinomio que pase exactamente por
            cada dato medido, sino la tendencia general — la recta que
            minimiza el error total frente a todos los puntos.
            """
        )

    st.title("🌡️ Regresión lineal: temperatura de ebullición del agua")
    st.subheader("Aplicación: altitud vs. temperatura de ebullición — caso La Paz, Bolivia")

    st.markdown(
        """
        Ingresa los datos medidos de altitud $h$ (en pies) y temperatura de
        ebullición $T_B$ (en °F). Se calculará la recta $T_B = mh + b$ que
        mejor se ajusta por mínimos cuadrados.
        """
    )

    # --------------------------------------------------------------
    # 1) Datos medidos
    # --------------------------------------------------------------
    st.markdown("### 1️⃣ Datos medidos (h en pies, T en °F)")

    default_h = [-1000, 0, 3000, 8000, 15000, 22000, 28000]
    default_T = [213.9, 212, 206.2, 196.2, 184.4, 172.6, 163.1]

    col_a, col_b = st.columns([1, 2])
    with col_a:
        n_puntos = st.number_input(
            "¿Cuántos puntos de medición tienes?",
            min_value=2,
            max_value=15,
            value=len(default_h),
            step=1,
            key="n_puntos_regresion",
        )

    st.caption("Tip: puedes editar la tabla directamente (doble clic en una celda).")

    data_default = pd.DataFrame(
        {
            "Altitud h (ft)": [
                default_h[i] if i < len(default_h) else 0 for i in range(n_puntos)
            ],
            "Temperatura T (°F)": [
                default_T[i] if i < len(default_T) else 0.0 for i in range(n_puntos)
            ],
        }
    )

    edited_df = st.data_editor(
        data_default,
        num_rows="fixed",
        use_container_width=True,
        key="tabla_datos_regresion",
    )

    h_vals = edited_df["Altitud h (ft)"].to_numpy(dtype=float)
    T_vals = edited_df["Temperatura T (°F)"].to_numpy(dtype=float)

    error = False
    if np.any(np.isnan(h_vals)) or np.any(np.isnan(T_vals)):
        st.error("⚠️ Completa todos los valores de la tabla antes de continuar.")
        error = True
    if len(set(h_vals)) < 2:
        st.error("⚠️ Se necesitan al menos 2 valores distintos de altitud.")
        error = True

    if error:
        st.stop()

    # --------------------------------------------------------------
    # 2) Ecuación de mejor ajuste
    # --------------------------------------------------------------
    st.markdown("### 2️⃣ Ecuación de mejor ajuste (mínimos cuadrados)")

    m, b = linear_fit(h_vals, T_vals)
    r2 = r_squared(h_vals, T_vals, m, b)

    col1, col2, col3 = st.columns(3)
    col1.metric("Pendiente m", f"{m:.6f} °F/ft")
    col2.metric("Intercepto b", f"{b:.4f} °F")
    col3.metric("R² (bondad de ajuste)", f"{r2:.5f}")

    st.code(f"T_B(h) = {m:.6f}·h + {b:.4f}", language="text")

    # --------------------------------------------------------------
    # 3) Estimación en una altitud específica
    # --------------------------------------------------------------
    st.markdown("### 3️⃣ Estima la temperatura de ebullición en una altitud")

    col1, col2 = st.columns([1, 1])
    with col1:
        unidad = st.radio("Unidad de la altitud a consultar:", ["metros (m)", "pies (ft)"], horizontal=True)
        if unidad == "metros (m)":
            alt_m = st.number_input("Altitud (m)", min_value=-500.0, max_value=9000.0, value=5000.0, step=100.0)
            alt_ft = alt_m * FT_PER_M
        else:
            alt_ft = st.number_input("Altitud (ft)", min_value=-2000.0, max_value=30000.0, value=16404.0, step=100.0)
            alt_m = alt_ft / FT_PER_M

    with col2:
        T_F = m * alt_ft + b
        T_C = (T_F - 32) * 5 / 9
        st.metric(
            label=f"Temperatura de ebullición a {alt_m:.0f} m ({alt_ft:.0f} ft)",
            value=f"{T_F:.2f} °F  /  {T_C:.2f} °C",
        )

    if alt_ft < min(h_vals) or alt_ft > max(h_vals):
        st.warning(
            "⚠️ Estás **extrapolando** (fuera del rango medido de altitudes). "
            "El resultado es menos confiable."
        )

    # --------------------------------------------------------------
    # 4) Caso aplicado: La Paz, El Alto y Zona Sur
    # --------------------------------------------------------------
    st.markdown("### 4️⃣ Caso aplicado: La Paz, El Alto y Zona Sur (Bolivia)")

    st.markdown(
        """
        Usando la misma ecuación de regresión, se estima la temperatura de
        ebullición del agua en distintas zonas de La Paz, Bolivia. Las
        altitudes son valores de referencia aproximados (la topografía de
        la ciudad varía bastante entre barrios).
        """
    )

    zonas_default = pd.DataFrame(
        {
            "Zona": ["La Paz (centro, Plaza Murillo)", "Zona Sur", "El Alto"],
            "Altitud (m)": [3640, 3200, 4150],
        }
    )

    zonas_df = st.data_editor(
        zonas_default,
        num_rows="dynamic",
        use_container_width=True,
        key="tabla_zonas",
        help="Puedes editar las altitudes o agregar más zonas.",
    )

    zonas_df = zonas_df.dropna()
    zonas_alt_m = zonas_df["Altitud (m)"].to_numpy(dtype=float)
    zonas_alt_ft = zonas_alt_m * FT_PER_M
    zonas_T_F = m * zonas_alt_ft + b
    zonas_T_C = (zonas_T_F - 32) * 5 / 9

    resultado_df = pd.DataFrame(
        {
            "Zona": zonas_df["Zona"].to_numpy(),
            "Altitud (m)": zonas_alt_m,
            "Altitud (ft)": np.round(zonas_alt_ft, 0),
            "T_B (°F)": np.round(zonas_T_F, 2),
            "T_B (°C)": np.round(zonas_T_C, 2),
        }
    )

    st.dataframe(resultado_df, use_container_width=True, hide_index=True)

    for _, fila in resultado_df.iterrows():
        if fila["Altitud (ft)"] < min(h_vals) or fila["Altitud (ft)"] > max(h_vals):
            st.warning(f"⚠️ {fila['Zona']}: fuera del rango medido — resultado extrapolado.")

    # --------------------------------------------------------------
    # 5) Gráfico
    # --------------------------------------------------------------
    st.markdown("### 5️⃣ Gráfico: puntos medidos, recta ajustada y zonas de La Paz")

    h_min_plot = min(np.min(h_vals), np.min(zonas_alt_ft)) - 1000
    h_max_plot = max(np.max(h_vals), np.max(zonas_alt_ft)) + 1000
    h_plot = np.linspace(h_min_plot, h_max_plot, 300)
    T_plot = m * h_plot + b

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=h_plot, y=T_plot,
            mode="lines",
            name="Recta de mejor ajuste",
            line=dict(color="#2563eb", width=3),
            hovertemplate="h: %{x:.0f} ft<br>T: %{y:.2f} °F<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=h_vals, y=T_vals,
            mode="markers",
            name="Datos medidos",
            marker=dict(color="#dc2626", size=11, symbol="circle"),
            hovertemplate="h: %{x:.0f} ft<br>T medido: %{y:.2f} °F<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=zonas_alt_ft, y=zonas_T_F,
            mode="markers+text",
            name="Zonas de La Paz",
            marker=dict(color="#16a34a", size=14, symbol="diamond"),
            text=zonas_df["Zona"].to_numpy(),
            textposition="top center",
            hovertemplate="%{text}<br>h: %{x:.0f} ft<br>T: %{y:.2f} °F<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[alt_ft], y=[T_F],
            mode="markers",
            name="Consulta actual",
            marker=dict(color="#f59e0b", size=14, symbol="star"),
            hovertemplate="h: %{x:.0f} ft<br>T: %{y:.2f} °F<extra></extra>",
        )
    )

    fig.update_layout(
        title="Regresión lineal: altitud vs. temperatura de ebullición",
        xaxis_title="Altitud h (ft)",
        yaxis_title="Temperatura de ebullición T_B (°F)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=40, r=20),
        height=480,
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 ¿Cómo se calculó esto? (paso a paso)"):
        st.markdown(
            r"""
            La recta de mínimos cuadrados $T_B = mh + b$ se obtiene
            minimizando la suma de errores cuadráticos entre los datos
            medidos y la recta:

            $$m = \frac{n\sum h_iT_i - \sum h_i \sum T_i}{n\sum h_i^2 - (\sum h_i)^2}$$

            $$b = \bar{T} - m\bar{h}$$

            Una vez obtenidos $m$ y $b$, la ecuación se evalúa en
            cualquier altitud $h$ (convertida a pies) para estimar la
            temperatura de ebullición correspondiente.

            Para convertir la respuesta de °F a °C se usa:

            $$T_{°C} = \frac{5}{9}(T_{°F} - 32)$$
            """
        )

    st.info(
        """
        💡 **Nota:** a diferencia de Lagrange, esta recta **no pasa
        exactamente** por todos los puntos medidos — minimiza el error
        total. Por eso R² (que mide qué tan bien se ajusta) casi nunca
        es exactamente 1, salvo que los datos sean perfectamente
        lineales.
        """
    )


# ========================================================================
# ENRUTADOR
# ========================================================================
if problema.startswith("1"):
    pagina_lagrange()
else:
    pagina_regresion()
