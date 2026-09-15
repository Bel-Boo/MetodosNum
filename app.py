"""
App de Streamlit: Menú de ejercicios de Métodos Numéricos
============================================================
1) Interpolación de Lagrange
   Caso: tiempo de respuesta de un servidor vs. usuarios concurrentes
2) Regresión Lineal (mínimos cuadrados)
   Caso: temperatura de ebullición del agua vs. altitud
        (aplicado a El Alto y la Zona Sur de La Paz, Bolivia)

Ejecutar con:
    streamlit run app_ejercicios.py

Dependencias:
    pip install streamlit numpy pandas plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Métodos Numéricos - Ejercicios Aplicados",
    page_icon="📊",
    layout="wide",
)

# ----------------------------------------------------------------------
# ESTILOS (menú lateral "bonito" hecho con CSS + botones nativos,
# sin depender de librerías externas que el usuario podría no tener)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Tarjetas de contenido */
    .card {
        background: #f8f9fc;
        border: 1px solid #e3e6f0;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .step-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #2c3e91;
        margin-bottom: 0.3rem;
    }
    /* Encabezado del menú lateral */
    .menu-header {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1a1a2e;
        padding-bottom: 0.2rem;
    }
    .menu-sub {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 0.8rem;
    }
    /* Botones del menú */
    div[data-testid="stSidebar"] button {
        border-radius: 12px !important;
        border: 1.5px solid #e3e6f0 !important;
        text-align: left !important;
        padding: 0.7rem 0.9rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.15rem !important;
    }
    div[data-testid="stSidebar"] button[kind="primary"] {
        border: 1.5px solid #4338ca !important;
        background: linear-gradient(135deg,#4f46e5,#6366f1) !important;
    }
    .badge {
        display:inline-block;
        background:#eef2ff;
        color:#4338ca;
        border-radius:999px;
        padding:2px 10px;
        font-size:0.75rem;
        font-weight:700;
        margin-bottom:0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FT_PER_M = 1 / 0.3048  # factor exacto metros -> pies

# ============================================================================
# MENÚ DE NAVEGACIÓN (SIDEBAR)
# ============================================================================
if "page" not in st.session_state:
    st.session_state.page = "lagrange"

with st.sidebar:
    st.markdown('<div class="menu-header">📚 Ejercicios</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="menu-sub">Selecciona un ejercicio para ver su resolución paso a paso</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "🖥️  1. Interpolación de Lagrange",
        use_container_width=True,
        type="primary" if st.session_state.page == "lagrange" else "secondary",
    ):
        st.session_state.page = "lagrange"
    st.caption("Tiempo de respuesta de un servidor")

    st.write("")

    if st.button(
        "💧  2. Regresión Lineal",
        use_container_width=True,
        type="primary" if st.session_state.page == "regresion" else "secondary",
    ):
        st.session_state.page = "regresion"
    st.caption("Ebullición del agua vs. altitud (El Alto / Zona Sur, La Paz)")

    st.divider()
    st.markdown(
        """
        <div style="font-size:0.78rem;color:#6b7280;">
        💡 Ambos ejercicios muestran la <b>resolución numérica paso a paso</b>
        (no solo el resultado final) y una gráfica interactiva.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# ============================================================================
#   EJERCICIO 1: INTERPOLACIÓN DE LAGRANGE
# ============================================================================
# ============================================================================
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
    """Construye el polinomio interpolador combinado (coeficientes)."""
    n = len(x_vals)
    coeffs = np.zeros(n)
    for i in range(n):
        roots = [x_vals[j] for j in range(n) if j != i]
        poly = np.poly(roots)
        denom = 1.0
        for j in range(n):
            if j != i:
                denom *= (x_vals[i] - x_vals[j])
        coeffs += (y_vals[i] / denom) * poly
    return np.poly1d(coeffs)


def lagrange_basis_latex(x_vals, i):
    """Construye la fórmula simbólica de L_i(x) con los x_j sustituidos."""
    xi = x_vals[i]
    num_terms = []
    denom = 1.0
    for j, xj in enumerate(x_vals):
        if j != i:
            num_terms.append(f"(x-{xj:g})")
            denom *= (xi - xj)
    numerator = "".join(num_terms)
    return f"L_{{{i}}}(x)=\\dfrac{{{numerator}}}{{{denom:g}}}"


def page_lagrange():
    with st.sidebar:
        st.header("📘 Contexto del problema")
        st.markdown(
            """
            Eres el encargado de **pruebas de rendimiento** de un servidor web.
            Durante pruebas de carga mediste el **tiempo de respuesta promedio
            (en milisegundos)** para distintos niveles de **usuarios concurrentes**.

            No probaste todos los niveles posibles (sería muy caro y lento),
            así que solo tienes algunos puntos de referencia.

            **Objetivo:** usar **interpolación de Lagrange** para estimar el
            tiempo de respuesta en niveles que *no* mediste directamente.
            """
        )
        st.divider()
        st.markdown(
            """
            **¿Por qué Lagrange y no una regresión lineal?**

            Porque no buscamos la "mejor recta promedio", sino un polinomio
            que **pase exactamente** por los datos medidos — útil cuando
            confiamos en que cada medición es precisa.
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

    # ------------------------------------------------------------------
    # PASO 1: Datos
    # ------------------------------------------------------------------
    st.markdown('<span class="badge">PASO 1</span>', unsafe_allow_html=True)
    st.markdown("### Ingresa tus datos medidos")

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

    n = len(x_vals)

    # ------------------------------------------------------------------
    # PASO 2: valor a estimar
    # ------------------------------------------------------------------
    st.markdown('<span class="badge">PASO 2</span>', unsafe_allow_html=True)
    st.markdown("### Elige el punto que quieres estimar")

    col1, col2 = st.columns([1, 1])
    with col1:
        x_min, x_max = float(min(x_vals)), float(max(x_vals))
        x_consulta = st.slider(
            "Número de usuarios concurrentes a estimar",
            min_value=x_min,
            max_value=x_max,
            value=(x_min + x_max) / 2,
            help="Puedes elegir cualquier valor dentro del rango medido.",
        )
    with col2:
        y_estimado = float(lagrange_interpolate(x_vals, y_vals, x_consulta)[0])
        st.metric(
            label=f"Tiempo de respuesta estimado para {x_consulta:.0f} usuarios",
            value=f"{y_estimado:.2f} ms",
        )

    if x_consulta < x_min or x_consulta > x_max:
        st.warning(
            "⚠️ Estás **extrapolando** (fuera del rango medido). "
            "La interpolación de Lagrange puede dar resultados poco confiables aquí."
        )

    # ------------------------------------------------------------------
    # PASO 3: Resolución paso a paso
    # ------------------------------------------------------------------
    st.markdown('<span class="badge">PASO 3</span>', unsafe_allow_html=True)
    st.markdown("### Resolución paso a paso")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">3.1 Fórmula general</div>', unsafe_allow_html=True)
        st.latex(r"P(x)=\sum_{i=0}^{n} y_i \cdot L_i(x) \qquad\qquad L_i(x)=\prod_{j \ne i}\frac{x-x_j}{x_i-x_j}")
        st.markdown(f"Con tus **n = {n}** puntos, se construyen **{n} polinomios base** $L_i(x)$, uno por cada punto medido.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">3.2 Polinomios base $L_i(x)$ (con tus datos)</div>', unsafe_allow_html=True)
        for i in range(n):
            st.latex(lagrange_basis_latex(x_vals, i))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="step-title">3.3 Evaluación de cada $L_i(x)$ en x = {x_consulta:.2f}</div>',
            unsafe_allow_html=True,
        )
        rows = []
        acumulado = 0.0
        for i in range(n):
            Li_val = float(lagrange_basis(x_vals, i, np.array([x_consulta]))[0])
            aporte = y_vals[i] * Li_val
            acumulado += aporte
            rows.append(
                {
                    "i": i,
                    "xᵢ": x_vals[i],
                    "yᵢ": y_vals[i],
                    f"Lᵢ({x_consulta:.2f})": round(Li_val, 6),
                    "yᵢ · Lᵢ(x)": round(aporte, 4),
                }
            )
        tabla_pasos = pd.DataFrame(rows)
        st.dataframe(tabla_pasos, use_container_width=True, hide_index=True)
        suma_txt = " + ".join([f"({r['yᵢ']:.2f})({r[f'Lᵢ({x_consulta:.2f})']:.4f})" for r in rows])
        st.latex(rf"P({x_consulta:.2f}) = {suma_txt} = \mathbf{{{y_estimado:.2f}}}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">3.4 Polinomio combinado final</div>', unsafe_allow_html=True)
        poly = build_polynomial_string(x_vals, y_vals)
        st.code(f"P(x) = {poly}", language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # PASO 4: gráfico
    # ------------------------------------------------------------------
    st.markdown('<span class="badge">PASO 4</span>', unsafe_allow_html=True)
    st.markdown("### Gráfico")

    x_plot = np.linspace(x_min, x_max, 300)
    y_plot = lagrange_interpolate(x_vals, y_vals, x_plot)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_plot, y=y_plot, mode="lines", name="Polinomio de Lagrange",
            line=dict(color="#2563eb", width=3),
            hovertemplate="Usuarios: %{x:.0f}<br>Tiempo estimado: %{y:.2f} ms<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=y_vals, mode="markers", name="Datos medidos",
            marker=dict(color="#dc2626", size=11, symbol="circle"),
            hovertemplate="Usuarios: %{x:.0f}<br>Tiempo medido: %{y:.2f} ms<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_consulta], y=[y_estimado], mode="markers", name="Estimación seleccionada",
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

    st.info(
        """
        💡 **Nota de ingeniería:** en un caso real, si tuvieras más de
        8-10 puntos, el polinomio de Lagrange puede oscilar mucho
        (fenómeno de Runge) y sería mejor usar **splines cúbicos** o
        un modelo de regresión. Lagrange es ideal cuando tienes
        **pocos puntos de medición confiables**.
        """
    )


# ============================================================================
# ============================================================================
#   EJERCICIO 2: REGRESIÓN LINEAL (MÍNIMOS CUADRADOS)
# ============================================================================
# ============================================================================
def linreg_steps(h, T):
    """Calcula la regresión lineal por mínimos cuadrados y devuelve
    todos los valores intermedios necesarios para mostrar el paso a paso."""
    n = len(h)
    sum_h = float(np.sum(h))
    sum_T = float(np.sum(T))
    sum_hT = float(np.sum(h * T))
    sum_h2 = float(np.sum(h ** 2))
    mean_h = sum_h / n
    mean_T = sum_T / n

    Sxy = sum_hT - n * mean_h * mean_T
    Sxx = sum_h2 - n * mean_h ** 2

    m = Sxy / Sxx
    b = mean_T - m * mean_h

    T_pred = m * h + b
    ss_res = float(np.sum((T - T_pred) ** 2))
    ss_tot = float(np.sum((T - mean_T) ** 2))
    r2 = 1 - ss_res / ss_tot

    return dict(
        n=n, sum_h=sum_h, sum_T=sum_T, sum_hT=sum_hT, sum_h2=sum_h2,
        mean_h=mean_h, mean_T=mean_T, Sxy=Sxy, Sxx=Sxx, m=m, b=b, r2=r2,
    )


PRESETS_ALTITUD = {
    "— (ninguno, usar el valor manual) —": None,
    "El Alto (4,150 m)": 4150,
    "Zona Sur, La Paz — promedio (2,750–3,200 m → 3,000 m)": 3000,
    "Zona Sur — Calacoto (3,280 m)": 3280,
    "La Paz centro, referencia (3,640 m)": 3640,
}


def page_regresion():
    with st.sidebar:
        st.header("📘 Contexto del problema")
        st.markdown(
            """
            La temperatura a la que hierve el agua **no es siempre 100 °C**:
            depende de la **presión atmosférica**, que baja a medida que
            aumenta la **altitud**.

            Tienes mediciones de la temperatura de ebullición $T_B$ del
            agua a distintas alturas $h$. Con esos datos quieres construir
            un **modelo lineal** $T_B = mh+b$ para predecir la temperatura
            de ebullición en cualquier altitud, incluso en lugares donde
            no mediste directamente — como **El Alto** o la **Zona Sur
            de La Paz**.
            """
        )
        st.divider()
        st.markdown(
            """
            **¿Por qué regresión lineal y no Lagrange?**

            Porque las mediciones tienen ligero ruido experimental y la
            relación $T_B$ vs. $h$ es aproximadamente una recta. No
            buscamos un polinomio que pase *exactamente* por cada punto,
            sino la **recta que mejor resume la tendencia** de todos
            los datos (mínimos cuadrados).
            """
        )

    st.title("💧 Regresión Lineal — Temperatura de ebullición del agua")
    st.subheader("Aplicación: El Alto y la Zona Sur de La Paz, Bolivia")

    st.markdown(
        """
        La tabla siguiente muestra la temperatura de ebullición del agua
        $T_B$ (°F) medida a distintas altitudes $h$ (ft). Ajusta una recta
        $T_B = mh + b$ que mejor se ajuste a los datos.
        """
    )

    # ------------------------------------------------------------------
    # PASO 1: Datos
    # ------------------------------------------------------------------
    st.markdown('<span class="badge">PASO 1</span>', unsafe_allow_html=True)
    st.markdown("### Datos medidos")

    data_default = pd.DataFrame(
        {
            "Altitud h (ft)": [-1000, 0, 3000, 8000, 15000, 22000, 28000],
            "T_B (°F)": [213.9, 212.0, 206.2, 196.2, 184.4, 172.6, 163.1],
        }
    )
    st.caption("Puedes editar la tabla directamente (doble clic en una celda).")
    edited_df = st.data_editor(
        data_default,
        num_rows="fixed",
        use_container_width=True,
        key="tabla_datos_regresion",
    )

    h = edited_df["Altitud h (ft)"].to_numpy(dtype=float)
    T = edited_df["T_B (°F)"].to_numpy(dtype=float)

    if np.any(np.isnan(h)) or np.any(np.isnan(T)):
        st.error("⚠️ Completa todos los valores de la tabla antes de continuar.")
        st.stop()
    if len(h) < 2:
        st.error("⚠️ Se necesitan al menos 2 puntos.")
        st.stop()

    r = linreg_steps(h, T)

    # ------------------------------------------------------------------
    # PASO 2: Resolución paso a paso (mínimos cuadrados)
    # ------------------------------------------------------------------
    st.markdown('<span class="badge">PASO 2</span>', unsafe_allow_html=True)
    st.markdown("### Resolución paso a paso (mínimos cuadrados)")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">2.1 Fórmulas del ajuste lineal</div>', unsafe_allow_html=True)
        st.latex(
            r"m=\dfrac{n\sum h_iT_i-\sum h_i\sum T_i}{n\sum h_i^2-\left(\sum h_i\right)^2}"
            r"\qquad\qquad b=\bar{T}-m\bar{h}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">2.2 Tabla de sumatorias</div>', unsafe_allow_html=True)
        tabla_sumas = pd.DataFrame(
            {
                "hᵢ": h,
                "Tᵢ": T,
                "hᵢ·Tᵢ": h * T,
                "hᵢ²": h ** 2,
            }
        )
        fila_sumas = pd.DataFrame(
            {
                "hᵢ": [r["sum_h"]],
                "Tᵢ": [r["sum_T"]],
                "hᵢ·Tᵢ": [r["sum_hT"]],
                "hᵢ²": [r["sum_h2"]],
            },
            index=["Σ (suma)"],
        )
        st.dataframe(
            pd.concat([tabla_sumas, fila_sumas]).round(3),
            use_container_width=True,
        )
        st.markdown(
            f"$n = {r['n']}$ &nbsp;&nbsp; "
            f"$\\bar{{h}} = {r['mean_h']:.4f}$ &nbsp;&nbsp; "
            f"$\\bar{{T}} = {r['mean_T']:.4f}$"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">2.3 Sustitución en las fórmulas</div>', unsafe_allow_html=True)
        st.latex(
            rf"m=\dfrac{{({r['n']})({r['sum_hT']:.2f})-({r['sum_h']:.2f})({r['sum_T']:.2f})}}"
            rf"{{({r['n']})({r['sum_h2']:.2f})-({r['sum_h']:.2f})^2}}=\mathbf{{{r['m']:.7f}}}"
        )
        st.latex(
            rf"b=\bar{{T}}-m\bar{{h}}={r['mean_T']:.4f}-({r['m']:.7f})({r['mean_h']:.4f})"
            rf"=\mathbf{{{r['b']:.4f}}}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">2.4 Ecuación final</div>', unsafe_allow_html=True)
        st.latex(rf"T_B = {r['m']:.7f}\,h + {r['b']:.4f}")
        st.markdown(f"**Bondad de ajuste:** $R^2 = {r['r2']:.4f}$ (muy cercano a 1 → excelente ajuste lineal)")
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # PASO 3: aplicar la ecuación
    # ------------------------------------------------------------------
    st.markdown('<span class="badge">PASO 3</span>', unsafe_allow_html=True)
    st.markdown("### Aplica la ecuación a una altitud")

    col1, col2 = st.columns([1, 1])
    with col1:
        preset = st.selectbox("Lugar predefinido (opcional)", list(PRESETS_ALTITUD.keys()))
        unidad = st.radio("Unidad de entrada manual", ["metros (m)", "pies (ft)"], horizontal=True)

    with col2:
        if PRESETS_ALTITUD[preset] is not None:
            alt_m_default = float(PRESETS_ALTITUD[preset])
        else:
            alt_m_default = 5000 / FT_PER_M  # ~1524 m, equivalente al ejemplo original en ft

        if unidad == "metros (m)":
            alt_m = st.number_input("Altitud (m)", min_value=-500.0, max_value=6000.0, value=round(alt_m_default, 1), step=10.0)
            alt_ft = alt_m * FT_PER_M
        else:
            alt_ft_input = st.number_input("Altitud (ft)", min_value=-1600.0, max_value=20000.0, value=round(alt_m_default * FT_PER_M, 1), step=50.0)
            alt_ft = alt_ft_input
            alt_m = alt_ft / FT_PER_M

    T_pred_F = r["m"] * alt_ft + r["b"]
    T_pred_C = (T_pred_F - 32) * 5 / 9

    m1, m2, m3 = st.columns(3)
    m1.metric("Altitud", f"{alt_m:,.0f} m", help=f"{alt_ft:,.0f} ft")
    m2.metric("Temperatura de ebullición", f"{T_pred_F:.2f} °F")
    m3.metric("Equivalente en °C", f"{T_pred_C:.2f} °C")

    st.latex(
        rf"T_B = {r['m']:.7f}({alt_ft:.0f}) + {r['b']:.4f} = {T_pred_F:.2f}\ °F"
        rf"\;\longrightarrow\; \dfrac{{({T_pred_F:.2f}-32)\times5}}{{9}} = {T_pred_C:.2f}\ °C"
    )

    if alt_ft < float(min(h)) or alt_ft > float(max(h)):
        st.warning(
            "⚠️ Este valor está **fuera del rango de las altitudes medidas** "
            "(extrapolación). El resultado es razonable porque la relación "
            "es muy lineal ($R^2$ alto), pero conviene tenerlo en cuenta."
        )

    # ------------------------------------------------------------------
    # PASO 4: gráfico
    # ------------------------------------------------------------------
    st.markdown('<span class="badge">PASO 4</span>', unsafe_allow_html=True)
    st.markdown("### Gráfico")

    h_min_plot = min(float(min(h)), alt_ft) - 1500
    h_max_plot = max(float(max(h)), alt_ft) + 1500
    h_plot = np.linspace(h_min_plot, h_max_plot, 300)
    T_plot = r["m"] * h_plot + r["b"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h_plot, y=T_plot, mode="lines", name=f"Recta: T_B = {r['m']:.6f}h + {r['b']:.2f}",
            line=dict(color="#2563eb", width=3),
            hovertemplate="h: %{x:.0f} ft<br>T: %{y:.2f} °F<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h, y=T, mode="markers", name="Datos medidos",
            marker=dict(color="#dc2626", size=11, symbol="circle"),
            hovertemplate="h: %{x:.0f} ft<br>T: %{y:.2f} °F<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[alt_ft], y=[T_pred_F], mode="markers", name="Altitud consultada",
            marker=dict(color="#16a34a", size=15, symbol="diamond"),
            hovertemplate=f"{alt_m:,.0f} m ({alt_ft:,.0f} ft)<br>T: %{{y:.2f}} °F<extra></extra>",
        )
    )

    # Marcadores de referencia para los lugares predefinidos
    colores_ref = ["#f59e0b", "#8b5cf6", "#0891b2"]
    idx_color = 0
    for nombre, alt in PRESETS_ALTITUD.items():
        if alt is None or nombre == preset:
            continue
        ft_ref = alt * FT_PER_M
        T_ref = r["m"] * ft_ref + r["b"]
        fig.add_trace(
            go.Scatter(
                x=[ft_ref], y=[T_ref], mode="markers", name=nombre,
                marker=dict(color=colores_ref[idx_color % len(colores_ref)], size=10, symbol="triangle-up"),
                hovertemplate=f"{nombre}<br>%{{y:.2f}} °F<extra></extra>",
            )
        )
        idx_color += 1

    fig.update_layout(
        title="Regresión lineal: temperatura de ebullición del agua vs. altitud",
        xaxis_title="Altitud h (ft)",
        yaxis_title="Temperatura de ebullición T_B (°F)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=40, l=40, r=20),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        """
        💡 **Nota:** los valores calculados para El Alto (≈86 °C) y la
        Zona Sur de La Paz (≈89–90 °C) coinciden bien con lo observado
        en la práctica: a mayor altitud, menor presión atmosférica y por
        lo tanto el agua hierve a una temperatura más baja — por eso
        cocinar toma más tiempo en El Alto que en la Zona Sur.
        """
    )


# ============================================================================
# ENRUTADOR PRINCIPAL
# ============================================================================
if st.session_state.page == "lagrange":
    page_lagrange()
else:
    page_regresion()
