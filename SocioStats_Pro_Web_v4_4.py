# -*- coding: utf-8 -*-
"""
SocioStats Pro Web v4.4
Versión refinada con mejoras de usabilidad, rendimiento y robustez.
"""

import io
import gc
import warnings
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, pearsonr

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    STATSMODELS_OK = True
except Exception:
    sm = None
    PerfectSeparationError = Exception
    STATSMODELS_OK = False

try:
    from prince import CA, MCA
    PRINCE_OK = True
except Exception:
    CA = None
    MCA = None
    PRINCE_OK = False

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Configuración de matplotlib optimizada
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 11,
    "figure.titleweight": "bold",
    "axes.titleweight": "bold",
    "figure.dpi": 110,
    "savefig.dpi": 160,
})

st.set_page_config(
    page_title="SocioStats Pro Web",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ESTILOS CSS MEJORADOS
# =============================================================================

CSS = """
<style>
.block-container {padding-top: 2.5rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {background: #f7f9fb;}
.ssp-title {font-size: 2rem; font-weight: 800; color: #2E86AB; margin-bottom: 0.2rem;}
.ssp-subtitle {color: #5b6770; margin-bottom: 1rem;}
.metric-card {border: 1px solid #e6edf2; border-radius: 14px; padding: 0.8rem 1rem; background: #ffffff;}
.result-box {background: #ffffff; border: 1px solid #e1e7ec; border-radius: 12px; padding: 1rem; white-space: pre-wrap; font-family: Consolas, monospace; font-size: 0.85rem; overflow-x: auto; max-height: 500px; overflow-y: auto;}
.small-muted {color:#667; font-size:0.88rem;}
.stButton > button {border-radius: 8px; font-weight: 500;}
.status-bar {background: #f0f2f6; padding: 0.3rem 1rem; border-radius: 8px; font-size: 0.85rem; color: #2c3e50;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# ESTADO Y UTILIDADES
# =============================================================================

def init_state():
    """Inicializa el estado de la sesión"""
    defaults = {
        "df": None,
        "df_original": None,
        "current_results": "📊 Bienvenido a SocioStats Pro Web v4.4\nCarga un CSV/XLSX o genera datos de ejemplo para empezar.",
        "current_fig": None,
        "analysis_generated_columns": set(),
        "status": "✅ Listo | SocioStats Pro Web v4.4",
        "processing": False,
        "last_analysis": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def set_status(msg: str) -> None:
    """Actualiza el mensaje de estado"""
    st.session_state.status = f"{msg}"


def set_results(text: str) -> None:
    """Actualiza los resultados mostrados"""
    st.session_state.current_results = str(text) if text is not None else ""
    set_status("✅ Resultados actualizados")


def set_fig(fig) -> None:
    """Establece la figura actual"""
    clear_fig_only()
    format_figure(fig)
    st.session_state.current_fig = fig
    set_status("📈 Gráfico actualizado")


def clear_fig_only() -> None:
    """Limpia solo la figura actual"""
    fig = st.session_state.get("current_fig")
    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass
    st.session_state.current_fig = None
    plt.close("all")
    gc.collect()


def reset_analysis() -> None:
    """Reinicia el análisis conservando los datos"""
    df = st.session_state.df
    cols = list(st.session_state.analysis_generated_columns or [])
    if df is not None and cols:
        st.session_state.df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
    st.session_state.analysis_generated_columns = set()
    st.session_state.current_results = "🔄 Análisis reiniciado. Los datos cargados se conservan."
    clear_fig_only()
    set_status("🔄 Análisis reiniciado")


def restore_original_data() -> None:
    """Restaura los datos originales"""
    if st.session_state.df_original is not None:
        st.session_state.df = st.session_state.df_original.copy()
        st.session_state.analysis_generated_columns = set()
        clear_fig_only()
        set_results("♻️ Dataset restaurado a la versión original.")


def is_categorical_like(s: pd.Series) -> bool:
    """Detecta si una serie es categórica"""
    try:
        # Tipos explícitamente categóricos
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_string_dtype(s):
            return True
        # Numéricas con pocos valores únicos
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= min(10, max(2, int(len(s) * 0.05))):
            return True
        return False
    except Exception:
        return False


def get_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Obtiene columnas numéricas y categóricas"""
    if df is None:
        return [], [], []
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if is_categorical_like(df[c]) and c not in numeric]
    categorical_like = [c for c in df.columns if is_categorical_like(df[c])]
    return numeric, categorical, categorical_like


def validate_dataframe(df: pd.DataFrame, min_rows: int = 3) -> bool:
    """Valida que el DataFrame sea apto para análisis"""
    if df is None:
        raise ValueError("No hay datos cargados")
    if df.empty:
        raise ValueError("El DataFrame está vacío")
    if len(df) < min_rows:
        raise ValueError(f"Se necesitan al menos {min_rows} filas (actual: {len(df)})")
    return True


def format_figure(fig) -> None:
    """Aplica formato consistente a las figuras"""
    try:
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_fontsize(11)
            fig._suptitle.set_fontweight("bold")
            fig._suptitle.set_y(0.985)
        
        for ax in fig.axes:
            try:
                ax.title.set_fontsize(9)
                ax.title.set_pad(7)
                ax.xaxis.label.set_size(8)
                ax.yaxis.label.set_size(8)
                ax.tick_params(axis="both", labelsize=7)
                leg = ax.get_legend()
                if leg is not None:
                    for t in leg.get_texts():
                        t.set_fontsize(7)
                    if leg.get_title() is not None:
                        leg.get_title().set_fontsize(7)
            except Exception:
                pass
        
        fig.subplots_adjust(
            top=0.90, bottom=0.10, left=0.08, right=0.97,
            hspace=0.55, wspace=0.35
        )
        try:
            fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92], pad=2.0)
        except Exception:
            pass
    except Exception:
        pass


def fig_to_bytes(fig, fmt: str = "png") -> bytes:
    """Convierte una figura a bytes para descarga"""
    bio = io.BytesIO()
    dpi = 220 if fmt == "png" else 150
    fig.savefig(bio, format=fmt, bbox_inches="tight", dpi=dpi)
    bio.seek(0)
    return bio.getvalue()


def read_uploaded_file(uploaded) -> pd.DataFrame:
    """Lee archivos CSV o Excel con detección automática"""
    name = uploaded.name.lower()
    
    # Excel
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded)
    
    # CSV robusto
    raw = uploaded.getvalue()
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            text = raw.decode(enc)
        except Exception:
            continue
        for sep in [None, ";", ",", "\t"]:
            try:
                if sep is None:
                    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
                else:
                    df = pd.read_csv(io.StringIO(text), sep=sep)
                if df.shape[1] > 1 or sep is not None:
                    return df
            except Exception:
                pass
    
    # Fallback
    return pd.read_csv(uploaded)


def create_sample_data() -> None:
    """Crea dataset de ejemplo"""
    np.random.seed(42)
    n = 300
    
    edad = np.random.normal(40, 12, n).clip(18, 80)
    ingresos = np.random.normal(1800, 600, n).clip(300, 6000)
    satisfaccion = np.random.randint(1, 6, n)
    sexo = np.random.choice(["Hombre", "Mujer"], size=n, p=[0.48, 0.52])
    nivel_estudios = np.random.choice(["Básicos", "Medios", "Superiores"], size=n, p=[0.35, 0.4, 0.25])
    region = np.random.choice(["Norte", "Centro", "Sur"], size=n, p=[0.3, 0.45, 0.25])
    
    z = (-2.0 + 0.03 * (edad - 40) + 0.0012 * (ingresos - 1800)
         + 0.35 * (satisfaccion - 3) + np.where(sexo == "Mujer", 0.25, 0)
         + np.where(nivel_estudios == "Superiores", 0.6, 0))
    p = 1 / (1 + np.exp(-z))
    participa = (np.random.rand(n) < p).astype(int)
    
    df = pd.DataFrame({
        "edad": edad.round(0).astype(int),
        "ingresos": ingresos.round(0).astype(int),
        "satisfaccion": satisfaccion.astype(int),
        "sexo": sexo,
        "nivel_estudios": nivel_estudios,
        "region": region,
        "participa": participa,
    })
    
    st.session_state.df = df.copy()
    st.session_state.df_original = df.copy()
    st.session_state.analysis_generated_columns = set()
    clear_fig_only()
    
    set_results(
        "🎲 DATOS DE EJEMPLO CREADOS\n" + "=" * 50 +
        f"\n📊 Dimensiones: {df.shape[0]}×{df.shape[1]}\n" +
        f"📋 Variables: {', '.join(df.columns)}\n\n" +
        "👀 Vista previa:\n" + df.head().to_string()
    )

# =============================================================================
# ANÁLISIS - FUNCIONES PURAS
# =============================================================================

def descriptive_stats(df: pd.DataFrame) -> str:
    """Estadísticas descriptivas"""
    num, _, _ = get_cols(df)
    if not num:
        raise ValueError("No hay variables numéricas para analizar.")
    
    out = ["📈 ESTADÍSTICAS DESCRIPTIVAS", "=" * 50, ""]
    for col in num:
        data = df[col].dropna()
        mean = data.mean()
        cv = (data.std() / mean * 100) if mean != 0 else np.nan
        out += [
            f"🎯 {col.upper()}:",
            f"• Media: {mean:.3f}",
            f"• Mediana: {data.median():.3f}",
            f"• Desviación: {data.std():.3f}",
            f"• Mínimo: {data.min():.3f}",
            f"• Máximo: {data.max():.3f}",
            f"• Rango: {(data.max() - data.min()):.3f}",
            f"• CV: {cv:.2f}%" if not np.isnan(cv) else "• CV: n/d",
            f"• Asimetría: {data.skew():.3f}",
            f"• Curtosis: {data.kurtosis():.3f}",
            f"• N: {len(data)} casos", ""
        ]
    return "\n".join(out)


def distribution_analysis(df: pd.DataFrame) -> str:
    """Análisis de distribuciones"""
    num, _, _ = get_cols(df)
    if not num:
        raise ValueError("No hay variables numéricas para analizar.")
    
    out = ["📊 ANÁLISIS DE DISTRIBUCIONES", "=" * 50, ""]
    for col in num:
        data = df[col].dropna()
        out.append(f"🎯 {col.upper()}:")
        
        if len(data) < 3:
            out.append("• Test normalidad: n<3 (insuficiente)")
        else:
            if len(data) < 8:
                stat, p = stats.shapiro(data)
                name = "Shapiro-Wilk"
            else:
                stat, p = stats.normaltest(data)
                name = "D'Agostino"
            out.append(f"• Normalidad ({name}): p = {p:.4f}")
            out.append(f"  → {'✅ Normal' if p > 0.05 else '❌ NO normal'}")
        
        sk = data.skew()
        if abs(sk) < 0.5:
            out.append(f"• Simetría: ~simétrica (skew={sk:.3f})")
        elif sk > 0:
            out.append(f"• Simetría: sesgada derecha (skew={sk:.3f})")
        else:
            out.append(f"• Simetría: sesgada izquierda (skew={sk:.3f})")
        out.append("")
    return "\n".join(out)


def outlier_analysis(df: pd.DataFrame) -> str:
    """Análisis de valores atípicos"""
    num, _, _ = get_cols(df)
    if not num:
        raise ValueError("No hay variables numéricas.")
    
    out = ["🎯 ANÁLISIS DE VALORES ATÍPICOS", "=" * 50, ""]
    for col in num:
        data = df[col].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outs = data[(data < low) | (data > high)]
        out += [
            f"📊 {col.upper()}:",
            f"• IQR: {iqr:.3f}",
            f"• Límites: [{low:.3f}, {high:.3f}]",
            f"• Outliers: {len(outs)} ({len(outs)/len(data)*100:.1f}%)"
        ]
        if 0 < len(outs) <= 10:
            out.append(f"• Valores: {outs.tolist()}")
        elif len(outs) > 10:
            out.append(f"• Valores: {outs.iloc[:10].tolist()}... ({len(outs)-10} más)")
        out.append("")
    return "\n".join(out)


def frequency_tables(df: pd.DataFrame, max_categories: int = 30) -> str:
    """Tablas de frecuencia"""
    _, cat, cat_like = get_cols(df)
    cols = cat if cat else [c for c in cat_like if df[c].nunique(dropna=True) <= max_categories]
    if not cols:
        raise ValueError("No hay variables categóricas.")
    
    out = ["📋 TABLAS DE FRECUENCIA", "=" * 50, ""]
    for col in cols:
        s = df[col].astype("object")
        freq_abs = s.value_counts(dropna=False).head(max_categories)
        freq_rel = (s.value_counts(dropna=False, normalize=True) * 100).round(2).head(max_categories)
        freq_cum = freq_rel.cumsum().round(2)
        table = pd.DataFrame({
            "Frec. Abs.": freq_abs,
            "Frec. Rel. %": freq_rel,
            "Acumulado %": freq_cum
        })
        out += [f"🎯 {col.upper()}:", table.to_string(), ""]
    return "\n".join(out)


def dataset_info(df: pd.DataFrame) -> str:
    """Información del dataset"""
    out = [
        "🔍 INFORMACIÓN DEL DATASET",
        "=" * 50, "",
        f"📊 FORMA:\n• Filas: {df.shape[0]}\n• Columnas: {df.shape[1]}",
        "", "📋 TIPOS DE DATOS:"
    ]
    for col in df.columns:
        out.append(f"• {col}: {df[col].dtype} ({df[col].nunique(dropna=True)} valores únicos)")
    
    out.append("\n⚠️ VALORES FALTANTES:")
    miss = df.isna().sum()
    for col in df.columns:
        pct = miss[col] / len(df) * 100 if len(df) else 0
        out.append(f"• {col}: {int(miss[col])} ({pct:.1f}%)")
    
    num = df.select_dtypes(include=[np.number]).columns
    if len(num):
        out += ["", "📈 ESTADÍSTICAS BÁSICAS:", df[num].describe().round(3).to_string()]
    return "\n".join(out)


def categorical_summary(df: pd.DataFrame) -> str:
    """Resumen de variables categóricas"""
    _, cat, _ = get_cols(df)
    if not cat:
        return "No hay variables categóricas"
    
    out = ["📊 RESUMEN DE CATEGÓRICAS", "=" * 50, ""]
    for col in cat:
        out.append(f"🎯 {col.upper()}:")
        out.append(f"• Niveles: {df[col].nunique(dropna=True)}")
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
        out.append(f"• Moda: {mode_val}")
        freq_high = df[col].value_counts().iloc[0] if not df[col].empty else 0
        out.append(f"• Frecuencia más alta: {freq_high}")
        out.append("")
    return "\n".join(out)


def contingency_result(df: pd.DataFrame, var1: str, var2: str) -> str:
    """Tabla de contingencia"""
    tab = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, exp = chi2_contingency(tab)
    n = tab.sum().sum()
    denom = n * (min(tab.shape) - 1)
    v = np.sqrt(chi2 / denom) if denom > 0 else np.nan
    
    txt = [
        f"📋 TABLA DE CONTINGENCIA: {var1} × {var2}",
        "=" * 50, "",
        "📊 TABLA OBSERVADA:",
        tab.to_string(), "",
        "🧮 ESTADÍSTICAS:",
        f"• χ² = {chi2:.3f}",
        f"• p = {p:.4f}",
        f"• gl = {dof}",
        f"• V de Cramer = {v:.3f}" if np.isfinite(v) else "• V de Cramer = n/d",
        "", "💡 INTERPRETACIÓN:"
    ]
    
    if p < 0.05:
        strength = "muy débil" if v < 0.1 else "débil" if v < 0.3 else "moderada" if v < 0.5 else "fuerte"
        txt += ["✅ Relación significativa", f"• Asociación: {strength}"]
    else:
        txt.append("❌ No hay relación significativa")
    return "\n".join(txt)


def correlation_analysis(df: pd.DataFrame) -> Tuple[str, plt.Figure]:
    """Análisis de correlaciones"""
    num, _, _ = get_cols(df)
    if len(num) < 2:
        raise ValueError("Se necesitan al menos 2 variables numéricas.")
    
    corr_matrix = df[num].corr()
    out = [
        "📈 ANÁLISIS DE CORRELACIONES",
        "=" * 50, "",
        "🔢 MATRIZ DE CORRELACIÓN (Pearson):",
        corr_matrix.round(3).to_string(), "",
        "🎯 CORRELACIONES SIGNIFICATIVAS (p < 0.05):"
    ]
    
    found = False
    for i in range(len(num)):
        for j in range(i + 1, len(num)):
            v1, v2 = num[i], num[j]
            clean = df[[v1, v2]].dropna()
            if len(clean) > 2 and clean[v1].nunique() > 1 and clean[v2].nunique() > 1:
                r, p = pearsonr(clean[v1], clean[v2])
                if p < 0.05:
                    found = True
                    strength = "débil" if abs(r) < 0.3 else "moderada" if abs(r) < 0.7 else "fuerte"
                    direction = "positiva" if r > 0 else "negativa"
                    out.append(f"• {v1} ↔ {v2}: r = {r:.3f} ({strength}, {direction})")
    
    if not found:
        out.append("No se encontraron correlaciones significativas")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, cmap="coolwarm", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8}, ax=ax
    )
    ax.set_title("Mapa de Calor de Correlaciones")
    
    return "\n".join(out), fig


def group_comparison(df: pd.DataFrame, group_var: str, numeric_var: str) -> Tuple[str, plt.Figure]:
    """Comparación de grupos"""
    clean = df[[group_var, numeric_var]].dropna()
    groups = [g[numeric_var] for _, g in clean.groupby(group_var) if len(g[numeric_var]) > 0]
    labels = [str(k) for k, g in clean.groupby(group_var) if len(g[numeric_var]) > 0]
    
    if len(groups) < 2:
        raise ValueError("No hay suficientes grupos.")
    
    if len(groups) == 2:
        stat, p = ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
        test = "t-test de Welch"
    else:
        stat, p = f_oneway(*groups)
        test = "ANOVA"
    
    stats_by = clean.groupby(group_var)[numeric_var].agg(["count", "mean", "std"]).round(3)
    
    txt = [
        f"👥 COMPARACIÓN DE GRUPOS ({test})",
        "=" * 50, "",
        f"Variable: {numeric_var} por {group_var}", "",
        "📊 ESTADÍSTICAS POR GRUPO:",
        stats_by.to_string(), "",
        f"🧮 {test}:",
        f"• Estadístico = {stat:.3f}",
        f"• p = {p:.4f}", "",
        "💡 INTERPRETACIÓN:",
        "✅ Diferencias significativas (p < 0.05)" if p < 0.05 else "❌ No hay diferencias significativas"
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.6)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'plum', 'lightskyblue'][:len(groups)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xlabel(group_var)
    ax.set_ylabel(numeric_var)
    ax.set_title(f"{numeric_var} por {group_var}")
    if len(labels) > 4:
        ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)
    
    return "\n".join(txt), fig


def simple_regression(df: pd.DataFrame, x_var: str, y_var: str) -> Tuple[str, plt.Figure]:
    """Regresión lineal simple"""
    clean = df[[x_var, y_var]].dropna()
    if len(clean) < 3:
        raise ValueError("No hay suficientes datos.")
    
    X = clean[[x_var]].values
    y = clean[y_var].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    r, p = pearsonr(clean[x_var], clean[y_var])
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    txt = [
        "📐 REGRESIÓN LINEAL SIMPLE",
        "=" * 50, "",
        f"Variable dependiente: {y_var}",
        f"Variable independiente: {x_var}",
        f"N: {len(clean)}", "",
        "🧮 ECUACIÓN:",
        f"{y_var} = {model.intercept_:.3f} + {model.coef_[0]:.3f}×{x_var}", "",
        "📊 MÉTRICAS:",
        f"• R² = {r2:.3f} ({r2*100:.1f}%)",
        f"• r = {r:.3f}",
        f"• p = {p:.4f}",
        f"• RMSE = {rmse:.3f}"
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(clean[x_var], clean[y_var], alpha=0.65, s=45)
    order = np.argsort(clean[x_var].values)
    ax.plot(clean[x_var].values[order], y_pred[order], 'r-', linewidth=2)
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f"Regresión simple: {y_var} por {x_var}")
    ax.grid(True, alpha=0.3)
    
    return "\n".join(txt), fig


def multiple_regression(df: pd.DataFrame, y_var: str, x_vars: List[str]) -> Tuple[str, plt.Figure]:
    """Regresión lineal múltiple"""
    clean = df[[y_var] + x_vars].dropna()
    if len(clean) < len(x_vars) + 2:
        raise ValueError("No hay suficientes datos para el número de predictoras.")
    
    X, y = clean[x_vars], clean[y_var]
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    r2 = model.score(X, y)
    n, k = len(clean), len(x_vars)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else np.nan
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    
    eq = f"{y_var} = {model.intercept_:.3f}" + "".join([f" + {coef:.3f}×{var}" for coef, var in zip(model.coef_, x_vars)])
    
    out = [
        "📈 REGRESIÓN LINEAL MÚLTIPLE",
        "=" * 60, "",
        f"Variable dependiente: {y_var}",
        f"Variables: {', '.join(x_vars)}",
        f"N: {n} casos", "",
        "🧮 ECUACIÓN:", eq, "",
        "📊 MÉTRICAS:",
        f"• R² = {r2:.3f} ({r2*100:.1f}%)",
        f"• R² ajustado = {adj_r2:.3f}",
        f"• RMSE = {rmse:.3f}", "",
        "🔍 COEFICIENTES:"
    ]
    
    for var, coef in zip(x_vars, model.coef_):
        out.append(f"• {var}: {coef:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y, pred, alpha=0.65)
    mn, mx = min(y.min(), pred.min()), max(y.max(), pred.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--')
    axes[0].set_title("Observado vs predicho")
    axes[0].set_xlabel("Observado")
    axes[0].set_ylabel("Predicho")
    
    resid = y - pred
    axes[1].scatter(pred, resid, alpha=0.65)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_title("Residuos")
    axes[1].set_xlabel("Predicho")
    axes[1].set_ylabel("Residuo")
    
    return "\n".join(out), fig


def logistic_regression(df: pd.DataFrame, y_var: str, x_vars: List[str]) -> str:
    """Regresión logística"""
    clean = df[[y_var] + x_vars].dropna()
    if clean[y_var].nunique() != 2:
        raise ValueError(f"'{y_var}' no es binaria.")
    
    unique = clean[y_var].drop_duplicates().tolist()
    mapping = {unique[0]: 0, unique[1]: 1}
    y = clean[y_var].map(mapping).astype(int)
    X = pd.get_dummies(clean[x_vars], drop_first=True)
    X = X.loc[:, X.nunique() > 1]
    
    if X.shape[1] == 0:
        raise ValueError("No hay predictoras válidas tras codificación.")
    
    out = [
        "🧠 REGRESIÓN LOGÍSTICA",
        "=" * 60, "",
        f"Variable: {y_var}",
        f"Predictoras: {', '.join(x_vars)}", "",
        "📌 Mapeo:",
        f"• 0 = '{unique[0]}'",
        f"• 1 = '{unique[1]}'", ""
    ]
    
    if STATSMODELS_OK:
        X_sm = sm.add_constant(X.astype(float), prepend=False)
        try:
            model = sm.Logit(y, X_sm).fit(disp=0)
        except Exception:
            model = sm.Logit(y, X_sm).fit_regularized(method="l1", disp=0)
        
        prob = model.predict(X_sm)
        pred = (prob >= 0.5).astype(int)
        acc = (pred.values == y.values).mean()
        
        try:
            ll_null = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0).llf
            pseudo_r2 = 1 - (model.llf / ll_null)
        except Exception:
            pseudo_r2 = np.nan
        
        out += [
            "📊 RESUMEN:",
            model.summary().as_text(), "",
            "📐 MÉTRICAS:",
            f"• Accuracy = {acc:.3f}",
            f"• Pseudo-R² = {pseudo_r2:.3f}" if not np.isnan(pseudo_r2) else "• Pseudo-R² = n/d", "",
            "🔍 ODDS RATIOS:"
        ]
        for pred_name, odds in np.exp(model.params).items():
            if pred_name != "const":
                out.append(f"• {pred_name}: {odds:.3f}")
    else:
        model = LogisticRegression(max_iter=1000).fit(X, y)
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        out += [
            "📊 RESUMEN:",
            "statsmodels no está instalado; usando sklearn.", "",
            "📐 MÉTRICAS:",
            f"• Accuracy = {acc:.3f}", "",
            "🔍 COEFICIENTES:"
        ]
        for name, coef in zip(X.columns, model.coef_[0]):
            out.append(f"• {name}: {coef:.3f}")
    
    return "\n".join(out)


def cluster_analysis(df: pd.DataFrame, x_vars: List[str]) -> Tuple[str, plt.Figure]:
    """Análisis de clusters"""
    X = df[x_vars].dropna()
    if len(X) < 10:
        raise ValueError("Se necesitan al menos 10 casos.")
    
    X_scaled = StandardScaler().fit_transform(X)
    max_k = max(2, min(8, len(X) - 1))
    k_vals = list(range(2, max_k + 1))
    scores = []
    inertias = []
    
    for k in k_vals:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0)
        inertias.append(km.inertia_)
    
    best_k = k_vals[int(np.argmax(scores))] if scores else 2
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    
    st.session_state.df.loc[X.index, "cluster"] = labels
    st.session_state.df["cluster"] = st.session_state.df["cluster"].astype("Int64")
    st.session_state.analysis_generated_columns.add("cluster")
    
    counts = pd.Series(labels).value_counts().sort_index()
    
    out = [
        "🔍 ANÁLISIS DE CLUSTERS",
        "=" * 60, "",
        "📊 CONFIGURACIÓN:",
        f"• Variables: {', '.join(x_vars)}",
        f"• Clusters: {best_k}",
        f"• N: {len(X)}", "",
        "📈 CALIDAD:",
        f"• Silhouette Score: {scores[k_vals.index(best_k)]:.3f}",
        f"• WCSS: {km.inertia_:.2f}", "",
        "👥 DISTRIBUCIÓN:"
    ]
    for cl, c in counts.items():
        out.append(f"• Cluster {cl}: {c} ({c/len(labels)*100:.1f}%)")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(k_vals, scores, 'bo-')
    axes[0].axvline(best_k, color='r', linestyle='--')
    axes[0].set_title("Silhouette por k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silhouette")
    axes[0].grid(True, alpha=0.3)
    
    if X_scaled.shape[1] >= 2:
        pca2 = PCA(n_components=2).fit_transform(X_scaled)
        sc = axes[1].scatter(pca2[:, 0], pca2[:, 1], c=labels, alpha=0.75, cmap='viridis')
        axes[1].set_title("Clusters en PCA 2D")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No hay suficientes dimensiones\npara visualización 2D", 
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_axis_off()
    
    return "\n".join(out), fig


def pca_analysis(df: pd.DataFrame, vars_: List[str]) -> Tuple[str, plt.Figure]:
    """Análisis de Componentes Principales"""
    X = df[vars_].dropna()
    if len(X) < 2 or len(vars_) < 2:
        raise ValueError("Se necesitan al menos 2 variables y 2 casos.")
    
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA().fit(X_scaled)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    
    out = ["📊 ANÁLISIS PCA", "=" * 60, "", "📈 VARIANZA EXPLICADA:"]
    for i, (v, c) in enumerate(zip(ev, cum), start=1):
        out.append(f"• PC{i}: {v*100:.1f}% (Acum: {c*100:.1f}%)")
    
    out.append("\n🔍 COMPONENTES PRINCIPALES:")
    for i in range(min(5, len(vars_))):
        out.append(f"\nPC{i+1} ({ev[i]*100:.1f}%):")
        load = pca.components_[i]
        for idx in np.argsort(np.abs(load))[-3:][::-1]:
            out.append(f"  - {vars_[idx]}: {load[idx]:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(range(1, len(ev)+1), ev*100, alpha=0.7)
    axes[0].plot(range(1, len(cum)+1), cum*100, 'ro-')
    axes[0].set_title("Varianza explicada")
    axes[0].set_xlabel("Componentes")
    axes[0].set_ylabel("%")
    axes[0].grid(True, alpha=0.3)
    
    scores = pca.transform(X_scaled)
    axes[1].scatter(scores[:, 0], scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores)), alpha=0.7)
    axes[1].set_title("Proyección PCA")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.3)
    
    return "\n".join(out), fig


def lda_analysis(df: pd.DataFrame, y_var: str, x_vars: List[str]) -> str:
    """Análisis Discriminante Lineal"""
    clean = df[[y_var] + x_vars].dropna()
    y = clean[y_var]
    X = clean[x_vars]
    
    if y.nunique() < 2:
        raise ValueError("La variable de grupo debe tener al menos 2 clases.")
    
    X_scaled = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=min(y.nunique() - 1, X.shape[1]))
    lda.fit(X_scaled, y)
    pred = lda.predict(X_scaled)
    
    labels = sorted(y.unique().tolist())
    acc = accuracy_score(y, pred)
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels)
    
    out = [
        "👥 ANÁLISIS DISCRIMINANTE LINEAL (LDA)",
        "=" * 60, "",
        f"Variable de grupos: {y_var} (clases: {y.nunique()})",
        f"Predictoras: {', '.join(x_vars)}",
        f"\n📊 PRECISIÓN: {acc:.3f} ({acc*100:.1f}%)", "",
        "📋 MATRIZ DE CONFUSIÓN:",
        cm.to_string(), "",
        "🔍 PESOS DISCRIMINANTES:"
    ]
    
    if lda.coef_.shape[0] == 1:
        out.append(pd.Series(lda.coef_[0], index=X.columns).sort_values(ascending=False).round(3).to_string())
    else:
        out.append("Múltiples funciones discriminantes")
    
    return "\n".join(out)


def correspondence_analysis(df: pd.DataFrame, var1: str, var2: str) -> Tuple[str, Optional[plt.Figure]]:
    """Análisis de Correspondencias"""
    if var1 == var2:
        raise ValueError("Selecciona dos variables distintas.")
    
    clean = df[[var1, var2]].dropna()
    if clean.empty:
        raise ValueError("No hay casos válidos tras eliminar valores perdidos.")
    
    tab_raw = pd.crosstab(clean[var1], clean[var2], dropna=False)
    row_sum = tab_raw.sum(axis=1)
    col_sum = tab_raw.sum(axis=0)
    excluded_rows = row_sum[row_sum == 0].index.tolist()
    excluded_cols = col_sum[col_sum == 0].index.tolist()
    tab = tab_raw.loc[row_sum > 0, col_sum > 0]
    
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        raise ValueError(f"La tabla debe ser al menos 2x2. Dimensión: {tab.shape[0]}x{tab.shape[1]}")
    
    out = [
        "↔️ ANÁLISIS DE CORRESPONDENCIAS (CA)",
        "=" * 60, "",
        f"Variables: {var1} × {var2}",
        f"Dimensiones: {tab.shape[0]}×{tab.shape[1]}",
        f"N válido: {int(tab.values.sum())}", "",
        "📊 TABLA DE CONTINGENCIA:",
        tab.to_string(), "",
    ]
    
    if excluded_rows or excluded_cols:
        out.append("⚠️ Categorías excluidas por frecuencia 0:")
        if excluded_rows:
            out.append(f"• {var1}: {', '.join(map(str, excluded_rows))}")
        if excluded_cols:
            out.append(f"• {var2}: {', '.join(map(str, excluded_cols))}")
        out.append("")
    
    chi2, p, dof, expected = chi2_contingency(tab)
    n = tab.values.sum()
    denom = n * (min(tab.shape) - 1)
    cramers_v = np.sqrt(chi2 / denom) if denom > 0 else np.nan
    
    out += [
        "🧮 CONTRASTE χ²:",
        f"• χ² = {chi2:.3f}",
        f"• p = {p:.4f}",
        f"• gl = {dof}",
        f"• V de Cramer = {cramers_v:.3f}" if np.isfinite(cramers_v) else "• V de Cramer = n/d",
        "",
    ]
    
    if not PRINCE_OK:
        out += [
            "⚠️ prince no está instalado.",
            "Para obtener el mapa factorial instala: pip install prince",
            "Sin prince se muestra solo la tabla y el contraste χ²."
        ]
        return "\n".join(out), None
    
    # Cálculo CA con prince
    n_components = min(2, min(tab.shape) - 1)
    ca = CA(n_components=n_components, random_state=42)
    ca = ca.fit(tab)
    
    row_coords = ca.row_coordinates(tab)
    col_coords = ca.column_coordinates(tab)
    
    eigenvalues = np.array(getattr(ca, "eigenvalues_", []), dtype=float)
    total_inertia = eigenvalues.sum() if eigenvalues.size else np.nan
    explained = np.array(getattr(ca, "explained_inertia_", []), dtype=float)
    
    if explained.size == 0 and eigenvalues.size and np.isfinite(total_inertia) and total_inertia > 0:
        explained = eigenvalues / total_inertia
    
    out.append("📊 INERCIA:")
    if eigenvalues.size:
        out.append(f"• Total: {total_inertia:.4f}")
        for i, eig in enumerate(eigenvalues[:n_components], start=1):
            pct = explained[i-1] * 100 if len(explained) >= i and np.isfinite(explained[i-1]) else np.nan
            if np.isfinite(pct):
                out.append(f"• Dimensión {i}: eigenvalue={eig:.4f} ({pct:.2f}%)")
            else:
                out.append(f"• Dimensión {i}: eigenvalue={eig:.4f}")
    else:
        out.append("• No disponible en esta versión de prince")
    
    out += [
        "", "📋 COORDENADAS DE FILAS:",
        row_coords.round(4).to_string(), "",
        "📋 COORDENADAS DE COLUMNAS:",
        col_coords.round(4).to_string(), "",
        "💡 INTERPRETACIÓN:",
        "Categorías próximas en el mapa tienden a estar asociadas.",
        "Categorías alejadas del origen suelen contribuir más a la estructura de asociación."
    ]
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(f"Correspondencias: {var1} vs {var2}")
    
    row_x = row_coords.iloc[:, 0]
    col_x = col_coords.iloc[:, 0]
    if row_coords.shape[1] >= 2:
        row_y = row_coords.iloc[:, 1]
        col_y = col_coords.iloc[:, 1]
    else:
        row_y = pd.Series(np.zeros(len(row_coords)), index=row_coords.index)
        col_y = pd.Series(np.zeros(len(col_coords)), index=col_coords.index)
    
    ax.scatter(row_x, row_y, marker="o", s=80, alpha=0.75, label=var1, c='blue')
    for idx in row_coords.index:
        ax.annotate(str(idx), (row_x.loc[idx], row_y.loc[idx]), 
                   xytext=(5, 5), textcoords="offset points", fontsize=8)
    
    ax.scatter(col_x, col_y, marker="^", s=80, alpha=0.75, label=var2, c='red')
    for idx in col_coords.index:
        ax.annotate(str(idx), (col_x.loc[idx], col_y.loc[idx]), 
                   xytext=(5, -8), textcoords="offset points", fontsize=8)
    
    x_pct = explained[0] * 100 if len(explained) >= 1 and np.isfinite(explained[0]) else None
    y_pct = explained[1] * 100 if len(explained) >= 2 and np.isfinite(explained[1]) else None
    ax.set_xlabel(f"Dimensión 1 ({x_pct:.1f}%)" if x_pct is not None else "Dimensión 1")
    ax.set_ylabel(f"Dimensión 2 ({y_pct:.1f}%)" if y_pct is not None else "Dimensión 2")
    ax.axhline(0, linestyle="--", linewidth=0.8, color='gray')
    ax.axvline(0, linestyle="--", linewidth=0.8, color='gray')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    all_x = pd.concat([row_x, col_x]).replace([np.inf, -np.inf], np.nan).dropna()
    all_y = pd.concat([row_y, col_y]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(all_x):
        x_margin = max(0.05, (all_x.max() - all_x.min()) * 0.15)
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    if len(all_y) and all_y.max() != all_y.min():
        y_margin = max(0.05, (all_y.max() - all_y.min()) * 0.15)
        ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)
    elif len(all_y):
        ax.set_ylim(-0.2, 0.2)
    
    return "\n".join(out), fig

# =============================================================================
# UI PRINCIPAL
# =============================================================================

def main():
    """Función principal de la aplicación"""
    init_state()
    
    # Sidebar
    st.sidebar.markdown("## 🧭 Navegación")
    section = st.sidebar.radio(
        "Selecciona módulo",
        [
            "📁 Gestión de Datos",
            "📊 Univariante",
            "🔗 Bivariante",
            "🔮 Avanzado",
            "📋 Exportación",
            "❓ Ayuda"
        ],
        label_visibility="collapsed",
    )
    
    # Panel de estado
    with st.sidebar.expander("📊 Estado", expanded=True):
        df = st.session_state.df
        if df is None:
            st.info("Sin dataset cargado")
        else:
            st.success(f"Dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
            num, cat, cat_like = get_cols(df)
            st.caption(f"Numéricas: {len(num)} | Categóricas: {len(cat_like)}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Reiniciar", use_container_width=True):
                reset_analysis()
                st.rerun()
        with col2:
            if st.button("♻️ Restaurar", use_container_width=True, disabled=st.session_state.df_original is None):
                restore_original_data()
                st.rerun()
    
    # Header
    st.markdown('<div class="ssp-title">📊 SocioStats Pro Web v4.4</div>', unsafe_allow_html=True)
    st.markdown('<div class="ssp-subtitle">Análisis socioestadístico con interfaz web, resultados, gráficos y exportación.</div>', unsafe_allow_html=True)
    
    # Tabs principales
    main_tab, plot_tab, data_tab = st.tabs(["📊 Resultados", "📈 Gráficos", "📋 Datos"])
    
    # ========================================================================
    # SECCIONES
    # ========================================================================
    
    with st.container():
        if section == "📁 Gestión de Datos":
            st.subheader("📁 Gestión de Datos")
            c1, c2 = st.columns([2, 1])
            
            with c1:
                up = st.file_uploader("Cargar CSV o Excel", type=["csv", "txt", "xlsx", "xls"])
                if up is not None:
                    try:
                        with st.spinner("Cargando archivo..."):
                            df_new = read_uploaded_file(up)
                            st.session_state.df = df_new.copy()
                            st.session_state.df_original = df_new.copy()
                            st.session_state.analysis_generated_columns = set()
                            clear_fig_only()
                            set_results(
                                "✅ ARCHIVO CARGADO EXITOSAMENTE\n" + "=" * 50 +
                                f"\n• Archivo: {up.name}\n" +
                                f"• Tamaño: {df_new.shape[0]} filas × {df_new.shape[1]} columnas\n" +
                                f"• Variables: {', '.join(map(str, df_new.columns))}\n" +
                                f"• Tipos: {', '.join([str(df_new[c].dtype) for c in df_new.columns])}"
                            )
                            st.success("Archivo cargado correctamente")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error cargando archivo: {e}")
            
            with c2:
                if st.button("🎲 Datos de ejemplo", use_container_width=True):
                    create_sample_data()
                    st.rerun()
                if st.button("🔍 Información", use_container_width=True, disabled=st.session_state.df is None):
                    set_results(dataset_info(st.session_state.df))
                    st.rerun()
                if st.button("👀 Vista previa", use_container_width=True, disabled=st.session_state.df is None):
                    df_prev = st.session_state.df
                    set_results(
                        f"👀 VISTA PREVIA DE DATOS\n{'='*50}\n\n" +
                        f"📊 Dimensiones: {df_prev.shape[0]}×{df_prev.shape[1]}\n\n" +
                        f"📋 Primeras 10 filas:\n\n{df_prev.head(10).to_string()}\n\n" +
                        f"📋 Últimas 5 filas:\n\n{df_prev.tail(5).to_string()}"
                    )
                    st.rerun()
            
            # Conversión de variables
            if st.session_state.df is not None:
                st.markdown("#### 🔄 Convertir variable")
                df_conv = st.session_state.df
                _, _, cat_like = get_cols(df_conv)
                
                if cat_like:
                    col_a, col_b, col_c = st.columns([1.4, 1.2, 1])
                    var = col_a.selectbox("Variable", cat_like)
                    method = col_b.selectbox("Método", ["Dicotómica (Dummy)", "Etiquetado Numérico"])
                    
                    if col_c.button("Ejecutar conversión", use_container_width=True):
                        try:
                            if method == "Etiquetado Numérico":
                                codes, uniques = pd.factorize(df_conv[var])
                                new_col = f"{var}_num"
                                st.session_state.df[new_col] = codes.astype(np.int64)
                                set_results(
                                    f"✅ Conversión exitosa: '{var}'\n{'='*50}\n" +
                                    f"• Método: Label Encoding\n" +
                                    f"• Nueva variable: '{new_col}'\n" +
                                    f"• Mapeo: {dict(enumerate(uniques))}"
                                )
                            else:
                                dummies = pd.get_dummies(df_conv[var], prefix=var)
                                st.session_state.df = pd.concat([
                                    df_conv.drop(
                                        columns=[c for c in df_conv.columns 
                                                if c.startswith(f'{var}_') and c in dummies.columns],
                                        errors='ignore'
                                    ),
                                    dummies
                                ], axis=1)
                                set_results(
                                    f"✅ Conversión exitosa: '{var}'\n{'='*50}\n" +
                                    f"• Método: Dummy Encoding\n" +
                                    f"• {dummies.shape[1]} nuevas columnas creadas\n" +
                                    f"• Columnas: {', '.join(dummies.columns)}"
                                )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error en conversión: {e}")
                else:
                    st.info("No hay variables categóricas para convertir.")
        
        elif section == "📊 Univariante":
            st.subheader("📊 Análisis Univariante")
            if st.session_state.df is None:
                st.warning("Primero carga los datos.")
            else:
                df_uni = st.session_state.df
                action = st.radio(
                    "Análisis",
                    [
                        "📈 Estadísticas Descriptivas",
                        "📊 Distribuciones",
                        "📉 Histogramas",
                        "🎯 Valores Atípicos",
                        "📋 Tablas Frecuencia",
                        "📋 Resumen Categóricas"
                    ],
                    horizontal=True
                )
                
                if st.button("Ejecutar análisis", type="primary"):
                    try:
                        if action.startswith("📈"):
                            set_results(descriptive_stats(df_uni))
                            clear_fig_only()
                        elif action.startswith("📊"):
                            set_results(distribution_analysis(df_uni))
                            clear_fig_only()
                        elif action.startswith("📉"):
                            num, _, _ = get_cols(df_uni)
                            if not num:
                                raise ValueError("No hay variables numéricas.")
                            ncols = 2
                            nrows = int(np.ceil(len(num) / ncols))
                            fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, 3.8 * nrows)))
                            axes = np.atleast_1d(axes).flatten()
                            fig.suptitle("Histogramas de Variables Numéricas")
                            for i, col in enumerate(num):
                                data = df_uni[col].dropna()
                                axes[i].hist(data, bins=15, alpha=0.75, edgecolor="white")
                                axes[i].axvline(data.mean(), linestyle="--", 
                                              label=f"Media: {data.mean():.1f}")
                                axes[i].axvline(data.median(), linestyle="--", 
                                              label=f"Mediana: {data.median():.1f}")
                                axes[i].set_title(col)
                                axes[i].set_xlabel(col)
                                axes[i].set_ylabel("Frecuencia")
                                axes[i].legend()
                                axes[i].grid(True, alpha=0.3)
                            for j in range(len(num), len(axes)):
                                axes[j].set_visible(False)
                            set_results(f"📉 HISTOGRAMAS GENERADOS\n{'='*50}\nVariables: {', '.join(num)}")
                            set_fig(fig)
                        elif action.startswith("🎯"):
                            set_results(outlier_analysis(df_uni))
                            clear_fig_only()
                        elif action.startswith("📋 Tablas"):
                            set_results(frequency_tables(df_uni))
                            clear_fig_only()
                        else:
                            set_results(categorical_summary(df_uni))
                            clear_fig_only()
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        
        elif section == "🔗 Bivariante":
            st.subheader("🔗 Análisis Bivariante")
            if st.session_state.df is None:
                st.warning("Primero carga los datos.")
            else:
                df_bi = st.session_state.df
                num, cat, cat_like = get_cols(df_bi)
                
                action = st.selectbox(
                    "Análisis",
                    [
                        "📋 Tablas Contingencia",
                        "📈 Correlaciones",
                        "👥 Comparar Grupos/ANOVA",
                        "📊 Scatter Plots",
                        "📐 Regresión Simple"
                    ]
                )
                
                try:
                    if action.startswith("📋"):
                        c1, c2 = st.columns(2)
                        v1 = c1.selectbox("Variable categórica 1", cat_like)
                        v2 = c2.selectbox("Variable categórica 2", [c for c in cat_like if c != v1] or cat_like)
                        if st.button("Calcular", type="primary"):
                            set_results(contingency_result(df_bi, v1, v2))
                            clear_fig_only()
                            st.rerun()
                    
                    elif action.startswith("📈"):
                        if st.button("Calcular correlaciones", type="primary"):
                            txt, fig = correlation_analysis(df_bi)
                            set_results(txt)
                            set_fig(fig)
                            st.rerun()
                    
                    elif action.startswith("👥"):
                        c1, c2 = st.columns(2)
                        gv = c1.selectbox("Variable de grupos", cat_like)
                        nv = c2.selectbox("Variable numérica", num)
                        if st.button("Comparar grupos", type="primary"):
                            txt, fig = group_comparison(df_bi, gv, nv)
                            set_results(txt)
                            set_fig(fig)
                            st.rerun()
                    
                    elif action.startswith("📊"):
                        c1, c2 = st.columns(2)
                        x = c1.selectbox("X", num)
                        y = c2.selectbox("Y", [c for c in num if c != x] or num)
                        if st.button("Crear dispersión", type="primary"):
                            clean = df_bi[[x, y]].dropna()
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(clean[x], clean[y], alpha=0.65, s=45)
                            ax.set_xlabel(x)
                            ax.set_ylabel(y)
                            ax.set_title(f"Dispersión: {x} vs {y}")
                            ax.grid(True, alpha=0.3)
                            if len(clean) > 2 and clean[x].nunique() > 1 and clean[y].nunique() > 1:
                                r, p = pearsonr(clean[x], clean[y])
                                ax.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.4f}", 
                                       transform=ax.transAxes,
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                                       fontsize=8)
                                set_results(f"📊 SCATTER PLOT\n{'='*50}\n{x} vs {y}\n• r = {r:.3f}\n• p = {p:.4f}")
                            else:
                                set_results(f"📊 SCATTER PLOT\n{'='*50}\n{x} vs {y}")
                            set_fig(fig)
                            st.rerun()
                    
                    else:
                        c1, c2 = st.columns(2)
                        x = c1.selectbox("Independiente X", num)
                        y = c2.selectbox("Dependiente Y", [c for c in num if c != x] or num)
                        if st.button("Calcular regresión simple", type="primary"):
                            txt, fig = simple_regression(df_bi, x, y)
                            set_results(txt)
                            set_fig(fig)
                            st.rerun()
                except Exception as e:
                    st.error(str(e))
        
        elif section == "🔮 Avanzado":
            st.subheader("🔮 Análisis Avanzado")
            if st.session_state.df is None:
                st.warning("Primero carga los datos.")
            else:
                df_adv = st.session_state.df
                num, cat, cat_like = get_cols(df_adv)
                
                action = st.selectbox(
                    "Análisis",
                    [
                        "📈 Regresión Múltiple",
                        "🧠 Regresión Logística",
                        "🔍 Análisis Cluster",
                        "📊 Componentes Principales",
                        "👥 Análisis Discriminante",
                        "↔️ Análisis de Correspondencias"
                    ]
                )
                
                try:
                    if action.startswith("📈"):
                        y = st.selectbox("Variable dependiente", num)
                        xs = st.multiselect("Variables independientes", [c for c in num if c != y],
                                           default=[c for c in num if c != y][:2])
                        if st.button("Calcular regresión múltiple", type="primary"):
                            if not xs:
                                raise ValueError("Selecciona al menos una predictora.")
                            txt, fig = multiple_regression(df_adv, y, xs)
                            set_results(txt)
                            set_fig(fig)
                            st.rerun()
                    
                    elif action.startswith("🧠"):
                        y = st.selectbox("Variable dependiente binaria", df_adv.columns.tolist())
                        xs = st.multiselect("Predictoras", [c for c in df_adv.columns if c != y],
                                          default=[c for c in df_adv.columns if c != y][:3])
                        if st.button("Calcular logística", type="primary"):
                            if not xs:
                                raise ValueError("Selecciona al menos una predictora.")
                            txt = logistic_regression(df_adv, y, xs)
                            set_results(txt)
                            clear_fig_only()
                            st.rerun()
                    
                    elif action.startswith("🔍"):
                        xs = st.multiselect("Variables para clustering", num, default=num[:min(4, len(num))])
                        if st.button("Calcular clusters", type="primary"):
                            if len(xs) < 2:
                                raise ValueError("Selecciona al menos dos variables.")
                            txt, fig = cluster_analysis(df_adv, xs)
                            set_results(txt)
                            set_fig(fig)
                            st.rerun()
                    
                    elif action.startswith("📊"):
                        xs = st.multiselect("Variables para PCA", num, default=num[:min(6, len(num))])
                        if st.button("Calcular PCA", type="primary"):
                            if len(xs) < 2:
                                raise ValueError("Selecciona al menos dos variables.")
                            txt, fig = pca_analysis(df_adv, xs)
                            set_results(txt)
                            set_fig(fig)
                            st.rerun()
                    
                    elif action.startswith("👥"):
                        y = st.selectbox("Variable de grupos", cat_like)
                        xs = st.multiselect("Predictoras numéricas", num, default=num[:min(4, len(num))])
                        if st.button("Calcular LDA", type="primary"):
                            if not xs:
                                raise ValueError("Selecciona al menos una predictora.")
                            txt = lda_analysis(df_adv, y, xs)
                            set_results(txt)
                            clear_fig_only()
                            st.rerun()
                    
                    else:  # Correspondencias
                        c1, c2 = st.columns(2)
                        v1 = c1.selectbox("Variable categórica 1", cat_like)
                        v2 = c2.selectbox("Variable categórica 2", [c for c in cat_like if c != v1] or cat_like)
                        if not PRINCE_OK:
                            st.info("Para mapa de correspondencias instala: pip install prince")
                        if st.button("Calcular correspondencias", type="primary"):
                            txt, fig = correspondence_analysis(df_adv, v1, v2)
                            set_results(txt)
                            if fig is not None:
                                set_fig(fig)
                            else:
                                clear_fig_only()
                            st.rerun()
                except Exception as e:
                    st.error(str(e))
        
        elif section == "📋 Exportación":
            st.subheader("📋 Exportación")
            
            res = st.session_state.current_results or ""
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "💾 Descargar resultados TXT",
                    data=res.encode("utf-8"),
                    file_name=f"resultados_sociostats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                if st.session_state.df is not None:
                    csv = st.session_state.df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📋 Descargar datos CSV",
                        data=csv,
                        file_name="datos_sociostats.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                report = dataset_info(st.session_state.df) + "\n\n📊 ÚLTIMOS RESULTADOS\n" + "-"*70 + "\n" + res
                st.download_button(
                    "📄 Descargar reporte completo",
                    data=report.encode("utf-8"),
                    file_name=f"reporte_sociostats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                if st.session_state.current_fig is not None:
                    fig = st.session_state.current_fig
                    st.download_button(
                        "🖼️ Descargar gráfico PNG",
                        data=fig_to_bytes(fig, "png"),
                        file_name="grafico_sociostats.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    st.download_button(
                        "🖼️ Descargar gráfico PDF",
                        data=fig_to_bytes(fig, "pdf"),
                        file_name="grafico_sociostats.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        else:  # Ayuda
            st.subheader("❓ Ayuda")
            st.markdown("""
**SocioStats Pro Web v4.4** conserva la lógica principal de la aplicación de escritorio y la adapta a Streamlit.

### 📁 Gestión de datos
- **Carga**: CSV/Excel con detección automática de codificación y separadores
- **Datos de ejemplo**: Dataset sintético con variables numéricas y categóricas
- **Conversión**: Codificación de variables categóricas (Label Encoding o Dummy)

### 📊 Univariante
- **Descriptivas**: Media, mediana, desviación, asimetría, curtosis
- **Distribuciones**: Test de normalidad (Shapiro-Wilk/D'Agostino)
- **Histogramas**: Visualización de distribuciones
- **Atípicos**: Detección por método IQR
- **Frecuencias**: Tablas para variables categóricas
- **Resumen categóricas**: Resumen compacto de variables categóricas

### 🔗 Bivariante
- **Contingencia**: Tablas, χ², V de Cramer
- **Correlaciones**: Matriz y mapa de calor
- **Comparación de grupos**: t-test/ANOVA con boxplots
- **Scatter**: Gráficos de dispersión con correlación
- **Regresión simple**: Modelo lineal con diagnóstico

### 🔮 Avanzado
- **Regresión múltiple**: Modelo lineal con múltiples predictoras
- **Logística**: Modelo logístico (requiere statsmodels)
- **Clustering**: K-means con selección óptima de k
- **PCA**: Componentes principales
- **LDA**: Análisis discriminante lineal
- **Correspondencias**: Análisis factorial de tablas (requiere prince)

### 💡 Consejos
- Usa **Reiniciar análisis** para limpiar resultados/gráficos y borrar columnas auxiliares
- Los gráficos se pueden descargar en PNG o PDF
- Los datos se exportan en CSV con la codificación adecuada
            """)
    
    # ========================================================================
    # TABS DE RESULTADOS, GRÁFICOS Y DATOS
    # ========================================================================
    
    with main_tab:
        st.markdown("#### Resultado actual")
        st.markdown(f"<div class='result-box'>{st.session_state.current_results}</div>", unsafe_allow_html=True)
    
    with plot_tab:
        st.markdown("#### Gráfico actual")
        if st.session_state.current_fig is None:
            st.info("No hay gráficos activos. Ejecuta un análisis gráfico para generarlo.")
        else:
            st.pyplot(st.session_state.current_fig, clear_figure=False, use_container_width=True)
    
    with data_tab:
        st.markdown("#### Dataset actual")
        if st.session_state.df is None:
            st.info("No hay datos cargados.")
        else:
            df_display = st.session_state.df
            c1, c2, c3 = st.columns(3)
            c1.metric("Filas", df_display.shape[0])
            c2.metric("Columnas", df_display.shape[1])
            c3.metric("Nulos", int(df_display.isna().sum().sum()))
            
            st.dataframe(df_display.head(500), use_container_width=True, height=420)
            if df_display.shape[0] > 500:
                st.caption("Se muestran las primeras 500 filas por rendimiento.")
    
    # Barra de estado
    st.markdown(f"<div class='status-bar'>{st.session_state.status}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
