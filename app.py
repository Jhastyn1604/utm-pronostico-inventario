# =====================================================================
# APP STREAMLIT — PROYECTO TESIS UT&M
# Sistema de Pronósticos y Gestión de Inventarios
# Melaminas Pelikano e Hispanos
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import itertools
import warnings
import io

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="UT&M — Pronósticos e Inventarios",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# FUNCIONES AUXILIARES
# =====================================================================

def redondear_05(x):
    """Redondear hacia arriba a múltiplo de 0.5"""
    return np.ceil(x * 2) / 2


def nombre_corto(producto, max_len=40):
    texto = str(producto).strip()
    if len(texto) <= max_len:
        return texto
    cortado = texto[:max_len]
    ultimo = cortado.rfind(" ")
    if ultimo > max_len // 2:
        return cortado[:ultimo] + "…"
    return cortado + "…"


# =====================================================================
# MÉTRICAS DE PRONÓSTICO
# =====================================================================

def calc_mape(r, p):
    r, p = np.array(r, dtype=float), np.array(p, dtype=float)
    m = r != 0
    return np.mean(np.abs((r[m] - p[m]) / r[m])) * 100 if m.sum() > 0 else 999


def calc_rmse(r, p):
    return np.sqrt(np.mean((np.array(r, dtype=float) - np.array(p, dtype=float)) ** 2))


# =====================================================================
# MODELOS DE PRONÓSTICO
# =====================================================================

def fit_ses(tr, h):
    n = len(tr)
    def obj(pa):
        a = pa[0]; L = tr[0]; s = 0
        for t in range(1, n):
            s += (tr[t] - L) ** 2
            L = a * tr[t] + (1 - a) * L
        return s
    r = minimize(obj, [0.3], bounds=[(0.01, 0.99)], method='L-BFGS-B')
    a = r.x[0]; L = tr[0]
    for t in range(1, n):
        L = a * tr[t] + (1 - a) * L
    return np.full(h, L), f'SES (α={a:.3f})'


def fit_holt(tr, h, damp=False):
    n = len(tr)
    def obj(pa):
        a, b = pa[0], pa[1]; phi = pa[2] if damp else 1.0
        L = tr[0]; T = tr[min(1, n - 1)] - tr[0]; s = 0
        for t in range(1, n):
            s += (tr[t] - (L + phi * T)) ** 2
            nL = a * tr[t] + (1 - a) * (L + phi * T)
            T = b * (nL - L) + (1 - b) * phi * T; L = nL
        return s
    if damp:
        r = minimize(obj, [0.3, 0.1, 0.9], bounds=[(0.01, 0.99)] * 3, method='L-BFGS-B')
        a, b, phi = r.x
    else:
        r = minimize(obj, [0.3, 0.1], bounds=[(0.01, 0.99)] * 2, method='L-BFGS-B')
        a, b = r.x; phi = 1.0
    L = tr[0]; T = tr[min(1, n - 1)] - tr[0]
    for t in range(1, n):
        nL = a * tr[t] + (1 - a) * (L + phi * T)
        T = b * (nL - L) + (1 - b) * phi * T; L = nL
    fc = [L + sum(phi ** j for j in range(1, i + 1)) * T if damp else L + i * T for i in range(1, h + 1)]
    return np.maximum(np.array(fc), 0), 'Holt amortiguado' if damp else 'Holt'


def fit_hw(tr, h, seas, trend, s):
    n = len(tr)
    if n < 2 * s:
        return None, ''
    try:
        def obj(pa):
            a, b_p, g = pa
            L = np.mean(tr[:s])
            T = (np.mean(tr[s:2 * s]) - np.mean(tr[:s])) / s if trend else 0
            S = np.zeros(n + s)
            for i in range(s):
                S[i] = (tr[i] - L) if seas == 'add' else (tr[i] / max(L, 1))
            ss = 0
            for t in range(s, n):
                if seas == 'add':
                    fc = (L + T + S[t - s]) if trend else (L + S[t - s])
                else:
                    fc = ((L + T) * S[t - s]) if trend else (L * S[t - s])
                ss += (tr[t] - fc) ** 2
                if seas == 'add':
                    nL = a * (tr[t] - S[t - s]) + (1 - a) * (L + T)
                    nS = g * (tr[t] - L - T) + (1 - g) * S[t - s]
                else:
                    den = max(L + T if trend else L, 1)
                    nL = a * (tr[t] / max(S[t - s], 0.01)) + (1 - a) * (L + T)
                    nS = g * (tr[t] / den) + (1 - g) * S[t - s]
                nT = (b_p * (nL - L) + (1 - b_p) * T) if trend else 0
                S[t] = nS; L = nL; T = nT
            return ss
        if trend:
            r = minimize(obj, [0.3, 0.1, 0.1], bounds=[(0.01, 0.99)] * 3, method='L-BFGS-B')
        else:
            r = minimize(obj, [0.3, 0.01, 0.1], bounds=[(0.01, 0.99), (0.001, 0.01), (0.01, 0.99)], method='L-BFGS-B')
        a, b_p, g = r.x
        L = np.mean(tr[:s]); T = (np.mean(tr[s:2 * s]) - np.mean(tr[:s])) / s if trend else 0
        S = np.zeros(n + s + h)
        for i in range(s):
            S[i] = (tr[i] - L) if seas == 'add' else (tr[i] / max(L, 1))
        for t in range(s, n):
            if seas == 'add':
                nL = a * (tr[t] - S[t - s]) + (1 - a) * (L + T)
                S[t] = g * (tr[t] - L - T) + (1 - g) * S[t - s]
            else:
                den = max(L + T if trend else L, 1)
                nL = a * (tr[t] / max(S[t - s], 0.01)) + (1 - a) * (L + T)
                S[t] = g * (tr[t] / den) + (1 - g) * S[t - s]
            nT = (b_p * (nL - L) + (1 - b_p) * T) if trend else 0
            L = nL; T = nT
        fc = []
        for i in range(h):
            idx = n - s + (i + 1) % s
            if seas == 'add':
                v = (L + (i + 1) * T + S[idx]) if trend else (L + S[idx])
            else:
                v = ((L + (i + 1) * T) * S[idx]) if trend else (L * S[idx])
            fc.append(v)
        ts = 'Con tend.' if trend else 'Sin tend.'
        ss_s = 'aditivo' if seas == 'add' else 'mult.'
        return np.maximum(np.array(fc), 0), f'HW {ts} {ss_s} (s={s})'
    except:
        return None, ''


def fit_arima(tr, h, order):
    p, d, q = order; y = tr.copy(); dv = []
    for _ in range(d):
        dv.append(y.copy()); y = np.diff(y)
    n = len(y)
    if n <= max(p, q) + 2:
        return None
    try:
        def obj(pa):
            c = pa[0]; phi = pa[1:1 + p]; theta = pa[1 + p:1 + p + q]
            res = np.zeros(n)
            for t in range(max(p, q, 1), n):
                pred = c
                for i in range(p): pred += phi[i] * y[t - 1 - i]
                for j in range(q): pred += theta[j] * res[t - 1 - j]
                res[t] = y[t] - pred
            return np.sum(res[max(p, q, 1):] ** 2)
        x0 = np.zeros(1 + p + q); x0[0] = np.mean(y)
        result = minimize(obj, x0, bounds=[(-1e6, 1e6)] + [(-0.99, 0.99)] * (p + q),
                          method='L-BFGS-B', options={'maxiter': 2000})
        c = result.x[0]; phi = result.x[1:1 + p]; theta = result.x[1 + p:1 + p + q]
        res = np.zeros(n)
        for t in range(max(p, q, 1), n):
            pred = c
            for i in range(p): pred += phi[i] * y[t - 1 - i]
            for j in range(q): pred += theta[j] * res[t - 1 - j]
            res[t] = y[t] - pred
        ye = np.concatenate([y, np.zeros(h)]); re = np.concatenate([res, np.zeros(h)])
        for t in range(n, n + h):
            pred = c
            for i in range(p): pred += phi[i] * ye[t - 1 - i]
            for j in range(q):
                if t - 1 - j < n: pred += theta[j] * re[t - 1 - j]
            ye[t] = pred
        fc = ye[n:]
        for dval in reversed(dv):
            fc2 = np.zeros(len(fc)); lv = dval[-1]
            for i in range(len(fc)):
                lv += fc[i]; fc2[i] = lv
            fc = fc2
        return np.maximum(fc, 0)
    except:
        return None


def fit_sarima(tr, h, order, sorder):
    p, d, q = order; P, D, Q, s = sorder; y = tr.copy()
    for _ in range(D):
        if len(y) > s:
            y = y[s:] - y[:-s]
        else:
            return None
    dr = []
    for _ in range(d):
        dr.append(y.copy()); y = np.diff(y)
    n = len(y)
    al = sorted(set(list(range(1, p + 1)) + [pp * s for pp in range(1, P + 1)] +
                     [pp * s + i for pp in range(1, P + 1) for i in range(1, p + 1)]))
    ml = sorted(set(list(range(1, q + 1)) + [qq * s for qq in range(1, Q + 1)] +
                     [qq * s + j for qq in range(1, Q + 1) for j in range(1, q + 1)]))
    mx = max(al + ml + [1]); na = len(al); nm = len(ml)
    if n <= mx + 2:
        return None
    try:
        def obj(pa):
            c = pa[0]; phi = pa[1:1 + na]; theta = pa[1 + na:]
            res = np.zeros(n)
            for t in range(mx, n):
                pred = c
                for i, lag in enumerate(al):
                    if t - lag >= 0: pred += phi[i] * y[t - lag]
                for j, lag in enumerate(ml):
                    if t - lag >= 0: pred += theta[j] * res[t - lag]
                res[t] = y[t] - pred
            return np.sum(res[mx:] ** 2)
        x0 = np.zeros(1 + na + nm); x0[0] = np.mean(y)
        result = minimize(obj, x0, bounds=[(-1e6, 1e6)] + [(-0.99, 0.99)] * (na + nm),
                          method='L-BFGS-B', options={'maxiter': 2000})
        c = result.x[0]; phi = result.x[1:1 + na]; theta = result.x[1 + na:]
        res = np.zeros(n)
        for t in range(mx, n):
            pred = c
            for i, lag in enumerate(al):
                if t - lag >= 0: pred += phi[i] * y[t - lag]
            for j, lag in enumerate(ml):
                if t - lag >= 0: pred += theta[j] * res[t - lag]
            res[t] = y[t] - pred
        ye = np.concatenate([y, np.zeros(h)]); re = np.concatenate([res, np.zeros(h)])
        for t in range(n, n + h):
            pred = c
            for i, lag in enumerate(al):
                if t - lag >= 0: pred += phi[i] * ye[t - lag]
            for j, lag in enumerate(ml):
                if t - lag >= 0 and t - lag < n: pred += theta[j] * re[t - lag]
            ye[t] = pred
        fc = ye[n:]
        for dval in reversed(dr):
            fc2 = np.zeros(len(fc)); lv = dval[-1]
            for i in range(len(fc)):
                lv += fc[i]; fc2[i] = lv
            fc = fc2
        for _ in range(D):
            fc2 = np.zeros(len(fc)); tail = tr[-(s * max(D, 1)):]
            for i in range(len(fc)):
                ref = tail[len(tail) - s + i] if 0 <= len(tail) - s + i < len(tail) else (fc2[i - s] if i >= s else tr[-1])
                fc2[i] = fc[i] + ref
            fc = fc2
        return np.maximum(fc, 0)
    except:
        return None


def ejecutar_pronostico(serie, horizonte_fc=6, progress_cb=None):
    """Ejecuta todos los modelos y devuelve ranking + forecast futuro."""
    y = serie.values.astype(float)
    n = len(y)
    n_test = min(3, max(1, n // 4))
    n_train = n - n_test
    train = y[:n_train]; test = y[n_train:]; h = len(test)

    modelos = []
    # Naive
    modelos.append(('Naive', 'Base', np.full(h, train[-1])))
    # Deriva
    pend = (train[-1] - train[0]) / max(n_train - 1, 1)
    modelos.append(('Deriva', 'Base', np.maximum(np.array([train[-1] + pend * (i + 1) for i in range(h)]), 0)))
    # Regresión lineal
    coefs = np.polyfit(np.arange(1, n_train + 1), train, 1)
    modelos.append(('Regresión lineal', 'Base', np.maximum(np.polyval(coefs, np.arange(n_train + 1, n_train + h + 1)), 0)))
    # SMA
    for k in range(2, min(7, n_train)):
        hist = train.tolist(); pred = []
        for _ in range(h):
            v = np.mean(hist[-k:]); pred.append(v); hist.append(v)
        modelos.append((f'Promedio móvil ({k}m)', 'SMA', np.maximum(np.array(pred), 0)))
    # SES, Holt
    p_, nm = fit_ses(train, h); modelos.append((nm, 'SES', p_))
    p_, nm = fit_holt(train, h, False); modelos.append((nm, 'Holt', p_))
    p_, nm = fit_holt(train, h, True); modelos.append((nm, 'Holt', p_))
    # HW
    for s in [3, 4]:
        if n_train >= 2 * s:
            for seas in ['add', 'mul']:
                for trend in [True, False]:
                    p_, nm = fit_hw(train, h, seas, trend, s)
                    if p_ is not None:
                        modelos.append((nm, 'HW', p_))
    # ARIMA
    for p, d, q in itertools.product(range(4), range(3), range(4)):
        if p == 0 and d == 0 and q == 0: continue
        pred = fit_arima(train, h, (p, d, q))
        if pred is not None and not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
            if calc_mape(test, pred) < 200:
                modelos.append((f'ARIMA({p},{d},{q})', 'ARIMA', pred))
    # SARIMA s=3
    s = 3
    if n_train >= 2 * s:
        for p, d, q in itertools.product(range(2), range(2), range(2)):
            for P, D, Q in itertools.product(range(2), range(2), range(2)):
                if p == 0 and q == 0 and P == 0 and Q == 0: continue
                pred = fit_sarima(train, h, (p, d, q), (P, D, Q, s))
                if pred is not None and not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
                    if calc_mape(test, pred) < 200:
                        modelos.append((f'SARIMA({p},{d},{q})({P},{D},{Q},{s})', 'SARIMA', pred))

    # Evaluar y rankear
    res = []
    for nombre, fam, pred in modelos:
        res.append({'Método': nombre, 'Familia': fam,
                    'MAPE': calc_mape(test, pred), 'RMSE': calc_rmse(test, pred), 'pred': pred})
    res.sort(key=lambda x: (x['MAPE'], x['RMSE']))

    best = res[0]
    nombre_ganador = best['Método']

    # Re-entrenar con todos los datos
    if nombre_ganador.startswith('SARIMA'):
        parts = nombre_ganador.replace('SARIMA', '').replace(' ', '')
        g1 = parts.split(')(')[0].replace('(', ''); g2 = parts.split(')(')[1].replace(')', '')
        n1 = [int(x) for x in g1.split(',')]; n2 = [int(x) for x in g2.split(',')]
        forecast_futuro = fit_sarima(y, horizonte_fc, tuple(n1), tuple(n2))
    elif nombre_ganador.startswith('ARIMA'):
        nums = [int(x) for x in nombre_ganador.replace('ARIMA(', '').replace(')', '').split(',')]
        forecast_futuro = fit_arima(y, horizonte_fc, tuple(nums))
    elif 'Promedio móvil' in nombre_ganador:
        k = int(nombre_ganador.split('(')[1].split('m')[0])
        hist = y.tolist(); forecast_futuro = []
        for _ in range(horizonte_fc):
            v = np.mean(hist[-k:]); forecast_futuro.append(v); hist.append(v)
        forecast_futuro = np.array(forecast_futuro)
    elif nombre_ganador.startswith('SES'):
        forecast_futuro, _ = fit_ses(y, horizonte_fc)
    elif nombre_ganador.startswith('Holt'):
        forecast_futuro, _ = fit_holt(y, horizonte_fc, 'amort' in nombre_ganador)
    elif nombre_ganador == 'Naive':
        forecast_futuro = np.full(horizonte_fc, y[-1])
    else:
        forecast_futuro = np.full(horizonte_fc, np.mean(y))

    if forecast_futuro is None:
        forecast_futuro = np.full(horizonte_fc, np.mean(y))

    return {
        'ranking': res[:20], 'modelo': nombre_ganador, 'mape': best['MAPE'],
        'rmse': best['RMSE'], 'forecast': forecast_futuro,
        'train': train, 'test': test, 'n_train': n_train, 'n_test': n_test,
    }


# =====================================================================
# MONTE CARLO DEL PRONÓSTICO (N°1) — 10,000 ITERACIONES
# =====================================================================
N_MC_PRONOSTICO = 10000


def ejecutar_mc_pronostico(forecast, mape_pct, horizonte_fc=6, seed=42):
    """
    Monte Carlo sobre el pronóstico: 10,000 iteraciones por mes.
    σ_t = Forecast_t × MAPE (en decimal).
    D_t^(i) ~ N(Forecast_t, σ_t), truncada en 0.
    """
    rng = np.random.default_rng(seed)
    mape_dec = mape_pct / 100.0
    sigma = forecast * mape_dec

    sim = np.zeros((horizonte_fc, N_MC_PRONOSTICO))
    for t in range(horizonte_fc):
        sim[t, :] = rng.normal(loc=forecast[t], scale=sigma[t], size=N_MC_PRONOSTICO)
        sim[t, :] = np.maximum(sim[t, :], 0)

    acum = np.sum(sim, axis=0)

    indicadores = []
    for t in range(horizonte_fc):
        d = sim[t, :]
        media = np.mean(d)
        mediana = np.median(d)
        std = np.std(d)
        cv = std / media if media > 0 else 0
        p2_5 = np.percentile(d, 2.5)
        p5 = np.percentile(d, 5)
        p50 = np.percentile(d, 50)
        p90 = np.percentile(d, 90)
        p95 = np.percentile(d, 95)
        p97_5 = np.percentile(d, 97.5)
        p99 = np.percentile(d, 99)
        var_inf_95 = p5       # VaR inferior 95%
        var_sup_95 = p95      # VaR superior 95%
        cvar_inf = np.mean(d[d <= p5]) if np.sum(d <= p5) > 0 else p5
        cvar_sup = np.mean(d[d >= p95]) if np.sum(d >= p95) > 0 else p95

        indicadores.append({
            'Forecast': forecast[t],
            'Sigma': sigma[t],
            'Media MC': media,
            'Mediana MC': mediana,
            'Desv. Std.': std,
            'CV': cv,
            'Mín': np.min(d),
            'Máx': np.max(d),
            'P2.5': p2_5,
            'P5': p5,
            'P50': p50,
            'P90': p90,
            'P95': p95,
            'P97.5': p97_5,
            'P99': p99,
            'VaR inf. 95%': var_inf_95,
            'VaR sup. 95%': var_sup_95,
            'CVaR inf. 95%': cvar_inf,
            'CVaR sup. 95%': cvar_sup,
            'Dif. Media vs Forecast': media - forecast[t],
            'Dif. P50 vs Forecast': p50 - forecast[t],
        })

    # Acumulado
    acum_stats = {
        'Forecast acum.': np.sum(forecast),
        'Media MC acum.': np.mean(acum),
        'Desv. Std. acum.': np.std(acum),
        'CV acum.': np.std(acum) / np.mean(acum) if np.mean(acum) > 0 else 0,
        'IC 95% inf': np.percentile(acum, 2.5),
        'IC 95% sup': np.percentile(acum, 97.5),
        'VaR inf. 95% acum.': np.percentile(acum, 5),
        'VaR sup. 95% acum.': np.percentile(acum, 95),
    }

    return {
        'sim': sim,
        'acum': acum,
        'indicadores': indicadores,
        'acum_stats': acum_stats,
    }


# =====================================================================
# SIMULACIÓN DE POLÍTICAS (CORREGIDA)
# =====================================================================

def simular_politica(p, politica, params_pol, horizonte, demanda_diaria=None, lt_diaria=None):
    """Simula política de inventario. Fix: BL = faltante - VP (conservación exacta)."""
    d_mean = p['d']; L_mean = p['L']; Pv = p['Pv']
    pct_vp = p['pct_vp']; H_d = p['H_diario']; b_d = p['b_d']; K = p['K']

    if demanda_diaria is None:
        demanda = np.full(horizonte, d_mean)
    else:
        demanda = demanda_diaria[:horizonte] if len(demanda_diaria) >= horizonte else \
            np.concatenate([demanda_diaria, np.full(horizonte - len(demanda_diaria), d_mean)])

    inventario = max(p['stock_ini'], 0)
    backlog = 0.0
    ordenes_en_transito = []

    registros = {k: [] for k in [
        'dia', 'demanda', 'atendida', 'vp', 'bl_nuevo', 'bl_final',
        'inv_inicio', 'inv_final', 'pedido', 'q_pedido', 'costo_ordenar',
        'costo_mant', 'costo_vp', 'costo_bl', 'costo_total', 'quiebre',
        'bl_atendido', 'recibido']}

    for dia in range(horizonte):
        llegadas = sum(q for (d_ll, q) in ordenes_en_transito if d_ll <= dia)
        ordenes_en_transito = [(d_ll, q) for (d_ll, q) in ordenes_en_transito if d_ll > dia]
        inventario += llegadas
        inv_inicio = inventario

        bl_atendido = 0.0
        if backlog > 0 and inventario > 0:
            atender_bl = min(backlog, inventario)
            backlog -= atender_bl
            inventario -= atender_bl
            bl_atendido = atender_bl

        dem_hoy = redondear_05(demanda[dia])
        atendida = min(dem_hoy, inventario)
        faltante = dem_hoy - atendida
        inventario -= atendida

        # FIX: VP con redondeo, BL = complemento exacto
        vp_hoy = redondear_05(faltante * pct_vp)
        bl_nuevo = faltante - vp_hoy
        backlog += bl_nuevo

        OO = sum(q for (_, q) in ordenes_en_transito)
        IP = inventario + OO - backlog

        pedido = 0; q_pedido = 0
        if politica == 'QS':
            if IP <= params_pol['s']:
                q_pedido = int(np.ceil(params_pol['Q'])); pedido = 1
        elif politica == 'TS':
            T = params_pol['T']
            if dia % T == 0:
                q_calc = params_pol['S'] - IP
                if q_calc > 0:
                    q_pedido = int(np.ceil(q_calc)); pedido = 1
        elif politica == 'sS':
            if IP <= params_pol['s']:
                q_calc = params_pol['S'] - IP
                if q_calc > 0:
                    q_pedido = int(np.ceil(q_calc)); pedido = 1

        if pedido:
            lt = int(np.ceil(lt_diaria[dia])) if lt_diaria is not None and dia < len(lt_diaria) else int(np.ceil(L_mean))
            ordenes_en_transito.append((dia + lt, q_pedido))

        c_ord = K if pedido else 0
        c_mant = max(inventario, 0) * H_d
        c_vp = vp_hoy * Pv
        c_bl = backlog * b_d

        registros['dia'].append(dia)
        registros['demanda'].append(dem_hoy)
        registros['atendida'].append(atendida)
        registros['vp'].append(vp_hoy)
        registros['bl_nuevo'].append(bl_nuevo)
        registros['bl_final'].append(backlog)
        registros['inv_inicio'].append(inv_inicio)
        registros['inv_final'].append(inventario)
        registros['pedido'].append(pedido)
        registros['q_pedido'].append(q_pedido)
        registros['costo_ordenar'].append(c_ord)
        registros['costo_mant'].append(c_mant)
        registros['costo_vp'].append(c_vp)
        registros['costo_bl'].append(c_bl)
        registros['costo_total'].append(c_ord + c_mant + c_vp + c_bl)
        registros['quiebre'].append(1 if faltante > 0 else 0)
        registros['bl_atendido'].append(bl_atendido)
        registros['recibido'].append(llegadas)

    return pd.DataFrame(registros)


def consolidar_mensual_calendario(sim_df, fecha_inicio):
    """Agrupa simulación diaria por mes calendario real."""
    sim_df = sim_df.copy()
    fechas = pd.date_range(start=fecha_inicio, periods=len(sim_df), freq='D')
    sim_df['fecha'] = fechas
    sim_df['mes_cal'] = sim_df['fecha'].dt.to_period('M')

    mensual = sim_df.groupby('mes_cal').agg(
        dias=('dia', 'count'),
        demanda=('demanda', 'sum'),
        atendida=('atendida', 'sum'),
        bl_atendido=('bl_atendido', 'sum'),
        vp=('vp', 'sum'),
        bl_nuevo=('bl_nuevo', 'sum'),
        bl_final=('bl_final', 'last'),
        inv_promedio=('inv_final', 'mean'),
        inv_final=('inv_final', 'last'),
        pedidos=('pedido', 'sum'),
        und_pedidas=('q_pedido', 'sum'),
        und_recibidas=('recibido', 'sum'),
        costo_ordenar=('costo_ordenar', 'sum'),
        costo_mant=('costo_mant', 'sum'),
        costo_vp=('costo_vp', 'sum'),
        costo_bl=('costo_bl', 'sum'),
        costo_total=('costo_total', 'sum'),
        dias_quiebre=('quiebre', 'sum'),
    ).reset_index()

    mensual['fill_rate'] = mensual['atendida'] / mensual['demanda']
    mensual['fill_rate'] = mensual['fill_rate'].clip(0, 1)
    mensual['mes_cal'] = mensual['mes_cal'].astype(str)
    return mensual


def ejecutar_mc(p, pol_type, pol_params, horizonte, n_mc=10000, seed=42):
    """Monte Carlo con bootstrap empírico."""
    np.random.seed(seed)
    dem_emp = p['dem_empirica']
    lt_emp = p['lt_empirica']

    mc = {k: np.zeros(n_mc) for k in [
        'costo_total', 'inv_prom', 'inv_max', 'vp', 'bl_max', 'bl_final',
        'fill_rate', 'pedidos']}

    for it in range(n_mc):
        dem_sim = np.random.choice(dem_emp, size=horizonte, replace=True)
        lt_sim = np.random.choice(lt_emp, size=horizonte, replace=True)
        sim = simular_politica(p, pol_type, pol_params, horizonte, dem_sim, lt_sim)
        mc['costo_total'][it] = sim['costo_total'].sum()
        mc['inv_prom'][it] = sim['inv_final'].mean()
        mc['inv_max'][it] = sim['inv_final'].max()
        mc['vp'][it] = sim['vp'].sum()
        mc['bl_max'][it] = sim['bl_final'].max()
        mc['bl_final'][it] = sim['bl_final'].iloc[-1]
        mc['fill_rate'][it] = sim['atendida'].sum() / sim['demanda'].sum() if sim['demanda'].sum() > 0 else 0
        mc['pedidos'][it] = sim['pedido'].sum()

    return mc


# =====================================================================
# GRÁFICOS
# =====================================================================

def grafico_pronostico(serie, resultado, nc):
    fechas = serie.index
    y = serie.values
    n_train = resultado['n_train']
    forecast = resultado['forecast']
    mape = resultado['mape']
    modelo = resultado['modelo']

    fechas_fut = pd.date_range(fechas[-1] + pd.DateOffset(months=1), periods=len(forecast), freq='MS')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fechas[:n_train], y=y[:n_train], mode='lines+markers',
                             name='Train', line=dict(color='#2196F3', width=2)))
    fig.add_trace(go.Scatter(x=fechas[n_train:], y=y[n_train:], mode='lines+markers',
                             name='Test', line=dict(color='#4CAF50', width=2)))
    fig.add_trace(go.Scatter(x=fechas_fut, y=forecast, mode='lines+markers',
                             name='Pronóstico', line=dict(color='#FF5722', width=2, dash='dash')))
    fig.update_layout(
        title=f'{nc} — {modelo} (MAPE={mape:.2f}%)',
        xaxis_title='Mes', yaxis_title='Demanda (und)',
        template='plotly_white', height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def grafico_diente_sierra(sim_df, nc, S_nivel, T_periodo, fecha_inicio):
    """Gráfico diente de sierra para política (T,S)."""
    fechas = pd.date_range(start=fecha_inicio, periods=len(sim_df), freq='D')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.08,
                        subplot_titles=[f'{nc} — Inventario diario (T,S) T={T_periodo}',
                                        'Backlog y ventas perdidas'])

    # Inventario
    fig.add_trace(go.Scatter(x=fechas, y=sim_df['inv_final'], mode='lines',
                             name='Inventario', line=dict(color='#2196F3', width=1.5),
                             fill='tozeroy', fillcolor='rgba(33,150,243,0.1)'), row=1, col=1)

    # Nivel S
    fig.add_hline(y=S_nivel, line_dash='dash', line_color='#4CAF50', line_width=1,
                  annotation_text=f'S = {S_nivel}', row=1, col=1)

    # Días de revisión (cada T días)
    for dia in range(0, len(sim_df), T_periodo):
        fig.add_vline(x=fechas[dia], line_dash='dot', line_color='rgba(128,128,128,0.3)',
                      line_width=0.5, row=1, col=1)

    # Pedidos emitidos
    pedidos_idx = sim_df[sim_df['pedido'] == 1].index
    if len(pedidos_idx) > 0:
        fig.add_trace(go.Scatter(
            x=fechas[pedidos_idx], y=sim_df.loc[pedidos_idx, 'inv_final'],
            mode='markers', name='Pedido emitido',
            marker=dict(symbol='triangle-up', size=8, color='#FF9800')), row=1, col=1)

    # Recepciones
    recep_idx = sim_df[sim_df['recibido'] > 0].index
    if len(recep_idx) > 0:
        fig.add_trace(go.Scatter(
            x=fechas[recep_idx], y=sim_df.loc[recep_idx, 'inv_final'],
            mode='markers', name='Recepción',
            marker=dict(symbol='diamond', size=7, color='#4CAF50')), row=1, col=1)

    # Backlog
    fig.add_trace(go.Scatter(x=fechas, y=sim_df['bl_final'], mode='lines',
                             name='Backlog', line=dict(color='#F44336', width=1.5),
                             fill='tozeroy', fillcolor='rgba(244,67,54,0.1)'), row=2, col=1)

    # VP acumulada
    fig.add_trace(go.Bar(x=fechas, y=sim_df['vp'], name='VP diaria',
                         marker_color='rgba(255,152,0,0.4)'), row=2, col=1)

    fig.update_layout(height=600, template='plotly_white',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02),
                      showlegend=True)
    fig.update_yaxes(title_text='Unidades', row=1, col=1)
    fig.update_yaxes(title_text='Unidades', row=2, col=1)
    return fig


def grafico_mc_histograma(mc_data, nc, indicador='costo_total', titulo='Costo total'):
    arr = mc_data[indicador]
    p5 = np.percentile(arr, 5); p95 = np.percentile(arr, 95)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=arr, nbinsx=80, name=titulo,
                               marker_color='rgba(33,150,243,0.6)',
                               marker_line_color='rgba(33,150,243,1)', marker_line_width=0.5))
    fig.add_vline(x=np.mean(arr), line_dash='solid', line_color='#F44336', line_width=2,
                  annotation_text=f'Media: S/{np.mean(arr):,.0f}')
    fig.add_vline(x=p5, line_dash='dash', line_color='#4CAF50',
                  annotation_text=f'P5: S/{p5:,.0f}')
    fig.add_vline(x=p95, line_dash='dash', line_color='#FF9800',
                  annotation_text=f'P95: S/{p95:,.0f}')
    fig.update_layout(title=f'{nc} — Distribución MC {titulo} (10,000 iter.)',
                      xaxis_title=titulo, yaxis_title='Frecuencia',
                      template='plotly_white', height=350, showlegend=False)
    return fig


def clasificar_estado_kanban(inv_final, backlog, vp, quiebre, SS, umbral_preventivo):
    """Clasifica el estado del semáforo Kanban para un día dado."""
    if inv_final <= SS or backlog > 0 or vp > 0 or quiebre == 1:
        estado = '🔴 Crítico'
        accion = 'Inventario crítico. Existe riesgo de quiebre antes de la próxima revisión.'
    elif inv_final <= umbral_preventivo:
        estado = '🟡 Alerta'
        accion = 'Inventario cercano al stock de seguridad. Preparar la próxima reposición.'
    else:
        estado = '🟢 Normal'
        accion = 'Inventario con cobertura suficiente.'
    return estado, accion


def proxima_revision_ts(dia_actual, T):
    """Calcula el próximo día de revisión en el ciclo T desde dia 0."""
    if dia_actual % T == 0:
        return dia_actual + T
    return dia_actual + (T - dia_actual % T)


def construir_historial_kanban(sim_df, nc, SS, S, T, margen_alerta, fecha_inicio):
    """Construye el historial completo de alertas para todos los días."""
    rows = []
    for idx, fila in sim_df.iterrows():
        dia = int(fila['dia'])
        fecha = fecha_inicio + pd.Timedelta(days=dia)
        umbral = SS * (1 + margen_alerta)
        estado, accion = clasificar_estado_kanban(
            fila['inv_final'], fila['bl_final'], fila['vp'],
            fila['quiebre'], SS, umbral)
        prox_rev_dia = proxima_revision_ts(dia, T)
        prox_rev_fecha = fecha_inicio + pd.Timedelta(days=prox_rev_dia)
        rows.append({
            'Día': dia + 1,
            'Fecha': fecha.strftime('%d-%b-%Y'),
            'Producto': nc,
            'Inv. inicial': fila['inv_inicio'],
            'Demanda': fila['demanda'],
            'Atendida': fila['atendida'],
            'Inv. final': fila['inv_final'],
            'SS': SS,
            'Umbral prev.': umbral,
            'S': S,
            'Backlog': fila['bl_final'],
            'VP': fila['vp'],
            'Pedido': int(fila['pedido']),
            'Cant. pedida': int(fila['q_pedido']),
            'Recibido': fila['recibido'],
            'Estado': estado,
            'Acción': accion,
            'Próx. revisión': prox_rev_fecha.strftime('%d-%b-%Y'),
        })
    return pd.DataFrame(rows)


def grafico_kanban(sim_df, nc, SS, S, umbral_prev, T, dia_sel, fecha_inicio):
    """Gráfico Plotly del Kanban con zonas de semáforo."""
    fechas = pd.date_range(start=fecha_inicio, periods=len(sim_df), freq='D')
    inv = sim_df['inv_final'].values
    y_max = max(S * 1.15, inv.max() * 1.1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.06,
                        subplot_titles=[f'{nc} — Kanban (T,S) T={T}',
                                        'Backlog y ventas perdidas'])

    # Zonas de color
    fig.add_hrect(y0=0, y1=SS, fillcolor='rgba(244,67,54,0.08)',
                  line_width=0, row=1, col=1)
    fig.add_hrect(y0=SS, y1=umbral_prev, fillcolor='rgba(255,193,7,0.08)',
                  line_width=0, row=1, col=1)
    fig.add_hrect(y0=umbral_prev, y1=y_max, fillcolor='rgba(76,175,80,0.06)',
                  line_width=0, row=1, col=1)

    # Inventario
    fig.add_trace(go.Scatter(x=fechas, y=inv, mode='lines', name='Inventario final',
                             line=dict(color='#1565C0', width=1.8)), row=1, col=1)

    # Líneas horizontales
    fig.add_hline(y=SS, line_dash='dash', line_color='#D32F2F', line_width=1.2,
                  annotation_text=f'SS = {SS}', annotation_position='top left', row=1, col=1)
    fig.add_hline(y=umbral_prev, line_dash='dot', line_color='#F9A825', line_width=1,
                  annotation_text=f'Umbral = {umbral_prev:.0f}', annotation_position='top left', row=1, col=1)
    fig.add_hline(y=S, line_dash='dash', line_color='#2E7D32', line_width=1.2,
                  annotation_text=f'S = {S}', annotation_position='top left', row=1, col=1)

    # Días de revisión
    for dia in range(0, len(sim_df), T):
        fig.add_vline(x=fechas[dia], line_dash='dot', line_color='rgba(128,128,128,0.25)',
                      line_width=0.5, row=1, col=1)

    # Pedidos emitidos
    ped_idx = sim_df[sim_df['pedido'] == 1].index
    if len(ped_idx) > 0:
        fig.add_trace(go.Scatter(
            x=fechas[ped_idx], y=sim_df.loc[ped_idx, 'inv_final'],
            mode='markers', name='Pedido emitido',
            marker=dict(symbol='triangle-up', size=9, color='#FF6F00')), row=1, col=1)

    # Recepciones
    rec_idx = sim_df[sim_df['recibido'] > 0].index
    if len(rec_idx) > 0:
        fig.add_trace(go.Scatter(
            x=fechas[rec_idx], y=sim_df.loc[rec_idx, 'inv_final'],
            mode='markers', name='Recepción',
            marker=dict(symbol='diamond', size=8, color='#2E7D32')), row=1, col=1)

    # Línea vertical del día seleccionado
    fecha_sel = fecha_inicio + pd.Timedelta(days=dia_sel)
    fig.add_vline(x=fecha_sel, line_dash='solid', line_color='#6A1B9A', line_width=2,
                  annotation_text=f'Día {dia_sel + 1}', row=1, col=1)

    # Backlog
    fig.add_trace(go.Scatter(x=fechas, y=sim_df['bl_final'], mode='lines',
                             name='Backlog', line=dict(color='#D32F2F', width=1.5),
                             fill='tozeroy', fillcolor='rgba(211,47,47,0.1)'), row=2, col=1)
    # VP
    fig.add_trace(go.Bar(x=fechas, y=sim_df['vp'], name='VP diaria',
                         marker_color='rgba(255,111,0,0.45)'), row=2, col=1)

    fig.update_layout(height=620, template='plotly_white',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02),
                      showlegend=True, margin=dict(t=60))
    fig.update_yaxes(title_text='Unidades', row=1, col=1, range=[0, y_max])
    fig.update_yaxes(title_text='Unidades', row=2, col=1)
    return fig


# =====================================================================
# APP PRINCIPAL
# =====================================================================

st.title("📦 Sistema de Pronósticos y Gestión de Inventarios")
st.caption("Proyecto de Tesis — UT&M | Melaminas Pelikano e Hispanos")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    archivo = st.file_uploader("Sube el Excel de demanda", type=["xlsx"],
                                help="Debe tener hoja DEMANDA con las columnas requeridas")

    with st.expander("📦 Parámetros de inventario"):
        nivel_servicio = st.slider("Nivel de servicio", 0.80, 0.99, 0.95, 0.01, format="%.0f%%")
        tasa_mant = st.slider("Tasa mantenimiento anual", 0.05, 0.40, 0.18, 0.01)
        horizonte_dias = st.slider("Horizonte simulación (días)", 90, 365, 180, 30)
        n_mc = st.select_slider("Iteraciones Monte Carlo", [1000, 5000, 10000], value=10000)
        horizonte_fc = st.slider("Horizonte pronóstico (meses)", 3, 12, 6)

    with st.expander("📋 Columnas del Excel"):
        st.info("Columnas esperadas: Fecha, Producto, Demanda_Ref, Stock_Inicial, "
                "LeadTime_Real_d, Precio_Unitario_S, Costo_Compra_Unitario_S, "
                "Costo_Pedido_S, etc.")

if not archivo:
    st.info("👆 Sube el archivo **DEMANDA_MELAMINA_COMPLETADA_ERP.xlsx** para comenzar.")
    st.markdown("""
    **Columnas requeridas en la hoja DEMANDA:**
    Fecha, Producto, Demanda_Ref, Stock_Inicial, Cantidad_Recibida, LeadTime_Real_d,
    Precio_Unitario_S, Costo_Falla_Abast_S, Costo_Compra_Unitario_S, Costo_Pedido_S,
    Tasa_Mantenimiento_Anual, Nivel_Servicio, Factor_Z, Pct_Ventas_Perdidas, Pct_Backlog,
    Demanda_Media_Diaria, Desv_Demanda_Diaria, LeadTime_Promedio_d, Desv_LeadTime_d,
    Precio_Venta_Promedio_S, LeadTime_Min_d, LeadTime_Max_d.
    """)
    st.stop()

# --- CARGA Y VALIDACIÓN ---
try:
    df = pd.read_excel(archivo, sheet_name='DEMANDA')
except Exception as e:
    st.error(f"Error leyendo la hoja DEMANDA: {e}")
    st.stop()

cols_requeridas = ['Fecha', 'Producto', 'Demanda_Ref', 'Stock_Inicial', 'LeadTime_Real_d',
    'Costo_Compra_Unitario_S', 'Costo_Pedido_S', 'Tasa_Mantenimiento_Anual',
    'Nivel_Servicio', 'Factor_Z', 'Pct_Ventas_Perdidas', 'Pct_Backlog',
    'Demanda_Media_Diaria', 'Desv_Demanda_Diaria', 'LeadTime_Promedio_d',
    'Desv_LeadTime_d', 'Precio_Venta_Promedio_S', 'LeadTime_Min_d', 'LeadTime_Max_d']

faltantes = [c for c in cols_requeridas if c not in df.columns]
if faltantes:
    st.error(f"Faltan columnas en el Excel: {', '.join(faltantes)}")
    st.stop()

df = df.dropna(subset=['Producto']).copy()
df['Fecha'] = pd.to_datetime(df['Fecha'])

st.success(f"✅ Archivo cargado: {len(df)} registros, {df['Producto'].nunique()} productos")

# --- PREPARAR PRODUCTOS ---
productos_list = df['Producto'].unique()
series = {}
productos = {}

for prod in productos_list:
    sub = df[df['Producto'] == prod].copy()
    nc = 'Pelikano' if 'PELIKANO' in prod.upper() else 'Hispanos' if 'HISPANOS' in prod.upper() else nombre_corto(prod)
    sub['Mes'] = sub['Fecha'].dt.to_period('M').dt.to_timestamp()
    serie = sub.groupby('Mes')['Demanda_Ref'].sum().sort_index()
    series[nc] = serie

    r = sub.iloc[0]
    d_mean = r['Demanda_Media_Diaria']; sigma_d = r['Desv_Demanda_Diaria']
    L_mean = r['LeadTime_Promedio_d']; sigma_L = r['Desv_LeadTime_d']
    C = r['Costo_Compra_Unitario_S']; K = r['Costo_Pedido_S']
    H = tasa_mant * C; z = r['Factor_Z']
    Pv = r['Precio_Venta_Promedio_S']
    pct_vp = r['Pct_Ventas_Perdidas']; pct_bl = r['Pct_Backlog']
    b_d = 0.50 * H / 365
    stock_ini = sub.sort_values('Fecha').iloc[-1]['Stock_Inicial']
    D_anual = d_mean * 365

    Q = int(np.ceil(np.sqrt(2 * D_anual * K / H)))
    SS = int(np.ceil(z * np.sqrt(L_mean * sigma_d ** 2 + d_mean ** 2 * sigma_L ** 2)))
    s_rop = int(np.ceil(d_mean * L_mean + SS))

    T_opciones = [7, 15, 30]
    TS = {}
    for T in T_opciones:
        S_ts = int(np.ceil(d_mean * (T + L_mean) + z * np.sqrt((T + L_mean) * sigma_d ** 2 + d_mean ** 2 * sigma_L ** 2)))
        SS_ts = int(np.ceil(z * np.sqrt((T + L_mean) * sigma_d ** 2 + d_mean ** 2 * sigma_L ** 2)))
        TS[T] = {'T': T, 'S': S_ts, 'SS': SS_ts}

    productos[nc] = {
        'd': d_mean, 'sigma_d': sigma_d, 'D_anual': D_anual,
        'L': L_mean, 'sigma_L': sigma_L,
        'L_min': r['LeadTime_Min_d'], 'L_max': r['LeadTime_Max_d'],
        'C': C, 'K': K, 'H': H, 'H_diario': H / 365, 'z': z, 'Pv': Pv,
        'pct_vp': pct_vp, 'pct_bl': pct_bl, 'b_d': b_d,
        'stock_ini': stock_ini,
        'dem_empirica': sub['Demanda_Ref'].values,
        'lt_empirica': sub['LeadTime_Real_d'].values,
        'QS': {'Q': Q, 's': s_rop, 'SS': SS},
        'TS': TS,
        'sS': {'s': s_rop, 'S': s_rop + Q, 'SS': SS},
        'sub_df': sub,
    }

# --- PESTAÑAS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Datos y pronóstico", "📦 Políticas de inventario",
    "💰 Evaluación económica", "🚦 Kanban", "📥 Descargar"])

# =====================================================================
# TAB 1: DATOS Y PRONÓSTICO
# =====================================================================
with tab1:
    st.header("Datos históricos y pronóstico de demanda")

    for nc, serie in series.items():
        with st.expander(f"📈 {nc}", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col2:
                st.markdown("**Serie mensual**")
                df_serie = pd.DataFrame({'Mes': [d.strftime('%b-%Y') for d in serie.index],
                                         'Demanda': serie.values})
                st.dataframe(df_serie, hide_index=True, use_container_width=True)

            with col1:
                if f'pron_{nc}' not in st.session_state:
                    with st.spinner(f'Evaluando modelos para {nc}...'):
                        np.random.seed(42)
                        st.session_state[f'pron_{nc}'] = ejecutar_pronostico(serie, horizonte_fc)

                res = st.session_state[f'pron_{nc}']
                fig = grafico_pronostico(serie, res, nc)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"**Modelo ganador:** {res['modelo']} — MAPE: {res['mape']:.2f}% | RMSE: {res['rmse']:.1f}")

            # Ranking top 10
            st.markdown("**Ranking de modelos (Top 10)**")
            ranking_df = pd.DataFrame([{
                '#': i + 1, 'Método': r['Método'], 'Familia': r['Familia'],
                'MAPE (%)': round(r['MAPE'], 2), 'RMSE': round(r['RMSE'], 1)
            } for i, r in enumerate(res['ranking'][:10])])
            st.dataframe(ranking_df, hide_index=True, use_container_width=True)

            # Pronóstico futuro
            fechas_fut = pd.date_range(serie.index[-1] + pd.DateOffset(months=1),
                                        periods=horizonte_fc, freq='MS')
            fc_df = pd.DataFrame({'Mes': [f.strftime('%b-%Y') for f in fechas_fut],
                                  'Pronóstico (und)': [f'{v:,.1f}' for v in res['forecast']]})
            st.markdown("**Pronóstico futuro**")
            st.dataframe(fc_df, hide_index=True, use_container_width=True)

            # --- Monte Carlo del pronóstico (N°1) ---
            mc_key = f'mc_pron_{nc}'
            if mc_key not in st.session_state:
                with st.spinner(f'Monte Carlo del pronóstico {nc}: {N_MC_PRONOSTICO:,} iteraciones por mes...'):
                    st.session_state[mc_key] = ejecutar_mc_pronostico(
                        res['forecast'], res['mape'], horizonte_fc, seed=42)

            mc_pron = st.session_state[mc_key]
            st.markdown(f"**Monte Carlo del pronóstico: {N_MC_PRONOSTICO:,} iteraciones por mes**")
            st.caption(f"σ_t = Forecast_t × MAPE ({res['mape']:.2f}%). "
                       f"D_t ~ N(Forecast_t, σ_t), truncada en 0. Semilla: 42.")

            # Tabla de indicadores por mes
            ind_rows = []
            for t, ind in enumerate(mc_pron['indicadores']):
                ind_rows.append({
                    'Mes': fechas_fut[t].strftime('%b-%Y'),
                    'Forecast': f"{ind['Forecast']:,.1f}",
                    'σ': f"{ind['Sigma']:,.1f}",
                    'Media MC': f"{ind['Media MC']:,.1f}",
                    'P2.5': f"{ind['P2.5']:,.1f}",
                    'P5': f"{ind['P5']:,.1f}",
                    'P50': f"{ind['P50']:,.1f}",
                    'P90': f"{ind['P90']:,.1f}",
                    'P95': f"{ind['P95']:,.1f}",
                    'P97.5': f"{ind['P97.5']:,.1f}",
                    'P99': f"{ind['P99']:,.1f}",
                    'VaR inf. 95%': f"{ind['VaR inf. 95%']:,.1f}",
                    'VaR sup. 95%': f"{ind['VaR sup. 95%']:,.1f}",
                    'CVaR inf. 95%': f"{ind['CVaR inf. 95%']:,.1f}",
                    'CVaR sup. 95%': f"{ind['CVaR sup. 95%']:,.1f}",
                    'Desv. Std.': f"{ind['Desv. Std.']:,.1f}",
                    'CV': f"{ind['CV']:.4f}",
                    'Dif. Media vs Fc': f"{ind['Dif. Media vs Forecast']:+,.1f}",
                    'Dif. P50 vs Fc': f"{ind['Dif. P50 vs Forecast']:+,.1f}",
                })
            st.dataframe(pd.DataFrame(ind_rows), hide_index=True, use_container_width=True)

            # Acumulado
            acs = mc_pron['acum_stats']
            st.markdown(f"**Acumulado {horizonte_fc} meses:** "
                        f"Forecast = {acs['Forecast acum.']:,.1f} und | "
                        f"Media MC = {acs['Media MC acum.']:,.1f} und | "
                        f"IC 95% = [{acs['IC 95% inf']:,.1f}, {acs['IC 95% sup']:,.1f}] | "
                        f"CV = {acs['CV acum.']:.4f}")

            # Gráfico de bandas
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(
                x=[f.strftime('%b-%Y') for f in fechas_fut],
                y=[ind['P97.5'] for ind in mc_pron['indicadores']],
                mode='lines', line=dict(width=0), showlegend=False))
            fig_mc.add_trace(go.Scatter(
                x=[f.strftime('%b-%Y') for f in fechas_fut],
                y=[ind['P2.5'] for ind in mc_pron['indicadores']],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(33,150,243,0.15)', name='IC 95%'))
            fig_mc.add_trace(go.Scatter(
                x=[f.strftime('%b-%Y') for f in fechas_fut],
                y=[ind['P95'] for ind in mc_pron['indicadores']],
                mode='lines', line=dict(width=0), showlegend=False))
            fig_mc.add_trace(go.Scatter(
                x=[f.strftime('%b-%Y') for f in fechas_fut],
                y=[ind['P5'] for ind in mc_pron['indicadores']],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(33,150,243,0.25)', name='IC 90%'))
            fig_mc.add_trace(go.Scatter(
                x=[f.strftime('%b-%Y') for f in fechas_fut],
                y=res['forecast'], mode='lines+markers',
                name='Forecast', line=dict(color='#F44336', width=2)))
            fig_mc.add_trace(go.Scatter(
                x=[f.strftime('%b-%Y') for f in fechas_fut],
                y=[ind['Media MC'] for ind in mc_pron['indicadores']],
                mode='lines+markers', name='Media MC',
                line=dict(color='#2196F3', width=2, dash='dash')))
            fig_mc.update_layout(
                title=f'{nc} — Monte Carlo del pronóstico ({N_MC_PRONOSTICO:,} iter.)',
                xaxis_title='Mes', yaxis_title='Demanda (und)',
                template='plotly_white', height=380,
                legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig_mc, use_container_width=True)


# =====================================================================
# TAB 2: POLÍTICAS DE INVENTARIO
# =====================================================================
with tab2:
    st.header("Políticas de inventario y simulación")

    if 'sim_done' not in st.session_state:
        if st.button("▶️ Ejecutar simulación y Monte Carlo", type="primary"):
            barra = st.progress(0, text="Iniciando simulación...")
            all_results = {}
            step = 0
            total_steps = len(productos) * 4  # det + 3 MC policies per product

            for nc, p in productos.items():
                # Determinística para 3 políticas
                barra.progress(step / total_steps, text=f"Simulación determinística {nc}...")
                det = {}
                fecha_inicio_sim = pd.Timestamp('2026-06-01')

                for pol_name, pol_type, pol_params in [
                    ('(Q,s)', 'QS', p['QS']),
                    ('(T,S) T=7', 'TS', p['TS'][7]),
                    ('(s,S)', 'sS', p['sS'])]:
                    sim = simular_politica(p, pol_type, pol_params, horizonte_dias)
                    mens = consolidar_mensual_calendario(sim, fecha_inicio_sim)
                    det[pol_name] = {'sim': sim, 'mensual': mens,
                                     'ct': sim['costo_total'].sum(),
                                     'fr': sim['atendida'].sum() / sim['demanda'].sum()}
                step += 1

                # MC para 3 políticas
                mc_results = {}
                for pol_name, pol_type, pol_params in [
                    ('(Q,s)', 'QS', p['QS']),
                    ('(T,S) T=7', 'TS', p['TS'][7]),
                    ('(s,S)', 'sS', p['sS'])]:
                    barra.progress(step / total_steps, text=f"Monte Carlo {nc} {pol_name} ({n_mc} iter.)...")
                    mc = ejecutar_mc(p, pol_type, pol_params, horizonte_dias, n_mc)
                    mc_results[pol_name] = mc
                    step += 1

                all_results[nc] = {'det': det, 'mc': mc_results}

            barra.progress(1.0, text="✅ Simulación completada")
            st.session_state['sim_done'] = True
            st.session_state['all_results'] = all_results
            st.rerun()
        else:
            st.info("Haz clic en el botón para ejecutar la simulación.")
            st.stop()

    all_results = st.session_state['all_results']

    for nc, p in productos.items():
        st.subheader(f"📦 {nc}")

        # Parámetros
        with st.expander("Parámetros calculados"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Q* (EOQ)", p['QS']['Q'])
                st.metric("SS", p['QS']['SS'])
                st.metric("s (punto reorden)", p['QS']['s'])
            with c2:
                st.metric("S (T,S T=7)", p['TS'][7]['S'])
                st.metric("S (s,S)", p['sS']['S'])
            with c3:
                st.metric("d̄ diaria", f"{p['d']:.2f}")
                st.metric("σ_d", f"{p['sigma_d']:.2f}")
                st.metric("L̄", f"{p['L']:.2f} días")

        # Tabla comparativa determinística
        st.markdown("**Simulación determinística (180 días)**")
        det = all_results[nc]['det']
        det_rows = []
        for pol_name in ['(Q,s)', '(T,S) T=7', '(s,S)']:
            d = det[pol_name]
            s = d['sim']
            det_rows.append({
                'Política': pol_name,
                'CT (S/)': f"{d['ct']:,.2f}",
                'Fill rate': f"{d['fr']:.2%}",
                'VP (und)': f"{s['vp'].sum():,.1f}",
                'Inv. prom.': f"{s['inv_final'].mean():,.1f}",
                'Pedidos': int(s['pedido'].sum()),
                'Días quiebre': int(s['quiebre'].sum()),
            })
        st.dataframe(pd.DataFrame(det_rows), hide_index=True, use_container_width=True)

        # Tabla MC
        st.markdown(f"**Monte Carlo ({n_mc:,} iteraciones)**")
        mc = all_results[nc]['mc']
        mc_rows = []
        for pol_name in ['(Q,s)', '(T,S) T=7', '(s,S)']:
            m = mc[pol_name]
            ct = m['costo_total']
            fr = m['fill_rate']
            v95 = np.percentile(ct, 95)
            mc_rows.append({
                'Política': pol_name,
                'CT medio': f"S/ {np.mean(ct):,.0f}",
                'CT σ': f"S/ {np.std(ct):,.0f}",
                'FR medio': f"{np.mean(fr):.2%}",
                'VP media': f"{np.mean(m['vp']):,.0f}",
                'VaR 95%': f"S/ {v95:,.0f}",
                'CVaR 95%': f"S/ {np.mean(ct[ct >= v95]):,.0f}",
            })
        mc_df = pd.DataFrame(mc_rows)
        st.dataframe(mc_df, hide_index=True, use_container_width=True)

        # Ganadora
        ct_medios = {pn: np.mean(mc[pn]['costo_total']) for pn in mc}
        ganadora = min(ct_medios, key=ct_medios.get)
        st.success(f"🏆 Política ganadora MC para {nc}: **{ganadora}** — "
                   f"CT medio: S/ {ct_medios[ganadora]:,.0f} | "
                   f"FR medio: {np.mean(mc[ganadora]['fill_rate']):.2%}")

        # Consolidación mensual de la ganadora
        st.markdown(f"**Consolidación mensual — {ganadora} (determinística)**")
        mens = det[ganadora]['mensual']
        cols_show = ['mes_cal', 'dias', 'demanda', 'atendida', 'bl_atendido', 'vp',
                     'bl_nuevo', 'bl_final', 'inv_promedio', 'inv_final', 'pedidos',
                     'und_pedidas', 'und_recibidas', 'costo_ordenar', 'costo_mant',
                     'costo_vp', 'costo_bl', 'costo_total', 'fill_rate', 'dias_quiebre']
        mens_show = mens[[c for c in cols_show if c in mens.columns]].copy()
        for c in ['demanda', 'atendida', 'vp', 'bl_nuevo', 'inv_promedio', 'inv_final',
                  'und_pedidas', 'und_recibidas', 'costo_ordenar', 'costo_mant',
                  'costo_vp', 'costo_bl', 'costo_total']:
            if c in mens_show.columns:
                mens_show[c] = mens_show[c].apply(lambda x: f'{x:,.1f}')
        if 'fill_rate' in mens_show.columns:
            mens_show['fill_rate'] = mens['fill_rate'].apply(lambda x: f'{x:.2%}')
        st.dataframe(mens_show, hide_index=True, use_container_width=True)

        # Gráfico diente de sierra
        st.markdown(f"**Gráfico diente de sierra — {ganadora}**")
        sim_ganadora = det[ganadora]['sim']
        fecha_inicio_sim = pd.Timestamp('2026-06-01')
        fig_ds = grafico_diente_sierra(sim_ganadora, nc, p['TS'][7]['S'], 7, fecha_inicio_sim)
        st.plotly_chart(fig_ds, use_container_width=True)

        # Histograma MC
        st.markdown(f"**Distribución Monte Carlo — {ganadora}**")
        c1, c2 = st.columns(2)
        with c1:
            fig_h1 = grafico_mc_histograma(mc[ganadora], nc, 'costo_total', 'Costo total (S/)')
            st.plotly_chart(fig_h1, use_container_width=True)
        with c2:
            fig_h2 = grafico_mc_histograma(mc[ganadora], nc, 'fill_rate', 'Fill rate')
            st.plotly_chart(fig_h2, use_container_width=True)


# =====================================================================
# TAB 3: EVALUACIÓN ECONÓMICA
# =====================================================================
with tab3:
    st.header("Evaluación económica AS-IS vs TO-BE")

    if 'sim_done' not in st.session_state:
        st.warning("Ejecuta primero la simulación en la pestaña de Políticas.")
        st.stop()

    all_results = st.session_state['all_results']

    for nc, p in productos.items():
        st.subheader(f"💰 {nc}")
        sub = p['sub_df']

        # AS-IS (datos históricos)
        total_reqs = len(sub)
        quiebres_hist = sub['Quiebre_Stock'].sum()
        pct_quiebre_hist = quiebres_hist / total_reqs * 100
        costo_falla_hist = sub['Costo_Falla_Abast_S'].sum()
        dem_total_hist = sub['Demanda_Ref'].sum()
        cant_recibida = sub['Cantidad_Recibida'].sum()
        # Fill rate histórico (proxy): cantidad atendida / demanda
        fr_hist = cant_recibida / dem_total_hist if dem_total_hist > 0 else 0
        meses_hist = sub['Fecha'].dt.to_period('M').nunique()

        # TO-BE (política ganadora MC)
        mc_ganadora = all_results[nc]['mc']['(T,S) T=7']
        fr_tobe = np.mean(mc_ganadora['fill_rate'])
        ct_tobe_6m = np.mean(mc_ganadora['costo_total'])
        vp_tobe = np.mean(mc_ganadora['vp'])
        vp_tobe_soles = vp_tobe * p['Pv']

        # Escalar AS-IS a 6 meses para comparación
        factor_escala = 6 / meses_hist if meses_hist > 0 else 1
        costo_falla_6m = costo_falla_hist * factor_escala
        quiebres_6m = quiebres_hist * factor_escala

        # Cálculos
        reduccion_quiebre = pct_quiebre_hist - (1 - fr_tobe) * 100
        ahorro_vp = costo_falla_6m - vp_tobe_soles
        ahorro_pct = ahorro_vp / costo_falla_6m * 100 if costo_falla_6m > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### AS-IS (situación actual)")
            st.metric("Quiebre de stock", f"{pct_quiebre_hist:.1f}%")
            st.metric("Fill rate (proxy)", f"{fr_hist:.2%}")
            st.metric(f"Costo falla ({meses_hist}m)", f"S/ {costo_falla_hist:,.0f}")
            st.metric("Costo falla (6m escalado)", f"S/ {costo_falla_6m:,.0f}")

        with col2:
            st.markdown("### TO-BE (política ganadora)")
            st.metric("Quiebre esperado", f"{(1 - fr_tobe) * 100:.1f}%")
            st.metric("Fill rate MC", f"{fr_tobe:.2%}")
            st.metric("CT esperado (6m)", f"S/ {ct_tobe_6m:,.0f}")
            st.metric("VP esperada (und)", f"{vp_tobe:,.0f}")

        with col3:
            st.markdown("### Impacto")
            st.metric("Reducción quiebre", f"{reduccion_quiebre:.1f} pp",
                      delta=f"-{reduccion_quiebre:.1f} pp")
            st.metric("Mejora fill rate", f"+{(fr_tobe - fr_hist) * 100:.1f} pp",
                      delta=f"+{(fr_tobe - fr_hist) * 100:.1f} pp")
            st.metric("Ahorro VP (6m)", f"S/ {ahorro_vp:,.0f}",
                      delta=f"{ahorro_pct:.0f}% reducción" if ahorro_vp > 0 else "N/A")

        # Tabla resumen
        st.markdown("**Tabla comparativa detallada**")
        comp = pd.DataFrame([
            {'Indicador': 'Quiebre de stock (%)', 'AS-IS': f'{pct_quiebre_hist:.1f}%',
             'TO-BE': f'{(1 - fr_tobe) * 100:.1f}%', 'Cambio': f'-{reduccion_quiebre:.1f} pp'},
            {'Indicador': 'Fill rate', 'AS-IS': f'{fr_hist:.2%}',
             'TO-BE': f'{fr_tobe:.2%}', 'Cambio': f'+{(fr_tobe - fr_hist) * 100:.1f} pp'},
            {'Indicador': f'Costo por falla/VP (6m)', 'AS-IS': f'S/ {costo_falla_6m:,.0f}',
             'TO-BE': f'S/ {vp_tobe_soles:,.0f}', 'Cambio': f'S/ {ahorro_vp:,.0f} ahorro'},
            {'Indicador': 'VaR 95% costo total', 'AS-IS': 'No disponible',
             'TO-BE': f'S/ {np.percentile(mc_ganadora["costo_total"], 95):,.0f}', 'Cambio': '—'},
            {'Indicador': 'CVaR 95% costo total', 'AS-IS': 'No disponible',
             'TO-BE': f'S/ {np.mean(mc_ganadora["costo_total"][mc_ganadora["costo_total"] >= np.percentile(mc_ganadora["costo_total"], 95)]):,.0f}',
             'Cambio': '—'},
        ])
        st.dataframe(comp, hide_index=True, use_container_width=True)

        # Beneficio anualizado
        ahorro_anual = ahorro_vp * 2  # extrapolación a 12 meses
        st.markdown(f"**Beneficio anualizado estimado:** S/ {ahorro_anual:,.0f}")
        st.caption("Nota: El costo de implementación (software, capacitación) no se incluye "
                   "porque no se dispone de esa información. Si se proporciona, se puede calcular "
                   "ROI y periodo de recuperación.")


# =====================================================================
# TAB 4: KANBAN
# =====================================================================
with tab4:
    st.header("🚦 Kanban — Estado de inventario")
    st.caption("Política (T,S) con T=7 días. Alerta visual y preventiva basada en el inventario "
               "diario de la simulación determinística. No modifica la lógica de reposición.")

    if 'sim_done' not in st.session_state:
        st.warning("Ejecuta primero la simulación en la pestaña de Políticas.")
        st.stop()

    all_results = st.session_state['all_results']
    fecha_inicio_sim = pd.Timestamp('2026-06-01')

    # --- Controles ---
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        dia_seleccionado = st.slider(
            "Día de simulación", min_value=1, max_value=horizonte_dias,
            value=1, step=1, help="Selecciona un día para ver el estado del inventario")
    with col_ctrl2:
        margen_alerta = st.slider(
            "Margen preventivo sobre el stock de seguridad",
            min_value=0.10, max_value=0.50, value=0.25, step=0.05,
            format="%.0f%%",
            help="Parámetro visual del Kanban. No afecta la lógica de reposición de (T,S).")

    fecha_simulada = fecha_inicio_sim + pd.Timedelta(days=dia_seleccionado - 1)
    st.markdown(f"**Fecha simulada:** {fecha_simulada.strftime('%d de %B de %Y')}  ·  "
                f"**Día:** {dia_seleccionado} / {horizonte_dias}")

    # --- Tarjetas Kanban por producto ---
    for nc, p in productos.items():
        sim = all_results[nc]['det']['(T,S) T=7']['sim']
        SS = p['TS'][7]['SS']
        S = p['TS'][7]['S']
        T = 7
        umbral = SS * (1 + margen_alerta)

        fila = sim.iloc[dia_seleccionado - 1]
        inv_final = fila['inv_final']
        inv_inicio = fila['inv_inicio']
        demanda = fila['demanda']
        atendida = fila['atendida']
        backlog = fila['bl_final']
        vp_dia = fila['vp']
        pedido_em = int(fila['pedido'])
        cant_pedida = int(fila['q_pedido'])
        recibido = fila['recibido']
        quiebre = int(fila['quiebre'])

        estado, accion = clasificar_estado_kanban(inv_final, backlog, vp_dia, quiebre, SS, umbral)

        dia_idx = dia_seleccionado - 1
        prox_rev_dia = proxima_revision_ts(dia_idx, T)
        prox_rev_fecha = fecha_inicio_sim + pd.Timedelta(days=prox_rev_dia)

        if '🟢' in estado:
            borde = '#4CAF50'; fondo = 'rgba(76,175,80,0.07)'
        elif '🟡' in estado:
            borde = '#FF9800'; fondo = 'rgba(255,152,0,0.07)'
        else:
            borde = '#F44336'; fondo = 'rgba(244,67,54,0.07)'

        st.markdown(f"""
        <div style="background:{fondo}; padding:18px; border-radius:10px; margin-bottom:14px;
                    border-left: 5px solid {borde};">
            <h3 style="margin:0 0 8px 0;">{estado} — {nc}</h3>
            <table style="width:100%; border-collapse:collapse; font-size:0.92em;">
                <tr>
                    <td><b>Día simulado:</b> {dia_seleccionado}</td>
                    <td><b>Fecha:</b> {fecha_simulada.strftime('%d-%b-%Y')}</td>
                    <td><b>Próxima revisión:</b> Día {prox_rev_dia + 1} ({prox_rev_fecha.strftime('%d-%b-%Y')})</td>
                </tr>
                <tr>
                    <td><b>Inv. inicial:</b> {inv_inicio:,.1f} und</td>
                    <td><b>Demanda:</b> {demanda:,.1f} und</td>
                    <td><b>Atendida:</b> {atendida:,.1f} und</td>
                </tr>
                <tr>
                    <td style="font-size:1.05em;"><b>Inv. final:</b> <span style="color:{borde}; font-weight:bold;">{inv_final:,.1f} und</span></td>
                    <td><b>SS:</b> {SS} und</td>
                    <td><b>Umbral preventivo:</b> {umbral:,.0f} und</td>
                </tr>
                <tr>
                    <td><b>Nivel S:</b> {S} und</td>
                    <td><b>Backlog:</b> {backlog:,.1f} und</td>
                    <td><b>Ventas perdidas:</b> {vp_dia:,.1f} und</td>
                </tr>
                <tr>
                    <td><b>Pedido emitido:</b> {'Sí' if pedido_em else 'No'}</td>
                    <td><b>Cantidad pedida:</b> {cant_pedida} und</td>
                    <td><b>Cantidad recibida:</b> {recibido:,.1f} und</td>
                </tr>
            </table>
            <p style="margin:10px 0 0 0;"><b>Acción:</b> {accion}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Gráfico Kanban por producto ---
    st.markdown("---")
    st.subheader("📊 Gráfico de semáforo Kanban")

    for nc, p in productos.items():
        sim = all_results[nc]['det']['(T,S) T=7']['sim']
        SS = p['TS'][7]['SS']
        S = p['TS'][7]['S']
        umbral = SS * (1 + margen_alerta)
        fig_k = grafico_kanban(sim, nc, SS, S, umbral, 7, dia_seleccionado - 1, fecha_inicio_sim)
        st.plotly_chart(fig_k, use_container_width=True)

    # --- Resumen de estados ---
    st.markdown("---")
    st.subheader("📋 Resumen de estados (180 días)")

    for nc, p in productos.items():
        sim = all_results[nc]['det']['(T,S) T=7']['sim']
        SS = p['TS'][7]['SS']
        S = p['TS'][7]['S']
        umbral = SS * (1 + margen_alerta)

        hist = construir_historial_kanban(sim, nc, SS, S, 7, margen_alerta, fecha_inicio_sim)

        n_verde = (hist['Estado'].str.contains('🟢')).sum()
        n_amarillo = (hist['Estado'].str.contains('🟡')).sum()
        n_rojo = (hist['Estado'].str.contains('🔴')).sum()
        n_quiebre = int(sim['quiebre'].sum())
        n_pedidos = int(sim['pedido'].sum())
        und_pedidas = int(sim['q_pedido'].sum())
        und_recibidas = sim['recibido'].sum()
        vp_total = sim['vp'].sum()
        bl_max = sim['bl_final'].max()

        st.markdown(f"**{nc}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🟢 Días verdes", n_verde)
        c2.metric("🟡 Días amarillos", n_amarillo)
        c3.metric("🔴 Días rojos", n_rojo)
        c4.metric("Días con quiebre", n_quiebre)
        c5.metric("Pedidos emitidos", n_pedidos)

        c6, c7, c8, c9 = st.columns(4)
        c6.metric("Und. pedidas", f"{und_pedidas:,}")
        c7.metric("Und. recibidas", f"{und_recibidas:,.0f}")
        c8.metric("VP acumulada", f"{vp_total:,.1f}")
        c9.metric("Backlog máximo", f"{bl_max:,.1f}")

    # --- Historial de alertas ---
    st.markdown("---")
    st.subheader("📜 Historial de alertas diario")

    historiales = []
    for nc, p in productos.items():
        sim = all_results[nc]['det']['(T,S) T=7']['sim']
        SS = p['TS'][7]['SS']
        S = p['TS'][7]['S']
        hist = construir_historial_kanban(sim, nc, SS, S, 7, margen_alerta, fecha_inicio_sim)
        historiales.append(hist)

    historial_completo = pd.concat(historiales, ignore_index=True)
    st.session_state['historial_kanban'] = historial_completo

    for nc in productos:
        with st.expander(f"📋 {nc} — Historial completo", expanded=False):
            hist_nc = historial_completo[historial_completo['Producto'] == nc]
            st.dataframe(hist_nc, hide_index=True, use_container_width=True, height=400)

    st.markdown("**Nota técnica:** El semáforo es un indicador visual y preventivo. "
                "La política (T,S) emite pedidos exclusivamente cada T=7 días, "
                "ordenando hasta el nivel S. No utiliza punto de reorden (s). "
                "El margen preventivo es un parámetro configurable del panel Kanban, "
                "no una regla matemática de la política.")


# =====================================================================
# TAB 5: DESCARGAR
# =====================================================================
with tab5:
    st.header("📥 Descargar resultados")

    if 'sim_done' not in st.session_state:
        st.warning("Ejecuta primero la simulación.")
        st.stop()

    all_results = st.session_state['all_results']
    fecha_inicio_sim = pd.Timestamp('2026-06-01')

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Parámetros
        params_rows = []
        for nc, p in productos.items():
            params_rows.append({
                'Producto': nc, 'd̄': p['d'], 'σ_d': p['sigma_d'],
                'L̄': p['L'], 'σ_L': p['sigma_L'],
                'C': p['C'], 'K': p['K'], 'H': p['H'],
                'Q*': p['QS']['Q'], 'SS': p['QS']['SS'], 's': p['QS']['s'],
                'S_TS7': p['TS'][7]['S'], 'SS_TS7': p['TS'][7]['SS'],
                'S_sS': p['sS']['S'],
            })
        pd.DataFrame(params_rows).to_excel(writer, sheet_name='Parametros', index=False)

        # Consolidación mensual por producto
        for nc in productos:
            det = all_results[nc]['det']
            ganadora = '(T,S) T=7'
            mens = det[ganadora]['mensual']
            mens.to_excel(writer, sheet_name=f'Mensual_{nc[:10]}', index=False)

        # MC resumen
        mc_rows = []
        for nc in productos:
            mc = all_results[nc]['mc']
            for pol_name in mc:
                ct = mc[pol_name]['costo_total']
                fr = mc[pol_name]['fill_rate']
                v95 = np.percentile(ct, 95)
                mc_rows.append({
                    'Producto': nc, 'Política': pol_name,
                    'CT_medio': np.mean(ct), 'CT_std': np.std(ct),
                    'FR_medio': np.mean(fr), 'VP_media': np.mean(mc[pol_name]['vp']),
                    'VaR95': v95,
                    'CVaR95': np.mean(ct[ct >= v95]),
                })
        pd.DataFrame(mc_rows).to_excel(writer, sheet_name='MC_Resumen', index=False)

        # MC Pronóstico
        mc_pron_rows = []
        for nc in productos:
            mc_key = f'mc_pron_{nc}'
            if mc_key in st.session_state:
                mc_pron = st.session_state[mc_key]
                pron_key = f'pron_{nc}'
                serie = series[nc]
                fechas_fut = pd.date_range(serie.index[-1] + pd.DateOffset(months=1),
                                            periods=horizonte_fc, freq='MS')
                for t, ind in enumerate(mc_pron['indicadores']):
                    row = {'Producto': nc, 'Mes': fechas_fut[t].strftime('%b-%Y')}
                    row.update(ind)
                    mc_pron_rows.append(row)
        if mc_pron_rows:
            pd.DataFrame(mc_pron_rows).to_excel(writer, sheet_name='MC_Pronostico', index=False)

        # Historial Kanban
        if 'historial_kanban' in st.session_state:
            st.session_state['historial_kanban'].to_excel(
                writer, sheet_name='Historial_Kanban', index=False)
        else:
            # Generar historial si no se visitó la pestaña Kanban aún
            historiales = []
            margen_default = 0.25
            for nc, p in productos.items():
                sim = all_results[nc]['det']['(T,S) T=7']['sim']
                SS = p['TS'][7]['SS']
                S = p['TS'][7]['S']
                hist = construir_historial_kanban(
                    sim, nc, SS, S, 7, margen_default, fecha_inicio_sim)
                historiales.append(hist)
            pd.concat(historiales, ignore_index=True).to_excel(
                writer, sheet_name='Historial_Kanban', index=False)

    st.download_button(
        label="📥 Descargar resultados completos (Excel)",
        data=buffer.getvalue(),
        file_name="RESULTADOS_UTM_TESIS.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.caption("El archivo contiene: parámetros, consolidación mensual, resumen Monte Carlo de políticas, "
               "Monte Carlo del pronóstico (10,000 iter.), e historial diario de alertas Kanban.")
