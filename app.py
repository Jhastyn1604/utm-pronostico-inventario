# =====================================================================
# APP STREAMLIT — PROYECTO TESIS UT&M  v3.0 (definitiva)
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools, warnings, io, time, hashlib

warnings.filterwarnings('ignore')
st.set_page_config(page_title="UT&M — Pronósticos e Inventarios", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")

N_MC_PRONOSTICO = 10000

# =====================================================================
# HELPERS
# =====================================================================
def redondear_05(x):
    return np.ceil(x * 2) / 2

def nombre_corto(producto, max_len=40):
    t = str(producto).strip()
    if len(t) <= max_len: return t
    c = t[:max_len]; u = c.rfind(" ")
    return (c[:u]+"…") if u > max_len//2 else (c+"…")

def calc_mape(r, p):
    r, p = np.array(r, float), np.array(p, float); m = r != 0
    return np.mean(np.abs((r[m]-p[m])/r[m]))*100 if m.sum()>0 else 999

def calc_rmse(r, p):
    return np.sqrt(np.mean((np.array(r,float)-np.array(p,float))**2))

def fmt_s(v): return f"S/ {v:,.2f}"
def fmt_pct(v): return f"{v:.2f}%"

def pol_key(pol_name):
    return str(pol_name).replace('(','').replace(')','').replace(',','_').replace(' ','_')

def invalidar_resultados():
    for k in list(st.session_state.keys()):
        if k.startswith(('pron_','mc_pron_','sim_done','all_results','historial_kanban','ganadoras')):
            del st.session_state[k]

def obtener_config_politica(p, nombre):
    """Mapea nombre de política a tipo, parámetros y niveles."""
    if nombre == '(Q,s)':
        return {'tipo':'QS','params':p['QS'],'T':None,'S':None,'SS':p['QS']['SS'],
                's':p['QS']['s'],'Q':p['QS']['Q'],'titulo':'(Q,s) Revisión continua'}
    elif nombre == '(T,S) T=7':
        return {'tipo':'TS','params':p['TS'][7],'T':7,'S':p['TS'][7]['S'],'SS':p['TS'][7]['SS'],
                's':None,'Q':None,'titulo':'(T,S) T=7 Revisión periódica'}
    elif nombre == '(s,S)':
        return {'tipo':'sS','params':p['sS'],'T':None,'S':p['sS']['S'],'SS':p['sS']['SS'],
                's':p['sS']['s'],'Q':None,'titulo':'(s,S) Mín-Máx'}
    return None


# =====================================================================
# MODELOS DE PRONÓSTICO (validados — no modificar lógica)
# =====================================================================
def fit_ses(tr, h):
    n = len(tr)
    def obj(pa):
        a=pa[0];L=tr[0];s=0
        for t in range(1,n): s+=(tr[t]-L)**2; L=a*tr[t]+(1-a)*L
        return s
    r=minimize(obj,[0.3],bounds=[(0.01,0.99)],method='L-BFGS-B')
    a=r.x[0];L=tr[0]
    for t in range(1,n): L=a*tr[t]+(1-a)*L
    return np.full(h,L), f'SES (α={a:.3f})'

def fit_holt(tr, h, damp=False):
    n=len(tr)
    def obj(pa):
        a,b=pa[0],pa[1];phi=pa[2] if damp else 1.0
        L=tr[0];T=tr[min(1,n-1)]-tr[0];s=0
        for t in range(1,n):
            s+=(tr[t]-(L+phi*T))**2; nL=a*tr[t]+(1-a)*(L+phi*T); T=b*(nL-L)+(1-b)*phi*T; L=nL
        return s
    if damp: r=minimize(obj,[0.3,0.1,0.9],bounds=[(0.01,0.99)]*3,method='L-BFGS-B'); a,b,phi=r.x
    else: r=minimize(obj,[0.3,0.1],bounds=[(0.01,0.99)]*2,method='L-BFGS-B'); a,b=r.x; phi=1.0
    L=tr[0];T=tr[min(1,n-1)]-tr[0]
    for t in range(1,n): nL=a*tr[t]+(1-a)*(L+phi*T);T=b*(nL-L)+(1-b)*phi*T;L=nL
    fc=[L+sum(phi**j for j in range(1,i+1))*T if damp else L+i*T for i in range(1,h+1)]
    return np.maximum(np.array(fc),0), 'Holt amortiguado' if damp else 'Holt'

def fit_hw(tr, h, seas, trend, s):
    n=len(tr)
    if n<2*s: return None,''
    try:
        def obj(pa):
            a,b_p,g=pa;L=np.mean(tr[:s]);T=(np.mean(tr[s:2*s])-np.mean(tr[:s]))/s if trend else 0
            S=np.zeros(n+s)
            for i in range(s): S[i]=(tr[i]-L) if seas=='add' else (tr[i]/max(L,1))
            ss=0
            for t in range(s,n):
                if seas=='add': fc=(L+T+S[t-s]) if trend else (L+S[t-s])
                else: fc=((L+T)*S[t-s]) if trend else (L*S[t-s])
                ss+=(tr[t]-fc)**2
                if seas=='add': nL=a*(tr[t]-S[t-s])+(1-a)*(L+T);nS=g*(tr[t]-L-T)+(1-g)*S[t-s]
                else: den=max(L+T if trend else L,1);nL=a*(tr[t]/max(S[t-s],0.01))+(1-a)*(L+T);nS=g*(tr[t]/den)+(1-g)*S[t-s]
                nT=(b_p*(nL-L)+(1-b_p)*T) if trend else 0;S[t]=nS;L=nL;T=nT
            return ss
        if trend: r=minimize(obj,[0.3,0.1,0.1],bounds=[(0.01,0.99)]*3,method='L-BFGS-B')
        else: r=minimize(obj,[0.3,0.01,0.1],bounds=[(0.01,0.99),(0.001,0.01),(0.01,0.99)],method='L-BFGS-B')
        a,b_p,g=r.x;L=np.mean(tr[:s]);T=(np.mean(tr[s:2*s])-np.mean(tr[:s]))/s if trend else 0
        S=np.zeros(n+s+h)
        for i in range(s): S[i]=(tr[i]-L) if seas=='add' else (tr[i]/max(L,1))
        for t in range(s,n):
            if seas=='add': nL=a*(tr[t]-S[t-s])+(1-a)*(L+T);S[t]=g*(tr[t]-L-T)+(1-g)*S[t-s]
            else: nL=a*(tr[t]/max(S[t-s],0.01))+(1-a)*(L+T);S[t]=g*(tr[t]/max(L+T if trend else L,1))+(1-g)*S[t-s]
            nT=(b_p*(nL-L)+(1-b_p)*T) if trend else 0;L=nL;T=nT
        fc=[]
        for i in range(h):
            idx=n-s+(i+1)%s
            if seas=='add': v=(L+(i+1)*T+S[idx]) if trend else (L+S[idx])
            else: v=((L+(i+1)*T)*S[idx]) if trend else (L*S[idx])
            fc.append(v)
        ts='Con tend.' if trend else 'Sin tend.';ss_s='aditivo' if seas=='add' else 'mult.'
        return np.maximum(np.array(fc),0), f'HW {ts} {ss_s} (s={s})'
    except: return None,''

def fit_arima(tr, h, order):
    p,d,q=order;y=tr.copy();dv=[]
    for _ in range(d): dv.append(y.copy());y=np.diff(y)
    n=len(y)
    if n<=max(p,q)+2: return None
    try:
        def obj(pa):
            c=pa[0];phi=pa[1:1+p];theta=pa[1+p:1+p+q];res=np.zeros(n)
            for t in range(max(p,q,1),n):
                pred=c
                for i in range(p): pred+=phi[i]*y[t-1-i]
                for j in range(q): pred+=theta[j]*res[t-1-j]
                res[t]=y[t]-pred
            return np.sum(res[max(p,q,1):]**2)
        x0=np.zeros(1+p+q);x0[0]=np.mean(y)
        result=minimize(obj,x0,bounds=[(-1e6,1e6)]+[(-0.99,0.99)]*(p+q),method='L-BFGS-B',options={'maxiter':2000})
        c=result.x[0];phi=result.x[1:1+p];theta=result.x[1+p:1+p+q]
        res=np.zeros(n)
        for t in range(max(p,q,1),n):
            pred=c
            for i in range(p): pred+=phi[i]*y[t-1-i]
            for j in range(q): pred+=theta[j]*res[t-1-j]
            res[t]=y[t]-pred
        ye=np.concatenate([y,np.zeros(h)]);re=np.concatenate([res,np.zeros(h)])
        for t in range(n,n+h):
            pred=c
            for i in range(p): pred+=phi[i]*ye[t-1-i]
            for j in range(q):
                if t-1-j<n: pred+=theta[j]*re[t-1-j]
            ye[t]=pred
        fc=ye[n:]
        for dval in reversed(dv):
            fc2=np.zeros(len(fc));lv=dval[-1]
            for i in range(len(fc)): lv+=fc[i];fc2[i]=lv
            fc=fc2
        return np.maximum(fc,0)
    except: return None

def fit_sarima(tr, h, order, sorder):
    p,d,q=order;P,D,Q,s=sorder;y=tr.copy()
    for _ in range(D):
        if len(y)>s: y=y[s:]-y[:-s]
        else: return None
    dr=[]
    for _ in range(d): dr.append(y.copy());y=np.diff(y)
    n=len(y)
    al=sorted(set(list(range(1,p+1))+[pp*s for pp in range(1,P+1)]+[pp*s+i for pp in range(1,P+1) for i in range(1,p+1)]))
    ml=sorted(set(list(range(1,q+1))+[qq*s for qq in range(1,Q+1)]+[qq*s+j for qq in range(1,Q+1) for j in range(1,q+1)]))
    mx=max(al+ml+[1]);na=len(al);nm=len(ml)
    if n<=mx+2: return None
    try:
        def obj(pa):
            c=pa[0];phi=pa[1:1+na];theta=pa[1+na:];res=np.zeros(n)
            for t in range(mx,n):
                pred=c
                for i,lag in enumerate(al):
                    if t-lag>=0: pred+=phi[i]*y[t-lag]
                for j,lag in enumerate(ml):
                    if t-lag>=0: pred+=theta[j]*res[t-lag]
                res[t]=y[t]-pred
            return np.sum(res[mx:]**2)
        x0=np.zeros(1+na+nm);x0[0]=np.mean(y)
        result=minimize(obj,x0,bounds=[(-1e6,1e6)]+[(-0.99,0.99)]*(na+nm),method='L-BFGS-B',options={'maxiter':2000})
        c=result.x[0];phi=result.x[1:1+na];theta=result.x[1+na:]
        res=np.zeros(n)
        for t in range(mx,n):
            pred=c
            for i,lag in enumerate(al):
                if t-lag>=0: pred+=phi[i]*y[t-lag]
            for j,lag in enumerate(ml):
                if t-lag>=0: pred+=theta[j]*res[t-lag]
            res[t]=y[t]-pred
        ye=np.concatenate([y,np.zeros(h)]);re=np.concatenate([res,np.zeros(h)])
        for t in range(n,n+h):
            pred=c
            for i,lag in enumerate(al):
                if t-lag>=0: pred+=phi[i]*ye[t-lag]
            for j,lag in enumerate(ml):
                if t-lag>=0 and t-lag<n: pred+=theta[j]*re[t-lag]
            ye[t]=pred
        fc=ye[n:]
        for dval in reversed(dr):
            fc2=np.zeros(len(fc));lv=dval[-1]
            for i in range(len(fc)): lv+=fc[i];fc2[i]=lv
            fc=fc2
        for _ in range(D):
            fc2=np.zeros(len(fc));tail=tr[-(s*max(D,1)):]
            for i in range(len(fc)):
                ref=tail[len(tail)-s+i] if 0<=len(tail)-s+i<len(tail) else (fc2[i-s] if i>=s else tr[-1])
                fc2[i]=fc[i]+ref
            fc=fc2
        return np.maximum(fc,0)
    except: return None

def reentrenar_modelo(nombre, y, h):
    """Re-entrena un modelo. Soporta HW con parámetros del nombre."""
    if nombre.startswith('SARIMA'):
        parts=nombre.replace('SARIMA','').replace(' ','')
        g1=parts.split(')(')[0].replace('(','');g2=parts.split(')(')[1].replace(')','')
        return fit_sarima(y,h,tuple(int(x) for x in g1.split(',')),tuple(int(x) for x in g2.split(',')))
    elif nombre.startswith('ARIMA'):
        nums=[int(x) for x in nombre.replace('ARIMA(','').replace(')','').split(',')]
        return fit_arima(y,h,tuple(nums))
    elif nombre.startswith('HW'):
        # Parse: "HW Con tend. aditivo (s=3)" or "HW Sin tend. mult. (s=4)"
        trend = 'Con tend.' in nombre
        seas = 'add' if 'aditivo' in nombre else 'mul'
        s_val = int(nombre.split('s=')[1].replace(')','')) if 's=' in nombre else 3
        fc, _ = fit_hw(y, h, seas, trend, s_val)
        return fc
    elif 'Promedio móvil' in nombre:
        k=int(nombre.split('(')[1].split('m')[0]);hist=y.tolist();fc=[]
        for _ in range(h): v=np.mean(hist[-k:]);fc.append(v);hist.append(v)
        return np.array(fc)
    elif nombre.startswith('SES'): fc,_=fit_ses(y,h);return fc
    elif nombre.startswith('Holt'): fc,_=fit_holt(y,h,'amort' in nombre);return fc
    elif nombre=='Naive': return np.full(h,y[-1])
    elif nombre=='Deriva':
        pend=(y[-1]-y[0])/max(len(y)-1,1)
        return np.maximum(np.array([y[-1]+pend*(i+1) for i in range(h)]),0)
    elif nombre=='Regresión lineal':
        coefs=np.polyfit(np.arange(1,len(y)+1),y,1)
        return np.maximum(np.polyval(coefs,np.arange(len(y)+1,len(y)+h+1)),0)
    return None

def ejecutar_pronostico(serie, horizonte_fc=6):
    y=serie.values.astype(float);n=len(y)
    n_test=min(3,max(1,n//4));n_train=n-n_test;train=y[:n_train];test=y[n_train:];h=len(test)
    modelos=[]
    modelos.append(('Naive','Base',np.full(h,train[-1])))
    pend=(train[-1]-train[0])/max(n_train-1,1)
    modelos.append(('Deriva','Base',np.maximum(np.array([train[-1]+pend*(i+1) for i in range(h)]),0)))
    coefs=np.polyfit(np.arange(1,n_train+1),train,1)
    modelos.append(('Regresión lineal','Base',np.maximum(np.polyval(coefs,np.arange(n_train+1,n_train+h+1)),0)))
    for k in range(2,min(7,n_train)):
        hist=train.tolist();pred=[]
        for _ in range(h): v=np.mean(hist[-k:]);pred.append(v);hist.append(v)
        modelos.append((f'Promedio móvil ({k}m)','SMA',np.maximum(np.array(pred),0)))
    p_,nm=fit_ses(train,h);modelos.append((nm,'SES',p_))
    p_,nm=fit_holt(train,h,False);modelos.append((nm,'Holt',p_))
    p_,nm=fit_holt(train,h,True);modelos.append((nm,'Holt',p_))
    for s in [3,4]:
        if n_train>=2*s:
            for seas in ['add','mul']:
                for trend in [True,False]:
                    p_,nm=fit_hw(train,h,seas,trend,s)
                    if p_ is not None: modelos.append((nm,'HW',p_))
    for p,d,q in itertools.product(range(4),range(3),range(4)):
        if p==0 and d==0 and q==0: continue
        pred=fit_arima(train,h,(p,d,q))
        if pred is not None and not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
            if calc_mape(test,pred)<200: modelos.append((f'ARIMA({p},{d},{q})','ARIMA',pred))
    s=3
    if n_train>=2*s:
        for p,d,q in itertools.product(range(2),range(2),range(2)):
            for P,D,Q in itertools.product(range(2),range(2),range(2)):
                if p==0 and q==0 and P==0 and Q==0: continue
                pred=fit_sarima(train,h,(p,d,q),(P,D,Q,s))
                if pred is not None and not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
                    if calc_mape(test,pred)<200: modelos.append((f'SARIMA({p},{d},{q})({P},{D},{Q},{s})','SARIMA',pred))
    res=[]
    for nombre,fam,pred in modelos:
        res.append({'Método':nombre,'Familia':fam,'MAPE':calc_mape(test,pred),'RMSE':calc_rmse(test,pred),'pred':pred})
    res.sort(key=lambda x:(x['MAPE'],x['RMSE']))
    best=res[0]; nombre_ganador=best['Método']
    forecast_futuro=reentrenar_modelo(nombre_ganador,y,horizonte_fc)
    if forecast_futuro is None: forecast_futuro=np.full(horizonte_fc,np.mean(y))
    top10_futuro={}
    for r in res[:10]:
        fc=reentrenar_modelo(r['Método'],y,horizonte_fc)
        if fc is not None and not np.any(np.isnan(fc)): top10_futuro[r['Método']]=fc
    return {'ranking':res[:20],'modelo':nombre_ganador,'mape':best['MAPE'],'rmse':best['RMSE'],
        'forecast':forecast_futuro,'train':train,'test':test,'n_train':n_train,'n_test':n_test,'top10_futuro':top10_futuro}

def ejecutar_mc_pronostico(forecast, mape_pct, horizonte_fc=6, seed=42):
    rng=np.random.default_rng(seed);mape_dec=mape_pct/100.0;sigma=forecast*mape_dec
    sim=np.zeros((horizonte_fc,N_MC_PRONOSTICO))
    for t in range(horizonte_fc):
        sim[t,:]=rng.normal(loc=forecast[t],scale=max(sigma[t],0.01),size=N_MC_PRONOSTICO)
        sim[t,:]=np.maximum(sim[t,:],0)
    acum=np.sum(sim,axis=0)
    indicadores=[]
    for t in range(horizonte_fc):
        d=sim[t,:];media=np.mean(d);std=np.std(d);p5=np.percentile(d,5);p95=np.percentile(d,95)
        indicadores.append({
            'Forecast':forecast[t],'Sigma':sigma[t],'Media MC':media,'Mediana MC':np.median(d),
            'Desv. Std.':std,'CV':std/media if media>0 else 0,'Mín':np.min(d),'Máx':np.max(d),
            'P2.5':np.percentile(d,2.5),'P5':p5,'P50':np.percentile(d,50),'P90':np.percentile(d,90),
            'P95':p95,'P97.5':np.percentile(d,97.5),'P99':np.percentile(d,99),
            'VaR inf. 95%':p5,'VaR sup. 95%':p95,
            'CVaR inf. 95%':np.mean(d[d<=p5]) if np.sum(d<=p5)>0 else p5,
            'CVaR sup. 95%':np.mean(d[d>=p95]) if np.sum(d>=p95)>0 else p95,
            'Dif. Media vs Forecast':media-forecast[t],'Dif. P50 vs Forecast':np.percentile(d,50)-forecast[t],
        })
    return {'sim':sim,'acum':acum,'indicadores':indicadores,
        'acum_stats':{'Forecast acum.':np.sum(forecast),'Media MC acum.':np.mean(acum),
            'Desv. Std. acum.':np.std(acum),'CV acum.':np.std(acum)/np.mean(acum) if np.mean(acum)>0 else 0,
            'IC 95% inf':np.percentile(acum,2.5),'IC 95% sup':np.percentile(acum,97.5)}}


# =====================================================================
# SIMULACIÓN DETERMINÍSTICA (con DataFrame — para Kanban, diente de sierra)
# =====================================================================
def simular_politica(p, politica, params_pol, horizonte, demanda_diaria=None, lt_diaria=None):
    d_mean=p['d'];L_mean=p['L'];Pv=p['Pv'];pct_vp=p['pct_vp'];H_d=p['H_diario'];b_d=p['b_d'];K=p['K']
    if demanda_diaria is None: demanda=np.full(horizonte,d_mean)
    else: demanda=demanda_diaria[:horizonte] if len(demanda_diaria)>=horizonte else np.concatenate([demanda_diaria,np.full(horizonte-len(demanda_diaria),d_mean)])
    inventario=max(p['stock_ini'],0);backlog=0.0;oot=[]
    reg={k:[] for k in ['dia','demanda','atendida','vp','bl_nuevo','bl_final','inv_inicio','inv_final',
        'pedido','q_pedido','costo_ordenar','costo_mant','costo_vp','costo_bl','costo_total','quiebre','bl_atendido','recibido']}
    for dia in range(horizonte):
        ll=sum(q for(d_ll,q)in oot if d_ll<=dia);oot=[(d_ll,q)for(d_ll,q)in oot if d_ll>dia];inventario+=ll
        inv_ini=inventario;bl_at=0.0
        if backlog>0 and inventario>0: att=min(backlog,inventario);backlog-=att;inventario-=att;bl_at=att
        dem=redondear_05(demanda[dia]);at=min(dem,inventario);ft=dem-at;inventario-=at
        vp=redondear_05(ft*pct_vp);vp=min(vp,ft);bn=max(ft-vp,0.0);backlog+=bn
        OO=sum(q for(_,q)in oot);IP=inventario+OO-backlog
        ped=0;qp=0
        if politica=='QS':
            if IP<=params_pol['s']:qp=int(np.ceil(params_pol['Q']));ped=1
        elif politica=='TS':
            if dia%params_pol['T']==0:
                qc=params_pol['S']-IP
                if qc>0:qp=int(np.ceil(qc));ped=1
        elif politica=='sS':
            if IP<=params_pol['s']:
                qc=params_pol['S']-IP
                if qc>0:qp=int(np.ceil(qc));ped=1
        if ped:
            lt=int(np.ceil(lt_diaria[dia])) if lt_diaria is not None and dia<len(lt_diaria) else int(np.ceil(L_mean))
            oot.append((dia+lt,qp))
        co=K if ped else 0;cm=max(inventario,0)*H_d;cv=vp*Pv;cb=backlog*b_d
        for k,v in zip(reg.keys(),[dia,dem,at,vp,bn,backlog,inv_ini,inventario,ped,qp,co,cm,cv,cb,co+cm+cv+cb,1 if ft>0 else 0,bl_at,ll]):
            reg[k].append(v)
    return pd.DataFrame(reg)

def consolidar_mensual(sim_df, fecha_inicio):
    sim_df=sim_df.copy();sim_df['fecha']=pd.date_range(start=fecha_inicio,periods=len(sim_df),freq='D')
    sim_df['mes_cal']=sim_df['fecha'].dt.to_period('M')
    m=sim_df.groupby('mes_cal').agg(dias=('dia','count'),demanda=('demanda','sum'),atendida=('atendida','sum'),
        bl_atendido=('bl_atendido','sum'),vp=('vp','sum'),bl_nuevo=('bl_nuevo','sum'),bl_final=('bl_final','last'),
        inv_promedio=('inv_final','mean'),inv_final=('inv_final','last'),pedidos=('pedido','sum'),
        und_pedidas=('q_pedido','sum'),und_recibidas=('recibido','sum'),costo_ordenar=('costo_ordenar','sum'),
        costo_mant=('costo_mant','sum'),costo_vp=('costo_vp','sum'),costo_bl=('costo_bl','sum'),
        costo_total=('costo_total','sum'),dias_quiebre=('quiebre','sum')).reset_index()
    m['fill_rate']=(m['atendida']/m['demanda']).clip(0,1);m['mes_cal']=m['mes_cal'].astype(str)
    return m

# =====================================================================
# MC RÁPIDO (sin DataFrames internos — punto 15)
# =====================================================================
def ejecutar_mc_rapido(p, pol_type, pol_params, horizonte, n_mc, dem_matrix, lt_matrix):
    """MC optimizado. dem_matrix/lt_matrix: shape (n_mc, horizonte). Escenarios compartidos."""
    Pv=p['Pv'];pct_vp=p['pct_vp'];H_d=p['H_diario'];b_d=p['b_d'];K=p['K'];L_m=p['L'];si=max(p['stock_ini'],0)
    keys=['costo_total','costo_ordenar','costo_mant','costo_vp','costo_bl','inv_prom','inv_max','vp',
          'bl_max','bl_final','bl_nuevo_total','fill_rate','pedidos','dem_total','atendida_total',
          'und_pedidas','und_recibidas','dias_quiebre']
    mc={k:np.zeros(n_mc) for k in keys}
    for it in range(n_mc):
        inv=si;bl=0.0;oot=[];s_ct=0;s_co=0;s_cm=0;s_cv=0;s_cb=0;s_inv=0;mx_inv=0
        s_vp=0;mx_bl=0;s_bn=0;s_dem=0;s_at=0;s_qp=0;s_rec=0;s_q=0;n_ped=0
        for dia in range(horizonte):
            ll=0;new_o=[]
            for(d_ll,q)in oot:
                if d_ll<=dia:ll+=q
                else:new_o.append((d_ll,q))
            oot=new_o;inv+=ll;s_rec+=ll
            if bl>0 and inv>0:att=min(bl,inv);bl-=att;inv-=att
            dem=np.ceil(dem_matrix[it,dia]*2)/2;s_dem+=dem;at=min(dem,inv);ft=dem-at;inv-=at;s_at+=at
            vp_h=np.ceil(ft*pct_vp*2)/2;vp_h=min(vp_h,ft);bn=max(ft-vp_h,0.0);bl+=bn;s_vp+=vp_h;s_bn+=bn
            OO=sum(q for(_,q)in oot);IP=inv+OO-bl;ped=0;qp=0
            if pol_type=='QS':
                if IP<=pol_params['s']:qp=int(np.ceil(pol_params['Q']));ped=1
            elif pol_type=='TS':
                if dia%pol_params['T']==0:
                    qc=pol_params['S']-IP
                    if qc>0:qp=int(np.ceil(qc));ped=1
            elif pol_type=='sS':
                if IP<=pol_params['s']:
                    qc=pol_params['S']-IP
                    if qc>0:qp=int(np.ceil(qc));ped=1
            if ped:lt=int(np.ceil(lt_matrix[it,dia]));oot.append((dia+lt,qp));n_ped+=1;s_qp+=qp
            co=K if ped else 0;cm=max(inv,0)*H_d;cv=vp_h*Pv;cb=bl*b_d
            s_co+=co;s_cm+=cm;s_cv+=cv;s_cb+=cb;s_ct+=co+cm+cv+cb
            s_inv+=inv;mx_inv=max(mx_inv,inv);mx_bl=max(mx_bl,bl)
            if ft>0:s_q+=1
        mc['costo_total'][it]=s_ct;mc['costo_ordenar'][it]=s_co;mc['costo_mant'][it]=s_cm
        mc['costo_vp'][it]=s_cv;mc['costo_bl'][it]=s_cb;mc['inv_prom'][it]=s_inv/horizonte
        mc['inv_max'][it]=mx_inv;mc['vp'][it]=s_vp;mc['bl_max'][it]=mx_bl;mc['bl_final'][it]=bl
        mc['bl_nuevo_total'][it]=s_bn;mc['fill_rate'][it]=s_at/s_dem if s_dem>0 else 0
        mc['pedidos'][it]=n_ped;mc['dem_total'][it]=s_dem;mc['atendida_total'][it]=s_at
        mc['und_pedidas'][it]=s_qp;mc['und_recibidas'][it]=s_rec;mc['dias_quiebre'][it]=s_q
    return mc

# =====================================================================
# SELECCIÓN MULTICRITERIO CON PUNTAJE PONDERADO (punto 14)
# =====================================================================
def seleccion_multicriterio(mc_resultados, nivel_servicio_obj, horizonte):
    pesos = {'fill_rate':0.25,'vp':0.20,'dias_quiebre':0.15,'costo_total':0.20,'VaR95':0.10,'CVaR95':0.10}
    datos = []
    for pol_name, mc in mc_resultados.items():
        ct=mc['costo_total'];v95=np.percentile(ct,95)
        datos.append({
            'Política':pol_name,'FR_medio':np.mean(mc['fill_rate']),
            'VP_media':np.mean(mc['vp']),'BL_max':np.mean(mc['bl_max']),
            'Dias_quiebre':np.mean(mc['dias_quiebre']),'Tasa_quiebre':np.mean(mc['dias_quiebre'])/horizonte,
            'Inv_prom':np.mean(mc['inv_prom']),'Pedidos':np.mean(mc['pedidos']),
            'CT_medio':np.mean(ct),'CT_std':np.std(ct),
            'VaR95':v95,'CVaR95':np.mean(ct[ct>=v95]),
            'Cumple_NS':np.mean(mc['fill_rate'])>=nivel_servicio_obj,
        })
    # Normalización min-max invertida donde menor es mejor
    vals = {k:np.array([d[k] for d in datos]) for k in ['FR_medio','VP_media','Dias_quiebre','CT_medio','VaR95','CVaR95']}
    scores = np.zeros(len(datos))
    for i in range(len(datos)):
        for crit, peso in pesos.items():
            arr = vals.get(crit, vals.get('CT_medio'))
            rng = arr.max()-arr.min()
            if rng == 0: s = 1.0
            elif crit == 'fill_rate': s = (arr[i]-arr.min())/rng  # mayor es mejor
            elif crit == 'FR_medio': s = (arr[i]-arr.min())/rng
            else: s = 1 - (arr[i]-arr.min())/rng  # menor es mejor
            scores[i] += s * peso
        datos[i]['Puntaje'] = scores[i]
    # Filtrar por NS, luego ordenar por puntaje
    cumplen = [d for d in datos if d['Cumple_NS']]
    pool = cumplen if cumplen else datos
    pool.sort(key=lambda x: -x['Puntaje'])
    ganadora = pool[0]['Política']; g = pool[0]
    motivos = []
    if g['Cumple_NS']: motivos.append(f"cumple el nivel de servicio ({g['FR_medio']:.2%} ≥ {nivel_servicio_obj:.0%})")
    motivos.append(f"mayor puntaje ponderado ({g['Puntaje']:.3f})")
    motivos.append(f"CT medio S/ {g['CT_medio']:,.0f}")
    explicacion = f"**{ganadora}** seleccionada: " + ", ".join(motivos) + "."
    return ganadora, datos, explicacion

# =====================================================================
# KANBAN
# =====================================================================
def clasificar_kanban(inv_final, backlog, vp, quiebre, SS, umbral):
    if inv_final<=SS or backlog>0 or vp>0 or quiebre==1:
        return '🔴 Crítico','Inventario crítico. Riesgo de quiebre antes de la próxima revisión.'
    elif inv_final<=umbral:
        return '🟡 Alerta','Inventario cercano al stock de seguridad. Preparar reposición.'
    return '🟢 Normal','Inventario con cobertura suficiente.'

def prox_revision(dia, T):
    if T is None: return None  # revisión continua
    if dia%T==0: return dia+T
    return dia+(T-dia%T)

def construir_historial(sim_df, nc, SS, S, T, margen, fecha_ini):
    rows=[];umbral=SS*(1+margen)
    for _,f in sim_df.iterrows():
        dia=int(f['dia']);fecha=fecha_ini+pd.Timedelta(days=dia)
        est,acc=clasificar_kanban(f['inv_final'],f['bl_final'],f['vp'],f['quiebre'],SS,umbral)
        pr=fecha_ini+pd.Timedelta(days=prox_revision(dia,T)) if T else None
        rows.append({'Día':dia+1,'Fecha':fecha.strftime('%d-%b-%Y'),'Producto':nc,
            'Inv. inicial':f['inv_inicio'],'Demanda':f['demanda'],'Atendida':f['atendida'],
            'Inv. final':f['inv_final'],'SS':SS,'Umbral':umbral,'S':S,
            'Backlog':f['bl_final'],'VP':f['vp'],'Pedido':int(f['pedido']),'Cant. pedida':int(f['q_pedido']),
            'Recibido':f['recibido'],'Estado':est,'Acción':acc,
            'Próx. revisión':pr.strftime('%d-%b-%Y') if pr else 'Continua'})
    return pd.DataFrame(rows)

# =====================================================================
# VALIDACIONES DE COHERENCIA (punto 21)
# =====================================================================
def validar_coherencia(mc, horizonte):
    errores = []
    fr = mc['fill_rate']
    if np.any(fr < 0) or np.any(fr > 1.001): errores.append("Fill rate fuera de [0,1]")
    if np.any(mc['bl_final'] < -0.01): errores.append("Backlog negativo detectado")
    if np.any(mc['vp'] < -0.01): errores.append("VP negativa detectada")
    ct = mc['costo_total']; p5=np.percentile(ct,5); p50=np.percentile(ct,50); p95=np.percentile(ct,95)
    v95 = np.percentile(ct,95); cv95 = np.mean(ct[ct>=v95])
    if not (p5 <= p50 + 0.01): errores.append("P5 > P50 en costo")
    if not (p50 <= p95 + 0.01): errores.append("P50 > P95 en costo")
    if not (np.mean(ct) <= cv95 + 0.01): errores.append("CT medio > CVaR95")
    at = mc['atendida_total']; dm = mc['dem_total']
    if np.any(at > dm + 0.1): errores.append("Demanda atendida > demanda")
    return errores


# =====================================================================
# GRÁFICOS
# =====================================================================
def graf_pronostico(serie, res, nc):
    f=serie.index;y=serie.values;nt=res['n_train']
    ff=pd.date_range(f[-1]+pd.DateOffset(months=1),periods=len(res['forecast']),freq='MS')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=f[:nt],y=y[:nt],mode='lines+markers',name='Train',line=dict(color='#1565C0',width=2,shape='spline',smoothing=0.8)))
    fig.add_trace(go.Scatter(x=f[nt:],y=y[nt:],mode='lines+markers',name='Test',line=dict(color='#2E7D32',width=2,shape='spline',smoothing=0.8)))
    fig.add_trace(go.Scatter(x=ff,y=res['forecast'],mode='lines+markers',name='Pronóstico',line=dict(color='#E65100',width=2.5,dash='dash',shape='spline',smoothing=0.8)))
    fig.update_layout(title=f'{nc} — {res["modelo"]} (MAPE={res["mape"]:.2f}%)',template='plotly_white',height=380,
        legend=dict(orientation='h',yanchor='bottom',y=1.02))
    return fig

def graf_top10_val(serie, res, nc):
    f=serie.index;nt=res['n_train'];ft=f[nt:]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ft,y=res['test'],mode='lines+markers',name='Real',line=dict(color='black',width=3)))
    for i,r in enumerate(res['ranking'][:10]):
        w=3 if i==0 else 1.2;op=1.0 if i==0 else 0.4
        fig.add_trace(go.Scatter(x=ft,y=r['pred'],mode='lines+markers',name=f"{r['Método']} ({r['MAPE']:.1f}%)",
            opacity=op,line=dict(width=w,shape='spline',smoothing=0.8)))
    fig.update_layout(title=f'{nc} — Top 10 validación',template='plotly_white',height=380,
        legend=dict(font=dict(size=8),orientation='h',yanchor='bottom',y=1.02))
    return fig

def graf_top10_fut(serie, res, nc, h):
    f=serie.index;y=serie.values;ff=pd.date_range(f[-1]+pd.DateOffset(months=1),periods=h,freq='MS')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=f[-6:],y=y[-6:],mode='lines+markers',name='Histórico',line=dict(color='#616161',width=2,shape='spline',smoothing=0.8)))
    gan=res['modelo']
    for nombre,fc in res.get('top10_futuro',{}).items():
        w=3 if nombre==gan else 1.2;op=1.0 if nombre==gan else 0.35
        fig.add_trace(go.Scatter(x=ff,y=fc,mode='lines+markers',name=nombre,opacity=op,line=dict(width=w,shape='spline',smoothing=0.8)))
    fig.update_layout(title=f'{nc} — Top 10 futuro',template='plotly_white',height=380,
        legend=dict(font=dict(size=8),orientation='h',yanchor='bottom',y=1.02))
    return fig

def graf_mc_bandas(ff, mc_p, fc, nc):
    ind=mc_p['indicadores']
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ff,y=[i['P97.5'] for i in ind],mode='lines',line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=ff,y=[i['P2.5'] for i in ind],mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(33,150,243,0.12)',name='IC 95%'))
    fig.add_trace(go.Scatter(x=ff,y=[i['P95'] for i in ind],mode='lines',line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=ff,y=[i['P5'] for i in ind],mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(33,150,243,0.22)',name='IC 90%'))
    fig.add_trace(go.Scatter(x=ff,y=fc,mode='lines+markers',name='Forecast',line=dict(color='#E65100',width=2.5,shape='spline',smoothing=0.8)))
    fig.add_trace(go.Scatter(x=ff,y=[i['Media MC'] for i in ind],mode='lines+markers',name='Media MC',line=dict(color='#1565C0',width=2,dash='dash',shape='spline',smoothing=0.8)))
    fig.update_layout(title=f'{nc} — MC pronóstico ({N_MC_PRONOSTICO:,} iter.)',template='plotly_white',height=360,
        legend=dict(orientation='h',yanchor='bottom',y=1.02))
    return fig

def graf_diente(sim, nc, config_pol, fecha_ini):
    """Diente de sierra adaptado a la política ganadora."""
    fechas=pd.date_range(start=fecha_ini,periods=len(sim),freq='D')
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],vertical_spacing=0.08,
        subplot_titles=[f'{nc} — {config_pol["titulo"]}','Backlog y VP'])
    fig.add_trace(go.Scatter(x=fechas,y=sim['inv_final'],mode='lines',name='Inventario',
        line=dict(color='#1565C0',width=1.5),fill='tozeroy',fillcolor='rgba(21,101,192,0.06)'),row=1,col=1)
    # Líneas según política
    if config_pol['S'] is not None:
        fig.add_hline(y=config_pol['S'],line_dash='dash',line_color='#2E7D32',line_width=1,
            annotation_text=f"S={config_pol['S']}",row=1,col=1)
    if config_pol['s'] is not None:
        fig.add_hline(y=config_pol['s'],line_dash='dot',line_color='#E65100',line_width=1,
            annotation_text=f"s={config_pol['s']}",row=1,col=1)
    if config_pol['T'] is not None:
        for dia in range(0,len(sim),config_pol['T']):
            fig.add_vline(x=fechas[dia],line_dash='dot',line_color='rgba(128,128,128,0.2)',line_width=0.5,row=1,col=1)
    pi=sim[sim['pedido']==1].index
    if len(pi)>0:
        fig.add_trace(go.Scatter(x=fechas[pi],y=sim.loc[pi,'inv_final'],mode='markers',name='Pedido',
            marker=dict(symbol='triangle-up',size=8,color='#FF6F00')),row=1,col=1)
    ri=sim[sim['recibido']>0].index
    if len(ri)>0:
        fig.add_trace(go.Scatter(x=fechas[ri],y=sim.loc[ri,'inv_final'],mode='markers',name='Recepción',
            marker=dict(symbol='diamond',size=7,color='#2E7D32')),row=1,col=1)
    fig.add_trace(go.Scatter(x=fechas,y=sim['bl_final'],mode='lines',name='Backlog',
        line=dict(color='#C62828',width=1.5),fill='tozeroy',fillcolor='rgba(198,40,40,0.06)'),row=2,col=1)
    fig.add_trace(go.Bar(x=fechas,y=sim['vp'],name='VP',marker_color='rgba(255,111,0,0.35)'),row=2,col=1)
    fig.update_layout(height=540,template='plotly_white',legend=dict(orientation='h',yanchor='bottom',y=1.02))
    return fig

def graf_mc_hist(arr, nc, titulo, unidad='S/'):
    fig=go.Figure();fig.add_trace(go.Histogram(x=arr,nbinsx=80,marker_color='rgba(21,101,192,0.5)'))
    m=np.mean(arr);p5=np.percentile(arr,5);p95=np.percentile(arr,95)
    if unidad=='%':
        fig.add_vline(x=m,line_color='#C62828',line_width=2,annotation_text=f'Media: {m:.2%}')
        fig.add_vline(x=p5,line_dash='dash',line_color='#2E7D32',annotation_text=f'P5: {p5:.2%}')
        fig.add_vline(x=p95,line_dash='dash',line_color='#E65100',annotation_text=f'P95: {p95:.2%}')
    elif unidad=='S/':
        fig.add_vline(x=m,line_color='#C62828',line_width=2,annotation_text=f'Media: S/{m:,.0f}')
        fig.add_vline(x=p5,line_dash='dash',line_color='#2E7D32',annotation_text=f'P5: S/{p5:,.0f}')
        fig.add_vline(x=p95,line_dash='dash',line_color='#E65100',annotation_text=f'P95: S/{p95:,.0f}')
    else:
        fig.add_vline(x=m,line_color='#C62828',line_width=2,annotation_text=f'Media: {m:,.1f}')
        fig.add_vline(x=p5,line_dash='dash',line_color='#2E7D32',annotation_text=f'P5: {p5:,.1f}')
        fig.add_vline(x=p95,line_dash='dash',line_color='#E65100',annotation_text=f'P95: {p95:,.1f}')
    fig.update_layout(title=f'{nc} — {titulo}',template='plotly_white',height=310,showlegend=False)
    return fig

def graf_kanban(sim, nc, SS, S, umbral, T, dia_sel, f0):
    fechas=pd.date_range(start=f0,periods=len(sim),freq='D');inv=sim['inv_final'].values;ym=max(S*1.15,inv.max()*1.1)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.72,0.28],vertical_spacing=0.06,
        subplot_titles=[f'{nc} — Kanban','Backlog y VP'])
    fig.add_hrect(y0=0,y1=SS,fillcolor='rgba(198,40,40,0.05)',line_width=0,row=1,col=1)
    fig.add_hrect(y0=SS,y1=umbral,fillcolor='rgba(249,168,37,0.05)',line_width=0,row=1,col=1)
    fig.add_hrect(y0=umbral,y1=ym,fillcolor='rgba(46,125,50,0.03)',line_width=0,row=1,col=1)
    fig.add_trace(go.Scatter(x=fechas,y=inv,mode='lines',name='Inventario',line=dict(color='#1565C0',width=1.8)),row=1,col=1)
    fig.add_hline(y=SS,line_dash='dash',line_color='#C62828',line_width=1,annotation_text=f'SS={SS}',row=1,col=1)
    fig.add_hline(y=umbral,line_dash='dot',line_color='#F9A825',line_width=1,annotation_text=f'Umbral={umbral:.0f}',row=1,col=1)
    fig.add_hline(y=S,line_dash='dash',line_color='#2E7D32',line_width=1,annotation_text=f'S={S}',row=1,col=1)
    if T:
        for d in range(0,len(sim),T): fig.add_vline(x=fechas[d],line_dash='dot',line_color='rgba(128,128,128,0.2)',line_width=0.5,row=1,col=1)
    pi=sim[sim['pedido']==1].index
    if len(pi)>0: fig.add_trace(go.Scatter(x=fechas[pi],y=sim.loc[pi,'inv_final'],mode='markers',name='Pedido',marker=dict(symbol='triangle-up',size=9,color='#FF6F00')),row=1,col=1)
    ri=sim[sim['recibido']>0].index
    if len(ri)>0: fig.add_trace(go.Scatter(x=fechas[ri],y=sim.loc[ri,'inv_final'],mode='markers',name='Recepción',marker=dict(symbol='diamond',size=8,color='#2E7D32')),row=1,col=1)
    fs=f0+pd.Timedelta(days=dia_sel);fig.add_vline(x=fs,line_color='#6A1B9A',line_width=2,annotation_text=f'Día {dia_sel+1}',row=1,col=1)
    fig.add_trace(go.Scatter(x=fechas,y=sim['bl_final'],mode='lines',name='Backlog',line=dict(color='#C62828',width=1.5),fill='tozeroy',fillcolor='rgba(198,40,40,0.06)'),row=2,col=1)
    fig.add_trace(go.Bar(x=fechas,y=sim['vp'],name='VP',marker_color='rgba(255,111,0,0.35)'),row=2,col=1)
    fig.update_layout(height=560,template='plotly_white',legend=dict(orientation='h',yanchor='bottom',y=1.02))
    fig.update_yaxes(range=[0,ym],row=1,col=1)
    return fig


# #####################################################################
# MAIN APP
# #####################################################################
st.title("📦 Sistema de Pronósticos y Gestión de Inventarios")
st.caption("Proyecto de Tesis — UT&M | Melaminas Pelikano e Hispanos")

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Configuración")
    archivo = st.file_uploader("Sube el Excel de demanda", type=["xlsx"])
    with st.expander("📦 Parámetros", expanded=True):
        nivel_servicio_pct = st.slider("Nivel de servicio (%)", 80, 99, 95, 1)
        nivel_servicio = nivel_servicio_pct / 100
        z_factor = norm.ppf(nivel_servicio)
        tasa_mant_pct = st.slider("Tasa mantenimiento (%)", 5, 40, 18, 1)
        tasa_mant = tasa_mant_pct / 100
        horizonte_dias = st.slider("Horizonte simulación (días)", 90, 365, 180, 30)
        n_mc = st.select_slider("Iteraciones MC", [1000, 5000, 10000], value=10000)
        horizonte_fc = st.slider("Horizonte pronóstico (meses)", 3, 12, 6)

    archivo_hash = hashlib.md5(archivo.getvalue()).hexdigest()[:8] if archivo else ''
    params_key = f"{archivo_hash}_{nivel_servicio_pct}_{tasa_mant_pct}_{horizonte_dias}_{n_mc}_{horizonte_fc}"
    if st.session_state.get('last_params', '') != params_key:
        invalidar_resultados()
        st.session_state['last_params'] = params_key
    if st.button("🗑️ Limpiar resultados"): invalidar_resultados()

if not archivo:
    st.info("👆 Sube **DEMANDA_MELAMINA_COMPLETADA_ERP.xlsx** para comenzar."); st.stop()

# === CARGA Y VALIDACIÓN ===
try: df = pd.read_excel(archivo, sheet_name='DEMANDA')
except Exception as e: st.error(f"Error: {e}"); st.stop()

cols_req = ['Fecha','Producto','Demanda_Ref','Stock_Inicial','Cantidad_Recibida','LeadTime_Real_d',
    'Precio_Unitario_S','Costo_Falla_Abast_S','Costo_Compra_Unitario_S','Costo_Pedido_S',
    'Tasa_Mantenimiento_Anual','Nivel_Servicio','Factor_Z','Pct_Ventas_Perdidas','Pct_Backlog',
    'Demanda_Media_Diaria','Desv_Demanda_Diaria','LeadTime_Promedio_d','Desv_LeadTime_d',
    'Precio_Venta_Promedio_S','LeadTime_Min_d','LeadTime_Max_d','Quiebre_Stock']
falt = [c for c in cols_req if c not in df.columns]
if falt: st.error(f"Faltan columnas: {', '.join(falt)}"); st.stop()

df = df.dropna(subset=['Producto']).copy()
df['Fecha'] = pd.to_datetime(df['Fecha'])
if (df['Demanda_Ref']<0).any():
    st.warning("⚠️ Valores negativos en Demanda_Ref → reemplazados por 0.")
    df['Demanda_Ref'] = df['Demanda_Ref'].clip(lower=0)

# Fecha inicio desde datos (punto 7)
FECHA_INICIO_SIM = df['Fecha'].max().normalize() + pd.Timedelta(days=1)
st.success(f"✅ {len(df)} registros, {df['Producto'].nunique()} productos. Simulación desde {FECHA_INICIO_SIM.strftime('%d-%b-%Y')}.")

# VP+BL validation
r0 = df.iloc[0]
suma = r0['Pct_Ventas_Perdidas'] + r0['Pct_Backlog']
if abs(suma - 1.0) > 0.01:
    st.warning(f"⚠️ Pct_VP + Pct_BL = {suma:.2f} ≠ 1.00 → se normalizará.")

# === PRODUCTOS ===
series = {}; productos = {}
for prod in df['Producto'].unique():
    sub = df[df['Producto']==prod].copy()
    nc = 'Pelikano' if 'PELIKANO' in prod.upper() else 'Hispanos' if 'HISPANOS' in prod.upper() else nombre_corto(prod)
    sub['Mes'] = sub['Fecha'].dt.to_period('M').dt.to_timestamp()
    series[nc] = sub.groupby('Mes')['Demanda_Ref'].sum().sort_index()
    r = sub.iloc[0]
    d_m=r['Demanda_Media_Diaria'];sd=r['Desv_Demanda_Diaria'];Lm=r['LeadTime_Promedio_d'];sL=r['Desv_LeadTime_d']
    C=r['Costo_Compra_Unitario_S'];K=r['Costo_Pedido_S'];H=tasa_mant*C;Pv=r['Precio_Venta_Promedio_S']
    pv=r['Pct_Ventas_Perdidas'];pb=r['Pct_Backlog']
    if abs(pv+pb-1.0)>0.01: t=pv+pb;pv/=t;pb/=t
    b_d=0.50*H/365;si=sub.sort_values('Fecha').iloc[-1]['Stock_Inicial'];Da=d_m*365
    Q=int(np.ceil(np.sqrt(2*Da*K/H)));SS=int(np.ceil(z_factor*np.sqrt(Lm*sd**2+d_m**2*sL**2)))
    s_r=int(np.ceil(d_m*Lm+SS))
    TS={}
    for T in [7,15,30]:
        S_ts=int(np.ceil(d_m*(T+Lm)+z_factor*np.sqrt((T+Lm)*sd**2+d_m**2*sL**2)))
        SS_ts=int(np.ceil(z_factor*np.sqrt((T+Lm)*sd**2+d_m**2*sL**2)))
        TS[T]={'T':T,'S':S_ts,'SS':SS_ts}
    productos[nc]={
        'd':d_m,'sigma_d':sd,'D_anual':Da,'L':Lm,'sigma_L':sL,
        'L_min':r['LeadTime_Min_d'],'L_max':r['LeadTime_Max_d'],
        'C':C,'K':K,'H':H,'H_diario':H/365,'z':z_factor,'Pv':Pv,
        'pct_vp':pv,'pct_bl':pb,'b_d':b_d,'stock_ini':si,
        'dem_empirica':sub['Demanda_Ref'].values,'lt_empirica':sub['LeadTime_Real_d'].values,
        'QS':{'Q':Q,'s':s_r,'SS':SS},'TS':TS,'sS':{'s':s_r,'S':s_r+Q,'SS':SS},'sub_df':sub}

tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["📊 Pronóstico","📦 Políticas","💰 Evaluación económica","🚦 Kanban","📊 Dashboard","📥 Descargar"])

# #####################################################################
# TAB 1: PRONÓSTICO
# #####################################################################
with tab1:
    st.header("Datos históricos y pronóstico")
    for nc,serie in series.items():
        with st.expander(f"📈 {nc}",expanded=True):
            pk=f'pron_{nc}'
            if pk not in st.session_state:
                with st.spinner(f'Evaluando modelos {nc}...'): np.random.seed(42); st.session_state[pk]=ejecutar_pronostico(serie,horizonte_fc)
            res=st.session_state[pk]
            c1,c2=st.columns([2.5,1])
            with c1: st.plotly_chart(graf_pronostico(serie,res,nc),use_container_width=True,key=f"c_pron_1_{nc}")
            with c2:
                st.dataframe(pd.DataFrame({'Mes':[d.strftime('%b-%Y') for d in serie.index],'Demanda':serie.values}),hide_index=True,use_container_width=True)
            st.markdown(f"**Ganador:** {res['modelo']} — MAPE: {res['mape']:.2f}% | RMSE: {res['rmse']:.1f} | Train: {res['n_train']}m | Test: {res['n_test']}m")
            # Validation table
            ft=serie.index[res['n_train']:];bp=res['ranking'][0]['pred']
            vr=[]
            for i in range(len(res['test'])):
                rl=res['test'][i];pr=bp[i];er=abs(rl-pr);ap=er/rl*100 if rl!=0 else 0
                vr.append({'Mes':ft[i].strftime('%b-%Y'),'Real':f'{rl:,.1f}','Pred':f'{pr:,.1f}','Error':f'{er:,.1f}','APE%':f'{ap:.2f}%'})
            vr.append({'Mes':'TOTAL','Real':'','Pred':'','Error':'',
                'APE%':f'MAPE: {res["mape"]:.2f}% | RMSE: {res["rmse"]:.1f}'})
            st.dataframe(pd.DataFrame(vr),hide_index=True,use_container_width=True)
            st.dataframe(pd.DataFrame([{'#':i+1,'Método':r['Método'],'Familia':r['Familia'],'MAPE%':round(r['MAPE'],2),'RMSE':round(r['RMSE'],1)} for i,r in enumerate(res['ranking'][:10])]),hide_index=True,use_container_width=True)
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(graf_top10_val(serie,res,nc),use_container_width=True,key=f"c_t10v_2_{nc}")
            with c2: st.plotly_chart(graf_top10_fut(serie,res,nc,horizonte_fc),use_container_width=True,key=f"c_t10f_3_{nc}")
            # Forecast + MC
            ff=pd.date_range(serie.index[-1]+pd.DateOffset(months=1),periods=horizonte_fc,freq='MS')
            st.dataframe(pd.DataFrame({'Mes':[f.strftime('%b-%Y') for f in ff],'Pronóstico':[f'{v:,.1f}' for v in res['forecast']]}),hide_index=True,use_container_width=True)
            mk=f'mc_pron_{nc}'
            if mk not in st.session_state:
                with st.spinner(f'MC pronóstico {nc}...'): st.session_state[mk]=ejecutar_mc_pronostico(res['forecast'],res['mape'],horizonte_fc,42)
            mcp=st.session_state[mk]
            st.markdown(f"**MC pronóstico: {N_MC_PRONOSTICO:,} iter/mes** — σ=Forecast×MAPE({res['mape']:.2f}%)")
            ir=[]
            for t,ind in enumerate(mcp['indicadores']):
                ir.append({'Mes':ff[t].strftime('%b-%Y'),'Fc':f"{ind['Forecast']:,.0f}",'Media':f"{ind['Media MC']:,.0f}",
                    'P5':f"{ind['P5']:,.0f}",'P50':f"{ind['P50']:,.0f}",'P95':f"{ind['P95']:,.0f}",
                    'VaR inf':f"{ind['VaR inf. 95%']:,.0f}",'VaR sup':f"{ind['VaR sup. 95%']:,.0f}",
                    'CVaR inf':f"{ind['CVaR inf. 95%']:,.0f}",'CVaR sup':f"{ind['CVaR sup. 95%']:,.0f}",
                    'CV':f"{ind['CV']:.4f}"})
            st.dataframe(pd.DataFrame(ir),hide_index=True,use_container_width=True)
            i0=mcp['indicadores'][0]
            if i0['CVaR inf. 95%']<i0['P5']<i0['P50']<i0['P95']<i0['CVaR sup. 95%']:
                st.success("✅ Pronóstico respaldado por MC. Orden estadístico validado.")
            else: st.warning("⚠️ Revisar dispersión del MC.")
            st.plotly_chart(graf_mc_bandas(ff,mcp,res['forecast'],nc),use_container_width=True,key=f"c_mcb_4_{nc}")


# #####################################################################
# TAB 2: POLÍTICAS
# #####################################################################
with tab2:
    st.header("Políticas de inventario y simulación")
    if 'sim_done' not in st.session_state:
        if st.button("▶️ Ejecutar simulación y Monte Carlo", type="primary"):
            t0=time.time();barra=st.progress(0,"Iniciando...");all_results={};ganadoras={}
            step=0;total=len(productos)*4
            for nc,p in productos.items():
                barra.progress(step/total,f"Determinística {nc}...");det={}
                for pn,pt,pp in [('(Q,s)','QS',p['QS']),('(T,S) T=7','TS',p['TS'][7]),('(s,S)','sS',p['sS'])]:
                    sim=simular_politica(p,pt,pp,horizonte_dias)
                    det[pn]={'sim':sim,'mensual':consolidar_mensual(sim,FECHA_INICIO_SIM),'ct':sim['costo_total'].sum(),
                        'fr':sim['atendida'].sum()/sim['demanda'].sum()}
                step+=1
                # Generar escenarios compartidos (punto 15)
                np.random.seed(42)
                dem_mat=np.zeros((n_mc,horizonte_dias));lt_mat=np.zeros((n_mc,horizonte_dias))
                for it in range(n_mc):
                    dem_mat[it]=np.random.choice(p['dem_empirica'],size=horizonte_dias,replace=True)
                    lt_mat[it]=np.random.choice(p['lt_empirica'],size=horizonte_dias,replace=True)
                mc_res={}
                for pn,pt,pp in [('(Q,s)','QS',p['QS']),('(T,S) T=7','TS',p['TS'][7]),('(s,S)','sS',p['sS'])]:
                    barra.progress(step/total,f"MC {nc} {pn} ({n_mc:,})...")
                    mc_res[pn]=ejecutar_mc_rapido(p,pt,pp,horizonte_dias,n_mc,dem_mat,lt_mat)
                    step+=1
                all_results[nc]={'det':det,'mc':mc_res}
                # Selección multicriterio (punto 1)
                gan,rank,expl=seleccion_multicriterio(mc_res,nivel_servicio,horizonte_dias)
                ganadoras[nc]={'politica':gan,'ranking':rank,'explicacion':expl}
            barra.progress(1.0,f"✅ Completado en {time.time()-t0:.1f}s")
            st.session_state['sim_done']=True;st.session_state['all_results']=all_results
            st.session_state['ganadoras']=ganadoras
        else: st.info("Haz clic para ejecutar."); st.stop()

    all_results=st.session_state['all_results'];ganadoras=st.session_state['ganadoras']

    # Validaciones de coherencia (punto 21)
    with st.expander("🔍 Validaciones internas"):
        all_ok=True
        for nc in productos:
            for pn,mc in all_results[nc]['mc'].items():
                errs=validar_coherencia(mc,horizonte_dias)
                if errs:
                    all_ok=False
                    for e in errs: st.error(f"{nc} / {pn}: {e}")
        if all_ok: st.success("✅ Todas las validaciones internas superadas.")

    for nc,p in productos.items():
        st.subheader(f"📦 {nc}")
        det=all_results[nc]['det'];mc=all_results[nc]['mc']
        pol_gan=ganadoras[nc]['politica'];cfg_gan=obtener_config_politica(p,pol_gan)

        with st.expander("Parámetros"):
            c1,c2,c3=st.columns(3)
            c1.metric("Q*",p['QS']['Q']);c1.metric("SS(Q,s)",p['QS']['SS']);c1.metric("s",p['QS']['s'])
            c2.metric("S(T,S T=7)",p['TS'][7]['S']);c2.metric("SS(T,S)",p['TS'][7]['SS'])
            c3.metric("S(s,S)",p['sS']['S']);c3.metric("d̄",f"{p['d']:.2f}");c3.metric("L̄",f"{p['L']:.2f}d")

        # Tabla MC unidades (punto 12)
        st.markdown("**MC — Unidades**")
        ur=[]
        for pn in ['(Q,s)','(T,S) T=7','(s,S)']:
            m=mc[pn]
            ur.append({'Política':pn,'Dem.total':f"{np.mean(m['dem_total']):,.0f}",
                'Atendida':f"{np.mean(m['atendida_total']):,.0f}",
                'VP media':f"{np.mean(m['vp']):,.0f}",
                'VP P5':f"{np.percentile(m['vp'],5):,.0f}",'VP P50':f"{np.percentile(m['vp'],50):,.0f}",
                'VP P95':f"{np.percentile(m['vp'],95):,.0f}",
                'BL máx':f"{np.mean(m['bl_max']):,.0f}",
                'Inv.prom':f"{np.mean(m['inv_prom']):,.0f}",
                'Inv.prom P5':f"{np.percentile(m['inv_prom'],5):,.0f}",
                'Inv.prom P95':f"{np.percentile(m['inv_prom'],95):,.0f}",
                'Pedidos':f"{np.mean(m['pedidos']):,.0f}",
                'Días quiebre':f"{np.mean(m['dias_quiebre']):,.1f}",
                'Tasa quiebre':f"{np.mean(m['dias_quiebre'])/horizonte_dias*100:.1f}%",
                'FR medio':f"{np.mean(m['fill_rate']):.2%}",
                'FR P5':f"{np.percentile(m['fill_rate'],5):.2%}",
                'FR P95':f"{np.percentile(m['fill_rate'],95):.2%}"})
        st.dataframe(pd.DataFrame(ur),hide_index=True,use_container_width=True)

        # Gráficos operativos
        c1,c2,c3=st.columns(3)
        with c1:
            fig=go.Figure([go.Bar(x=[pn],y=[np.mean(mc[pn]['vp'])],name=pn,error_y=dict(type='data',array=[np.std(mc[pn]['vp'])])) for pn in mc])
            fig.update_layout(title='VP media (und)',template='plotly_white',height=300,showlegend=False);st.plotly_chart(fig,use_container_width=True,key=f"pol_fr_{nc}")
        with c2:
            fig=go.Figure([go.Bar(x=[pn],y=[np.mean(mc[pn]['fill_rate'])*100]) for pn in mc])
            fig.update_layout(title='Fill rate (%)',template='plotly_white',height=300,showlegend=False,yaxis_ticksuffix='%');st.plotly_chart(fig,use_container_width=True,key=f"pol_dq_{nc}")
        with c3:
            fig=go.Figure([go.Bar(x=[pn],y=[np.mean(mc[pn]['dias_quiebre'])]) for pn in mc])
            fig.update_layout(title='Días quiebre',template='plotly_white',height=300,showlegend=False);st.plotly_chart(fig,use_container_width=True,key=f"c_dq_5_{nc}")

        # Tabla costos (punto 13)
        st.markdown("**MC — Costos (S/)**")
        cr=[]
        for pn in ['(Q,s)','(T,S) T=7','(s,S)']:
            m=mc[pn];ct=m['costo_total'];v95=np.percentile(ct,95)
            cr.append({'Política':pn,'C.ordenar':fmt_s(np.mean(m['costo_ordenar'])),'C.mantener':fmt_s(np.mean(m['costo_mant'])),
                'C.VP':fmt_s(np.mean(m['costo_vp'])),'C.BL':fmt_s(np.mean(m['costo_bl'])),
                'CT medio':fmt_s(np.mean(ct)),'CT σ':fmt_s(np.std(ct)),
                'CT P5':fmt_s(np.percentile(ct,5)),'CT P50':fmt_s(np.percentile(ct,50)),'CT P95':fmt_s(np.percentile(ct,95)),
                'VaR95':fmt_s(v95),'CVaR95':fmt_s(np.mean(ct[ct>=v95]))})
        st.dataframe(pd.DataFrame(cr),hide_index=True,use_container_width=True)

        # Stacked costs + CT vs VaR vs CVaR
        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure()
            pols=list(mc.keys())
            for comp,color,nm in [('costo_ordenar','#1565C0','Ordenar'),('costo_mant','#2E7D32','Mantener'),('costo_vp','#E65100','VP'),('costo_bl','#C62828','BL')]:
                fig.add_trace(go.Bar(name=nm,x=pols,y=[np.mean(mc[pn][comp]) for pn in pols],marker_color=color))
            fig.update_layout(barmode='stack',title='Costos desagregados',template='plotly_white',height=340);st.plotly_chart(fig,use_container_width=True,key=f"c_stk_6_{nc}")
        with c2:
            fig=go.Figure()
            for pn in pols:
                ct=mc[pn]['costo_total'];v95=np.percentile(ct,95)
                fig.add_trace(go.Bar(name=pn,x=['CT medio','VaR 95%','CVaR 95%'],
                    y=[np.mean(ct),v95,np.mean(ct[ct>=v95])]))
            fig.update_layout(barmode='group',title='CT vs VaR vs CVaR',template='plotly_white',height=340);st.plotly_chart(fig,use_container_width=True,key=f"c_rsk_7_{nc}")

        # Histograms of winner
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(graf_mc_hist(mc[pol_gan]['costo_total'],nc,f'Costo total {pol_gan}','S/'),use_container_width=True,key=f"c_mch_8_{nc}")
        with c2: st.plotly_chart(graf_mc_hist(mc[pol_gan]['fill_rate'],nc,f'Fill rate {pol_gan}','%'),use_container_width=True,key=f"c_mch_9_{nc}")

        # Selección multicriterio (punto 14)
        st.success(f"🏆 {ganadoras[nc]['explicacion']}")
        rk=pd.DataFrame([{'Política':r['Política'],'Cumple NS':'✅' if r['Cumple_NS'] else '❌',
            'Puntaje':f"{r['Puntaje']:.3f}",'FR':f"{r['FR_medio']:.2%}",'CT medio':fmt_s(r['CT_medio']),
            'VP':f"{r['VP_media']:,.0f}",'Días quiebre':f"{r['Dias_quiebre']:.1f}",
            'VaR95':fmt_s(r['VaR95']),'CVaR95':fmt_s(r['CVaR95'])} for r in ganadoras[nc]['ranking']])
        st.dataframe(rk,hide_index=True,use_container_width=True)

        # Consolidación y diente de sierra de la ganadora
        st.markdown(f"**Consolidación mensual — {pol_gan}**")
        st.dataframe(det[pol_gan]['mensual'],hide_index=True,use_container_width=True)
        st.plotly_chart(graf_diente(det[pol_gan]['sim'],nc,cfg_gan,FECHA_INICIO_SIM),use_container_width=True,key=f"c_dnt_10_{nc}")


# #####################################################################
# TAB 3: EVALUACIÓN ECONÓMICA + ROI
# #####################################################################
with tab3:
    st.header("Evaluación económica")
    if 'sim_done' not in st.session_state: st.warning("Ejecuta la simulación primero."); st.stop()
    all_results=st.session_state['all_results'];ganadoras=st.session_state['ganadoras']

    for nc,p in productos.items():
        st.subheader(f"💰 {nc}")
        pol_gan=ganadoras[nc]['politica'];cfg=obtener_config_politica(p,pol_gan)
        mc_g=all_results[nc]['mc'][pol_gan];det_g=all_results[nc]['det'][pol_gan]
        sub=p['sub_df']

        # AS-IS (punto 3 — misma fórmula)
        dem_hist=sub['Demanda_Ref'].sum();rec_hist=sub['Cantidad_Recibida'].sum()
        precision_asis=rec_hist/dem_hist if dem_hist>0 else 0  # atendida/requerida
        quiebres_hist=sub['Quiebre_Stock'].sum();total_regs=len(sub)
        tasa_quiebre_asis=quiebres_hist/total_regs  # punto 4
        meses_hist=sub['Fecha'].dt.to_period('M').nunique()
        costo_falla_hist=sub['Costo_Falla_Abast_S'].sum()
        costo_falla_6m=costo_falla_hist*6/meses_hist if meses_hist>0 else 0

        # TO-BE (punto 3 — misma fórmula: atendida/demanda)
        precision_tobe=np.mean(mc_g['fill_rate'])  # atendida_total/dem_total por iteración
        tasa_quiebre_tobe=np.mean(mc_g['dias_quiebre'])/horizonte_dias  # punto 4
        vp_und=np.mean(mc_g['vp']);vp_s=vp_und*p['Pv'];ct_tobe=np.mean(mc_g['costo_total'])
        dem_no_atendida_pct=(1-precision_tobe)*100
        ahorro_vp=costo_falla_6m-vp_s

        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown(f"### AS-IS")
            st.metric("Precisión de requerimiento",f"{precision_asis:.2%}")
            st.metric("Tasa de quiebre",f"{tasa_quiebre_asis:.2%}")
            st.metric(f"Costo falla ({meses_hist}m)",fmt_s(costo_falla_hist))
        with c2:
            st.markdown(f"### TO-BE ({pol_gan})")
            st.metric("Precisión de requerimiento",f"{precision_tobe:.2%}")
            st.metric("Tasa de quiebre",f"{tasa_quiebre_tobe:.2%}")
            st.metric("CT esperado (6m)",fmt_s(ct_tobe))
        with c3:
            st.markdown("### Impacto")
            dp=(precision_tobe-precision_asis)*100
            st.metric("Mejora precisión",f"+{dp:.1f} pp",delta=f"+{dp:.1f} pp")
            st.metric("Ahorro VP (6m)",fmt_s(ahorro_vp))
            st.metric("Dem. no atendida",f"{dem_no_atendida_pct:.1f}%")

        v95=np.percentile(mc_g['costo_total'],95)
        comp=pd.DataFrame([
            {'Indicador':'Precisión de requerimiento','AS-IS':f'{precision_asis:.2%}','TO-BE':f'{precision_tobe:.2%}','Cambio':f'+{dp:.1f} pp'},
            {'Indicador':'Tasa de quiebre','AS-IS':f'{tasa_quiebre_asis:.2%}','TO-BE':f'{tasa_quiebre_tobe:.2%}',
             'Cambio':f'{(tasa_quiebre_tobe-tasa_quiebre_asis)*100:+.1f} pp'},
            {'Indicador':'Fill rate TO-BE','AS-IS':'No disponible','TO-BE':f'{precision_tobe:.2%}','Cambio':'—'},
            {'Indicador':'Días con quiebre (6m)','AS-IS':f'{quiebres_hist}','TO-BE':f'{np.mean(mc_g["dias_quiebre"]):.1f}','Cambio':'—'},
            {'Indicador':'VP (und)','AS-IS':'—','TO-BE':f'{vp_und:,.0f}','Cambio':'—'},
            {'Indicador':'Costo falla/VP (6m)','AS-IS':fmt_s(costo_falla_6m),'TO-BE':fmt_s(vp_s),'Cambio':fmt_s(ahorro_vp)},
            {'Indicador':'VaR 95%','AS-IS':'—','TO-BE':fmt_s(v95),'Cambio':'—'},
            {'Indicador':'CVaR 95%','AS-IS':'—','TO-BE':fmt_s(np.mean(mc_g['costo_total'][mc_g['costo_total']>=v95])),'Cambio':'—'},
        ])
        st.dataframe(comp,hide_index=True,use_container_width=True)

        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure([go.Bar(x=['AS-IS','TO-BE'],y=[precision_asis*100,precision_tobe*100],
                marker_color=['#C62828','#2E7D32'],text=[f'{precision_asis:.1%}',f'{precision_tobe:.1%}'],textposition='auto')])
            fig.update_layout(title='Precisión de requerimiento',template='plotly_white',height=300,yaxis_title='%');st.plotly_chart(fig,use_container_width=True,key=f"c_prc_11_{nc}")
        with c2:
            fig=go.Figure([go.Bar(x=['AS-IS','TO-BE'],y=[costo_falla_6m,vp_s],
                marker_color=['#C62828','#2E7D32'],text=[fmt_s(costo_falla_6m),fmt_s(vp_s)],textposition='auto')])
            fig.update_layout(title='Costo falla / VP (6m)',template='plotly_white',height=300);st.plotly_chart(fig,use_container_width=True,key=f"c_cfv_12_{nc}")

    # ROI (punto 19)
    st.markdown("---");st.subheader("📊 Módulo ROI (editable)")
    st.caption("⚠️ Valores referenciales editables, no validados. Serán investigados.")
    ahorro_total=sum((p['sub_df']['Costo_Falla_Abast_S'].sum()*6/p['sub_df']['Fecha'].dt.to_period('M').nunique()
        -np.mean(all_results[nc]['mc'][ganadoras[nc]['politica']]['vp'])*p['Pv']) for nc,p in productos.items())
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Inversión**")
        dev=st.number_input("Desarrollo (S/)",value=0.0,min_value=0.0,step=500.0,key='r1')
        cap=st.number_input("Capacitación (S/)",value=0.0,min_value=0.0,step=100.0,key='r2')
        mig=st.number_input("Migración datos (S/)",value=0.0,min_value=0.0,step=100.0,key='r3')
        otr=st.number_input("Otros (S/)",value=0.0,min_value=0.0,step=100.0,key='r4')
        cont=st.number_input("Contingencia (S/)",value=0.0,min_value=0.0,step=100.0,key='r5')
    with c2:
        st.markdown("**Operativos anuales**")
        host=st.number_input("Hosting/mes (S/)",value=0.0,min_value=0.0,step=10.0,key='r6')
        git=st.number_input("GitHub/año (S/)",value=0.0,min_value=0.0,step=50.0,key='r7')
        mant=st.number_input("Mantenimiento/año (S/)",value=0.0,min_value=0.0,step=100.0,key='r8')
        ahorro_inst.number_input("Ahorro anual (S/)",value=max(0.0,round(float(ahorro_total)*2,0)),min_value=0.0,step=1000.0,key='r9')
    inv=dev+cap+mig+otr+cont;cop=host*12+git+mant;ben=ahorro_in-cop
    roi_v=((ben-inv)/inv*100) if inv>0 else 0;pay=(inv/(ben/12)) if ben>0 else float('inf')
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Inversión",fmt_s(inv));c2.metric("Beneficio neto/año",fmt_s(ben))
    c3.metric("ROI",f"{roi_v:.1f}%");c4.metric("Payback",f"{pay:.1f}m" if pay<999 else "N/A")
    esc=[]
    for nm,f in [('Bajo',0.7),('Medio',1.0),('Alto',1.3)]:
        ah=ahorro_in*f;bn=ah-cop;r=((bn-inv)/inv*100) if inv>0 else 0;pb=(inv/(bn/12)) if bn>0 else float('inf')
        esc.append({'Escenario':nm,'Ahorro':fmt_s(ah),'Beneficio':fmt_s(bn),'ROI':f'{r:.1f}%','Payback':f'{pb:.1f}m' if pb<999 else 'N/A'})
    st.dataframe(pd.DataFrame(esc),hide_index=True,use_container_width=True)

# #####################################################################
# TAB 4: KANBAN (adaptado a ganadora — punto 9)
# #####################################################################
with tab4:
    st.header("🚦 Kanban")
    if 'sim_done' not in st.session_state: st.warning("Ejecuta la simulación."); st.stop()
    all_results=st.session_state['all_results'];ganadoras=st.session_state['ganadoras']
    c1,c2=st.columns(2)
    with c1: dia_sel=st.slider("Día",1,horizonte_dias,1,1)
    with c2: margen_pct=st.slider("Margen preventivo (%)",10,50,25,5);margen=margen_pct/100
    fecha_sim=FECHA_INICIO_SIM+pd.Timedelta(days=dia_sel-1)
    st.markdown(f"**{fecha_sim.strftime('%d-%b-%Y')}** — Día {dia_sel}/{horizonte_dias}")
    st.caption("🔴 Inv≤SS ó BL>0 ó VP>0 ó quiebre · 🟡 SS<Inv≤SS×(1+margen) · 🟢 Inv>SS×(1+margen)")

    hists_k=[]
    for nc,p in productos.items():
        pol_gan=ganadoras[nc]['politica'];cfg=obtener_config_politica(p,pol_gan)
        sim=all_results[nc]['det'][pol_gan]['sim']
        SS=cfg['SS'];S_val=cfg['S'] if cfg['S'] else cfg['s'];T_val=cfg['T']
        umbral=SS*(1+margen);f=sim.iloc[dia_sel-1]
        est,acc=clasificar_kanban(f['inv_final'],f['bl_final'],f['vp'],f['quiebre'],SS,umbral)
        pr_d=prox_revision(dia_sel-1,T_val)
        pr_f=(FECHA_INICIO_SIM+pd.Timedelta(days=pr_d)).strftime('%d-%b-%Y') if pr_d else 'Revisión continua'
        borde='#C62828' if '🔴' in est else '#F9A825' if '🟡' in est else '#2E7D32'
        fondo=f'rgba({",".join(str(int(borde[i:i+2],16)) for i in (1,3,5))},0.05)'
        if '🔴' in est:
            st.error(f"⚠️ {nc}: {est} — Inv: {f['inv_final']:,.1f} | SS: {SS} | BL: {f['bl_final']:,.1f} | VP: {f['vp']:,.1f}")
        st.markdown(f"""<div style="background:{fondo};padding:14px;border-radius:8px;margin-bottom:10px;border-left:5px solid {borde};">
        <h3 style="margin:0">{est} — {nc} ({pol_gan})</h3>
        <table style="width:100%;font-size:0.9em;border-collapse:collapse;">
        <tr><td><b>Inv.ini:</b> {f['inv_inicio']:,.1f}</td><td><b>Demanda:</b> {f['demanda']:,.1f}</td><td><b>Atendida:</b> {f['atendida']:,.1f}</td></tr>
        <tr><td><b>Inv.final:</b> <span style="color:{borde};font-weight:bold">{f['inv_final']:,.1f}</span></td><td><b>SS:</b> {SS}</td><td><b>Umbral:</b> {umbral:.0f}</td></tr>
        <tr><td><b>{'S' if S_val else 's'}:</b> {S_val}</td><td><b>BL:</b> {f['bl_final']:,.1f}</td><td><b>VP:</b> {f['vp']:,.1f}</td></tr>
        <tr><td><b>Pedido:</b> {'Sí' if f['pedido'] else 'No'}</td><td><b>Cant:</b> {int(f['q_pedido'])}</td><td><b>Próx.rev:</b> {pr_f}</td></tr>
        </table><p style="margin:8px 0 0"><b>Acción:</b> {acc}</p></div>""",unsafe_allow_html=True)
        st.plotly_chart(graf_kanban(sim,nc,SS,S_val or SS*2,umbral,T_val,dia_sel-1,FECHA_INICIO_SIM),use_container_width=True,key=f"c_kan_13_{nc}")
        hist=construir_historial(sim,nc,SS,S_val or SS*2,T_val,margen,FECHA_INICIO_SIM);hists_k.append(hist)
        nv=hist['Estado'].str.contains('🟢').sum();na_k=hist['Estado'].str.contains('🟡').sum();nr=hist['Estado'].str.contains('🔴').sum()
        c1,c2,c3=st.columns(3)
        c1.metric(f"🟢 {nv} ({nv/len(hist)*100:.0f}%)","");c2.metric(f"🟡 {na_k} ({na_k/len(hist)*100:.0f}%)","");c3.metric(f"🔴 {nr} ({nr/len(hist)*100:.0f}%)","")
    hc=pd.concat(hists_k,ignore_index=True);st.session_state['historial_kanban']=hc
    filtro=st.selectbox("Filtrar",['Todos','🔴 Crítico','🟡 Alerta','🟢 Normal','Pedidos'])
    for nc in productos:
        with st.expander(nc):
            h=hc[hc['Producto']==nc]
            if filtro=='🔴 Crítico':h=h[h['Estado'].str.contains('🔴')]
            elif filtro=='🟡 Alerta':h=h[h['Estado'].str.contains('🟡')]
            elif filtro=='🟢 Normal':h=h[h['Estado'].str.contains('🟢')]
            elif filtro=='Pedidos':h=h[h['Pedido']==1]
            st.dataframe(h,hide_index=True,use_container_width=True,height=350)


# #####################################################################
# TAB 5: DASHBOARD (punto 10, 12, 17)
# #####################################################################
with tab5:
    st.header("📊 Dashboard ejecutivo")
    if 'sim_done' not in st.session_state: st.warning("Ejecuta la simulación."); st.stop()
    all_results=st.session_state['all_results'];ganadoras=st.session_state['ganadoras']
    dtabs=st.tabs([f"Dashboard {nc}" for nc in productos])
    for i_nc,(nc,p) in enumerate(productos.items()):
        with dtabs[i_nc]:
            pol_gan=ganadoras[nc]['politica'];cfg=obtener_config_politica(p,pol_gan)
            mc_g=all_results[nc]['mc'][pol_gan];det_g=all_results[nc]['det'][pol_gan]
            pr=st.session_state.get(f'pron_{nc}',{});v95=np.percentile(mc_g['costo_total'],95)
            sub=p['sub_df'];prec_a=sub['Cantidad_Recibida'].sum()/sub['Demanda_Ref'].sum()
            prec_t=np.mean(mc_g['fill_rate'])
            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Modelo",pr.get('modelo','—'));c2.metric("MAPE",f"{pr.get('mape',0):.2f}%")
            c3.metric("Política",pol_gan);c4.metric("Fill rate",f"{prec_t:.2%}");c5.metric("CT medio",fmt_s(np.mean(mc_g['costo_total'])))
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Precisión req.",f"{prec_t:.2%}");c2.metric("VaR 95%",fmt_s(v95))
            c3.metric("CVaR 95%",fmt_s(np.mean(mc_g['costo_total'][mc_g['costo_total']>=v95])))
            c4.metric("VP media",f"{np.mean(mc_g['vp']):,.0f} und")
            serie=series[nc]
            c1,c2=st.columns(2)
            with c1:
                if pr: st.plotly_chart(graf_pronostico(serie,pr,nc),use_container_width=True,key=f"c_pron_14_{nc}")
            with c2:
                mcp=st.session_state.get(f'mc_pron_{nc}')
                if mcp and pr:
                    ff=pd.date_range(serie.index[-1]+pd.DateOffset(months=1),periods=horizonte_fc,freq='MS')
                    st.plotly_chart(graf_mc_bandas(ff,mcp,pr['forecast'],nc),use_container_width=True,key=f"dash_hct_{nc}_{pol_key(pol_gan)}")
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(graf_diente(det_g['sim'],nc,cfg,FECHA_INICIO_SIM),use_container_width=True,key=f"c_dnt_15_{nc}")
            with c2: st.plotly_chart(graf_mc_hist(mc_g['costo_total'],nc,f'CT {pol_gan}','S/'),use_container_width=True,key=f"c_mch_16_{nc}")
            c1,c2=st.columns(2)
            with c1:
                fig=go.Figure([go.Bar(x=['AS-IS','TO-BE'],y=[prec_a*100,prec_t*100],marker_color=['#C62828','#2E7D32'],
                    text=[f'{prec_a:.1%}',f'{prec_t:.1%}'],textposition='auto')])
                fig.update_layout(title='Precisión AS-IS vs TO-BE',template='plotly_white',height=300);st.plotly_chart(fig,use_container_width=True,key=f"c_prc_17_{nc}")
            with c2:
                mc_all=all_results[nc]['mc']
                fig=go.Figure()
                for comp,color,nm in [('costo_ordenar','#1565C0','Ord'),('costo_mant','#2E7D32','Mant'),('costo_vp','#E65100','VP'),('costo_bl','#C62828','BL')]:
                    fig.add_trace(go.Bar(name=nm,x=list(mc_all.keys()),y=[np.mean(mc_all[pn][comp]) for pn in mc_all],marker_color=color))
                fig.update_layout(barmode='stack',title='Costos por política',template='plotly_white',height=300);st.plotly_chart(fig,use_container_width=True,key=f"c_stk_18_{nc}")

            # Descarga HTML resumen (punto 17)
            html=f"""<html><head><meta charset='utf-8'><title>Dashboard {nc}</title>
            <style>body{{font-family:Arial;margin:20px}}table{{border-collapse:collapse;width:100%}}
            td,th{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#1565C0;color:white}}</style></head>
            <body><h1>Dashboard Ejecutivo — {nc}</h1>
            <table><tr><th>Indicador</th><th>Valor</th></tr>
            <tr><td>Modelo</td><td>{pr.get('modelo','')}</td></tr>
            <tr><td>MAPE</td><td>{pr.get('mape',0):.2f}%</td></tr>
            <tr><td>Pronóstico acum.</td><td>{np.sum(pr.get('forecast',[])):,.0f} und</td></tr>
            <tr><td>Política ganadora</td><td>{pol_gan}</td></tr>
            <tr><td>Precisión req.</td><td>{prec_t:.2%}</td></tr>
            <tr><td>Fill rate</td><td>{prec_t:.2%}</td></tr>
            <tr><td>Tasa quiebre</td><td>{np.mean(mc_g['dias_quiebre'])/horizonte_dias:.2%}</td></tr>
            <tr><td>CT medio</td><td>{fmt_s(np.mean(mc_g['costo_total']))}</td></tr>
            <tr><td>VaR 95%</td><td>{fmt_s(v95)}</td></tr>
            <tr><td>CVaR 95%</td><td>{fmt_s(np.mean(mc_g['costo_total'][mc_g['costo_total']>=v95]))}</td></tr>
            <tr><td>VP media</td><td>{np.mean(mc_g['vp']):,.0f} und</td></tr>
            </table><p>{ganadoras[nc]['explicacion']}</p></body></html>"""
            st.download_button(f"📥 Dashboard {nc} (HTML)",html,f"dashboard_{nc}.html","text/html",key=f"dl_dash_{nc}")

# #####################################################################
# TAB 6: DESCARGAR (punto 18)
# #####################################################################
with tab6:
    st.header("📥 Descargar resultados")
    if 'sim_done' not in st.session_state: st.warning("Ejecuta la simulación."); st.stop()
    all_results=st.session_state['all_results'];ganadoras=st.session_state['ganadoras']
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='xlsxwriter') as w:
        # Resumen ejecutivo
        res_ej=[]
        for nc,p in productos.items():
            pr=st.session_state.get(f'pron_{nc}',{});pg=ganadoras[nc]['politica']
            mc_g=all_results[nc]['mc'][pg];v95=np.percentile(mc_g['costo_total'],95)
            sub=p['sub_df'];mh=sub['Fecha'].dt.to_period('M').nunique()
            cf6=sub['Costo_Falla_Abast_S'].sum()*6/mh if mh>0 else 0
            vps=np.mean(mc_g['vp'])*p['Pv']
            res_ej.append({'Producto':nc,'Modelo':pr.get('modelo',''),'MAPE%':pr.get('mape',0),
                'RMSE':pr.get('rmse',0),'Pronóstico acum.':np.sum(pr.get('forecast',[])),
                'Política ganadora':pg,'Precisión req. AS-IS':sub['Cantidad_Recibida'].sum()/sub['Demanda_Ref'].sum(),
                'Precisión req. TO-BE':np.mean(mc_g['fill_rate']),
                'Tasa quiebre AS-IS':sub['Quiebre_Stock'].sum()/len(sub),
                'Tasa quiebre TO-BE':np.mean(mc_g['dias_quiebre'])/horizonte_dias,
                'VP media (und)':np.mean(mc_g['vp']),'CT medio':np.mean(mc_g['costo_total']),
                'VaR95':v95,'CVaR95':np.mean(mc_g['costo_total'][mc_g['costo_total']>=v95]),
                'Ahorro 6m':cf6-vps,'Explicación':ganadoras[nc]['explicacion']})
        pd.DataFrame(res_ej).to_excel(w,sheet_name='Resumen_Ejecutivo',index=False)

        # Pronóstico por producto
        for nc in productos:
            pr=st.session_state.get(f'pron_{nc}')
            if pr:
                serie=series[nc]
                pd.DataFrame({'Mes':[d.strftime('%b-%Y') for d in serie.index],'Demanda':serie.values}).to_excel(w,sheet_name=f'Pronostico_{nc[:8]}',index=False)

        # Validación
        val_rows=[]
        for nc in productos:
            pr=st.session_state.get(f'pron_{nc}')
            if pr:
                ft=series[nc].index[pr['n_train']:];bp=pr['ranking'][0]['pred']
                for i in range(len(pr['test'])):
                    rl=pr['test'][i];pred=bp[i]
                    val_rows.append({'Producto':nc,'Mes':ft[i].strftime('%b-%Y'),'Real':rl,'Predicción':pred,
                        'Error':abs(rl-pred),'APE%':abs(rl-pred)/rl*100 if rl!=0 else 0})
        if val_rows: pd.DataFrame(val_rows).to_excel(w,sheet_name='Validacion_Pronostico',index=False)

        # Ranking modelos
        rk_rows=[]
        for nc in productos:
            pr=st.session_state.get(f'pron_{nc}')
            if pr:
                for i,r in enumerate(pr['ranking']): rk_rows.append({'Producto':nc,'#':i+1,'Método':r['Método'],'MAPE':r['MAPE'],'RMSE':r['RMSE']})
        if rk_rows: pd.DataFrame(rk_rows).to_excel(w,sheet_name='Ranking_Modelos',index=False)

        # Top10 futuro
        t10=[]
        for nc in productos:
            pr=st.session_state.get(f'pron_{nc}')
            if pr:
                ff=pd.date_range(series[nc].index[-1]+pd.DateOffset(months=1),periods=horizonte_fc,freq='MS')
                for nombre,fc in pr.get('top10_futuro',{}).items():
                    for t in range(len(fc)): t10.append({'Producto':nc,'Modelo':nombre,'Mes':ff[t].strftime('%b-%Y'),'Pronóstico':fc[t]})
        if t10: pd.DataFrame(t10).to_excel(w,sheet_name='Top10_Futuro',index=False)

        # MC pronóstico
        mcp_rows=[]
        for nc in productos:
            mcp=st.session_state.get(f'mc_pron_{nc}')
            if mcp:
                ff=pd.date_range(series[nc].index[-1]+pd.DateOffset(months=1),periods=horizonte_fc,freq='MS')
                for t,ind in enumerate(mcp['indicadores']):
                    row={'Producto':nc,'Mes':ff[t].strftime('%b-%Y')};row.update(ind);mcp_rows.append(row)
        if mcp_rows: pd.DataFrame(mcp_rows).to_excel(w,sheet_name='MC_Pronostico',index=False)

        # Mensual ganadora
        for nc in productos:
            pg=ganadoras[nc]['politica']
            all_results[nc]['det'][pg]['mensual'].to_excel(w,sheet_name=f'Mensual_{nc[:8]}',index=False)

        # Políticas unidades + costos
        pu=[];pc=[]
        for nc in productos:
            mc=all_results[nc]['mc']
            for pn in mc:
                m=mc[pn];ct=m['costo_total'];v95=np.percentile(ct,95)
                pu.append({'Producto':nc,'Política':pn,'Dem_total':np.mean(m['dem_total']),
                    'Atendida':np.mean(m['atendida_total']),'VP':np.mean(m['vp']),
                    'BL_max':np.mean(m['bl_max']),'Inv_prom':np.mean(m['inv_prom']),
                    'Pedidos':np.mean(m['pedidos']),'Dias_quiebre':np.mean(m['dias_quiebre']),
                    'FR':np.mean(m['fill_rate'])})
                pc.append({'Producto':nc,'Política':pn,'C_ord':np.mean(m['costo_ordenar']),
                    'C_mant':np.mean(m['costo_mant']),'C_vp':np.mean(m['costo_vp']),
                    'C_bl':np.mean(m['costo_bl']),'CT_medio':np.mean(ct),'CT_std':np.std(ct),
                    'VaR95':v95,'CVaR95':np.mean(ct[ct>=v95])})
        pd.DataFrame(pu).to_excel(w,sheet_name='Politicas_Unidades',index=False)
        pd.DataFrame(pc).to_excel(w,sheet_name='Politicas_Costos',index=False)

        # Ranking políticas + ganadora
        rp=[]
        for nc in productos:
            for r in ganadoras[nc]['ranking']:
                row={'Producto':nc};row.update(r);rp.append(row)
        pd.DataFrame(rp).to_excel(w,sheet_name='Ranking_Politicas',index=False)
        pg_rows=[{'Producto':nc,'Política':ganadoras[nc]['politica'],'Explicación':ganadoras[nc]['explicacion']} for nc in productos]
        pd.DataFrame(pg_rows).to_excel(w,sheet_name='Politica_Ganadora',index=False)

        # AS-IS vs TO-BE
        at_rows=[]
        for nc,p in productos.items():
            sub=p['sub_df'];pg=ganadoras[nc]['politica'];mc_g=all_results[nc]['mc'][pg]
            mh=sub['Fecha'].dt.to_period('M').nunique()
            at_rows.append({'Producto':nc,'Prec_ASIS':sub['Cantidad_Recibida'].sum()/sub['Demanda_Ref'].sum(),
                'Prec_TOBE':np.mean(mc_g['fill_rate']),'Tasa_Q_ASIS':sub['Quiebre_Stock'].sum()/len(sub),
                'Tasa_Q_TOBE':np.mean(mc_g['dias_quiebre'])/horizonte_dias,
                'CF_ASIS_6m':sub['Costo_Falla_Abast_S'].sum()*6/mh if mh>0 else 0,
                'VP_TOBE_S':np.mean(mc_g['vp'])*p['Pv'],'CT_TOBE':np.mean(mc_g['costo_total'])})
        pd.DataFrame(at_rows).to_excel(w,sheet_name='ASIS_TOBE',index=False)

        # ROI
        pd.DataFrame([{'Inversión':inv,'Costo_op_anual':cop,'Ahorro_anual':ahorro_in,'Beneficio_neto':ben,'ROI%':roi_v,'Payback_meses':pay}]).to_excel(w,sheet_name='ROI',index=False)

        # Parámetros
        prm=[]
        for nc,p in productos.items():
            prm.append({'Producto':nc,'d':p['d'],'sigma_d':p['sigma_d'],'L':p['L'],'sigma_L':p['sigma_L'],
                'C':p['C'],'K':p['K'],'H':p['H'],'z':p['z'],'Q':p['QS']['Q'],'SS_QS':p['QS']['SS'],
                's':p['QS']['s'],'S_TS7':p['TS'][7]['S'],'SS_TS7':p['TS'][7]['SS'],'S_sS':p['sS']['S']})
        pd.DataFrame(prm).to_excel(w,sheet_name='Parametros',index=False)

        # Historial Kanban
        hk=st.session_state.get('historial_kanban')
        if hk is not None: hk.to_excel(w,sheet_name='Historial_Kanban',index=False)
        else:
            hists=[]
            for nc,p in productos.items():
                pg=ganadoras[nc]['politica'];cfg=obtener_config_politica(p,pg)
                sim=all_results[nc]['det'][pg]['sim']
                hists.append(construir_historial(sim,nc,cfg['SS'],cfg['S'] or cfg['s'],cfg['T'],0.25,FECHA_INICIO_SIM))
            pd.concat(hists,ignore_index=True).to_excel(w,sheet_name='Historial_Kanban',index=False)

    st.download_button("📥 Descargar Excel completo",buf.getvalue(),"RESULTADOS_UTM_TESIS.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("17 hojas: Resumen_Ejecutivo, Pronóstico×producto, Validación, Ranking_Modelos, Top10_Futuro, "
        "MC_Pronostico, Mensual×producto, Politicas_Unidades, Politicas_Costos, Ranking_Politicas, "
        "Politica_Ganadora, ASIS_TOBE, ROI, Parametros, Historial_Kanban.")

