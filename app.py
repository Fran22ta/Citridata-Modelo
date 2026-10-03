from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from model import load_data, parcel_histories, predict, metrics, HISTORY, COEF, VEG

st.set_page_config(page_title='Citridata · Rendimiento', page_icon='🍊', layout='wide')
st.markdown('''<style>
.stApp {background: #f5f8f6;} h1,h2,h3 {color: #174f43;}
[data-testid="stMetric"] {background:white;border:1px solid #dde8e2;border-radius:14px;padding:16px;}
</style>''', unsafe_allow_html=True)
st.title('🍊 Citridata · Rendimiento y clima')
st.caption('Explora campañas, parcelas y variedades · Modelo lineal con déficit ETo − precipitación')
with st.sidebar:
    st.header('Datos y selección')
    upload = st.file_uploader('Cargar JSON', type=['json'])
    source = st.radio('Históricos del modelo', ['Variedad: campos del JSON', 'Parcela: calcular campañas anteriores'])
    policy = st.radio('Históricos ausentes', ['Excluir predicciones incompletas', 'Sustituir por medianas acordadas'])
try:
    raw = upload.getvalue() if upload else Path(__file__).with_name('datos.json').read_bytes()
    original = load_data(raw)
except FileNotFoundError:
    st.info('Carga tu JSON en el panel lateral para comenzar.'); st.stop()
except (ValueError, TypeError) as exc:
    st.error(f'No se pudo cargar el archivo: {exc}'); st.stop()
base = parcel_histories(original) if source.startswith('Parcela') else original
# Calculate histories BEFORE filtering, to preserve prior seasons.
full, contributions = predict(base, policy.startswith('Sustituir'))
f = full.copy()
with st.sidebar:
    for field, label in [('season','Campañas'), ('farmName','Fincas'), ('variety','Variedades'), ('parcel','Parcelas')]:
        options = sorted(f[field].unique())
        selected = st.multiselect(label, options, default=options)
        f = f[f[field].isin(selected)]
    comparable = st.checkbox('Mostrar solo pares observado/predicho', False)
    if comparable: f = f.dropna(subset=['parcelYield','predictedParcelYield'])
    st.caption('Media aritmética por registro; las bandas muestran desviación típica muestral, no incertidumbre predictiva.')
if f.empty:
    st.info('No hay registros con esta selección. Amplía los filtros.'); st.stop()

def chart(fig, key):
    fig.update_layout(template='plotly_white', font=dict(family='Arial',color='#24443b'),
                      colorway=['#177d68','#f2a33b','#4a7ab7','#a96591'],
                      margin=dict(l=30,r=30,t=50,b=40), legend_title_text='')
    st.plotly_chart(fig, width='stretch', key=key)

def fmt(v): return '—' if pd.isna(v) else f'{v:.2f}'

m = metrics(f)
cols = st.columns(5)
for c, label, val in zip(cols, ['Registros','Predicciones disponibles','MAE (t/ha)','RMSE (t/ha)','R²'],
                         [str(len(f)),str(f.predictedParcelYield.notna().sum()),fmt(m['MAE']),fmt(m['RMSE']),fmt(m['R²'])]):
    c.metric(label,val)
st.caption(f"Métricas descriptivas sobre {m['n']} pares disponibles. No representan validación independiente del modelo.")
tabs = st.tabs(['Rendimiento','Ajuste y residuos','Clima','Vegetación','Simulador','Datos y modelo'])
with tabs[0]:
    st.subheader('Rendimiento por campaña · media ± desviación típica')
    g = f.groupby('season').agg(observado=('parcelYield','mean'), sd=('parcelYield','std'),
                               predicho=('predictedParcelYield','mean'), sd_pred=('predictedParcelYield','std'), n=('parcelYield','count')).reset_index()
    fig = go.Figure()
    for name, col, sd, color in [('Observado','observado','sd','#177d68'), ('Predicho','predicho','sd_pred','#f2a33b')]:
        fig.add_trace(go.Scatter(x=g.season, y=g[col], name=name, mode='lines+markers',
                                line=dict(color=color),error_y=dict(type='data',array=g[sd],visible=True)))
    fig.update_layout(yaxis_title='Rendimiento (t/ha)',xaxis_title='Campaña')
    chart(fig,'yield')
    st.caption('Para comparar medias sobre los mismos registros, activa «Mostrar solo pares observado/predicho». Con un único dato, la desviación típica no está definida.')
    chart(px.box(f,x='variety',y='parcelYield',color='variety',points='all',labels={'parcelYield':'Rendimiento observado (t/ha)','variety':'Variedad'}),'box')
    st.dataframe(g,hide_index=True,width='stretch')
with tabs[1]:
    pairs=f.dropna(subset=['parcelYield','predictedParcelYield'])
    if pairs.empty: st.info('No hay pares completos para evaluar.')
    else:
        left,right=st.columns(2)
        with left:
            fig=px.scatter(pairs,x='parcelYield',y='predictedParcelYield',color='variety',hover_data=['parcel','season'],labels={'parcelYield':'Observado (t/ha)','predictedParcelYield':'Predicho (t/ha)'})
            lo=min(pairs.parcelYield.min(),pairs.predictedParcelYield.min()); hi=max(pairs.parcelYield.max(),pairs.predictedParcelYield.max())
            fig.add_shape(type='line',x0=lo,y0=lo,x1=hi,y1=hi,line=dict(color='gray',dash='dash'))
            chart(fig,'scatter')
        with right:
            fig=px.scatter(pairs,x='predictedParcelYield',y='residual',color='season',hover_data=['parcel'],labels={'residual':'Residuo: observado − predicho (t/ha)','predictedParcelYield':'Predicho (t/ha)'})
            fig.add_hline(y=0,line_dash='dash'); chart(fig,'resid')
        chart(px.histogram(pairs,x='residual',nbins=25,labels={'residual':'Residuo (t/ha)'}),'hist')
with tabs[2]:
    climate=['referenceEvapotranspiration','accumulatedPrecipitation','waterDeficit','growingDegreeDays','daysAbove35C','frostRiskDays']
    cg=f.groupby('season')[climate].mean().reset_index()
    water=cg.melt('season',value_vars=climate[:3],var_name='Variable',value_name='mm')
    water.Variable=water.Variable.replace({'referenceEvapotranspiration':'ETo','accumulatedPrecipitation':'Precipitación','waterDeficit':'ETo − precipitación'})
    chart(px.line(water,x='season',y='mm',color='Variable',markers=True,labels={'season':'Campaña'}),'water')
    choose=st.selectbox('Indicador climático',['growingDegreeDays','daysAbove35C','frostRiskDays'])
    joint=f.groupby('season').agg(yield_mean=('parcelYield','mean'),climate_mean=(choose,'mean')).reset_index()
    fig=make_subplots(specs=[[{'secondary_y':True}]])
    fig.add_trace(go.Scatter(x=joint.season,y=joint.yield_mean,name='Rendimiento (t/ha)',mode='lines+markers'),secondary_y=False)
    fig.add_trace(go.Bar(x=joint.season,y=joint.climate_mean,name=choose,opacity=.45),secondary_y=True)
    fig.update_yaxes(title_text='Rendimiento (t/ha)',secondary_y=False)
    fig.update_yaxes(title_text='°C·día' if choose=='growingDegreeDays' else 'Días',secondary_y=True)
    chart(fig,'dual');st.dataframe(cg,hide_index=True,width='stretch')
with tabs[3]:
    vi=st.selectbox('Índice de vegetación',VEG)
    if f[vi].notna().sum()==0: st.info('No hay datos de este índice.')
    else:
        vg=f.groupby('season')[vi].agg(['mean','std']).reset_index()
        chart(px.line(vg,x='season',y='mean',error_y='std',markers=True,labels={'mean':vi,'season':'Campaña'}),'vegseason')
        chart(px.scatter(f,x=vi,y='parcelYield',color='season',hover_data=['parcel','variety'],labels={'parcelYield':'Rendimiento observado (t/ha)'}),'vegscatter')
        rows=[]
        for v in VEG:
            p=f[[v,'parcelYield']].dropna()
            rows.append({'Índice':v,'Pares':len(p),'Pearson r':p[v].corr(p.parcelYield) if len(p)>2 and p[v].std()>0 and p.parcelYield.std()>0 else np.nan})
        correlations=pd.DataFrame(rows)
        correlations['|r|']=correlations['Pearson r'].abs()
        st.dataframe(correlations.sort_values('|r|',ascending=False),hide_index=True)
        st.caption('Correlación exploratoria de los registros seleccionados; no demuestra causalidad ni capacidad predictiva fuera de muestra. Los índices no forman parte de esta ecuación.')
with tabs[4]:
    st.subheader('Escenario de una parcela')
    labels=(f.season+' · '+f.parcel+' · '+f.variety).to_dict()
    ix=st.selectbox('Registro de partida',list(f.index),format_func=lambda i:labels[i])
    row=base.loc[ix].copy()
    st.caption('Los cambios afectan únicamente al escenario; no modifican el JSON ni reajustan los coeficientes.')
    inputs={};cs=st.columns(3)
    for i,k in enumerate(COEF):
        default=row[k]
        if pd.isna(default): default=0.0
        inputs[k]=cs[i%3].number_input(k,value=float(default),key='scenario_'+k)
    estimate=53.19+sum(COEF[k]*inputs[k] for k in COEF)
    st.metric('Rendimiento del escenario (t/ha)',fmt(estimate))
    st.caption('Los campos ausentes del registro se muestran inicialmente como 0: introduce un valor antes de interpretar el escenario.')
    c=pd.DataFrame({'Variable':['Intercepto']+list(COEF),'Contribución (t/ha)':[53.19]+[COEF[k]*inputs[k] for k in COEF]})
    chart(px.bar(c,x='Contribución (t/ha)',y='Variable',orientation='h',color='Contribución (t/ha)',color_continuous_scale='Tealrose'),'contrib')
    st.caption('Estas contribuciones dependen del valor y las unidades; no son un ranking de importancia estadística.')
with tabs[5]:
    st.subheader('Modelo y calidad de datos')
    st.code('Yield = 53.19 − 0.0070 × (ETo − precipitación)\n + 0.0051 × GDD − 0.5983 × Heat35 − 0.4747 × Frost\n − 0.1574 × parcelArea + 0.1687 × Yieldₜ₋₁\n + 0.0695 × Yieldₜ₋₂ + 0.5589 × MeanYield')
    st.write('Fuente de históricos:',source,'· Tratamiento:',policy)
    st.caption('Los campos históricos del JSON son de variedad. La alternativa de parcela usa la referencia catastral, los años exactos t−1/t−2 y la media anterior al año evaluado. Las duplicidades no se resuelven escogiendo un registro arbitrario.')
    if policy.startswith('Sustituir'):
        st.info(f'{int(f.imputedHistory.sum())} registros seleccionados tienen algún histórico sustituido. Las medianas son las acordadas para el JSON original y no se recalculan al filtrar.')
    st.dataframe(f[list(COEF)].isna().sum().rename('Nulos').to_frame(),width='stretch')
    cols=['season','farmName','producerName','parcel','variety','parcelYield','predictedParcelYield','residual','imputedHistory']+list(COEF)+VEG
    st.dataframe(f[cols],hide_index=True,width='stretch')
    st.download_button('Descargar datos y predicciones CSV',f[cols].to_csv(index=False).encode('utf-8-sig'),'predicciones_citridata.csv','text/csv')
    st.caption('Se aplican coeficientes fijos. No se entrena Random Forest ni se reproducen las métricas citadas en el informe. Las predicciones negativas, si aparecen, se conservan sin recorte.')
