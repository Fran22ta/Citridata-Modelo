import json
import numpy as np
import pandas as pd

HISTORY = ['previousSeasonVarietyYield', 'twoSeasonsAgoVarietyYield', 'historicalMeanVarietyYield']
MEDIANS = dict(zip(HISTORY, [35.664381315, 36.165105995, 37.489863321]))
COEF = {'waterDeficit': -.007, 'growingDegreeDays': .0051, 'daysAbove35C': -.5983,
        'frostRiskDays': -.4747, 'parcelArea': -.1574, HISTORY[0]: .1687,
        HISTORY[1]: .0695, HISTORY[2]: .5589}
VEG = ['meanNDVI', 'meanNDRE', 'meanNDMI', 'meanMSI', 'meanEVI2', 'meanNIRv']

def load_data(raw):
    records = json.loads(raw)
    if not isinstance(records, list) or not records or not all(isinstance(x, dict) for x in records):
        raise ValueError('El JSON debe contener una lista de registros CitrusYieldRecord.')
    records = [{k: (v.get('value') if isinstance(v, dict) and 'value' in v else v)
                for k, v in r.items()} for r in records]
    df = pd.DataFrame(records)
    required = ['season', 'parcelYield', 'parcelArea', 'referenceEvapotranspiration',
                'accumulatedPrecipitation', 'growingDegreeDays', 'daysAbove35C', 'frostRiskDays']
    missing = [k for k in required if k not in df]
    if missing:
        raise ValueError('Faltan campos: ' + ', '.join(missing))
    for k in required[1:] + HISTORY + VEG + ['varietyYield', 'parcelProduction']:
        if k not in df: df[k] = np.nan
        df[k] = pd.to_numeric(df[k], errors='coerce').replace([np.inf, -np.inf], np.nan)
    def parcel(v):
        if isinstance(v, list): return ' | '.join(sorted(map(str, v)))
        return str(v) if pd.notna(v) else 'Sin referencia'
    df['parcel'] = df.get('cadastralReference', pd.Series(index=df.index, dtype=object)).map(parcel)
    for k in ['variety', 'farmName', 'producerName']:
        if k not in df: df[k] = 'Sin dato'
        df[k] = df[k].fillna('Sin dato').astype(str)
    df['season'] = df['season'].astype(str)
    df['year'] = pd.to_numeric(df['season'].str.extract(r'(\d{4})')[0], errors='coerce')
    df['waterDeficit'] = df['referenceEvapotranspiration'] - df['accumulatedPrecipitation']
    return df

def parcel_histories(df):
    # Resolve exact previous years; ambiguous duplicate parcel/year records stay missing.
    out = df.copy()
    out[HISTORY] = np.nan
    for ref, group in df.groupby('parcel'):
        if ref == 'Sin referencia': continue
        byyear = group.groupby('year')['parcelYield'].agg(['count', 'size', 'first'])
        valid = byyear.loc[(byyear['size'] == 1) & (byyear['count'] == 1), 'first']
        for ix, row in group.iterrows():
            year = row['year']
            if pd.isna(year): continue
            out.loc[ix, HISTORY[0]] = valid.get(year - 1, np.nan)
            out.loc[ix, HISTORY[1]] = valid.get(year - 2, np.nan)
            # Do not silently choose among duplicate historical records.
            if not (byyear.loc[byyear.index < year, 'size'] > 1).any():
                out.loc[ix, HISTORY[2]] = valid.loc[valid.index < year].mean()
    return out

def predict(df, impute=False):
    out = df.copy()
    out['imputedHistory'] = out[HISTORY].isna().any(axis=1)
    X = out[list(COEF)].copy()
    if impute: X[HISTORY] = X[HISTORY].fillna(MEDIANS)
    contributions = X.mul(pd.Series(COEF))
    out['predictedParcelYield'] = 53.19 + contributions.sum(axis=1, min_count=len(COEF))
    out['residual'] = out['parcelYield'] - out['predictedParcelYield']
    return out, contributions

def metrics(df):
    pairs = df[['parcelYield', 'predictedParcelYield']].dropna()
    if pairs.empty: return {'n': 0, 'MAE': np.nan, 'RMSE': np.nan, 'R²': np.nan}
    y, p = pairs.to_numpy().T
    err = y-p
    den = ((y-y.mean())**2).sum()
    return {'n': len(y), 'MAE': np.abs(err).mean(), 'RMSE': np.sqrt((err**2).mean()),
            'R²': 1-(err**2).sum()/den if len(y)>1 and den>0 else np.nan}
