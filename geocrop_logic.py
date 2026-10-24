"""GeoCrop AI - application logic (no Streamlit imports, so it can be unit-tested outside the app)."""
import numpy as np, pandas as pd
import geocrop_core as gc

CLASS_ORDER = {'No reference': -1, 'Unsuitable': 0, 'Marginal': 1, 'Moderate': 2, 'High': 3}
FACTORS = {'temperature': ('TMIN', 'TOPMN', 'TOPMX', 'TMAX', 'deg C (season mean)', 'temp_mean'),
           'rainfall': ('RMIN', 'ROPMN', 'ROPMX', 'RMAX', 'mm (annual)', 'rain_annual_mm'),
           'soil pH': ('PHMIN', 'PHOPMN', 'PHOPMX', 'PHMAX', 'pH (0-30 cm)', 'soil_ph')}

def site_row(dist_row, clim_row, static, dyn, season, overrides=None):
    """One-row feature frame for Layer A. overrides: {feature: value} for what-if analysis."""
    d = {**{c: dist_row[c] for c in static}, **{c: clim_row[c] for c in dyn}, 'season': season}
    d.update(overrides or {})
    return pd.DataFrame([d])

def occurrence_parts(pipe, cfg, tables, state, season, row):
    """Components of the blend, for display only: state x season history share and the Extra-Trees probability."""
    crops = list(cfg['crops']); nC = len(crops)
    et = np.zeros(nC); et[np.array(cfg['present_class_codes'])] = pipe.predict_proba(row[cfg['numeric_features'] + ['season']])[0]
    ss = tables['state_season']; m = ss[(ss.state == state) & (ss.season == season)]
    hist = np.full(nC, np.nan)
    if len(m): hist = np.zeros(nC); hist[m.y_crop.values] = m.p.values
    return pd.DataFrame({'history_share': hist, 'extra_trees': et}, index=crops)

def recommend(pipe, cfg, tables, req, rule, obs_local, state, season, row):
    """Table over all crops: occurrence score (Layer A) next to EcoCrop suitability (rule engine) and local evidence."""
    occ, mode = gc.occurrence_scores(pipe, cfg, tables, state, season, row)
    es = gc.eco_suitability(req, rule, row.temp_mean.iloc[0], row.rain_annual_mm.iloc[0], row.soil_ph.iloc[0], row.tmin_abs.iloc[0])
    res = pd.DataFrame({'occurrence': occ}).join(es[['si', 's_temp', 's_rain', 's_ph', 'suit_class', 'limiting_factor']])
    if obs_local is not None and len(obs_local):
        res = res.join(obs_local.set_index('crop')[['years_recorded', 'mean_yield_t_ha']])
    for c in ['years_recorded', 'mean_yield_t_ha']:
        if c not in res: res[c] = np.nan
    res['suit_class'] = res.suit_class.fillna('No reference')
    res['limiting_factor'] = res.limiting_factor.fillna('-')
    res['rank_occurrence'] = res.occurrence.rank(ascending=False, method='first').astype(int)
    return res.rename_axis('crop').reset_index(), mode

def buckets(res, n=10):
    """Three transparent lists; none of them is a combined score."""
    top = res.nsmallest(n, 'rank_occurrence')
    ok = top[top.suit_class.isin(['Moderate', 'High'])]
    caution = top[top.suit_class.isin(['Unsuitable', 'Marginal'])]
    opp = res[(res.suit_class == 'High') & (res.rank_occurrence > n)].sort_values('si', ascending=False).head(n)
    return ok, caution, opp

def explain_table(req, rule, crop, values):
    """Local value vs the crop's EcoCrop limits for the three agro-climatic factors. values: {'temp_mean':..,'rain_annual_mm':..,'soil_ph':..}"""
    r = req.set_index('crop').loc[crop]; ru = rule.reindex([crop]).iloc[0]
    used = {'temperature': bool(ru.use_temp) if pd.notna(ru.use_temp) else True, 'rainfall': bool(ru.use_rain) if pd.notna(ru.use_rain) else True,
            'soil pH': bool(ru.use_ph) if pd.notna(ru.use_ph) else True}
    rows = []
    for f, (a, b, c, d, unit, col) in FACTORS.items():
        x = values[col]; s = float(gc.trap(x, r[a], r[b], r[c], r[d]))
        rows.append({'factor': f, 'unit': unit, 'local value': round(float(x), 2), 'absolute range': f'{r[a]:g} - {r[d]:g}',
                     'optimum range': f'{r[b]:g} - {r[c]:g}', 'score (0-1)': round(s, 3),
                     'used in index': 'yes' if used[f] else 'no (>=50% of recorded data fall outside the limits)'})
    return pd.DataFrame(rows)

def curve(req, crop, factor, n=200):
    r = req.set_index('crop').loc[crop]; a, b, c, d = [r[k] for k in FACTORS[factor][:4]]
    span = d - a; x = np.linspace(a - 0.15 * span, d + 0.15 * span, n)
    return x, gc.trap(x, a, b, c, d)

def suitability_map(clim, dist, req, rule, crop, season, obs):
    """EcoCrop suitability of one crop in one season for every district that has climatology for it."""
    cl = clim[clim.season == season].merge(dist[['district_key', 'state', 'district', 'lat', 'lon', 'soil_ph']], on='district_key')
    cl = cl.reset_index(drop=True)
    r = req.set_index('crop').reindex([crop] * len(cl)); r.index = cl.index
    ru = rule.reindex([crop] * len(cl)); ru.index = cl.index
    out = gc._score(r, ru, cl.temp_mean.values, cl.rain_annual_mm.values, cl.soil_ph.values, cl.tmin_abs.values)
    cl = cl.join(out[['si', 'suit_class', 'limiting_factor']])
    seen = obs[(obs.crop == crop) & (obs.season == season)][['district_key', 'years_recorded']]
    cl = cl.merge(seen, on='district_key', how='left')
    cl['recorded'] = np.where(cl.years_recorded.notna(), 'recorded', 'not recorded')
    return cl
