"""GeoCrop AI - scoring core shared by the notebook smoke test and the Streamlit app (no Streamlit imports here)."""
import numpy as np, pandas as pd

LAB = ['Unsuitable', 'Marginal', 'Moderate', 'High']

def trap(x, a, b, c, d):
    """EcoCrop trapezoid: 0 outside [a,d], 1 inside [b,c], linear in between (identical to Stage 3)."""
    x, a, b, c, d = [np.asarray(v, float) for v in (x, a, b, c, d)]
    with np.errstate(divide='ignore', invalid='ignore'):
        up = np.where(b > a, (x - a) / (b - a), (x >= a).astype(float))
        dn = np.where(d > c, (d - x) / (d - c), (x <= d).astype(float))
    return np.clip(np.minimum(up, dn), 0, 1)

def classify(si):
    si = np.asarray(si, float)
    return np.select([si < 1e-9, si < 0.5, si < 0.8], [0, 1, 2], default=3)   # index into LAB

def _score(r, ru, temp_mean, rain_annual_mm, soil_ph, tmin_abs=None):
    """Vectorised EcoCrop score. r / ru are row-aligned requirement and rule tables; inputs are scalars or arrays of the same length."""
    ub = lambda c: ru[c].fillna(True).astype(bool).values
    out = pd.DataFrame(index=r.index)
    out['s_temp'] = trap(temp_mean, r.TMIN, r.TOPMN, r.TOPMX, r.TMAX)
    out['s_rain'] = trap(rain_annual_mm, r.RMIN, r.ROPMN, r.ROPMX, r.RMAX)
    out['s_ph'] = trap(soil_ph, r.PHMIN, r.PHOPMN, r.PHOPMX, r.PHMAX)
    comp = np.column_stack([np.where(ub('use_temp'), out.s_temp, 1.0), np.where(ub('use_rain'), out.s_rain, 1.0),
                            np.where(ub('use_ph'), out.s_ph, 1.0)])
    si = comp.min(axis=1)
    kill = np.zeros(len(r), bool)
    if tmin_abs is not None:
        kill = (np.asarray(tmin_abs, float) < r.KTMP.values) & r.KTMP.notna().values & ub('use_frost')
        si = np.where(kill, 0.0, si)
    out['frost_kill'] = kill; out['si'] = si
    out['suit_class'] = [LAB[i] for i in classify(si)]
    out['limiting_factor'] = np.where(si >= 0.999, '-', np.array(['temperature', 'rainfall', 'soil pH'])[comp.argmin(axis=1)])
    out.loc[kill, 'limiting_factor'] = 'frost'
    return out

def eco_suitability(req, rule, temp_mean, rain_annual_mm, soil_ph, tmin_abs=None):
    """Suitability of EVERY crop at one location (scalar inputs). Returns a DataFrame indexed by crop.
    req: crop_requirements_v2; rule: stage3_component_rule (index crop; use_temp/use_rain/use_ph/use_frost)."""
    r = req.set_index('crop') if 'crop' in req.columns else req
    return _score(r, rule.reindex(r.index), temp_mean, rain_annual_mm, soil_ph, tmin_abs)

def si_for_records(df, req, rule):
    """Suitability index for observed records (columns: crop, temp_mean, rain_annual_mm, soil_ph, tmin_abs) - used to verify
    that the app reproduces the Stage 3 labels exactly."""
    r = req.set_index('crop').reindex(df.crop.values); ru = rule.reindex(df.crop.values)
    r.index = df.index; ru.index = df.index
    return _score(r, ru, df.temp_mean.values, df.rain_annual_mm.values, df.soil_ph.values, df.tmin_abs.values)

def occurrence_scores(pipe, cfg, tables, state, season, feats_row):
    """Layer A: blend of state x season prior and Extra-Trees probabilities -> Series over all crops.
    feats_row: one-row DataFrame holding cfg['numeric_features'] and 'season'."""
    crops = list(cfg['crops']); nC = len(crops)
    present = np.array(cfg['present_class_codes'])
    P_et = np.zeros(nC); P_et[present] = pipe.predict_proba(feats_row[cfg['numeric_features'] + ['season']])[0]
    def lookup(tab, **kw):
        m = tab
        for k, v in kw.items(): m = m[m[k] == v]
        v = np.zeros(nC); v[m.y_crop.values] = m.p.values
        return v if m.shape[0] else None
    glob = tables['global'].reindex(range(nC)).fillna(0).values
    p_season = lookup(tables['season'], season=season)
    p_season = glob if p_season is None else p_season
    p_ss = lookup(tables['state_season'], state=state, season=season)
    alpha = cfg['alpha_state_season_prior']
    if p_ss is None:                       # state unseen in training -> ET only (validated in leave-state-out, Cell 6-3)
        P, mode = P_et, 'extra-trees only (no training history for this state/season)'
    else:
        P, mode = alpha * p_ss + (1 - alpha) * P_et, f'blend: {alpha:.1f} x state-season history + {1 - alpha:.1f} x Extra-Trees'
    return pd.Series(P, index=crops, name='occurrence_score'), mode
