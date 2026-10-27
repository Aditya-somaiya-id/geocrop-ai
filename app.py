"""GeoCrop AI - Streamlit dashboard. Run: streamlit run app.py"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(
    os.environ.get("GEOCROP_APP_DIR")
    or (
        Path(__file__).resolve().parent
        if "__file__" in globals()
        else "/content/drive/MyDrive/GeoCropAI/app"
    )
)
sys.path.insert(0, str(HERE))

import geocrop_core as gc
import geocrop_logic as gl

DATA = HERE / "app_data"
st.set_page_config(page_title="GeoCrop AI", page_icon=None, layout="wide")


def log_memory(stage):
    """Write current and peak resident memory to Render logs when available."""
    try:
        status = Path("/proc/self/status").read_text()
        values = {}
        for line in status.splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                values[key] = value.strip()
        print(
            f"[MEMORY] {stage}: "
            f"RSS={values.get('VmRSS', 'unavailable')}; "
            f"peak={values.get('VmHWM', 'unavailable')}",
            flush=True,
        )
    except Exception as exc:
        print(f"[MEMORY] {stage}: unable to read memory stats ({exc})", flush=True)


# ------------------------------------------------------------------ data access (cached)
@st.cache_data(show_spinner=False)
def load_bundle():
    return dict(
        dist=pd.read_parquet(DATA / "districts.parquet"),
        clim=pd.read_parquet(DATA / "climatology.parquet"),
        obs=pd.read_parquet(DATA / "observed_crops.parquet"),
        req=pd.read_csv(DATA / "crop_requirements_v2.csv"),
        rule=pd.read_csv(DATA / "component_rule.csv", index_col=0),
        cfg=json.loads((DATA / "layerA_config.json").read_text()),
        man=json.loads((DATA / "bundle_manifest.json").read_text()),
    )


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    # joblib is imported only when the recommendation page actually needs the model.
    import joblib

    et_model = joblib.load(DATA / "models" / "layerA_et_compact.joblib")
    prior_tables = joblib.load(DATA / "models" / "layerA_prior_tables.joblib")
    log_memory("after model loading")
    return et_model, prior_tables


def report(name):
    p = DATA / "reports" / name
    if not p.exists():
        return None
    if name.endswith(".csv") and "per_class" in name:
        return pd.read_csv(p, index_col=0)
    return pd.read_csv(p) if name.endswith(".csv") else json.loads(p.read_text())


def sdf(df, **kw):
    # Works on old and new Streamlit versions.
    try:
        return st.dataframe(df, width="stretch", **kw)
    except TypeError:
        return st.dataframe(df, use_container_width=True, **kw)


def spc(fig, **kw):
    try:
        return st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        return st.plotly_chart(fig, use_container_width=True, **kw)


def image(name, caption=None):
    p = DATA / "plots" / name
    if not p.exists():
        return st.caption(f"(plot {name} not found in the bundle)")
    try:
        st.image(str(p), caption=caption, width="stretch")
    except TypeError:
        st.image(str(p), caption=caption, use_container_width=True)


def geo_scatter(df, color, hover, title, continuous=True, zoom=3.6, **kw):
    args = dict(
        lat="lat",
        lon="lon",
        color=color,
        hover_name=hover,
        zoom=zoom,
        height=560,
        title=title,
        center={"lat": 22.5, "lon": 80.0},
        **kw,
    )
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(df, map_style="open-street-map", **args)
    else:
        fig = px.scatter_mapbox(df, mapbox_style="open-street-map", **args)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ------------------------------------------------------------------ startup
log_memory("after standard imports")

B = load_bundle()
log_memory("after data bundle")

dist, clim, obs, req, rule, cfg, man = (
    B[k] for k in ["dist", "clim", "obs", "req", "rule", "cfg", "man"]
)
STATIC, DYN = man["static_features"], man["season_features"]
SEASONS = ["kharif", "rabi", "summer", "whole_year", "autumn", "winter"]
CLASS_COLORS = {
    "High": "#1a7f4b",
    "Moderate": "#8fbf4d",
    "Marginal": "#e3b341",
    "Unsuitable": "#c0392b",
    "No reference": "#999999",
}

PAGES = [
    "Overview",
    "Data explorer",
    "Crop recommendation",
    "Maps",
    "Model comparison",
    "Explainability",
    "Limitations",
]
page = st.sidebar.radio("GeoCrop AI", PAGES, key="page")

# Load Plotly only for pages that create interactive charts/maps.
if page in ("Data explorer", "Crop recommendation", "Maps", "Model comparison"):
    import plotly.express as px
    import plotly.graph_objects as go

st.sidebar.caption(
    f"{man['districts']} districts | {man['district_seasons']} district-seasons | "
    f"{len(cfg['crops'])} crops"
)

# ================================================================== Overview
if page == "Overview":
    st.title("GeoCrop AI")
    st.subheader(
        "A geo-spatial machine-learning framework for crop suitability using soil and climate"
    )
    st.markdown(
        """
This dashboard answers two **different** questions and keeps them apart:

* **What is typically grown here? (Layer A, machine learning).** An Extra-Trees model blended with each state's historical
  crop mix ranks all crops for a district and season. It is trained on district-level crop statistics, so it describes
  *historical practice*, not agronomic potential.
* **What does the environment allow? (EcoCrop rule engine).** The FAO EcoCrop temperature, rainfall and soil-pH
  requirements are applied to the district's NASA POWER climate and SoilGrids soil. The result is an index from 0 to 1 and a
  class (Unsuitable, Marginal, Moderate, High), plus the limiting factor. This is independent of whether the crop is grown.

Recommendations show both side by side. They are **not merged into one score**, because any weighting would be arbitrary.
Open *Limitations* before relying on any output."""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts", man["districts"])
    c2.metric("District-season climatologies", man["district_seasons"])
    c3.metric("Crops ranked", len(cfg["crops"]))
    c4.metric("Crops with EcoCrop reference", req.crop.nunique())
    fm = report("stage6_layerA_final_metrics.csv")
    if fm is not None:
        st.markdown("**Layer A on unseen districts and years (hold-outs)**")
        show = fm[
            fm.model.str.startswith(("FINAL", "state x season", "CEILING"))
        ][["split", "model", "hit@3", "hit@5", "prec@5", "mAP"]]
        sdf(show.round(3), hide_index=True)
    st.caption(
        "Data: India district-level crop statistics 1997-2022, NASA POWER daily climate, "
        "SoilGrids 2.0, FAO EcoCrop. Seed 42."
    )

# ================================================================== Data explorer
elif page == "Data explorer":
    st.title("Data explorer")
    t1, t2, t3, t4 = st.tabs(
        ["Districts", "Distributions", "Correlation", "Recorded crops"]
    )
    with t1:
        states = sorted(dist.state.unique())
        sel = st.multiselect("State", states, default=[])
        d = dist[dist.state.isin(sel)] if sel else dist
        sdf(d, hide_index=True, height=360)
        st.download_button(
            "Download table (CSV)",
            d.to_csv(index=False).encode(),
            "districts.csv",
            "text/csv",
        )
        spc(
            geo_scatter(
                d,
                "soil_ph",
                "district_key",
                "Soil pH by district (SoilGrids 0-30 cm)",
                color_continuous_scale="PuOr",
            )
        )
    with t2:
        c1, c2 = st.columns(2)
        season = c1.selectbox(
            "Season", [s for s in SEASONS if s in clim.season.unique()], key="dist_season"
        )
        var = c2.selectbox(
            "Variable",
            [c for c in DYN if c not in ("n_years",)],
            index=DYN.index("rain_annual_mm") if "rain_annual_mm" in DYN else 0,
        )
        cs = clim[clim.season == season]
        spc(
            px.histogram(
                cs, x=var, nbins=60,
                title=f"{var} - district climatology, {season} (n={len(cs)})",
            )
        )
        sdf(cs[DYN].describe(percentiles=[.01, .5, .99]).T.round(3))
    with t3:
        season = st.selectbox(
            "Season", [s for s in SEASONS if s in clim.season.unique()], key="corr_season"
        )
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        m = clim[clim.season == season][DYN].corr(method=method)
        spc(
            px.imshow(
                m, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                aspect="auto", height=720,
                title=f"{method.title()} correlation, district climatologies, {season}",
            )
        )
        with st.expander("Covariance matrix"):
            sdf(clim[clim.season == season][DYN].cov().round(3))
    with t4:
        o = (
            obs.groupby(["crop", "season"]).years_recorded.sum()
            .rename("district_years").reset_index()
        )
        spc(
            px.bar(
                o, x="district_years", y="crop", color="season",
                orientation="h", height=900,
                title="Recorded district-years per crop and season",
            )
        )

# ================================================================== Crop recommendation
elif page == "Crop recommendation":
    st.title("Crop recommendation")
    pipe, tables = load_models()
    c1, c2, c3 = st.columns(3)
    state = c1.selectbox("State", sorted(dist.state.unique()), key="rec_state")
    dsub = dist[dist.state == state].sort_values("district")
    dk = c2.selectbox(
        "District", dsub.district_key.tolist(),
        format_func=lambda k: k.split("|", 1)[1], key="rec_dist",
    )
    avail = [
        s for s in SEASONS
        if ((clim.district_key == dk) & (clim.season == s)).any()
    ]
    season = c3.selectbox("Season", avail, key=f"rec_season_{dk}")
    dr = dist.set_index("district_key").loc[dk]
    cr = clim[(clim.district_key == dk) & (clim.season == season)].iloc[0]
    st.caption(
        f"Climatology = mean of {int(cr.n_years)} recorded seasons "
        f"({int(cr.year_min)}-{int(cr.year_max)}); soil from SoilGrids at the district point."
    )
    with st.expander("What-if: override site conditions", expanded=False):
        o1, o2, o3, o4 = st.columns(4)
        ov = {}
        for col, (lab, v, step) in zip(
            [o1, o2, o3, o4],
            [
                ("temp_mean", "Season mean temp (C)", 0.5),
                ("rain_annual_mm", "Annual rainfall (mm)", 25.0),
                ("soil_ph", "Soil pH", 0.1),
                ("tmin_abs", "Coldest day min (C)", 0.5),
            ],
        ):
            base = float(dr[lab]) if lab in STATIC else float(cr[lab])
            x = col.number_input(
                v, value=round(base, 2), step=step,
                key=f"ov_{lab}_{dk}_{season}",
            )
            if abs(x - round(base, 2)) > 1e-9:
                ov[lab] = float(x)
        if ov:
            st.warning(
                f"Using overridden values: {ov}. The model was not trained on such combinations - treat as exploratory."
            )
    row = gl.site_row(dr, cr, STATIC, DYN, season, ov)
    ol = obs[(obs.district_key == dk) & (obs.season == season)]
    res, mode = gl.recommend(pipe, cfg, tables, req, rule, ol, state, season, row)
    st.caption("Ranking method: " + mode)
    ok, caution, opp = gl.buckets(res, 10)
    cols = [
        "crop", "occurrence", "rank_occurrence", "si", "suit_class",
        "limiting_factor", "years_recorded", "mean_yield_t_ha",
    ]
    ren = {
        "occurrence": "occurrence score",
        "rank_occurrence": "rank",
        "si": "EcoCrop index",
        "suit_class": "EcoCrop class",
        "limiting_factor": "limiting factor",
        "years_recorded": "years recorded here",
        "mean_yield_t_ha": "median yield (t/ha)",
    }
    a, b = st.columns([3, 2])
    with a:
        top = res.nsmallest(12, "rank_occurrence").sort_values("occurrence")
        fig = px.bar(
            top, x="occurrence", y="crop", orientation="h",
            color="suit_class", color_discrete_map=CLASS_COLORS,
            title="Top 12 by occurrence score (colour = EcoCrop class)", height=460,
        )
        spc(fig)
    with b:
        st.markdown("**Site inputs**")
        sdf(
            pd.DataFrame(
                {
                    "value": [
                        row.temp_mean.iloc[0], row.rain_annual_mm.iloc[0],
                        row.soil_ph.iloc[0], row.tmin_abs.iloc[0], row.elev_m.iloc[0],
                    ]
                },
                index=[
                    "season mean temp (C)", "annual rain (mm)", "soil pH",
                    "coldest day min (C)", "elevation (m)",
                ],
            ).round(2)
        )
    t1, t2, t3, t4 = st.tabs(
        [
            "Typically grown and suitable",
            "Typically grown, environment-limited",
            "Suitable but rarely recorded (hypotheses)",
            "All crops",
        ]
    )
    with t1:
        sdf(ok[cols].rename(columns=ren).round(3), hide_index=True)
    with t2:
        st.caption(
            "Commonly recorded here, yet EcoCrop rates the local climate or soil as Marginal or Unsuitable. "
            "Possible causes: irrigation, management, rule limits, or coarse district-mean inputs."
        )
        sdf(caution[cols].rename(columns=ren).round(3), hide_index=True)
    with t3:
        st.caption(
            "High EcoCrop suitability but outside the top 10 by occurrence. These are agronomic hypotheses only: "
            "economics, markets, pests and water are not modelled."
        )
        sdf(opp[cols].rename(columns=ren).round(3), hide_index=True)
    with t4:
        sort = st.radio("Sort by", ["occurrence score", "EcoCrop index"], horizontal=True)
        mn = st.select_slider(
            "Minimum EcoCrop class",
            ["No reference", "Unsuitable", "Marginal", "Moderate", "High"],
            value="No reference",
        )
        v = res[
            res.suit_class.map(gl.CLASS_ORDER) >= gl.CLASS_ORDER[mn]
        ].sort_values("occurrence" if sort.startswith("occ") else "si", ascending=False)
        sdf(v[cols].rename(columns=ren).round(3), hide_index=True)
        st.download_button(
            "Download (CSV)", v[cols].to_csv(index=False).encode(),
            f"recommendation_{dk.replace('|', '_')}_{season}.csv", "text/csv",
        )

    st.divider()
    st.subheader("Why? Explain one crop")
    crop = st.selectbox(
        "Crop", res.sort_values("rank_occurrence").crop.tolist(), key=f"why_{dk}_{season}"
    )
    rr = res.set_index("crop").loc[crop]
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            f"**EcoCrop suitability: {rr.suit_class}** (index {rr.si:.2f}; limiting factor: {rr.limiting_factor})"
            if rr.suit_class != "No reference"
            else "**No EcoCrop reference for this crop** (aggregate or unmatched species)."
        )
        if crop in set(req.crop):
            vals = {
                "temp_mean": float(row.temp_mean.iloc[0]),
                "rain_annual_mm": float(row.rain_annual_mm.iloc[0]),
                "soil_ph": float(row.soil_ph.iloc[0]),
            }
            sdf(gl.explain_table(req, rule, crop, vals), hide_index=True)
            fac = st.radio("Response curve", list(gl.FACTORS), horizontal=True, key=f"fac_{crop}")
            x, y = gl.curve(req, crop, fac)
            fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name="EcoCrop score"))
            col = gl.FACTORS[fac][5]
            fig.add_vline(x=vals[col], line_dash="dash", annotation_text="this site")
            fig.update_layout(
                height=300, xaxis_title=gl.FACTORS[fac][4], yaxis_title="score",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            spc(fig)
    with e2:
        parts = gl.occurrence_parts(pipe, cfg, tables, state, season, row).loc[crop]
        st.markdown(
            f"**Occurrence score {rr.occurrence:.3f}** (rank {int(rr.rank_occurrence)} of {len(res)})"
        )
        sdf(
            pd.DataFrame(
                {
                    "component": [
                        f"share of {state} {season} records that are {crop}",
                        "Extra-Trees probability from this site's climate, soil and location",
                    ],
                    "value": [parts.history_share, parts.extra_trees],
                }
            ).round(4),
            hide_index=True,
        )
        st.caption(
            "The history share is what the state grew; the Extra-Trees term lets this district's environment "
            "shift the ranking. See *Explainability* for which inputs matter."
        )
        if pd.notna(rr.years_recorded):
            st.info(f"Recorded in this district and season in {int(rr.years_recorded)} year(s).")
        else:
            st.info("Never recorded here in this season in the data.")

# ================================================================== Maps
elif page == "Maps":
    st.title("Maps")
    t1, t2 = st.tabs(["Crop suitability map", "District summaries"])
    with t1:
        c1, c2 = st.columns(2)
        crop = c1.selectbox(
            "Crop (EcoCrop reference available)", sorted(req.crop.unique()), key="map_crop"
        )
        season = c2.selectbox(
            "Season", [s for s in SEASONS if s in clim.season.unique()], key="map_season"
        )
        m = gl.suitability_map(clim, dist, req, rule, crop, season, obs)
        mode = st.radio(
            "Colour by",
            ["EcoCrop index", "EcoCrop class", "recorded vs not recorded"],
            horizontal=True,
        )
        if mode == "EcoCrop index":
            fig = geo_scatter(
                m, "si", "district_key", f"{crop} - EcoCrop index, {season}",
                color_continuous_scale="RdYlGn", range_color=[0, 1],
            )
        elif mode == "EcoCrop class":
            fig = geo_scatter(
                m, "suit_class", "district_key", f"{crop} - EcoCrop class, {season}",
                color_discrete_map=CLASS_COLORS,
            )
        else:
            fig = geo_scatter(
                m, "recorded", "district_key",
                f"{crop} - district-seasons where it is recorded, {season}",
                color_discrete_map={"recorded": "#1f77b4", "not recorded": "#cccccc"},
            )
        spc(fig)
        grown = m[m.recorded == "recorded"]
        st.caption(
            f"{len(grown)} of {len(m)} district-seasons record {crop}. Share rated Moderate/High among recorded: "
            f"{(grown.suit_class.isin(['Moderate', 'High']).mean() if len(grown) else float('nan')):.2f}; "
            f"among not recorded: "
            f"{(m[m.recorded != 'recorded'].suit_class.isin(['Moderate', 'High']).mean()):.2f}. "
            "Absence from statistics is not proof of unsuitability."
        )
    with t2:
        ds = report("stage4_district_summary.csv")
        if ds is None:
            st.info("District summary not found in the bundle.")
        else:
            opts = [
                c for c in ["n_crops", "mean_si", "rain", "temp", "soil_ph", "soil_oc"]
                if c in ds.columns
            ]
            v = st.selectbox("Variable", opts)
            spc(geo_scatter(ds, v, "district_key", v, color_continuous_scale="Viridis"))
        image("eda_spatial_split.png", "Spatial hold-out: red districts were never seen in training")

# ================================================================== Model comparison
elif page == "Model comparison":
    st.title("Model comparison")
    st.info(
        "All scores below are on districts and years the models never saw (spatial, temporal and "
        "spatio-temporal hold-outs). Hyper-parameter choices and the blend weight were fixed on separate data."
    )
    t1, t2, t3, t4 = st.tabs(
        [
            "Layer A: crop ranking",
            "Layer A: model leaderboard",
            "EcoCrop class prediction (Task B)",
            "Label validation",
        ]
    )
    with t1:
        fm = report("stage6_layerA_final_metrics.csv")
        if fm is not None:
            sdf(fm.round(3), hide_index=True)
        bl = report("stage5_baselines.csv")
        if bl is not None:
            with st.expander("Task A baselines (single-crop prediction)"):
                sdf(bl[bl.task == "A"].round(3), hide_index=True)
        pc = report("stage6_layerA_per_crop.csv")
        if pc is not None:
            st.markdown("**Per-crop results (pooled hold-outs)**")
            sdf(pc.round(3), hide_index=True)
        st.caption(
            "hit@k: a grown crop is in the top k. prec@k / rec@k: precision / recall of the top k "
            "against all crops grown there. mAP: mean average precision. CEILING is the best any model "
            "could do from the same inputs."
        )
    with t2:
        lb = report("stage5_taskA_leaderboard.csv")
        if lb is not None:
            st.markdown(
                "**Single-crop prediction (55 classes)** - Random Forest, Extra Trees, XGBoost, "
                "LightGBM, CatBoost, logistic regression"
            )
            sdf(lb.round(3), hide_index=True)
        lbB = report("stage5_taskB_leaderboard.csv")
        if lbB is not None:
            st.markdown("**EcoCrop class prediction**")
            sdf(lbB.round(3), hide_index=True)
        ss = report("stage6_taskA_set_ranking.csv")
        if ss is not None:
            st.markdown("**Crop-set ranking, all models**")
            sdf(ss.round(3), hide_index=True)
    with t3:
        st.warning(
            "B_full re-derives the EcoCrop rule because it sees the rule's inputs, so its ~99% measures "
            "rule reproduction, not prediction. B_strict withholds the rule inputs and their close proxies: "
            "that is the honest predictive score."
        )
        rb = report("stage6_taskB_final.json")
        if rb:
            rows = [
                {
                    "feature set": fs,
                    "hold-out": hs,
                    **{
                        k: v for k, v in m.items()
                        if k in (
                            "accuracy", "balanced_acc", "macro_f1", "macro_precision",
                            "macro_recall", "qwk", "within_one",
                        )
                    },
                }
                for fs, d in rb.items()
                for hs, m in d.items()
            ]
            sdf(pd.DataFrame(rows).round(3), hide_index=True)
        image("taskB_confusion_matrices.png")
        c1, c2 = st.columns(2)
        pcl = report("stage6_taskB_strict_per_class.csv")
        pcr = report("stage6_taskB_strict_per_crop.csv")
        if pcl is not None:
            c1.markdown("**B_strict per class**")
            c1.dataframe(pcl.round(3))
        if pcr is not None:
            c2.markdown("**B_strict per crop**")
            c2.dataframe(pcr.round(3))
    with t4:
        au = report("stage3_presence_auc.csv")
        if au is not None:
            au = au.sort_values("auc")
            spc(
                px.bar(
                    au, x="auc", y=au.columns[0], orientation="h", height=900,
                    title="Per-crop AUC: does the EcoCrop index separate districts that grow a crop "
                    "from those that do not? (0.5 = no information)",
                )
            )
            st.caption(
                f"Median AUC {au.auc.median():.2f}. Absence from the statistics is not proof of "
                "unsuitability, so this is a lower-bound signal."
            )

# ================================================================== Explainability
elif page == "Explainability":
    st.title("Explainability")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Layer A: permutation importance")
        image(
            "layerA_permutation_importance.png",
            "Drop in mAP on unseen districts when a feature group is shuffled",
        )
    with c2:
        st.subheader("Task B surrogate: SHAP")
        image("taskB_shap_importance.png", "Mean |SHAP| for the B_strict LightGBM")
    sh = report("stage7_taskB_shap_importance.csv")
    if sh is not None:
        sdf(sh.head(20).round(4))
    st.markdown(
        """
**How to read this.** Season dominates Layer A because crops are strongly seasonal. Season length, climate and location come next.
Soil and nutrient inputs matter least. Spatial hold-out tests show that much of the gain over the state-season prior can also be achieved by a
plain nearest-neighbour baseline on latitude/longitude, so the model should be read as learning regional pattern more than agronomy.
For any single recommendation, the *Why? Explain one crop* panel on the recommendation page shows the exact rule-based reasoning."""
    )
    st.subheader("Exploratory maps from the analysis")
    image("eda_maps_districts.png")
    image("eda_maps_top_crops.png")

# ================================================================== Limitations
else:
    st.title("Limitations and responsible use")
    st.markdown(
        """
1. **Occurrence is not suitability.** Layer A learns where crops were recorded. Markets, subsidies, irrigation and tradition shape those records.
2. **EcoCrop is a coarse reference.** It uses temperature, annual rainfall and soil pH only (a component is dropped for a crop if at least half of its recorded data fall outside the limits, e.g. rainfall for Jowar and Black Pepper, pH for Linseed). Irrigation can make dry areas suitable; the rule cannot see it.
3. **Independent validation is modest.** The EcoCrop index separates districts that grow a crop from those that do not with a median AUC of about 0.63, and it shows no meaningful relationship with relative yield. Treat classes as screening, not agronomic advice.
4. **Inputs are district-level.** Soil is one point per district (SoilGrids median in a small buffer); climate is NASA POWER at the same point. Within-district variation is invisible.
5. **Suitability-class ML is a surrogate.** Held back from the rule inputs it reaches about 63-77% accuracy, below an 80% target; the ~99% model only reproduces the rule. The app therefore computes suitability with the rule itself.
6. **Geographic transfer.** Much of Layer A's advantage over the state-season history comes from regional pattern; in leave-state-out tests a latitude/longitude neighbour baseline performs similarly.
7. **What-if overrides** move inputs outside the combinations seen in training; results are exploratory.
8. **Not for financial or planting decisions** without local agronomic advice. Catch-all crops (Other Pulses, Other Cereals, Other Oilseeds) and several aggregates were excluded."""
    )
