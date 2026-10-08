import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Churn Segmentation Analytics", layout="wide")

AGE = ["<30", "30-45", "46-60", "60+"]
CREDIT = ["Low", "Medium", "High"]
TENURE = ["New (0-2 yrs)", "Mid-term (3-6 yrs)", "Long-term (7-10 yrs)"]
BALANCE = ["Zero", "Low", "High"]
DIMS = {
    "Geography": ["France", "Germany", "Spain"],
    "Age Group": AGE,
    "Credit Band": CREDIT,
    "Tenure Group": TENURE,
    "Balance Segment": BALANCE,
    "Gender": ["Female", "Male"],
    "Activity": ["Active", "Inactive"],
}


# ---------- Data ingestion, validation, preparation ----------
@st.cache_data
def load():
    here = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(here, "**", "*.csv"), recursive=True)
    df = pd.read_csv(files[0])
    df = df.drop(columns=[c for c in ["Year", "Surname"] if c in df.columns])  # non-analytical fields
    hb = df.loc[df.Balance > 0, "Balance"].quantile(0.75)
    df["Age Group"] = pd.cut(df.Age, [0, 29, 45, 60, 200], labels=AGE).astype(str)
    df["Credit Band"] = pd.cut(df.CreditScore, [0, 579, 669, 900], labels=CREDIT).astype(str)
    df["Tenure Group"] = pd.cut(df.Tenure, [-1, 2, 6, 10], labels=TENURE).astype(str)
    df["Balance Segment"] = np.where(df.Balance == 0, "Zero", np.where(df.Balance >= hb, "High", "Low"))
    df["Activity"] = np.where(df.IsActiveMember == 1, "Active", "Inactive")
    return df, hb


df, hb_cut = load()
checks = {
    "Binary columns are 0/1": all(df[c].isin([0, 1]).all() for c in ["HasCrCard", "IsActiveMember", "Exited"]),
    "No missing values": df.isna().sum().sum() == 0,
    "CustomerId unique": df.CustomerId.is_unique,
    "NumOfProducts in 1-4": df.NumOfProducts.between(1, 4).all(),
    "Geography has 3 countries": df.Geography.nunique() == 3,
}

# ---------- Sidebar: segment filters ----------
st.sidebar.header("Segment filters")
sel = {k: st.sidebar.multiselect(k, v, default=v) for k, v in DIMS.items()}
f = df.copy()
for k, v in sel.items():
    f = f[f[k].isin(v)]

st.title("Customer Segmentation & Churn Pattern Analytics")
st.caption("European retail banking | filters in the sidebar update every KPI and chart")
if f.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

with st.expander("Data validation"):
    for k, v in checks.items():
        st.write(("✅ " if v else "❌ ") + k)


def churn_table(d, col, order=None):
    g = d.groupby(col).Exited.agg(["count", "sum", "mean"]).rename(
        columns={"count": "Customers", "sum": "Churned", "mean": "Churn rate %"})
    g["Churn rate %"] = (g["Churn rate %"] * 100).round(1)
    g["Share of customers %"] = (g.Customers / g.Customers.sum() * 100).round(1)
    g["Share of churn %"] = (g.Churned / max(g.Churned.sum(), 1) * 100).round(1)
    g["Risk index"] = (g["Churn rate %"] / (d.Exited.mean() * 100) * 100).round(0)
    if order:
        g = g.reindex([o for o in order if o in g.index])
    return g.reset_index()


def bar(d, col, order, title):
    t = churn_table(d, col, order)
    fig = px.bar(t, x=col, y="Churn rate %", text="Churn rate %", title=title,
                 hover_data=["Customers", "Churned"], category_orders={col: order})
    fig.add_hline(y=d.Exited.mean() * 100, line_dash="dash", annotation_text="Selection average")
    return fig, t


# ---------- Dynamic KPIs ----------
overall = f.Exited.mean()
hv = f[f["Balance Segment"] == "High"]
hv_ratio = (hv.Exited.mean() / overall) if len(hv) and overall > 0 else float("nan")
act, ina = f[f.Activity == "Active"].Exited.mean(), f[f.Activity == "Inactive"].Exited.mean()
edi = (ina / act) if act and act > 0 and not np.isnan(ina) else float("nan")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers selected", f"{len(f):,}", f"{len(f)/len(df):.0%} of all", delta_color="off")
k2.metric("Overall Churn Rate", f"{overall:.1%}")
k3.metric("High-Value Churn Ratio", "n/a" if np.isnan(hv_ratio) else f"{hv_ratio:.2f}x",
          help="Churn of high-balance customers ÷ churn of the selection")
k4.metric("Engagement Drop Indicator", "n/a" if np.isnan(edi) else f"{edi:.2f}x",
          help="Inactive churn ÷ active churn")
k5.metric("Balance lost to churn", f"{f[f.Exited == 1].Balance.sum()/1e6:,.1f}M",
          help="Sum of balances of churned customers in the selection")

t0, t1, t2, t3, t4 = st.tabs(["Overall Summary", "Geography", "Age & Tenure", "High-Value Explorer", "Drill-down"])

# ---------- Tab: overall ----------
with t0:
    c1, c2 = st.columns(2)
    fig, _ = bar(f, "Gender", DIMS["Gender"], "Churn by gender")
    c1.plotly_chart(fig)
    fig, _ = bar(f, "Activity", DIMS["Activity"], "Churn by activity status")
    c2.plotly_chart(fig)
    c3, c4 = st.columns(2)
    fig, _ = bar(f, "Credit Band", CREDIT, "Churn by credit score band")
    c3.plotly_chart(fig)
    fig, _ = bar(f, "Balance Segment", BALANCE, "Churn by balance segment")
    c4.plotly_chart(fig)
    st.subheader("Churned vs retained customer profile")
    prof = f.groupby("Exited")[["Age", "CreditScore", "Tenure", "Balance", "EstimatedSalary",
                                "NumOfProducts", "IsActiveMember", "HasCrCard"]].mean().round(2).T
    prof.columns = ["Retained", "Churned"][: len(prof.columns)] if len(prof.columns) == 2 else prof.columns
    st.dataframe(prof)

# ---------- Tab: geography ----------
with t1:
    fig, t = bar(f, "Geography", DIMS["Geography"], "Churn rate by country")
    st.plotly_chart(fig)
    t["Geographic Risk Index"] = t["Risk index"]
    st.subheader("Geographic Risk Index (100 = selection average)")
    st.dataframe(t.drop(columns=["Risk index"]), hide_index=True)
    st.subheader("Geography × age interaction")
    pv = f.pivot_table(index="Geography", columns="Age Group", values="Exited", aggfunc="mean").reindex(
        columns=[a for a in AGE if a in f["Age Group"].unique()]).mul(100).round(1)
    st.plotly_chart(px.imshow(pv, text_auto=True, color_continuous_scale="Reds", aspect="auto",
                              title="Churn % by country and age group"))
    bal = f[f.Exited == 1].groupby("Geography").Balance.sum().div(1e6).round(1).reset_index(name="Balance lost (M)")
    st.plotly_chart(px.bar(bal, x="Geography", y="Balance lost (M)", text="Balance lost (M)",
                           title="Balance lost to churn by country (millions)"))

# ---------- Tab: age & tenure ----------
with t2:
    c1, c2 = st.columns(2)
    fig, ta = bar(f, "Age Group", AGE, "Churn by age group")
    c1.plotly_chart(fig)
    fig, tt = bar(f, "Tenure Group", TENURE, "Churn by tenure group")
    c2.plotly_chart(fig)
    st.subheader("Churn contribution by age group")
    st.dataframe(ta, hide_index=True)
    st.subheader("Age × tenure")
    pv = f.pivot_table(index="Age Group", columns="Tenure Group", values="Exited", aggfunc="mean")
    pv = pv.reindex(index=[a for a in AGE if a in pv.index], columns=[t for t in TENURE if t in pv.columns]).mul(100).round(1)
    st.plotly_chart(px.imshow(pv, text_auto=True, color_continuous_scale="Reds", aspect="auto",
                              title="Churn % by age and tenure"))

# ---------- Tab: high value ----------
with t3:
    st.caption(f"High-value = top 25% of non-zero balances (balance ≥ {hb_cut:,.0f}).")
    min_bal = st.slider("Minimum balance", 0, int(df.Balance.max()), int(hb_cut), step=5000)
    hvx = f[f.Balance >= min_bal]
    if hvx.empty:
        st.info("No customers above this balance in the current selection.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Customers above threshold", f"{len(hvx):,}")
        m2.metric("Churn rate", f"{hvx.Exited.mean():.1%}")
        m3.metric("Balance at risk (churned)", f"{hvx[hvx.Exited == 1].Balance.sum()/1e6:,.1f}M")
        samp = hvx.sample(min(3000, len(hvx)), random_state=1).assign(
            Status=lambda d: d.Exited.map({0: "Retained", 1: "Churned"}))
        st.plotly_chart(px.scatter(samp, x="EstimatedSalary", y="Balance", color="Status", opacity=0.55,
                                   title="Salary vs balance, high-value customers"))
        c1, c2 = st.columns(2)
        fig, _ = bar(hvx, "Geography", DIMS["Geography"], "High-value churn by country")
        c1.plotly_chart(fig)
        fig, _ = bar(hvx, "Age Group", AGE, "High-value churn by age group")
        c2.plotly_chart(fig)
        churners = hvx[hvx.Exited == 1].sort_values("Balance", ascending=False)[
            ["CustomerId", "Geography", "Gender", "Age", "Tenure", "Balance", "EstimatedSalary",
             "NumOfProducts", "Activity"]]
        st.subheader(f"High-balance churners: {len(churners):,}")
        st.dataframe(churners, hide_index=True)
        st.download_button("Download list (CSV)", churners.to_csv(index=False), "high_value_churners.csv")

# ---------- Tab: drill-down ----------
with t4:
    a, b = st.columns(2)
    rows = a.selectbox("Rows", list(DIMS), index=0)
    cols = b.selectbox("Columns", [d for d in DIMS if d != rows], index=0)
    pv = f.pivot_table(index=rows, columns=cols, values="Exited", aggfunc="mean")
    pv = pv.reindex(index=[x for x in DIMS[rows] if x in pv.index], columns=[x for x in DIMS[cols] if x in pv.columns]).mul(100).round(1)
    st.plotly_chart(px.imshow(pv, text_auto=True, color_continuous_scale="Reds", aspect="auto",
                              title=f"Churn % by {rows} and {cols}"))
    cnt = f.pivot_table(index=rows, columns=cols, values="Exited", aggfunc="count")
    cnt = cnt.reindex(index=pv.index, columns=pv.columns)
    st.caption("Customers in each cell (small cells give unreliable rates):")
    st.dataframe(cnt)
