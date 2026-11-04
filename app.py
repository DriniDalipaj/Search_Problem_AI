import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from assignment import default_instance, random_instance, search

st.set_page_config(page_title="Ride Staffing Optimizer", layout="wide")
st.title("Ride Staffing Optimizer")
st.caption("Assign one trained employee to each ride and maximize favorite-ride matches. Compares UCS and A*.")

# ---- sidebar controls ----
mode = st.sidebar.radio("Instance", ["Park data", "Random"])
if mode == "Random":
    n = st.sidebar.slider("Rides", 4, 10, 7)
    extra = st.sidebar.slider("Extra employees", 0, 5, 3)
    seed = st.sidebar.number_input("Seed", 0, 999, 0)
    p = random_instance(n, n + extra, int(seed))
else:
    p = default_instance()

# ---- run all three searches ----
algos = {"UCS": p.h_zero, "A* (favorite)": p.h_favorite, "A* (Hungarian)": p.h_hungarian}
results, rows = {}, []
for name, h in algos.items():
    t = time.perf_counter()
    res = search(p, h)
    ms = (time.perf_counter() - t) * 1000
    if res is None:
        st.error("No valid assignment exists for this instance. Try another seed.")
        st.stop()
    results[name] = res
    rows.append({"Algorithm": name, "Cost": res["cost"],
                 "Nodes expanded": res["expansions"], "Runtime (ms)": round(ms, 2)})
df = pd.DataFrame(rows)
best = results["A* (favorite)"]

# ---- layout ----
c1, c2 = st.columns(2)
with c1:
    st.subheader("Assignment")
    st.dataframe(pd.DataFrame({
        "Ride": p.rides,
        "Employee": best["state"],
        "Got favorite": ["Yes" if p.favorite[e] == r else "No" for r, e in zip(p.rides, best["state"])],
    }), hide_index=True)
    st.metric("Rides staffed by a favorite employee", f"{len(p.rides) - best['cost']} / {len(p.rides)}")

with c2:
    st.subheader("Algorithm comparison")
    st.dataframe(df, hide_index=True)
    st.bar_chart(df.set_index("Algorithm")["Nodes expanded"])
    st.bar_chart(df.set_index("Algorithm")["Runtime (ms)"])

# ---- bipartite graph: green = favorite ride, orange = not ----
st.subheader("Assignment graph")
pos, edges, colors = {}, [], []
for j, e in enumerate(p.employees):
    pos[e] = (0, -j)
for i, r in enumerate(p.rides):
    pos[r] = (1, -i * len(p.employees) / len(p.rides))  # spread rides to match employee column
for r, e in zip(p.rides, best["state"]):
    edges.append((e, r))
    colors.append("green" if p.favorite[e] == r else "orange")
G = nx.Graph(edges)
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors, width=3, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                        bbox=dict(boxstyle="round", fc="lightblue", ec="none"))
ax.axis("off")
st.pyplot(fig)