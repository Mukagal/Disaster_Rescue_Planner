"""
Disaster Zone Rescue Planner — A* Search on Real OSM Data
INF375 Final Project
"""

import math
import heapq
import time
import random
from collections import defaultdict

import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
NODES_CSV  = "output_nodes.csv"
EDGES_CSV  = "output_edges.csv"

MAP_CENTER = [43.2075, 76.6364]
MAP_ZOOM   = 13

# Road-type colours for the legend
ROAD_COLORS = {
    "trunk": "#e74c3c",
    "trunk_link": "#e74c3c",
    "primary": "#e67e22",
    "primary_link": "#e67e22",
    "secondary": "#f1c40f",
    "secondary_link": "#f1c40f",
    "tertiary": "#2ecc71",
    "tertiary_link": "#2ecc71",
    "residential": "#3498db",
    "living_street": "#9b59b6",
    "service": "#95a5a6",
    "unclassified": "#bdc3c7",
    "footway": "#1abc9c",
    "path": "#1abc9c",
    "cycleway": "#1abc9c",
    "steps": "#1abc9c",
}

HEURISTIC_LABELS = {
    "haversine":  "Haversine distance ÷ max-speed  (admissible, fast)",
    "euclidean":  "Euclidean degree-distance  (non-admissible, aggressive)",
    "zero":       "Zero heuristic  (= Dijkstra / UCS)",
}

WEIGHT_LABELS = {
    "travel_time": "Travel time  (seconds)",
    "distance":    "Physical distance  (metres)",
}

# ─────────────────────────────────────────────────────────────
#  Data loading  (cached so the CSVs are only read once)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading map data…")
def load_graph():
    nodes_df = pd.read_csv(NODES_CSV, dtype={"node_id": str})
    edges_df = pd.read_csv(EDGES_CSV, dtype={"from_node": str, "to_node": str})

    # node_id → (lat, lon)
    coords = {
        row.node_id: (float(row.lat), float(row.lon))
        for row in nodes_df.itertuples(index=False)
    }

    # adjacency list: node_id → list of (neighbour, distance_m, travel_time_s, highway, way_id)
    graph = defaultdict(list)
    for row in edges_df.itertuples(index=False):
        graph[row.from_node].append({
            "to":           row.to_node,
            "distance_m":   float(row.distance_m),
            "travel_time_s":float(row.travel_time_s),
            "highway":      row.highway,
            "way_id":       str(row.way_id),
        })

    return coords, graph, nodes_df, edges_df


# ─────────────────────────────────────────────────────────────
#  Heuristics
# ─────────────────────────────────────────────────────────────
EARTH_R = 6_371_000  # metres

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))

def make_heuristic(heuristic_name, weight_mode, goal_lat, goal_lon, max_speed_ms):
    """Return a heuristic function h(node_id) → estimated cost to goal."""
    def h_haversine(coords, nid):
        lat, lon = coords[nid]
        dist = haversine_m(lat, lon, goal_lat, goal_lon)
        if weight_mode == "travel_time":
            return dist / max_speed_ms        # optimistic: assume max speed
        return dist                            # pure distance

    def h_euclidean(coords, nid):
        lat, lon = coords[nid]
        # Degree-distance (not admissible but fast for demo)
        dlat = math.radians(goal_lat - lat) * EARTH_R
        dlon = math.radians(goal_lon - lon) * EARTH_R * math.cos(math.radians(lat))
        dist = math.sqrt(dlat**2 + dlon**2)
        if weight_mode == "travel_time":
            return dist / max_speed_ms
        return dist

    def h_zero(coords, nid):
        return 0.0

    return {"haversine": h_haversine,
            "euclidean": h_euclidean,
            "zero":      h_zero}[heuristic_name]


# ─────────────────────────────────────────────────────────────
#  A* Search
# ─────────────────────────────────────────────────────────────
def astar(coords, graph, start, goal,
          heuristic_name="haversine",
          weight_mode="travel_time",
          blocked_ways=None,
          blocked_nodes=None):
    """
    Returns:
        path        – list of node_ids from start to goal (or [])
        stats       – dict with nodes_explored, edges_relaxed, cost, time_s
    """
    if blocked_ways  is None: blocked_ways  = set()
    if blocked_nodes is None: blocked_nodes = set()

    if start not in coords or goal not in coords:
        return [], {"error": "Start or goal node not in graph"}

    goal_lat, goal_lon = coords[goal]
    # 110 km/h in m/s — used by admissible haversine heuristic
    max_speed_ms = 110 * 1000 / 3600

    h = make_heuristic(heuristic_name, weight_mode, goal_lat, goal_lon, max_speed_ms)

    # g_cost[n] = best known cost from start to n
    g_cost = {start: 0.0}
    came_from = {start: None}

    # Priority queue: (f, tie-breaker, node)
    counter = 0
    pq = [(h(coords, start), counter, start)]

    nodes_explored = 0
    edges_relaxed  = 0
    t0 = time.perf_counter()

    while pq:
        f, _, current = heapq.heappop(pq)

        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            elapsed = time.perf_counter() - t0
            return path, {
                "nodes_explored": nodes_explored,
                "edges_relaxed":  edges_relaxed,
                "cost":           g_cost[goal],
                "time_s":         elapsed,
                "weight_mode":    weight_mode,
            }

        # Skip stale entries
        if f > g_cost.get(current, math.inf) + h(coords, current):
            continue

        nodes_explored += 1

        for edge in graph.get(current, []):
            neighbour = edge["to"]
            if neighbour in blocked_nodes:
                continue
            if edge["way_id"] in blocked_ways:
                continue

            edges_relaxed += 1
            w = edge["travel_time_s"] if weight_mode == "travel_time" else edge["distance_m"]
            new_g = g_cost[current] + w

            if new_g < g_cost.get(neighbour, math.inf):
                g_cost[neighbour] = new_g
                came_from[neighbour] = current
                counter += 1
                f_new = new_g + h(coords, neighbour)
                heapq.heappush(pq, (f_new, counter, neighbour))

    elapsed = time.perf_counter() - t0
    return [], {
        "nodes_explored": nodes_explored,
        "edges_relaxed":  edges_relaxed,
        "cost":           None,
        "time_s":         elapsed,
        "weight_mode":    weight_mode,
        "error":          "No path found — destination is unreachable",
    }


# ─────────────────────────────────────────────────────────────
#  Nearest node finder
# ─────────────────────────────────────────────────────────────
def nearest_node(coords, lat, lon):
    """Return the node_id whose coordinates are closest to (lat, lon)."""
    best_id, best_d = None, math.inf
    for nid, (nlat, nlon) in coords.items():
        d = (nlat - lat)**2 + (nlon - lon)**2   # squared OK for comparison
        if d < best_d:
            best_d, best_id = d, nid
    return best_id


# ─────────────────────────────────────────────────────────────
#  Map builder
# ─────────────────────────────────────────────────────────────
def build_map(coords, graph, edges_df,
              path=None,
              start_node=None, goal_node=None,
              blocked_ways=None, blocked_nodes=None,
              show_all_edges=False):
    blocked_ways  = blocked_ways  or set()
    blocked_nodes = blocked_nodes or set()

    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM,
                   tiles="CartoDB positron")

    # ── Optional: draw all road edges lightly ──────────────────────────
    if show_all_edges:
        plotted = set()
        for row in edges_df.itertuples(index=False):
            key = tuple(sorted([row.from_node, row.to_node]))
            if key in plotted:
                continue
            plotted.add(key)
            if row.from_node not in coords or row.to_node not in coords:
                continue
            is_blocked = (str(row.way_id) in blocked_ways or
                          row.from_node in blocked_nodes or
                          row.to_node in blocked_nodes)
            color = "#e74c3c" if is_blocked else ROAD_COLORS.get(row.highway, "#bdc3c7")
            weight = 1 if is_blocked else 1
            opacity = 0.6 if is_blocked else 0.3
            folium.PolyLine(
                [[coords[row.from_node][0], coords[row.from_node][1]],
                 [coords[row.to_node][0],   coords[row.to_node][1]]],
                color=color, weight=weight, opacity=opacity,
                tooltip=f"{row.highway} | {row.distance_m:.0f}m"
            ).add_to(m)

    # ── Draw A* path ──────────────────────────────────────────────────
    if path and len(path) > 1:
        path_coords = [list(coords[n]) for n in path if n in coords]
        folium.PolyLine(
            path_coords,
            color="#2ecc71", weight=6, opacity=0.95,
            tooltip="A* rescue path"
        ).add_to(m)

        # Animate with arrows every few segments
        for i in range(0, len(path_coords) - 1, max(1, len(path_coords)//20)):
            mid_lat = (path_coords[i][0] + path_coords[i+1][0]) / 2
            mid_lon = (path_coords[i][1] + path_coords[i+1][1]) / 2
            folium.Marker(
                [mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html='<div style="font-size:14px;color:#2ecc71;">▶</div>',
                    icon_size=(14, 14), icon_anchor=(7, 7)
                )
            ).add_to(m)

    # ── Start marker ────────────────────────────────────────────────
    if start_node and start_node in coords:
        lat, lon = coords[start_node]
        folium.Marker(
            [lat, lon],
            tooltip=f"START  (node {start_node})",
            icon=folium.Icon(color="green", icon="ambulance", prefix="fa")
        ).add_to(m)

    # ── Goal marker ─────────────────────────────────────────────────
    if goal_node and goal_node in coords:
        lat, lon = coords[goal_node]
        folium.Marker(
            [lat, lon],
            tooltip=f"GOAL  (node {goal_node})",
            icon=folium.Icon(color="red", icon="flag", prefix="fa")
        ).add_to(m)

    # ── Blocked nodes ────────────────────────────────────────────────
    for nid in list(blocked_nodes)[:200]:   # cap rendering
        if nid in coords:
            lat, lon = coords[nid]
            folium.CircleMarker(
                [lat, lon], radius=5,
                color="#e74c3c", fill=True, fill_color="#e74c3c",
                fill_opacity=0.8,
                tooltip=f"Blocked node {nid}"
            ).add_to(m)

    return m


# ─────────────────────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Zone Rescue Planner",
    page_icon="",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────
coords, graph, nodes_df, edges_df = load_graph()
all_node_ids = list(coords.keys())

# ── Session state defaults ────────────────────────────────────
for key, default in {
    "start_node":    None,
    "goal_node":     None,
    "path":          [],
    "stats":         {},
    "blocked_ways":  set(),
    "blocked_nodes": set(),
    "ran_search":    False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#c0392b,#e74c3c);
            padding:1.2rem 1.5rem;border-radius:10px;margin-bottom:1rem'>
  <h1 style='color:white;margin:0;font-size:1.8rem'>🚑 Disaster Zone Rescue Planner</h1>
  <p style='color:#fadbd8;margin:0.3rem 0 0'>
      A* pathfinding on real OpenStreetMap data · Almaty, Kazakhstan
  </p>
</div>
""", unsafe_allow_html=True)

# ── Layout: sidebar + main ────────────────────────────────────
sidebar, main_col = st.columns([1, 2.8])

with sidebar:
    st.subheader("Search Settings")

    heuristic = st.selectbox(
        "Heuristic",
        options=list(HEURISTIC_LABELS.keys()),
        format_func=lambda k: HEURISTIC_LABELS[k],
    )

    weight_mode = st.selectbox(
        "Edge weight (cost)",
        options=list(WEIGHT_LABELS.keys()),
        format_func=lambda k: WEIGHT_LABELS[k],
    )

    st.markdown("---")
    st.subheader("Waypoints")
    st.caption("Pick nodes by ID or use the random buttons.")

    # Start node
    c1, c2 = st.columns([3, 1])
    with c1:
        start_input = st.text_input(
            "Start node ID",
            value=str(st.session_state.start_node) if st.session_state.start_node else "",
            key="start_input_box",
        )
    with c2:
        st.write("")
        st.write("")
        if st.button("🎲", key="rand_start", help="Random start"):
            st.session_state.start_node = random.choice(all_node_ids)
            st.rerun()

    # Goal node
    c3, c4 = st.columns([3, 1])
    with c3:
        goal_input = st.text_input(
            "Goal node ID",
            value=str(st.session_state.goal_node) if st.session_state.goal_node else "",
            key="goal_input_box",
        )
    with c4:
        st.write("")
        st.write("")
        if st.button("🎲", key="rand_goal", help="Random goal"):
            st.session_state.goal_node = random.choice(all_node_ids)
            st.rerun()

    # Apply typed IDs
    if start_input.strip() and start_input.strip() in coords:
        st.session_state.start_node = start_input.strip()
    if goal_input.strip()  and goal_input.strip()  in coords:
        st.session_state.goal_node  = goal_input.strip()

    # Show coords
    if st.session_state.start_node:
        lat, lon = coords[st.session_state.start_node]
        st.caption(f"Start: {lat:.5f}, {lon:.5f}")
    if st.session_state.goal_node:
        lat, lon = coords[st.session_state.goal_node]
        st.caption(f"Goal:  {lat:.5f}, {lon:.5f}")

    st.markdown("---")
    st.subheader("Disaster Simulation")

    block_pct = st.slider(
        "Block random roads (%)",
        0, 60, 0, step=5,
        help="Simulates damaged/flooded roads"
    )
    block_pct_nodes = st.slider(
        "Block random intersections (%)",
        0, 30, 0, step=5,
        help="Simulates collapsed intersections"
    )

    if st.button("🔀 Apply road blockages", use_container_width=True):
        total_ways   = edges_df["way_id"].nunique()
        total_nodes  = len(all_node_ids)
        n_block_ways  = int(total_ways  * block_pct        / 100)
        n_block_nodes = int(total_nodes * block_pct_nodes  / 100)
        all_way_ids = edges_df["way_id"].astype(str).unique().tolist()
        st.session_state.blocked_ways  = set(random.sample(all_way_ids,   n_block_ways))
        st.session_state.blocked_nodes = set(random.sample(all_node_ids,  n_block_nodes))
        st.session_state.path = []
        st.session_state.ran_search = False
        st.success(f"Blocked {n_block_ways} roads + {n_block_nodes} nodes")

    if st.button("Clear all blockages", use_container_width=True):
        st.session_state.blocked_ways  = set()
        st.session_state.blocked_nodes = set()
        st.session_state.path = []
        st.session_state.ran_search = False

    st.markdown("---")
    show_all = st.checkbox("Show road network on map", value=False,
                           help="Renders all edges — slower for large maps")

    run_btn = st.button("Run A* Search", type="primary", use_container_width=True)

# ── Run search ────────────────────────────────────────────────
if run_btn:
    if not st.session_state.start_node or not st.session_state.goal_node:
        st.error("Please set both a start and a goal node first.")
    elif st.session_state.start_node == st.session_state.goal_node:
        st.error("Start and goal must be different nodes.")
    else:
        with st.spinner("Running A* search…"):
            path, stats = astar(
                coords, graph,
                st.session_state.start_node,
                st.session_state.goal_node,
                heuristic_name=heuristic,
                weight_mode=weight_mode,
                blocked_ways=st.session_state.blocked_ways,
                blocked_nodes=st.session_state.blocked_nodes,
            )
        st.session_state.path = path
        st.session_state.stats = stats
        st.session_state.ran_search = True

# ── Main area ─────────────────────────────────────────────────
with main_col:
    # Stats panel
    if st.session_state.ran_search:
        stats = st.session_state.stats
        if "error" in stats and not st.session_state.path:
            st.error(f"{stats['error']}")
        else:
            cost_val = stats.get("cost", 0)
            unit = "s" if stats.get("weight_mode") == "travel_time" else "m"
            cost_str = (f"{cost_val/60:.1f} min" if unit == "s"
                        else f"{cost_val/1000:.2f} km")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Path nodes",       len(st.session_state.path))
            m2.metric("Route cost",        cost_str)
            m3.metric("Nodes explored",   f"{stats.get('nodes_explored',0):,}")
            m4.metric("Edges relaxed",    f"{stats.get('edges_relaxed',0):,}")
            m5.metric("CPU time",         f"{stats.get('time_s',0)*1000:.1f} ms")

    # Map
    fmap = build_map(
        coords, graph, edges_df,
        path=st.session_state.path,
        start_node=st.session_state.start_node,
        goal_node=st.session_state.goal_node,
        blocked_ways=st.session_state.blocked_ways,
        blocked_nodes=st.session_state.blocked_nodes,
        show_all_edges=show_all,
    )
    map_data = st_folium(fmap, width=None, height=580, returned_objects=["last_clicked"])

    # Click-to-set waypoint
    if map_data and map_data.get("last_clicked"):
        click = map_data["last_clicked"]
        clat, clon = click["lat"], click["lng"]
        nid = nearest_node(coords, clat, clon)

        col_set1, col_set2, _ = st.columns([1, 1, 3])
        with col_set1:
            if st.button(f"Set as START  ({nid})", key="set_start_click"):
                st.session_state.start_node = nid
                st.rerun()
        with col_set2:
            if st.button(f"Set as GOAL  ({nid})", key="set_goal_click"):
                st.session_state.goal_node = nid
                st.rerun()

    # ── Path table ─────────────────────────────────────────────
    if st.session_state.path:
        with st.expander(f"Path details  ({len(st.session_state.path)} nodes)", expanded=False):
            rows = []
            path = st.session_state.path
            for i, nid in enumerate(path):
                lat, lon = coords[nid]
                seg_dist, seg_time, seg_hw = "—", "—", "—"
                if i < len(path) - 1:
                    nxt = path[i+1]
                    for edge in graph.get(nid, []):
                        if edge["to"] == nxt:
                            seg_dist = f"{edge['distance_m']:.1f} m"
                            seg_time = f"{edge['travel_time_s']:.1f} s"
                            seg_hw   = edge["highway"]
                            break
                rows.append({
                    "Step": i + 1,
                    "Node ID": nid,
                    "Lat": round(lat, 6),
                    "Lon": round(lon, 6),
                    "Seg dist": seg_dist,
                    "Seg time": seg_time,
                    "Road type": seg_hw,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

    # ── Algorithm explainer ────────────────────────────────────
    with st.expander("How A* works in this planner", expanded=False):
        st.markdown("""
### A\\* Search Algorithm

A\\* finds the shortest path from **start → goal** by combining:

| Term | Meaning |
|------|---------|
| **g(n)** | Actual cost paid from start to node *n* |
| **h(n)** | Heuristic estimate of cost from *n* to goal |
| **f(n) = g(n) + h(n)** | Priority in the open list |

**Why A\\* is optimal:**  
When the heuristic is *admissible* (never over-estimates), A\\* is guaranteed to find the shortest path.  
Here the Haversine heuristic uses *straight-line distance ÷ max road speed*, which always under-estimates travel time.

**Three heuristics available:**

| Heuristic | Admissible? | Speed |
|-----------|-------------|-------|
| Haversine ÷ max speed | ✅ Yes | Fast |
| Euclidean degree distance | ❌ No | Faster (fewer nodes) |
| Zero (= Dijkstra / UCS) | ✅ Yes | Slowest (explores most) |

**Disaster simulation:**  
Randomly blocked roads and intersections are removed from the graph before search runs.  
A\\* automatically re-routes around them if an alternative path exists.
        """)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<hr style='margin-top:2rem'>
<p style='text-align:center;color:#7f8c8d;font-size:0.8rem'>
INF375 · AI Final Project · Search Algorithms · OpenStreetMap data © ODbL
</p>
""", unsafe_allow_html=True)