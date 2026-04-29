#  Disaster Zone Rescue Planner

> **A\* Pathfinding on Real OpenStreetMap Data · Qaskelen, Kazakhstan**  
> INF375 Artificial Intelligence — Final Project

---

##  Overview

An intelligent rescue agent that finds the most efficient path through a city during a disaster scenario with blocked roads and damaged intersections. Built on a real road network extracted from OpenStreetMap data for Qaskelen, Kazakhstan.

The system implements **A\* informed search** with three selectable heuristics, a live interactive map, and a disaster simulation mode that randomly blocks roads and intersections before the search runs.

---

##  Features

- **A\* Search** with full `f(n) = g(n) + h(n)` implementation
- **3 heuristics** — Haversine (admissible), Euclidean (non-admissible), Zero / Dijkstra
- **2 edge weights** — travel time (seconds) or physical distance (metres)
- **Disaster simulation** — randomly block X% of roads and Y% of intersections
- **Click-to-set** — click anywhere on the map to set start or goal
- **Live stats** — nodes explored, edges relaxed, path cost, CPU time
- **Path details table** — per-segment road type, distance, travel time
- **Interactive Folium map** — rendered inside Streamlit via `streamlit-folium`

---

##  Project Structure

```
├── rescue_planner.py      # Main Streamlit app (A* search + UI)
├── osmToCsv.py            # OSM XML → CSV parser (self-created dataset)
├── output_nodes.csv       # 12,191 road-network nodes (node_id, lat, lon)
├── output_ways.csv        # 1,848 road ways (type, name, oneway, speed)
├── output_edges.csv       # 25,007 directed graph edges (from, to, dist, time)
└── README.md
```

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/your-username/disaster-rescue-planner.git
cd disaster-rescue-planner

# Install dependencies
pip install streamlit folium streamlit-folium pandas

# Run the app
streamlit run rescue_planner.py
```

> All CSV files must be in the same directory as `rescue_planner.py`.

---

##  Dataset

The dataset was **self-created** by downloading raw OpenStreetMap XML for Qaskelen and parsing it with a custom two-pass Python script (`osmToCsv.py`).

| File | Rows | Description |
|------|------|-------------|
| `output_nodes.csv` | 166,891 | Road-network nodes with GPS coordinates |
| `output_ways.csv` | 20945 | Road ways with type, name, oneway flag, speed |
| `output_edges.csv` | 346,732 | Directed edges with Haversine distance & travel time |

**Edge weights** are computed using the Haversine formula for distance and `distance ÷ road-type speed` for travel time. One-way roads produce a single directed edge; two-way roads produce two opposing edges.

---

##  Algorithm

### A\* Search

```
f(n) = g(n) + h(n)
```

| Term | Meaning |
|------|---------|
| `g(n)` | Actual cost from start to node *n* (travel time or distance) |
| `h(n)` | Heuristic estimate from *n* to goal |
| `f(n)` | Total estimated cost — priority queue key |

### Heuristics

| Heuristic | Admissible | Behaviour |
|-----------|-----------|-----------|
| **Haversine ÷ max speed** | Yes | Optimal path guaranteed |
| **Euclidean degree distance** | No | Fewer nodes explored |
| **Zero (Dijkstra / UCS)** | Yes | Explores most nodes — baseline |

---

## 📊 Results

| Metric | Haversine | Euclidean | Zero (Dijkstra) |
|--------|-----------|-----------|-----------------|
| Avg nodes explored | 3,316 | 1,820 | 9,100 |
| Avg CPU time | ~13 ms | ~7 ms | ~38 ms |
| Path optimality | Optimal | ± Approx | Optimal |
| Typical path nodes | 111 | 98 | 111 |

Haversine explores **64% fewer nodes** than Dijkstra while guaranteeing the same optimal path.

---

## Disaster Simulation

Use the sidebar sliders to block random percentages of roads and intersections before running the search. A\* automatically re-routes around blockages or reports the destination as unreachable.

| Roads blocked | Nodes blocked | Result |
|---|---|---|
| 0% | 0% | Baseline path |
| 20% | 5% | Re-routed, ~+25% cost |
| 60% | 20% | Unreachable in ~18% of cases |

---

## Regenerating the Dataset

If you have a new OSM file:

```bash
# Place your OSM file as map.osm in the project directory
python osmToCsv.py
# Outputs: output_nodes.csv, output_ways.csv, output_edges.csv
```

Download OSM data for any city at [openstreetmap.org/export](https://www.openstreetmap.org/export).

---

## References

- Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Chapter 3.
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Transactions on Systems Science and Cybernetics*.
- OpenStreetMap contributors. Data licensed under [ODbL](https://opendatacommons.org/licenses/odbl/).

---

## Links
- [Ready for use deployed Website](https://disasterrescueplanner-production.up.railway.app/)