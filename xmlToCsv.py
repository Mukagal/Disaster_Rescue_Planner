import xml.etree.ElementTree as ET
import csv
import math

ROAD_TYPES = {
    'motorway', 'motorway_link',
    'trunk', 'trunk_link',
    'primary', 'primary_link',
    'secondary', 'secondary_link',
    'tertiary', 'tertiary_link',
    'unclassified', 'residential',
    'service', 'living_street', 'road',
    'pedestrian', 'footway', 'path', 'steps', 'cycleway',
}

SPEED_TABLE = {
    'motorway': 110, 'motorway_link': 60,
    'trunk': 90,     'trunk_link': 50,
    'primary': 70,   'primary_link': 40,
    'secondary': 50, 'secondary_link': 30,
    'tertiary': 40,  'tertiary_link': 25,
    'unclassified': 30, 'residential': 25,
    'service': 15,   'living_street': 10,
    'road': 30,      'pedestrian': 5,
    'footway': 5,    'path': 5,
    'steps': 2,      'cycleway': 15,
}
DEFAULT_SPEED = 20


def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def osm_to_csv(osm_file,
               nodes_csv='output_nodes.csv',
               ways_csv='output_ways.csv',
               edges_csv='output_edges.csv'):
    # Pass 1 – collect all node positions
    print("Pass 1: loading node coordinates...")
    all_nodes = {}
    for _, elem in ET.iterparse(osm_file, events=('end',)):
        if elem.tag == 'node':
            nid = elem.attrib.get('id')
            lat = elem.attrib.get('lat')
            lon = elem.attrib.get('lon')
            if nid and lat and lon:
                all_nodes[nid] = (float(lat), float(lon))
    print(f"  -> {len(all_nodes):,} nodes loaded")

    # Pass 2 – process ways (fresh parse; avoids elem.clear() bug)
    print("Pass 2: processing road ways...")
    road_node_ids = set()
    ways_data = []
    edges_data = []

    for _, elem in ET.iterparse(osm_file, events=('end',)):
        if elem.tag != 'way':
            continue
        tags = {}
        nd_refs = []
        for child in elem:
            if child.tag == 'tag' and 'k' in child.attrib:
                tags[child.attrib['k']] = child.attrib.get('v', '')
            elif child.tag == 'nd' and 'ref' in child.attrib:
                nd_refs.append(child.attrib['ref'])

        highway = tags.get('highway', '')
        if highway not in ROAD_TYPES:
            continue

        way_id = elem.attrib.get('id', '')
        oneway_raw = tags.get('oneway', 'no').lower()
        is_oneway = oneway_raw in ('yes', '1', 'true')
        if highway in ('motorway', 'motorway_link'):
            is_oneway = True

        name = tags.get('name', tags.get('name:en', ''))
        maxspeed_str = tags.get('maxspeed', '')
        try:
            maxspeed = int(maxspeed_str.split()[0])
        except (ValueError, IndexError):
            maxspeed = SPEED_TABLE.get(highway, DEFAULT_SPEED)

        ways_data.append({
            'way_id': way_id, 'highway': highway, 'name': name,
            'oneway': 'yes' if is_oneway else 'no',
            'maxspeed_kmh': maxspeed,
            'node_refs': ' '.join(nd_refs),
        })

        speed_ms = maxspeed * 1000 / 3600
        for i in range(len(nd_refs) - 1):
            u, v = nd_refs[i], nd_refs[i+1]
            if u not in all_nodes or v not in all_nodes:
                continue
            road_node_ids.add(u)
            road_node_ids.add(v)
            lat1, lon1 = all_nodes[u]
            lat2, lon2 = all_nodes[v]
            dist = haversine(lat1, lon1, lat2, lon2)
            travel_time = dist / speed_ms if speed_ms > 0 else 9999.0
            base = {
                'way_id': way_id, 'highway': highway,
                'distance_m': round(dist, 2), 'speed_kmh': maxspeed,
                'travel_time_s': round(travel_time, 2),
                'oneway': 'yes' if is_oneway else 'no',
            }
            edges_data.append({'from_node': u, 'to_node': v, **base})
            if not is_oneway:
                edges_data.append({'from_node': v, 'to_node': u, **base})

    print(f"  -> {len(ways_data):,} road ways")
    print(f"  -> {len(edges_data):,} directed edges")
    print(f"  -> {len(road_node_ids):,} unique road nodes")

    print(f"\nWriting {nodes_csv}...")
    with open(nodes_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['node_id', 'lat', 'lon', 'highway'])
        w.writeheader()
        for nid in sorted(road_node_ids, key=int):
            lat, lon = all_nodes[nid]
            w.writerow({'node_id': nid, 'lat': lat, 'lon': lon, 'highway': ''})

    print(f"Writing {ways_csv}...")
    with open(ways_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['way_id','highway','name','oneway','maxspeed_kmh','node_refs'])
        w.writeheader()
        w.writerows(ways_data)

    print(f"Writing {edges_csv}...")
    with open(edges_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['from_node','to_node','way_id','highway',
                                          'distance_m','speed_kmh','travel_time_s','oneway'])
        w.writeheader()
        w.writerows(edges_data)

    print("\nDone! Summary:")
    print(f"  {nodes_csv:<22} -> {len(road_node_ids):>6,} rows")
    print(f"  {ways_csv:<22} -> {len(ways_data):>6,} rows")
    print(f"  {edges_csv:<22} -> {len(edges_data):>6,} rows")
    print("\nColumn guide (edges CSV):")
    print("  distance_m     - Haversine distance in metres")
    print("  travel_time_s  - distance / speed  -> use as A* g-cost")
    print("  oneway         - 'yes' means reverse direction is blocked")


if __name__ == '__main__':
    osm_to_csv('map.osm', 'output_nodes.csv', 'output_ways.csv', 'output_edges.csv')
    