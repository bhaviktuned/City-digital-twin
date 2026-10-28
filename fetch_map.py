import sys
import osmnx as ox

city = sys.argv[1] if len(sys.argv) > 1 else "noida"
dist = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
centers = {"noida": (28.4744, 77.5040), "gurgaon": (28.4950, 77.0890)}
mirrors = ["https://overpass-api.de/api", "https://overpass.kumi.systems/api", "https://overpass.private.coffee/api"]

ox.settings.log_console = True
ox.settings.requests_timeout = 60

for url in mirrors:
    for attr in ("overpass_url", "overpass_endpoint"):
        setattr(ox.settings, attr, url)
    print("trying", url, flush=True)
    try:
        G = ox.graph_from_point(centers[city], dist=dist, network_type="drive")
        ox.save_graphml(G, f"{city}.graphml")
        print("saved", f"{city}.graphml", len(G.edges), "edges")
        break
    except Exception as e:
        print("failed:", e, flush=True)
else:
    print("all mirrors failed")
