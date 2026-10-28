import sys, json, time, queue, argparse, threading
import numpy as np
from multiprocessing import Pool, shared_memory

VMAX = 5
P_SLOW = 0.18
CELL_M = 7.5
CENTERS = {"noida": (28.4744, 77.5040), "gurgaon": (28.4950, 77.0890)}
GRID = {1: (1, 1), 2: (2, 1), 4: (2, 2), 8: (4, 2)}
HEAD = 1          # cells at road start that must be empty to enter
CAP = 0.30        # slider 100% == 30% of all cells
MAXV = 3000       # max vehicles sent to browser per frame

_cells = None
_off = None
_parts = None
_shm = None


class Net:
    def __init__(self, xy, src, dst, ln):
        self.xy = np.asarray(xy, dtype=float)
        self.src = np.asarray(src)
        self.dst = np.asarray(dst)
        self.len = np.asarray(ln)
        self.off = np.concatenate([[0], np.cumsum(self.len)])
        self.out = [[] for _ in range(len(self.xy))]
        for r, s in enumerate(self.src):
            self.out[s].append(r)
        self.closed = np.zeros(len(self.src), bool)
        self.mid = (self.xy[self.src] + self.xy[self.dst]) / 2

    @classmethod
    def osm(cls, city, dist):
        import os
        import osmnx as ox
        ox.settings.log_console = False
        path = f"{city}.graphml"
        if os.path.exists(path):
            G = ox.load_graphml(path)
        else:
            G = ox.graph_from_point(CENTERS[city], dist=dist, network_type="drive")
            ox.save_graphml(G, path)
        ids = {n: i for i, n in enumerate(G.nodes)}
        xy = [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in G.nodes]
        src, dst, ln = [], [], []
        for u, v, d in G.edges(data=True):
            src.append(ids[u])
            dst.append(ids[v])
            ln.append(min(300, max(5, int(d["length"] / CELL_M))))
        return cls(xy, src, dst, ln)

    @classmethod
    def planned(cls, city):
        w, h, sp, jit, drop, art, seed = {"noida-sim": (32, 24, 130, 0.12, 0.18, 8, 5),
                                          "gurgaon-sim": (30, 24, 120, 0.30, 0.30, 7, 6)}[city]
        rng = np.random.default_rng(seed)
        lat0, lon0 = CENTERS[city.split("-")[0]]
        m = []
        for j in range(h):
            for i in range(w):
                m.append([(i - w / 2) * sp + rng.normal(0, jit * sp), (j - h / 2) * sp + rng.normal(0, jit * sp)])
        m = np.array(m)
        xy = np.stack([lon0 + m[:, 0] / (111320 * np.cos(np.radians(lat0))), lat0 + m[:, 1] / 110540], 1)
        edges = []
        for j in range(h):
            for i in range(w):
                a = j * w + i
                if i + 1 < w:
                    edges.append((a, a + 1, j % art == 0))
                if j + 1 < h:
                    edges.append((a, a + w, i % art == 0))
        for j in range(h - 1):
            edges.append((j * w + j, (j + 1) * w + j + 1, True))
            edges.append((j * w + (w - 1 - j), (j + 1) * w + (w - 2 - j), True))
        deg = np.zeros(w * h, int)
        for a, b, _ in edges:
            deg[a] += 1
            deg[b] += 1
        src, dst, ln = [], [], []
        for a, b, art_road in edges:
            if not art_road and deg[a] > 2 and deg[b] > 2 and rng.random() < drop:
                deg[a] -= 1
                deg[b] -= 1
                continue
            n = int(min(300, max(5, np.linalg.norm(m[a] - m[b]) / CELL_M)))
            src += [a, b]
            dst += [b, a]
            ln += [n, n]
        return cls(xy, src, dst, ln)

    @classmethod
    def grid(cls, w=40, h=30):
        rng = np.random.default_rng(3)
        xy = [[x / w, y / h] for y in range(h) for x in range(w)]
        src, dst, ln = [], [], []
        for y in range(h):
            for x in range(w):
                for dx, dy in ((1, 0), (0, 1)):
                    if x + dx < w and y + dy < h:
                        a, b = y * w + x, (y + dy) * w + x + dx
                        n = int(rng.integers(12, 40))
                        src += [a, b]
                        dst += [b, a]
                        ln += [n, n]
        return cls(xy, src, dst, ln)

    def tiles(self, k):
        cx, cy = GRID[k]
        lo, hi = self.xy.min(0), self.xy.max(0)
        t = np.minimum(((self.mid - lo) / (hi - lo) * [cx, cy]).astype(int), [cx - 1, cy - 1])
        part = t[:, 1] * cx + t[:, 0]
        return [np.flatnonzero(part == i) for i in range(k)]


def _bind(cells, off, parts):
    global _cells, _off, _parts
    _cells, _off, _parts = cells, off, parts


def _init(name, size, off, parts):
    global _shm
    _shm = shared_memory.SharedMemory(name=name)
    _bind(np.ndarray((size,), dtype=np.int8, buffer=_shm.buf), off, parts)


def _work(args):
    i, open_end, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for r in _parts[i]:
        road = _cells[_off[r]:_off[r + 1]]
        idx = np.flatnonzero(road >= 0)
        if not idx.size:
            continue
        v = np.minimum(road[idx].astype(np.int64) + 1, VMAX)
        gap = np.empty(idx.size, np.int64)
        gap[:-1] = idx[1:] - idx[:-1] - 1
        gap[-1] = road.size - 1 - idx[-1] + (VMAX if open_end[r] else 0)
        v = np.minimum(v, gap)
        v = np.where((v > 0) & (rng.random(v.size) < P_SLOW), v - 1, v)
        new = idx + v
        road[idx] = -1
        gone = new >= road.size
        road[new[~gone]] = v[~gone]
        if gone.any():
            out.append((r, v[gone].tolist())) 
    return out


class Sim:
    def __init__(self, net, k=4, density=0.2, seed=1):
        self.net = net
        self.density = density
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.pool = None
        size = int(net.off[-1])
        self.shm = shared_memory.SharedMemory(create=True, size=size)
        self.cells = np.ndarray((size,), dtype=np.int8, buffer=self.shm.buf)
        self.cells[:] = -1
        self.hidx = net.off[:-1, None] + np.arange(HEAD)
        cnt = np.array([len(o) for o in net.out])
        self.cnt = cnt
        self.ptr = np.concatenate([[0], np.cumsum(cnt)])
        self.out_flat = np.array([r for o in net.out for r in o], dtype=np.int64)
        self.set_workers(k)

    def set_workers(self, k):
        if self.pool:
            self.pool.terminate()
            self.pool = None
        self.k = k
        self.parts = self.net.tiles(k)
        if k == 1:
            _bind(self.cells, self.net.off, self.parts)
        else:
            self.pool = Pool(k, initializer=_init, initargs=(self.shm.name, self.cells.size, self.net.off, self.parts))

    def toggle(self, r):
        net = self.net
        twins = [t for t in net.out[net.dst[r]] if net.dst[t] == net.src[r]]
        state = not net.closed[r]
        for x in [r] + twins:
            net.closed[x] = state
            if state:
                self.cells[net.off[x]:net.off[x + 1]] = -1

    def close_area(self, lon, lat, r_m):
        net = self.net
        dx = (net.mid[:, 0] - lon) * 111320 * np.cos(np.radians(lat))
        dy = (net.mid[:, 1] - lat) * 110540
        hit = np.flatnonzero((dx * dx + dy * dy) <= r_m * r_m)
        for x in hit:
            net.closed[x] = True
            self.cells[net.off[x]:net.off[x + 1]] = -1
        return len(hit)

    def reset(self):
        self.cells[:] = -1
        self.t = 0

    def reopen(self):
        self.net.closed[:] = False

    def open_flags(self):
        net = self.net
        head = (self.cells[self.hidx] < 0).all(1) & ~net.closed
        s = np.zeros(len(net.out), int)
        nz = self.cnt > 0
        s[nz] = np.add.reduceat(head[self.out_flat].astype(int), self.ptr[:-1][nz])
        return s[net.dst] > 0

    def tick(self):
        net, c = self.net, self.cells
        open_end = self.open_flags()
        tasks = [(i, open_end, int(self.rng.integers(1 << 31))) for i in range(self.k)]
        res = self.pool.map(_work, tasks) if self.pool else [_work(tasks[0])]
        for part in res:
            for r, speeds in part:
                cand = [t for t in net.out[net.dst[r]] if not net.closed[t] and c[net.off[t]] < 0]
                for sp in speeds:
                    if not cand:
                        break
                    t = cand.pop(int(self.rng.integers(len(cand))))
                    c[net.off[t]] = max(1, sp)
        open_cells = int(net.len[~net.closed].sum())
        target = int(self.density * CAP * open_cells)
        n = int((c >= 0).sum())
        for _ in range(60):
            if n >= target:
                break
            r = int(self.rng.integers(len(net.src)))
            if not net.closed[r] and c[net.off[r]] < 0:
                c[net.off[r]] = 0
                n += 1
        if n > target:
            occ = np.flatnonzero(c >= 0)
            c[self.rng.choice(occ, min(30, n - target, occ.size), replace=False)] = -1
        self.t += 1

    def frame(self, ms):
        net, c = self.net, self.cells
        occ = c >= 0
        starts = net.off[:-1]
        cnt = np.add.reduceat(occ.astype(np.int64), starts)
        sv = np.add.reduceat(np.where(occ, c, 0).astype(np.int64), starts)
        mean = np.divide(sv, cnt, out=np.zeros(len(cnt)), where=cnt > 0)
        dens = cnt / net.len
        cong = np.clip((1 - mean / VMAX) * 0.8 + np.minimum(1, dens * 2) * 0.2, 0, 1)
        cong = np.where(cnt > 0, cong, 0)
        out = (cong * 100).astype(int)
        out[net.closed] = -1
        n = int(occ.sum())
        idx = np.flatnonzero(occ)
        if idx.size > MAXV:
            idx = idx[:: int(np.ceil(idx.size / MAXV))]
        vr = np.searchsorted(net.off, idx, side="right") - 1
        vp = (idx - net.off[vr] + 0.5) / net.len[vr]
        return {"type": "frame", "veh": [vr.tolist(), np.round(vp, 3).tolist(), c[idx].tolist()],
                "tick": self.t, "ms": round(ms, 2), "k": self.k, "n": n,
                "avg": round(float(sv.sum() / max(n, 1)), 2), "cong": out.tolist()}

    def close(self):
        if self.pool:
            self.pool.terminate()
        self.shm.close()
        self.shm.unlink()


def emit(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def reader(q):
    for line in sys.stdin:
        try:
            q.put(json.loads(line))
        except ValueError:
            pass


def bench(net, ticks, out=None):
    res = []
    for k in (1, 2, 4, 8):
        s = Sim(net, k, 0.25)
        for _ in range(60):
            s.tick()
        t0 = time.perf_counter()
        for _ in range(ticks):
            s.tick()
        res.append({"k": k, "sec": round((time.perf_counter() - t0) / ticks * 1000, 2)})
        s.close()
    base = res[0]["sec"]
    for r in res:
        sp = base / r["sec"]
        print(f'{r["k"]} workers  speedup {sp:.2f}x  efficiency {sp / r["k"]:.0%}', file=sys.stderr)
    print(json.dumps(res))
    if out:
        with open(out, "w") as f:
            json.dump(res, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="noida")
    ap.add_argument("--dist", type=int, default=3500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tps", type=float, default=15)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    net = Net.grid() if a.city == "grid" else Net.planned(a.city) if a.city.endswith("-sim") else Net.osm(a.city, a.dist)
    if a.bench:
        bench(net, a.ticks, a.out)
        return
    sim = Sim(net, a.workers)
    paused = False
    q = queue.Queue()
    threading.Thread(target=reader, args=(q,), daemon=True).start()
    emit({"type": "net", "roads": [[round(float(net.xy[s][0]), 5), round(float(net.xy[s][1]), 5),
                                    round(float(net.xy[d][0]), 5), round(float(net.xy[d][1]), 5)]
                                   for s, d in zip(net.src, net.dst)]})
    try:
        while True:
            t0 = time.perf_counter()
            while not q.empty():
                m = q.get()
                c = m.get("cmd")
                if c == "density":
                    sim.density = min(1.0, max(0.02, float(m["v"])))
                elif c == "workers" and int(m["v"]) in GRID:
                    sim.set_workers(int(m["v"]))
                elif c == "toggle" and 0 <= int(m["road"]) < len(net.src):
                    sim.toggle(int(m["road"]))
                elif c == "reopen":
                    sim.reopen()
                elif c == "tps":
                    a.tps = min(60, max(1, float(m["v"])))
                elif c == "pause":
                   paused = bool(m["v"])
                elif c == "reset":
                    sim.reset()
                elif c == "closeArea":
                     sim.close_area(float(m["lon"]), float(m["lat"]), min(1500, max(50, float(m["r"]))))

            if not paused:
                sim.tick()
            ms = (time.perf_counter() - t0) * 1000
            emit(sim.frame(ms))
            time.sleep(max(0, 1 / a.tps - (time.perf_counter() - t0)))
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        sim.close()


if __name__ == "__main__":
     import multiprocessing
     multiprocessing.freeze_support()
     main()
