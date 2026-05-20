from collections import deque
import time
import tracemalloc
import heapq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 0 = libre, 1 = mur
maze = {
    (0, 0): 0, (0, 1): 1, (0, 2): 0, (0, 3): 0,
    (1, 0): 0, (1, 1): 1, (1, 2): 0, (1, 3): 1,
    (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 1,
    (3, 0): 0, (3, 1): 1, (3, 2): 0, (3, 3): 0,
}

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def get_neighbors(node):
    x, y = node
    return [
        (x + dx, y + dy)
        for dx, dy in directions
        if (x + dx, y + dy) in maze and maze[(x + dx, y + dy)] == 0
    ]

def reconstruct(came_from, goal):
    if goal not in came_from:
        return []
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    return path[::-1]

# Affichage console
def print_maze(maze, path=None):
    for x in range(4):
        for y in range(4):
            if path and (x, y) in path:
                print("P", end=" ")
            elif maze[(x, y)] == 1:
                print("#", end=" ")
            else:
                print(".", end=" ")
        print()

# ── Algorithmes ──────────────────────────────────────────────────────────────

def bfs(start, goal):
    queue = deque([start])
    came_from = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for nb in get_neighbors(cur):
            if nb not in came_from:
                came_from[nb] = cur
                queue.append(nb)
    return reconstruct(came_from, goal)

def dfs(start, goal):
    stack = [start]
    came_from = {start: None}
    while stack:
        cur = stack.pop()
        if cur == goal:
            break
        for nb in get_neighbors(cur):
            if nb not in came_from:
                came_from[nb] = cur
                stack.append(nb)
    return reconstruct(came_from, goal)

def astar(start, goal):
    """A* — chemin optimal avec heuristique Manhattan."""
    def h(n): return abs(n[0] - goal[0]) + abs(n[1] - goal[1])
    heap = [(h(start), 0, start)]
    came_from = {start: None}
    g_cost = {start: 0}
    visited = set()
    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            break
        for nb in get_neighbors(cur):
            ng = g + 1
            if nb not in g_cost or ng < g_cost[nb]:
                g_cost[nb] = ng
                came_from[nb] = cur
                heapq.heappush(heap, (ng + h(nb), ng, nb))
    return reconstruct(came_from, goal)

def gbfs(start, goal):
    """Greedy Best-First Search — heuristique uniquement."""
    def h(n): return abs(n[0] - goal[0]) + abs(n[1] - goal[1])
    heap = [(h(start), start)]
    came_from = {start: None}
    visited = set()
    while heap:
        _, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            break
        for nb in get_neighbors(cur):
            if nb not in came_from:
                came_from[nb] = cur
                heapq.heappush(heap, (h(nb), nb))
    return reconstruct(came_from, goal)

# ── Exécution console ────────────────────────────────────────────────────────

start, goal = (0, 0), (3, 3)

print("Labyrinthe initial:")
print_maze(maze)

for fn, label in [(bfs, "BFS"), (dfs, "DFS"), (astar, "A*"), (gbfs, "GBFS")]:
    print(f"\nChemin {label}:")
    print_maze(maze, fn(start, goal))

# ── Mesure de performance ────────────────────────────────────────────────────

def measure_performance(algorithm, start, goal):
    tracemalloc.start()
    t0 = time.time()
    path = algorithm(start, goal)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return path, time.time() - t0, peak / 1024

algorithms = [
    (bfs,   "BFS"),
    (dfs,   "DFS"),
    (astar, "A*"),
    (gbfs,  "GBFS"),
]

results = []
for algo, name in algorithms:
    path, t, mem = measure_performance(algo, start, goal)
    results.append((name, len(path), t, mem))

# ── Tableau console ──────────────────────────────────────────────────────────

print("\nComparaison des performances:")
print(f"{'Algorithme':<12}| {'Temps (ms)':<14}| {'Mémoire (KB)':<15}| Longueur chemin")
print("-" * 60)
for name, length, t, mem in results:
    print(f"{name:<12}| {t * 1000:<14.5f}| {mem:<15.2f}| {length}")

# ── Graphiques ───────────────────────────────────────────────────────────────

names   = [r[0] for r in results]
times   = [r[2] * 1000 for r in results]
memory  = [r[3] for r in results]
lengths = [r[1] for r in results]

GREEN  = "#90EE90"   # lightgreen
SALMON = "#E8836E"   # salmon

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("white")

def style_ax(ax, data, title, ylabel, color):
    bars = ax.bar(names, data, color=color, width=0.55, zorder=3)

    # Grille pointillée grise
    ax.set_facecolor("white")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    ax.grid(axis='y', linestyle='--', linewidth=0.8, color='#bbbbbb', zorder=0)
    ax.set_axisbelow(True)

    # Supprimer les bordures inutiles
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # Titres et labels
    ax.set_title(title, fontsize=13, pad=12, color="#222222")
    ax.set_ylabel(ylabel, fontsize=11, color="#444444")
    ax.tick_params(axis='both', labelsize=10, colors="#444444")

    # Valeur au-dessus de chaque barre
    for bar, val in zip(bars, data):
        if val != 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(data) * 0.02,
                f"{val:.4f}" if isinstance(val, float) else str(val),
                ha='center', va='bottom', fontsize=9, color="#333333"
            )

style_ax(ax1, times,   "Temps d'exécution",  "Millisecondes",   GREEN)
style_ax(ax2, memory,  "Mémoire utilisée",   "Kilobytes",       GREEN)
style_ax(ax3, lengths, "Longueur du chemin", "Nombre de noeuds", SALMON)

plt.tight_layout(pad=2.5)
plt.savefig("comparaison_algos.png", dpi=150, bbox_inches="tight")
plt.show()
