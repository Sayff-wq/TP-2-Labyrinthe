import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import tracemalloc
import heapq
from collections import deque
import threading

# ── Color Theme ──────────────────────────────────────────────────────────────
COLORS = {
    "bg":           "#0d0d14",
    "panel":        "#13131f",
    "border":       "#2a2a40",
    "wall":         "#1a1a2e",
    "wall_border":  "#0d0d1a",
    "free":         "#1e1e30",
    "free_border":  "#252538",
    "path":         "#f5c518",
    "path_border":  "#e6b800",
    "start":        "#00e5a0",
    "start_border": "#00c88c",
    "goal":         "#ff4f7b",
    "goal_border":  "#e6003d",
    "visited":      "#2d2d50",
    "visited_border":"#3a3a60",
    "frontier":     "#4a3060",
    "frontier_b":   "#5a3a70",
    "text":         "#e8e8f0",
    "text_dim":     "#7070a0",
    "accent":       "#f5c518",
    "accent2":      "#00e5a0",
    "accent3":      "#ff4f7b",
    "btn_bg":       "#1e1e35",
    "btn_hover":    "#2a2a4a",
    "btn_active":   "#f5c518",
    "stat_bg":      "#111120",
}

CELL_MIN = 20
CELL_MAX = 80

# ── Maze Generation (Recursive Backtracker) ───────────────────────────────────
def generate_maze(rows, cols):
    maze = {(r, c): 1 for r in range(rows) for c in range(cols)}
    visited = set()

    def carve(r, c):
        visited.add((r, c))
        maze[(r, c)] = 0
        dirs = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                maze[(r + dr // 2, c + dc // 2)] = 0
                carve(nr, nc)

    carve(0, 0)
    # Guarantee start/goal are free
    maze[(0, 0)] = 0
    maze[(rows - 1, cols - 1)] = 0
    return maze

# ── Pathfinding Algorithms ────────────────────────────────────────────────────
def get_neighbors(maze, rows, cols, node):
    x, y = node
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and maze.get((nx, ny), 1) == 0:
            yield (nx, ny)

def reconstruct(came_from, goal):
    if goal not in came_from:
        return []
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    return path[::-1]

def bfs_gen(maze, rows, cols, start, goal):
    queue = deque([start])
    came_from = {start: None}
    while queue:
        cur = queue.popleft()
        yield cur, set(came_from.keys()), False
        if cur == goal:
            yield cur, set(came_from.keys()), True
            return
        for nb in get_neighbors(maze, rows, cols, cur):
            if nb not in came_from:
                came_from[nb] = cur
                queue.append(nb)
    yield None, set(came_from.keys()), True

def dfs_gen(maze, rows, cols, start, goal):
    stack = [start]
    came_from = {start: None}
    while stack:
        cur = stack.pop()
        yield cur, set(came_from.keys()), False
        if cur == goal:
            yield cur, set(came_from.keys()), True
            return
        for nb in get_neighbors(maze, rows, cols, cur):
            if nb not in came_from:
                came_from[nb] = cur
                stack.append(nb)
    yield None, set(came_from.keys()), True

def astar_gen(maze, rows, cols, start, goal):
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
        yield cur, visited, False
        if cur == goal:
            yield cur, visited, True
            return
        for nb in get_neighbors(maze, rows, cols, cur):
            ng = g + 1
            if nb not in g_cost or ng < g_cost[nb]:
                g_cost[nb] = ng
                came_from[nb] = cur
                heapq.heappush(heap, (ng + h(nb), ng, nb))
    yield None, visited, True

def gbfs_gen(maze, rows, cols, start, goal):
    def h(n): return abs(n[0] - goal[0]) + abs(n[1] - goal[1])
    heap = [(h(start), start)]
    came_from = {start: None}
    visited = set()
    while heap:
        _, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        yield cur, visited, False
        if cur == goal:
            yield cur, visited, True
            return
        for nb in get_neighbors(maze, rows, cols, cur):
            if nb not in came_from:
                came_from[nb] = cur
                heapq.heappush(heap, (h(nb), nb))
    yield None, visited, True

# Full (non-generator) versions for stats
def bfs_full(maze, rows, cols, start, goal):
    queue = deque([start])
    came_from = {start: None}
    visited_count = 0
    while queue:
        cur = queue.popleft()
        visited_count += 1
        if cur == goal:
            break
        for nb in get_neighbors(maze, rows, cols, cur):
            if nb not in came_from:
                came_from[nb] = cur
                queue.append(nb)
    return reconstruct(came_from, goal), visited_count

def dfs_full(maze, rows, cols, start, goal):
    stack = [start]
    came_from = {start: None}
    visited_count = 0
    while stack:
        cur = stack.pop()
        visited_count += 1
        if cur == goal:
            break
        for nb in get_neighbors(maze, rows, cols, cur):
            if nb not in came_from:
                came_from[nb] = cur
                stack.append(nb)
    return reconstruct(came_from, goal), visited_count

def astar_full(maze, rows, cols, start, goal):
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
        for nb in get_neighbors(maze, rows, cols, cur):
            ng = g + 1
            if nb not in g_cost or ng < g_cost[nb]:
                g_cost[nb] = ng
                came_from[nb] = cur
                heapq.heappush(heap, (ng + h(nb), ng, nb))
    return reconstruct(came_from, goal), len(visited)

def gbfs_full(maze, rows, cols, start, goal):
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
        for nb in get_neighbors(maze, rows, cols, cur):
            if nb not in came_from:
                came_from[nb] = cur
                heapq.heappush(heap, (h(nb), nb))
    return reconstruct(came_from, goal), len(visited)

ALGO_GENS = {
    "BFS":   bfs_gen,
    "DFS":   dfs_gen,
    "A*":    astar_gen,
    "GBFS":  gbfs_gen,
}
ALGO_FULL = {
    "BFS":   bfs_full,
    "DFS":   dfs_full,
    "A*":    astar_full,
    "GBFS":  gbfs_full,
}

# ── Main Application ──────────────────────────────────────────────────────────
class MazeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Maze — Algorithmes de Recherche")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        # State
        self.rows = 15
        self.cols = 20
        self.maze = {}
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.path = []
        self.visited_cells = set()
        self.algo_var = tk.StringVar(value="A*")
        self.speed_var = tk.IntVar(value=30)
        self.animating = False
        self.anim_job = None
        self.draw_mode = None   # "wall", "free", "start", "goal"
        self.cell_rects = {}    # (r,c) -> canvas rect id
        self.cell_size = 30

        self._build_ui()
        self.generate_new_maze()

    # ── UI Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Left panel
        left = tk.Frame(self, bg=COLORS["panel"], width=240)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        left.pack_propagate(False)

        self._build_left_panel(left)

        # Right: canvas + stats bar
        right = tk.Frame(self, bg=COLORS["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.Frame(right, bg=COLORS["bg"])
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg=COLORS["bg"],
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Stats bar at bottom right
        self.stats_frame = tk.Frame(right, bg=COLORS["stat_bg"], height=60)
        self.stats_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.stats_frame.pack_propagate(False)
        self._build_stats_bar()

    def _build_left_panel(self, parent):
        # Title
        tk.Label(
            parent, text="MAZE", font=("Courier", 22, "bold"),
            bg=COLORS["panel"], fg=COLORS["accent"]
        ).pack(pady=(20, 2))
        tk.Label(
            parent, text="Algorithmes de Recherche",
            font=("Courier", 8), bg=COLORS["panel"], fg=COLORS["text_dim"]
        ).pack(pady=(0, 16))

        sep = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep.pack(fill=tk.X, padx=14, pady=6)

        # Maze size
        self._section(parent, "CONFIGURATION DU LABYRINTHE")
        self._spinbox_row(parent, "Hauteur (lignes):", 5, 40, "rows_var", self.rows)
        self._spinbox_row(parent, "Largeur (cols):", 5, 60, "cols_var", self.cols)

        tk.Button(
            parent, text="⟳  Générer Labyrinthe",
            command=self.generate_new_maze,
            bg=COLORS["accent2"], fg=COLORS["bg"],
            font=("Courier", 10, "bold"),
            relief=tk.FLAT, cursor="hand2",
            padx=8, pady=8, bd=0,
        ).pack(fill=tk.X, padx=14, pady=(10, 4))

        tk.Button(
            parent, text="✕  Effacer Chemin",
            command=self.clear_path,
            bg=COLORS["btn_bg"], fg=COLORS["text"],
            font=("Courier", 9),
            relief=tk.FLAT, cursor="hand2",
            padx=8, pady=6, bd=0,
        ).pack(fill=tk.X, padx=14, pady=2)

        sep2 = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep2.pack(fill=tk.X, padx=14, pady=10)

        # Algorithm
        self._section(parent, "ALGORITHME")
        for algo in ["BFS", "DFS", "A*", "GBFS"]:
            desc = {
                "BFS":  "Largeur en premier · Optimal",
                "DFS":  "Profondeur en premier · Rapide",
                "A*":   "A-Star · Optimal + Heuristique",
                "GBFS": "Greedy BFS · Heuristique pure",
            }[algo]
            f = tk.Frame(parent, bg=COLORS["btn_bg"], cursor="hand2")
            f.pack(fill=tk.X, padx=14, pady=2)
            rb = tk.Radiobutton(
                f, text=algo, variable=self.algo_var, value=algo,
                bg=COLORS["btn_bg"], fg=COLORS["accent"],
                selectcolor=COLORS["panel"],
                font=("Courier", 11, "bold"),
                activebackground=COLORS["btn_hover"],
                relief=tk.FLAT, bd=0, padx=10, pady=4,
            )
            rb.pack(side=tk.LEFT)
            tk.Label(
                f, text=desc, bg=COLORS["btn_bg"],
                fg=COLORS["text_dim"], font=("Courier", 7),
            ).pack(side=tk.LEFT, padx=(0, 6))

        sep3 = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep3.pack(fill=tk.X, padx=14, pady=10)

        # Speed
        self._section(parent, "VITESSE D'ANIMATION")
        spd_frame = tk.Frame(parent, bg=COLORS["panel"])
        spd_frame.pack(fill=tk.X, padx=14)
        tk.Label(spd_frame, text="Lent", bg=COLORS["panel"],
                 fg=COLORS["text_dim"], font=("Courier", 8)).pack(side=tk.LEFT)
        tk.Scale(
            spd_frame, from_=1, to=100, variable=self.speed_var,
            orient=tk.HORIZONTAL, bg=COLORS["panel"],
            fg=COLORS["accent"], troughcolor=COLORS["border"],
            highlightthickness=0, showvalue=False, bd=0,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(spd_frame, text="Vite", bg=COLORS["panel"],
                 fg=COLORS["text_dim"], font=("Courier", 8)).pack(side=tk.LEFT)

        sep4 = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep4.pack(fill=tk.X, padx=14, pady=10)

        # Draw tools
        self._section(parent, "OUTILS DE DESSIN")
        tools = [
            ("✎ Dessiner Mur",   "wall",  COLORS["wall"]),
            ("✎ Effacer Mur",    "free",  COLORS["free"]),
            ("▶ Départ",         "start", COLORS["start"]),
            ("■ Arrivée",        "goal",  COLORS["goal"]),
        ]
        self.tool_buttons = {}
        for label, mode, color in tools:
            btn = tk.Button(
                parent, text=label,
                command=lambda m=mode: self._set_draw_mode(m),
                bg=COLORS["btn_bg"], fg=COLORS["text"],
                font=("Courier", 9), relief=tk.FLAT,
                cursor="hand2", padx=8, pady=5, bd=0,
            )
            btn.pack(fill=tk.X, padx=14, pady=2)
            self.tool_buttons[mode] = btn

        sep5 = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep5.pack(fill=tk.X, padx=14, pady=10)

        # Run button
        self.run_btn = tk.Button(
            parent, text="▶  LANCER L'ALGORITHME",
            command=self.run_algorithm,
            bg=COLORS["accent3"], fg="white",
            font=("Courier", 11, "bold"),
            relief=tk.FLAT, cursor="hand2",
            padx=8, pady=10, bd=0,
        )
        self.run_btn.pack(fill=tk.X, padx=14, pady=(4, 2))

        self.stop_btn = tk.Button(
            parent, text="■  ARRÊTER",
            command=self.stop_animation,
            bg=COLORS["btn_bg"], fg=COLORS["text_dim"],
            font=("Courier", 9),
            relief=tk.FLAT, cursor="hand2",
            padx=8, pady=6, bd=0,
        )
        self.stop_btn.pack(fill=tk.X, padx=14, pady=2)

        # Benchmark
        sep6 = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep6.pack(fill=tk.X, padx=14, pady=10)
        tk.Button(
            parent, text="⚡  COMPARER TOUS",
            command=self.benchmark_all,
            bg="#2a2040", fg=COLORS["accent"],
            font=("Courier", 9, "bold"),
            relief=tk.FLAT, cursor="hand2",
            padx=8, pady=7, bd=0,
        ).pack(fill=tk.X, padx=14, pady=2)

        # Legend
        sep7 = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep7.pack(fill=tk.X, padx=14, pady=10)
        self._section(parent, "LÉGENDE")
        legend = [
            (COLORS["start"],   "Départ"),
            (COLORS["goal"],    "Arrivée"),
            (COLORS["path"],    "Chemin trouvé"),
            (COLORS["visited"], "Cellule visitée"),
            (COLORS["wall"],    "Mur"),
            (COLORS["free"],    "Libre"),
        ]
        for color, text in legend:
            f = tk.Frame(parent, bg=COLORS["panel"])
            f.pack(fill=tk.X, padx=18, pady=1)
            tk.Label(f, bg=color, width=2, relief=tk.FLAT).pack(side=tk.LEFT)
            tk.Label(f, text=text, bg=COLORS["panel"],
                     fg=COLORS["text_dim"], font=("Courier", 8)).pack(side=tk.LEFT, padx=6)

    def _build_stats_bar(self):
        self.stat_labels = {}
        items = [
            ("algo",    "Algorithme", "—"),
            ("path",    "Longueur",   "—"),
            ("visited", "Visités",    "—"),
            ("time",    "Temps (ms)", "—"),
            ("mem",     "Mémoire (KB)", "—"),
        ]
        for key, label, val in items:
            f = tk.Frame(self.stats_frame, bg=COLORS["stat_bg"])
            f.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2, pady=4)
            tk.Label(f, text=label, bg=COLORS["stat_bg"],
                     fg=COLORS["text_dim"], font=("Courier", 7)).pack()
            lbl = tk.Label(f, text=val, bg=COLORS["stat_bg"],
                           fg=COLORS["accent"], font=("Courier", 11, "bold"))
            lbl.pack()
            self.stat_labels[key] = lbl

    def _section(self, parent, title):
        tk.Label(
            parent, text=title,
            bg=COLORS["panel"], fg=COLORS["text_dim"],
            font=("Courier", 7, "bold"),
        ).pack(anchor=tk.W, padx=18, pady=(4, 2))

    def _spinbox_row(self, parent, label, from_, to, attr, init):
        f = tk.Frame(parent, bg=COLORS["panel"])
        f.pack(fill=tk.X, padx=14, pady=2)
        tk.Label(f, text=label, bg=COLORS["panel"],
                 fg=COLORS["text"], font=("Courier", 9), width=18, anchor=tk.W).pack(side=tk.LEFT)
        var = tk.IntVar(value=init)
        setattr(self, attr, var)
        sb = tk.Spinbox(
            f, from_=from_, to=to, textvariable=var, width=5,
            bg=COLORS["btn_bg"], fg=COLORS["accent"],
            buttonbackground=COLORS["border"],
            highlightthickness=1, highlightcolor=COLORS["border"],
            relief=tk.FLAT, font=("Courier", 10), insertbackground=COLORS["accent"],
        )
        sb.pack(side=tk.LEFT)

    # ── Draw Mode ─────────────────────────────────────────────────────────────
    def _set_draw_mode(self, mode):
        self.draw_mode = mode if self.draw_mode != mode else None
        for m, btn in self.tool_buttons.items():
            if m == self.draw_mode:
                btn.configure(bg=COLORS["accent"], fg=COLORS["bg"])
            else:
                btn.configure(bg=COLORS["btn_bg"], fg=COLORS["text"])

    # ── Maze Generation ───────────────────────────────────────────────────────
    def generate_new_maze(self):
        self.stop_animation()
        self.rows = self.rows_var.get()
        self.cols = self.cols_var.get()
        self.start = (0, 0)
        self.goal = (self.rows - 1, self.cols - 1)
        self.maze = generate_maze(self.rows, self.cols)
        self.path = []
        self.visited_cells = set()
        self._update_cell_size()
        self._draw_maze()
        self._reset_stats()

    def _update_cell_size(self):
        w = self.canvas.winfo_width() or 700
        h = self.canvas.winfo_height() or 600
        cs = min(w // self.cols, h // self.rows)
        self.cell_size = max(CELL_MIN, min(CELL_MAX, cs))

    def _on_canvas_resize(self, event):
        self._update_cell_size()
        self._draw_maze()

    def clear_path(self):
        self.stop_animation()
        self.path = []
        self.visited_cells = set()
        self._draw_maze()
        self._reset_stats()

    # ── Canvas Draw ───────────────────────────────────────────────────────────
    def _draw_maze(self):
        self.canvas.delete("all")
        self.cell_rects.clear()
        cs = self.cell_size
        pad_x = (self.canvas.winfo_width()  - cs * self.cols) // 2
        pad_y = (self.canvas.winfo_height() - cs * self.rows) // 2
        pad_x = max(4, pad_x)
        pad_y = max(4, pad_y)
        self._pad_x = pad_x
        self._pad_y = pad_y

        for r in range(self.rows):
            for c in range(self.cols):
                self._draw_cell(r, c)

    def _cell_color(self, r, c):
        pos = (r, c)
        if pos == self.start:
            return COLORS["start"], COLORS["start_border"]
        if pos == self.goal:
            return COLORS["goal"], COLORS["goal_border"]
        if pos in self.path:
            return COLORS["path"], COLORS["path_border"]
        if pos in self.visited_cells:
            return COLORS["visited"], COLORS["visited_border"]
        if self.maze.get(pos, 1) == 1:
            return COLORS["wall"], COLORS["wall_border"]
        return COLORS["free"], COLORS["free_border"]

    def _draw_cell(self, r, c):
        cs = self.cell_size
        px, py = self._pad_x, self._pad_y
        x1 = px + c * cs
        y1 = py + r * cs
        x2 = x1 + cs
        y2 = y1 + cs
        fill, outline = self._cell_color(r, c)

        rid = self.canvas.create_rectangle(
            x1 + 1, y1 + 1, x2 - 1, y2 - 1,
            fill=fill, outline=outline, width=1,
        )
        self.cell_rects[(r, c)] = rid

        pos = (r, c)
        if pos == self.start:
            self.canvas.create_text(
                x1 + cs // 2, y1 + cs // 2,
                text="S", fill=COLORS["bg"],
                font=("Courier", max(7, cs // 3), "bold"),
            )
        elif pos == self.goal:
            self.canvas.create_text(
                x1 + cs // 2, y1 + cs // 2,
                text="G", fill="white",
                font=("Courier", max(7, cs // 3), "bold"),
            )

    def _update_cell(self, r, c):
        rid = self.cell_rects.get((r, c))
        if rid:
            fill, outline = self._cell_color(r, c)
            self.canvas.itemconfig(rid, fill=fill, outline=outline)

    # ── Mouse Interaction ─────────────────────────────────────────────────────
    def _canvas_to_cell(self, x, y):
        cs = self.cell_size
        r = (y - self._pad_y) // cs
        c = (x - self._pad_x) // cs
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None

    def _on_canvas_click(self, event):
        cell = self._canvas_to_cell(event.x, event.y)
        if not cell:
            return
        r, c = cell
        if self.draw_mode == "wall":
            if (r, c) not in (self.start, self.goal):
                self.maze[(r, c)] = 1
        elif self.draw_mode == "free":
            self.maze[(r, c)] = 0
        elif self.draw_mode == "start":
            self.start = (r, c)
            self.maze[(r, c)] = 0
        elif self.draw_mode == "goal":
            self.goal = (r, c)
            self.maze[(r, c)] = 0
        self._draw_maze()

    def _on_canvas_drag(self, event):
        cell = self._canvas_to_cell(event.x, event.y)
        if not cell:
            return
        r, c = cell
        if self.draw_mode == "wall":
            if (r, c) not in (self.start, self.goal):
                self.maze[(r, c)] = 1
                self._draw_maze()
        elif self.draw_mode == "free":
            self.maze[(r, c)] = 0
            self._draw_maze()

    # ── Run Algorithm ─────────────────────────────────────────────────────────
    def run_algorithm(self):
        self.stop_animation()
        algo_name = self.algo_var.get()
        self.path = []
        self.visited_cells = set()
        self._draw_maze()

        gen_fn = ALGO_GENS[algo_name]
        self._gen = gen_fn(self.maze, self.rows, self.cols, self.start, self.goal)
        self._came_from_path = {}
        self._t_start = time.time()
        self.animating = True
        self._animate_step(algo_name)

    def _animate_step(self, algo_name):
        if not self.animating:
            return
        speed = self.speed_var.get()
        delay = max(1, int(110 - speed))
        try:
            cur, visited, done = next(self._gen)
            self.visited_cells = visited
            for cell in visited:
                if cell not in (self.start, self.goal):
                    self._update_cell(*cell)
            if done:
                self.animating = False
                self._finalize(algo_name)
                return
            self.anim_job = self.after(delay, lambda: self._animate_step(algo_name))
        except StopIteration:
            self.animating = False
            self._finalize(algo_name)

    def _finalize(self, algo_name):
        full_fn = ALGO_FULL[algo_name]
        tracemalloc.start()
        t0 = time.time()
        path, visited_count = full_fn(self.maze, self.rows, self.cols, self.start, self.goal)
        elapsed = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.path = set(path)
        self._draw_maze()

        self.stat_labels["algo"].config(text=algo_name)
        self.stat_labels["path"].config(text=str(len(path)) if path else "—")
        self.stat_labels["visited"].config(text=str(visited_count))
        self.stat_labels["time"].config(text=f"{elapsed * 1000:.3f}")
        self.stat_labels["mem"].config(text=f"{peak / 1024:.2f}")

        if not path:
            messagebox.showwarning("Résultat", "Aucun chemin trouvé !")

    def stop_animation(self):
        self.animating = False
        if self.anim_job:
            self.after_cancel(self.anim_job)
            self.anim_job = None

    def _reset_stats(self):
        for lbl in self.stat_labels.values():
            lbl.config(text="—")

    # ── Benchmark All ─────────────────────────────────────────────────────────
    def benchmark_all(self):
        self.stop_animation()
        lines = [f"{'Algo':<6} | {'Chemin':>8} | {'Visités':>8} | {'Temps ms':>10} | {'Mém KB':>8}"]
        lines.append("-" * 52)
        for name, fn in ALGO_FULL.items():
            tracemalloc.start()
            t0 = time.time()
            path, vis = fn(self.maze, self.rows, self.cols, self.start, self.goal)
            elapsed = time.time() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            lines.append(
                f"{name:<6} | {len(path):>8} | {vis:>8} | {elapsed*1000:>10.4f} | {peak/1024:>8.2f}"
            )
        win = tk.Toplevel(self)
        win.title("Comparaison des Algorithmes")
        win.configure(bg=COLORS["bg"])
        win.geometry("520x280")
        tk.Label(win, text="BENCHMARK", font=("Courier", 14, "bold"),
                 bg=COLORS["bg"], fg=COLORS["accent"]).pack(pady=(16, 4))
        txt = tk.Text(
            win, bg=COLORS["stat_bg"], fg=COLORS["accent2"],
            font=("Courier", 11), relief=tk.FLAT, padx=16, pady=12,
            highlightthickness=0,
        )
        txt.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)
        txt.insert(tk.END, "\n".join(lines))
        txt.config(state=tk.DISABLED)

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = MazeApp()
    app.mainloop()
