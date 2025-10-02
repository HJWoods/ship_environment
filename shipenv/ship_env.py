# ship_env.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import importlib.resources as ir


# =========================
# Physics / Env Parameters
# =========================
@dataclass
class ShipParams:
    # Simulation
    dt: float = 0.1                 # seconds per step
    max_steps: int = 250

    # World bounds (square [-L, L] x [-L, L])
    world_size: float = 10.0

    # Control (accelerations, not velocities)
    thrust_accel: float = 1.5       # m/s^2 (forward accel)
    brake_accel: float = 2.0        # m/s^2 (deceleration when braking)
    torque_accel: float = 2.5       # rad/s^2 (left/right yaw acceleration)

    # Limits
    v_max: float = 2.5              # m/s (non-negative; no reverse)
    w_max: float = 2.0              # rad/s

    # Drag (first-order viscous)
    lin_drag: float = 0.1           # on v
    ang_drag: float = 0.12          # on w

    # Episode / reward
    goal_radius: float = 0.3
    start_noise: float = 0.2        # randomization at reset
    random_goal: bool = True        # if True, pick a random goal each reset
    goal_margin: float = 1.0        # min distance from world edge for random goal

    # Rocks (hazards)
    num_rocks: int = 6
    rock_radius: float = 0.5
    rock_clearance: float = 1.2       # min distance from start & goal (center-to-center)
    rock_min_separation: float = 1.0  # min distance between rocks

    # Rewards
    success_bonus: float = 100.0
    crash_penalty: float = -100.0     # penalty when hitting a rock or AI ship

    # ---- Raycasting (does NOT affect physics) ----
    num_rays: int = 8                 # 0 keeps old observation without rays
    include_goal_in_rays: bool = False  # treat goal as a hit (small radius) if True

    # Ship-ship collision model (for AI collision with player & ray hits)
    ship_radius: float = 0.3


# =========================
# AI Parameters
# =========================
@dataclass
class AIParams:
    # Toggles
    enable_backforth: bool = True
    enable_waypoints: bool = True
    enable_chasers: bool = True

    # Counts (default 1 of each)
    num_backforth: int = 1
    num_waypoints: int = 1
    num_chasers: int = 1

    # Behavior tuning
    backforth_min_len: float = 3.0
    backforth_max_len: float = 6.0
    waypoint_count_min: int = 3   # will choose 3–4 far-apart waypoints
    waypoint_count_max: int = 4
    waypoint_stop_radius: float = 0.25

    # Simple speed preference for AI (they use same physics)
    pref_speed: float = 0.16


# =========================
# Graphics Parameters
# =========================
@dataclass
class GraphicsParams:
    # Canvas mapping
    render_scale_px_per_m: int = 50
    render_padding_px: int = 40

    # Ship geometry (fallback triangle)
    ship_length_m: float = 0.6
    ship_width_m: float = 0.35

    # Water look & feel
    water_primary: Tuple[int, int, int] = (120, 170, 220)
    water_secondary: Tuple[int, int, int] = (90, 140, 200)
    water_wave_px: int = 24
    water_scroll_px_per_step: int = 0

    # Trail
    trail_len: int = 100
    trail_color: Tuple[int, int, int] = (127, 0, 0)

    # FPS (human mode)
    fps_limit: int = 60

    # Optional sprite (None => try shipenv/assets/ship.png)
    sprite_path: Optional[str] = None
    sprite_meters_long: float = 0.9

    # Sprite alignment tweaks
    sprite_heading_deg_offset: float = 0.0
    sprite_px_offset: Tuple[int, int] = (0, 0)

    # Debug overlays
    show_velocity_vector: bool = False
    show_debug_triangle: bool = False

    # Rays (debug rendering)
    show_rays: bool = False
    ray_color: Tuple[int, int, int] = (255, 255, 255)
    ray_hit_color: Tuple[int, int, int] = (0, 0, 0)

    # Rocks appearance
    rock_fill: Tuple[int, int, int] = (110, 110, 110)
    rock_stroke: Tuple[int, int, int] = (40, 40, 40)

    # Goal appearance
    goal_color: Tuple[int, int, int] = (0, 180, 0)

    # Optional overlays
    show_grid: bool = False

    # AI ship colors
    ai_backforth_color: Tuple[int, int, int] = (230, 160, 50)
    ai_waypoint_color:  Tuple[int, int, int] = (80, 200, 120)
    ai_chaser_color:    Tuple[int, int, int] = (200, 80, 80)


class ShipEnv(gym.Env):
    """
    No lateral slip. One action per step; turning and moving are exclusive.

    Action space: Discrete(4)
        0: Turn left
        1: Turn right
        2: Accelerate
        3: Brake

    Kinematics (unchanged):
        * Turn steps (0/1): rotate (dx,dy) by ±torque*dt; drag on v; no x,y move.
        * Move steps (2/3): lock rotation; update v; move x+=v*dx*dt, y-=v*dy*dt.
        * v ∈ [0, v_max], no reverse.

    Observation:
        [x, y, dx, dy, v, goal_x, goal_y] + (optional rays)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    __slots__ = (
        "P","G","AI","render_mode","_render_on","_fast_headless",
        "action_space","observation_space",
        "x","y","v","dx","dy","goal","steps","rocks",
        "_screen","_clock","_surf","_water_surf","_water_offset",
        "_sprite_img","_sprite_img_scaled","_trail","_obs_buf",
        "_ray_dirs","_last_ray_ds","_world_px","_win_w","_win_h",
        "_dt","_vmax","_lin_drag_dt","_thrust_dt","_brake_dt",
        "_cos_rot","_sin_rot","_neg_sin_rot",
        "_ai_ships"
    )

    def __init__(
        self,
        params: Optional[ShipParams] = None,
        graphics: Optional[GraphicsParams] = None,
        render_mode: Optional[str] = None,
        goal: Optional[Tuple[float, float]] = None,
        ai: Optional[AIParams] = None,
    ):
        super().__init__()
        self.P = params or ShipParams()
        self.G = graphics or GraphicsParams()
        self.AI = ai or AIParams()
        self.render_mode = render_mode

        # Flags for rendering / headless
        self._render_on = render_mode in ("human", "rgb_array")
        self._fast_headless = not self._render_on

        # Discrete(4) actions
        self.action_space = spaces.Discrete(4)

        # Observation space: [x, y, dx, dy, v, goal_x, goal_y] + (optional rays)
        L = self.P.world_size
        base_low  = [-L, -L, -1.0, -1.0, 0.0, -2*L, -2*L]
        base_high = [ L,  L,  1.0,  1.0, self.P.v_max, 2*L, 2*L]

        if self.P.num_rays > 0:
            ray_high = 2.0 * math.sqrt(2.0) * L
            base_low  += [0.0] * self.P.num_rays
            base_high += [ray_high] * self.P.num_rays

        self.observation_space = spaces.Box(
            low=np.array(base_low, dtype=np.float32),
            high=np.array(base_high, dtype=np.float32),
            dtype=np.float32
        )

        # State
        self.x = self.y = 0.0
        self.v = 0.0
        self.dx = 1.0
        self.dy = 0.0
        self.goal = np.array(goal if goal is not None else [0.0, 0.0], dtype=np.float32)
        self.steps = 0

        # Rocks
        self.rocks: List[Tuple[float, float]] = []

        # Rendering
        self._screen = None
        self._clock = None
        self._surf = None
        a = self.G.render_scale_px_per_m
        pad = self.G.render_padding_px
        self._world_px = int(2 * L * a)
        self._win_w = self._world_px + 2 * pad
        self._win_h = self._world_px + 2 * pad
        self._water_surf = None
        self._water_offset = 0
        self._sprite_img = None
        self._sprite_img_scaled = None
        self._trail: List[Tuple[float, float]] = []

        # Preallocated observation buffer (7 base dims + rays)
        self._obs_buf = np.zeros(7 + max(0, self.P.num_rays), dtype=np.float32)

        # Precompute ray directions
        self._ray_dirs: List[Tuple[float, float]] = []
        if self.P.num_rays > 0:
            two_pi = 2.0 * math.pi
            for k in range(self.P.num_rays):
                ang = two_pi * (k / self.P.num_rays)
                self._ray_dirs.append((math.cos(ang), math.sin(ang)))

        # Cache of last ray distances for rendering
        self._last_ray_ds = np.zeros(self.P.num_rays, dtype=np.float32) if self.P.num_rays > 0 else None

        # AI ships list
        self._ai_ships: List[dict] = []

        # Cache step constants
        self._refresh_step_consts()

    # Cache constants derived from params/dt to reduce per-step overhead
    def _refresh_step_consts(self):
        dt = self.P.dt
        self._dt = dt
        self._vmax = self.P.v_max
        self._lin_drag_dt = self.P.lin_drag * dt
        self._thrust_dt   = self.P.thrust_accel * dt
        self._brake_dt    = self.P.brake_accel  * dt
        ang = self.P.torque_accel * dt
        self._cos_rot = math.cos(ang)
        self._sin_rot = math.sin(ang)
        self._neg_sin_rot = -self._sin_rot

    # -------------------
    # Gym API
    # -------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._refresh_step_consts()
        rng = np.random.default_rng(seed)
        L = self.P.world_size

        # Start pose/speeds
        self.x = float(-0.5 * L + rng.normal(0, self.P.start_noise))
        self.y = float(rng.normal(0, self.P.start_noise))
        self.v = max(0.0, float(rng.normal(0, 0.1)))
        self.dx = 1.0
        self.dy = 0.0
        self.steps = 0

        # Goal
        if options and "goal" in options:
            self.goal = np.array(options["goal"], dtype=np.float32)
        elif self.P.random_goal:
            self.goal = self._sample_goal(rng)

        # Trail
        self._trail = [(self.x, self.y)]

        # Rocks
        self.rocks = self._sample_rocks(rng)

        # AI spawn
        self._spawn_ai(rng)

        # Rendering setup - only if rendering enabled
        if self._render_on:
            if not pygame.get_init():
                pygame.init()
            if self.render_mode == "human":
                if self._screen is None:
                    try:
                        pygame.display.set_caption("ShipEnv")
                        self._screen = pygame.display.set_mode((self._win_w, self._win_h))
                    except pygame.error as e:
                        print(f"Warning: Could not initialize display: {e}")
                        print("Falling back to headless mode")
                        self.render_mode = None
                        self._render_on = False
                        return self._obs(), {"goal": self.goal.copy(), "rocks": list(self.rocks)}
                if self._clock is None:
                    self._clock = pygame.time.Clock()
            self._surf = pygame.Surface((self._win_w, self._win_h)).convert()
            self._water_surf = pygame.Surface((self._win_w, self._win_h)).convert()
            self._build_water(self._water_surf)
            self._water_offset = 0

            # Sprite (packaged default if none provided)
            self._sprite_img = self._sprite_img_scaled = None
            sprite_path = self.G.sprite_path
            if sprite_path is None:
                try:
                    candidate = ir.files("shipenv").joinpath("assets/ship.png")
                    if candidate.is_file():
                        sprite_path = str(candidate)
                except Exception:
                    sprite_path = None
            if sprite_path:
                try:
                    raw = pygame.image.load(sprite_path).convert_alpha()
                    self._sprite_img = raw
                    px_per_m = self.G.render_scale_px_per_m
                    target_len_px = max(1, int(self.G.sprite_meters_long * px_per_m))
                    w0, h0 = raw.get_size()
                    scale = target_len_px / max(w0, h0)
                    self._sprite_img_scaled = pygame.transform.smoothscale(
                        raw, (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
                    )
                except Exception as e:
                    print(f"[ShipEnv] Sprite load failed ({sprite_path}): {e}")
                    self._sprite_img = self._sprite_img_scaled = None
        else:
            # Fast mode: skip all graphics initialization
            self._screen = None
            self._clock = None
            self._surf = None
            self._water_surf = None
            self._sprite_img = None
            self._sprite_img_scaled = None

        return self._obs(), {"goal": self.goal.copy(), "rocks": list(self.rocks)}

    def step(self, action: int):
        if action not in (0, 1, 2, 3):
            raise ValueError("Invalid action for Discrete(4).")

        # --- Player physics (unchanged) using cached-step constants ---
        x, y, dx, dy, v = self.x, self.y, self.dx, self.dy, self.v
        x, y, dx, dy, v = self._apply_action(x, y, dx, dy, v, action)
        self.x, self.y, self.dx, self.dy, self.v = x, y, dx, dy, v

        # Trail (rendering only)
        if self._render_on:
            self._trail.append((x, y))
            if len(self._trail) > self.G.trail_len:
                self._trail.pop(0)

        self.steps += 1

        # Distances / termination
        dxg = x - float(self.goal[0])
        dyg = y - float(self.goal[1])
        dist_goal = math.hypot(dxg, dyg)
        destroyed = self._check_rock_collision(x, y)  # player vs rocks only

        reward = 0
        # If not moving, penalize
        if self.v < 0.01:
            reward -= 0.5

        # Reward based on distance to goal
        reward -= dist_goal * 0.1

        terminated = False
        if destroyed:
            reward += self.P.crash_penalty
            terminated = True

        reached = (dist_goal <= self.P.goal_radius) and not destroyed
        if reached:
            reward += self.P.success_bonus
            terminated = True

        out_of_bounds = (abs(x) > self.P.world_size or abs(y) > self.P.world_size)
        truncated = out_of_bounds or (self.steps >= self.P.max_steps)

        # --- Advance AI and check player-vs-AI collision (AI ignore rocks) ---
        self._step_ai()
        if not terminated and self._check_ai_collision_with_player():
            reward += self.P.crash_penalty
            terminated = True

        info = {
            "distance": dist_goal,
            "out_of_bounds": out_of_bounds,
            "goal": self.goal.copy(),
            "destroyed": destroyed,
            "reached_goal": reached,
        }

        # Render - only if rendering is enabled
        if self._render_on:
            frame = self._render_frame()
            if self.render_mode == "rgb_array":
                info["frame"] = frame

        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self._render_on:
            self._render_frame()

    def close(self):
        try:
            if pygame.get_init():
                pygame.display.quit()
                pygame.quit()
        except Exception:
            pass
        self._screen = self._surf = self._clock = None
        self._water_surf = None
        self._sprite_img = self._sprite_img_scaled = None

    # -------------------
    # Internals
    # -------------------
    def _obs(self):
        buf = self._obs_buf
        # base features
        buf[0] = self.x
        buf[1] = self.y
        buf[2] = self.dx
        buf[3] = self.dy
        buf[4] = self.v
        # goal relative position
        buf[5] = float(self.goal[0] - self.x)
        buf[6] = float(self.goal[1] - self.y)
        # rays (if any)
        if self.P.num_rays > 0:
            self._compute_rays_into(buf, offset=7)
        return buf

    # Physics application used by both player and AI (logic unchanged)
    def _apply_action(self, x, y, dx, dy, v, action):
        dt = self._dt
        v_max = self._vmax
        lin_drag_dt = self._lin_drag_dt
        thrust_dt = self._thrust_dt
        brake_dt = self._brake_dt
        c = self._cos_rot
        s = self._sin_rot
        ns = self._neg_sin_rot

        if action == 0:   # turn left
            ndx = dx * c + dy * s
            ndy = -dx * s + dy * c
            dx, dy = ndx, ndy
            v -= lin_drag_dt * v
            if v < 0.0:
                v = 0.0
        elif action == 1:  # turn right
            ndx = dx * c + dy * ns
            ndy = -dx * ns + dy * c
            dx, dy = ndx, ndy
            v -= lin_drag_dt * v
            if v < 0.0:
                v = 0.0
        elif action == 2:  # accelerate
            v += thrust_dt - lin_drag_dt * v
            if v > v_max:
                v = v_max
        else:  # 3: brake
            v += -brake_dt - lin_drag_dt * v
            if v < 0.0:
                v = 0.0

        x += v * dx * dt
        y -= v * dy * dt
        return x, y, dx, dy, v

    # --- Raycasting (geometry only; no physics effects) ---
    def _compute_rays_into(self, out: np.ndarray, offset: int):
        if not self._ray_dirs:
            return
        x0 = self.x
        y0 = self.y
        L = self.P.world_size
        rr = self.P.rock_radius
        ship_r = self.P.ship_radius
        min_x = -L; max_x = +L
        min_y = -L; max_y = +L

        include_goal = self.P.include_goal_in_rays
        r2_rock = rr * rr
        r2_ship = ship_r * ship_r

        ray_dirs = self._ray_dirs
        rocks = self.rocks
        last_ray = self._last_ray_ds
        ai = self._ai_ships

        for i, (dx, dy) in enumerate(ray_dirs):
            # Distance to box edges
            t_edge = self._ray_to_box(x0, y0, dx, dy, min_x, max_x, min_y, max_y)

            # Nearest object: rocks or AI ships (treated as circles)
            t_obj = float('inf')
            for (cx, cy) in rocks:
                t = self._ray_circle_intersect(x0, y0, dx, dy, cx, cy, r2_rock)
                if t is not None and 0.0 <= t < t_obj:
                    t_obj = t
            if ai:
                for s in ai:
                    cx, cy = s["x"], s["y"]
                    t = self._ray_circle_intersect(x0, y0, dx, dy, cx, cy, r2_ship)
                    if t is not None and 0.0 <= t < t_obj:
                        t_obj = t

            # Optional: goal as small target
            if include_goal:
                gx = float(self.goal[0]); gy = float(self.goal[1])
                gr = max(0.05, rr * 0.25)
                t = self._ray_circle_intersect(x0, y0, dx, dy, gx, gy, gr * gr)
                if t is not None and 0.0 <= t < t_obj:
                    t_obj = t

            d_hit = t_obj if t_obj < t_edge else t_edge
            out[offset + i] = d_hit
            if last_ray is not None:
                last_ray[i] = d_hit

    @staticmethod
    def _ray_circle_intersect(px: float, py: float, dx: float, dy: float,
                              cx: float, cy: float, r2: float) -> Optional[float]:
        mx = px - cx
        my = py - cy
        b = mx * dx + my * dy
        c = mx * mx + my * my - r2

        if c <= 0.0:
            return 0.0

        if b > 0.0:
            return None

        disc = b * b - c
        if disc < 0.0:
            return None

        t = -b - math.sqrt(disc)
        return t if t >= 0.0 else None

    @staticmethod
    def _ray_to_box(px: float, py: float, dx: float, dy: float,
                    min_x: float, max_x: float, min_y: float, max_y: float) -> float:
        candidates: List[float] = []

        if abs(dx) > 1e-12:
            tx1 = (min_x - px) / dx
            tx2 = (max_x - px) / dx
            if tx1 > 0.0: candidates.append(tx1)
            if tx2 > 0.0: candidates.append(tx2)
        if abs(dy) > 1e-12:
            ty1 = (min_y - py) / dy
            ty2 = (max_y - py) / dy
            if ty1 > 0.0: candidates.append(ty1)
            if ty2 > 0.0: candidates.append(ty2)

        if not candidates:
            return float('inf')

        for t in sorted(candidates):
            if t <= 0.0:
                continue
            x = px + t * dx
            y = py + t * dy
            if (min_x - 1e-9) <= x <= (max_x + 1e-9) and (min_y - 1e-9) <= y <= (max_y + 1e-9):
                return t

        return min(candidates)

    # --- Random sampling helpers
    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        L = self.P.world_size
        m = self.P.goal_margin
        gx = float(rng.uniform(-L + m, L - m))
        gy = float(rng.uniform(-L + m, L - m))
        return np.array([gx, gy], dtype=np.float32)

    def _sample_rocks(self, rng: np.random.Generator) -> List[Tuple[float, float]]:
        # Fast mode: simplified rock placement for speed
        if self._fast_headless:
            rocks: List[Tuple[float, float]] = []
            L = self.P.world_size
            rr = self.P.rock_radius
            min_xy = -L + rr + 0.05
            max_xy = L - rr - 0.05
            for _ in range(self.P.num_rocks):
                rx = float(rng.uniform(min_xy, max_xy))
                ry = float(rng.uniform(min_xy, max_xy))
                rocks.append((rx, ry))
            return rocks

        # Original placement for rendering mode
        rocks: List[Tuple[float, float]] = []
        L = self.P.world_size
        rr = self.P.rock_radius
        clear = max(self.P.rock_clearance, rr + 0.1)
        sep = max(self.P.rock_min_separation, 0.0)
        max_tries = 2000
        min_xy = -L + rr + 0.05
        max_xy = L - rr - 0.05

        def far_enough(ax, ay, bx, by, dmin):
            return (ax - bx) ** 2 + (ay - by) ** 2 >= dmin * dmin

        for _ in range(self.P.num_rocks):
            placed = False
            for _attempt in range(max_tries):
                rx = float(rng.uniform(min_xy, max_xy))
                ry = float(rng.uniform(min_xy, max_xy))
                if not far_enough(rx, ry, self.x, self.y, clear):
                    continue
                if not far_enough(rx, ry, float(self.goal[0]), float(self.goal[1]), clear):
                    continue
                ok = True
                for (ox, oy) in rocks:
                    if not far_enough(rx, ry, ox, oy, max(sep, 2 * rr * 0.9)):
                        ok = False
                        break
                if ok:
                    rocks.append((rx, ry))
                    placed = True
                    break
            if not placed:
                rx = float(rng.uniform(min_xy, max_xy))
                ry = float(rng.uniform(min_xy, max_xy))
                rocks.append((rx, ry))
        return rocks

    def _check_rock_collision(self, x: float, y: float) -> bool:
        if not self.rocks:
            return False
        rr = self.P.rock_radius
        rr2 = rr * rr
        for (rx, ry) in self.rocks:
            dx = x - rx
            dy = y - ry
            if dx * dx + dy * dy <= rr2:
                return True
        return False

    # -------------------
    # AI: spawn / control / collisions
    # -------------------
    def _spawn_ai(self, rng: np.random.Generator):
        self._ai_ships.clear()
        if not (self.AI.enable_backforth or self.AI.enable_waypoints or self.AI.enable_chasers):
            return

        L = self.P.world_size
        rr = max(self.P.rock_radius, self.P.ship_radius)
        min_xy = -L + rr + 0.05
        max_xy =  L - rr - 0.05

        def rand_pos():
            return float(rng.uniform(min_xy, max_xy)), float(rng.uniform(min_xy, max_xy))

        # Back-and-forth ships
        if self.AI.enable_backforth and self.AI.num_backforth > 0:
            for _ in range(self.AI.num_backforth):
                x0, y0 = rand_pos()
                ang = rng.uniform(0, 2*math.pi)
                seg_len = rng.uniform(self.AI.backforth_min_len, self.AI.backforth_max_len)
                ux = math.cos(ang); uy = math.sin(ang)
                x1 = x0 + 0.5 * seg_len * ux; y1 = y0 + 0.5 * seg_len * uy
                x2 = x0 - 0.5 * seg_len * ux; y2 = y0 - 0.5 * seg_len * uy
                self._ai_ships.append(dict(
                    kind="bf",
                    x=x1, y=y1, dx=ux, dy=uy, v=0.0,
                    a=(x1, y1), b=(x2, y2), target="b"
                ))

        # Waypoint ships (3–4 waypoints far apart)
        if self.AI.enable_waypoints and self.AI.num_waypoints > 0:
            for _ in range(self.AI.num_waypoints):
                x, y = rand_pos()
                # choose count in [min, max]
                n_wp = int(rng.integers(self.AI.waypoint_count_min, self.AI.waypoint_count_max + 1))
                waypts: List[Tuple[float, float]] = []

                # target pairwise separation: ~ half the world size
                min_sep = 0.5 * L
                tries = 0
                while len(waypts) < n_wp and tries < 5000:
                    tries += 1
                    c = rand_pos()
                    if all((c[0]-wx)**2 + (c[1]-wy)**2 >= (min_sep*min_sep) for (wx, wy) in waypts):
                        waypts.append(c)

                if not waypts:
                    waypts = [rand_pos() for __ in range(n_wp)]  # fallback

                self._ai_ships.append(dict(
                    kind="wp",
                    x=x, y=y, dx=1.0, dy=0.0, v=0.0,
                    waypoints=waypts, idx=0, stopped=False
                ))

        # Chasers
        if self.AI.enable_chasers and self.AI.num_chasers > 0:
            for _ in range(self.AI.num_chasers):
                x, y = rand_pos()
                self._ai_ships.append(dict(
                    kind="chase",
                    x=x, y=y, dx=1.0, dy=0.0, v=0.0
                ))

    def _ai_pick_action_towards(self, sx, sy, sdx, sdy, tx, ty):
        # Desired heading vector towards (tx,ty)
        vx = tx - sx; vy = ty - sy
        if vx*vx + vy*vy < 1e-12:
            return 3  # brake
        desired = math.atan2(-vy, vx)   # screen-inverted y matches integrator
        cur = math.atan2(-sdy, sdx)
        d = desired - cur
        d = (d + math.pi) % (2*math.pi) - math.pi
        if abs(d) > 1e-4:
            return 0 if d > 0.0 else 1
        return 2

    def _step_ai(self):
        if not self._ai_ships:
            return
        L = self.P.world_size
        pref_v = self.AI.pref_speed
        r = self.P.ship_radius

        for s in self._ai_ships:
            kind = s["kind"]
            x, y, dx, dy, v = s["x"], s["y"], s["dx"], s["dy"], s["v"]

            if kind == "bf":
                ax, ay = s["a"]; bx, by = s["b"]
                tgt = (bx, by) if s["target"] == "b" else (ax, ay)
                act = self._ai_pick_action_towards(x, y, dx, dy, tgt[0], tgt[1])
                x, y, dx, dy, v = self._apply_action(x, y, dx, dy, v, act)
                # regulate speed
                if v < pref_v * 0.98:
                    x, y, dx, dy, v = self._apply_action(x, y, dx, dy, v, 2)
                elif v > pref_v * 1.02:
                    x, y, dx, dy, v = self._apply_action(x, y, dx, dy, v, 3)
                # switch end when close
                if (x - tgt[0])**2 + (y - tgt[1])**2 <= (r*r):
                    s["target"] = "a" if s["target"] == "b" else "b"

            elif kind == "wp":
                if s["stopped"]:
                    x, y, dx, dy, v = self._apply_action(x, y, dx, dy, v, 3)
                else:
                    tx, ty = s["waypoints"][s["idx"]]
                    act = self._ai_pick_action_towards(x, y, dx, dy, tx, ty)
                    x, y, dx, dy, v = self._apply_action(x, y, dx, dy, v, act)
                    x, y, dx, dy, v = self._apply_action(x, y, dx, dy, v, 2)
                    if (x - tx)**2 + (y - ty)**2 <= (self.AI.waypoint_stop_radius**2):
                        if s["idx"] + 1 < len(s["waypoints"]):
                            s["idx"] += 1
                        else:
                            s["stopped"] = True

            else:  # "chase" — always target the PLAYER (x,y)
                tx, ty = self.x, self.y
                act = self._ai_pick_action_towards(x, y, dx, dy, tx, ty)
                x, y, dx, dy, v = self._apply_action(x, y, dx, dy, v, act)
                x, y, dx, dy, v = self._apply_action(x, y, dx, dy, v, 2)

            # keep in bounds with simple reflect
            if abs(x) > L:
                x = max(-L, min(L, x))
                dx = -dx
            if abs(y) > L:
                y = max(-L, min(L, y))
                dy = -dy

            s["x"], s["y"], s["dx"], s["dy"], s["v"] = x, y, dx, dy, v

    def _check_ai_collision_with_player(self) -> bool:
        if not self._ai_ships:
            return False
        r2 = (self.P.ship_radius * 2.0) ** 2
        x, y = self.x, self.y
        for s in self._ai_ships:
            dx = x - s["x"]; dy = y - s["y"]
            if dx*dx + dy*dy <= r2:
                return True
        return False

    # --- Rendering helpers
    def _world_to_screen(self, x, y):
        s = self.G.render_scale_px_per_m
        pad = self.G.render_padding_px
        L = self.P.world_size
        sx = pad + int((x + L) * s)
        sy = pad + int((L - y) * s)  # invert y for screen coords
        return sx, sy

    def _build_water(self, surf: pygame.Surface):
        w, h = surf.get_size()
        cell = self.G.water_wave_px
        base = self.G.water_primary
        dark = self.G.water_secondary
        surf.fill(base)

        import math as _m
        for y in range(0, h, cell):
            amp = cell * 0.35
            for x in range(0, w, 2):
                y2 = int(y + amp * _m.sin((x / cell) * 2.2) * 0.6)
                if 0 <= y2 < h:
                    surf.set_at((x, y2), dark)

        highlight = (min(255, base[0] + 28), min(255, base[1] + 28), min(255, base[2] + 28))
        step = max(6, cell // 2)
        for d in range(-h, w, step):
            pygame.draw.aaline(surf, highlight, (d, 0), (d + h, h))

    def _handle_input(self):
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFTBRACKET:
                        self.G.sprite_heading_deg_offset -= 5.0
                        print(f"sprite_heading_deg_offset = {self.G.sprite_heading_deg_offset}")
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.G.sprite_heading_deg_offset += 5.0
                        print(f"sprite_heading_deg_offset = {self.G.sprite_heading_deg_offset}")
                    elif event.key == pygame.K_LEFT:
                        x, y = self.G.sprite_px_offset
                        self.G.sprite_px_offset = (x - 1, y)
                        print(f"sprite_px_offset = {self.G.sprite_px_offset}")
                    elif event.key == pygame.K_RIGHT:
                        x, y = self.G.sprite_px_offset
                        self.G.sprite_px_offset = (x + 1, y)
                        print(f"sprite_px_offset = {self.G.sprite_px_offset}")
                    elif event.key == pygame.K_UP:
                        x, y = self.G.sprite_px_offset
                        self.G.sprite_px_offset = (x, y - 1)
                        print(f"sprite_px_offset = {self.G.sprite_px_offset}")
                    elif event.key == pygame.K_DOWN:
                        x, y = self.G.sprite_px_offset
                        self.G.sprite_px_offset = (x, y + 1)
                        print(f"sprite_px_offset = {self.G.sprite_px_offset}")
                    elif event.key == pygame.K_v:
                        self.G.show_velocity_vector = not self.G.show_velocity_vector
                        print(f"show_velocity_vector = {self.G.show_velocity_vector}")
                    elif event.key == pygame.K_t:
                        self.G.show_debug_triangle = not self.G.show_debug_triangle
                        self.G.show_rays = not self.G.show_rays
                        print(f"show_debug_triangle = {self.G.show_debug_triangle}, show_rays = {self.G.show_rays}")
                    elif event.key == pygame.K_0:
                        self.G.sprite_heading_deg_offset = 0.0
                        self.G.sprite_px_offset = (0, 0)
                        print("sprite_heading_deg_offset=0, sprite_px_offset=(0,0)")
        except pygame.error:
            pass

    def _draw_water(self):
        if not self._water_surf:
            return
        self._water_offset = (self._water_offset + self.G.water_scroll_px_per_step) % self._win_h
        off = self._water_offset
        self._surf.blit(self._water_surf, (0, off - self._win_h))
        self._surf.blit(self._water_surf, (0, off))

    def _draw_goal(self, surf):
        gx, gy = self.goal
        sx, sy = self._world_to_screen(gx, gy)
        radius = max(2, int(self.P.goal_radius * self.G.render_scale_px_per_m))
        pygame.draw.circle(surf, self.G.goal_color, (sx, sy), radius, 2)

    def _draw_trail(self, surf):
        if len(self._trail) < 2:
            return
        pts = [self._world_to_screen(px, py) for (px, py) in self._trail]
        pygame.draw.lines(surf, self.G.trail_color, False, pts, 2)

    def _draw_rocks(self, surf):
        if not self.rocks:
            return
        r_px = max(1, int(self.P.rock_radius * self.G.render_scale_px_per_m))
        for (rx, ry) in self.rocks:
            sx, sy = self._world_to_screen(rx, ry)
            pygame.draw.circle(surf, self.G.rock_fill, (sx, sy), r_px)
            pygame.draw.circle(surf, self.G.rock_stroke, (sx, sy), r_px, 2)

    def _draw_rays(self, surf):
        if not self._render_on or not self.P.num_rays or not self.G.show_rays or self._last_ray_ds is None:
            return
        x0, y0 = self.x, self.y
        sx0, sy0 = self._world_to_screen(x0, y0)
        for i, (dx, dy) in enumerate(self._ray_dirs):
            d = float(self._last_ray_ds[i])
            if d <= 0.0 or not math.isfinite(d):
                continue
            ex = x0 + d * dx
            ey = y0 + d * dy
            sxe, sye = self._world_to_screen(ex, ey)
            pygame.draw.aaline(surf, self.G.ray_color, (sx0, sy0), (sxe, sye))
            pygame.draw.circle(surf, self.G.ray_hit_color, (sxe, sye), 3)

    def _draw_ai_ships(self, surf):
        if not self._ai_ships:
            return
        Lb = self.G.ship_length_m
        Wb = self.G.ship_width_m
        body = np.array([[+0.5 * Lb, 0.0], [-0.5 * Lb, +0.5 * Wb], [-0.5 * Lb, -0.5 * Wb]])

        for s in self._ai_ships:
            color = (150, 150, 150)
            if s["kind"] == "bf":
                color = self.G.ai_backforth_color
            elif s["kind"] == "wp":
                color = self.G.ai_waypoint_color
            elif s["kind"] == "chase":
                color = self.G.ai_chaser_color

            R = np.array([[s["dx"], s["dy"]], [-s["dy"], s["dx"]]])
            world = (R @ body.T).T + np.array([s["x"], s["y"]])
            pts = [self._world_to_screen(px, py) for (px, py) in world]
            pygame.draw.polygon(surf, color, pts)
            pygame.draw.polygon(surf, (0, 0, 0), pts, 2)

    def _draw_ship(self, surf):
        cx, cy = self._world_to_screen(self.x, self.y)

        if self._sprite_img_scaled is not None:
            angle_deg = math.degrees(math.atan2(-self.dy, self.dx)) + self.G.sprite_heading_deg_offset
            img = pygame.transform.rotozoom(self._sprite_img_scaled, angle_deg, 1.0)
            rect = img.get_rect()
            rect.center = (cx + self.G.sprite_px_offset[0], cy + self.G.sprite_px_offset[1])
            surf.blit(img, rect)

            if self.G.show_debug_triangle:
                Lb = self.G.ship_length_m
                Wb = self.G.ship_width_m
                body = np.array([[+0.5 * Lb, 0.0], [-0.5 * Lb, +0.5 * Wb], [-0.5 * Lb, -0.5 * Wb]])
                R = np.array([[self.dx, self.dy], [-self.dy, self.dx]])
                world = (R @ body.T).T + np.array([self.x, self.y])
                pts = [self._world_to_screen(px, py) for (px, py) in world]
                pygame.draw.polygon(surf, (255, 100, 100), pts)
                pygame.draw.polygon(surf, (180, 0, 0), pts, 2)
        else:
            Lb = self.G.ship_length_m
            Wb = self.G.ship_width_m
            body = np.array([[+0.5 * Lb, 0.0], [-0.5 * Lb, +0.5 * Wb], [-0.5 * Lb, -0.5 * Wb]])
            R = np.array([[self.dx, self.dy], [-self.dy, self.dx]])
            world = (R @ body.T).T + np.array([self.x, self.y])
            pts = [self._world_to_screen(px, py) for (px, py) in world]
            pygame.draw.polygon(surf, (30, 144, 255), pts)
            pygame.draw.polygon(surf, (0, 60, 120), pts, 2)

        if self.G.show_velocity_vector and self.v > 1e-6:
            arrow_len = max(10, int(self.v * self.G.render_scale_px_per_m * 0.3))
            tip_x = cx + int(arrow_len * self.dx)
            tip_y = cy - int(arrow_len * self.dy)
            pygame.draw.line(surf, (0, 0, 0), (cx, cy), (tip_x, tip_y), 2)
            pygame.draw.circle(surf, (0, 0, 0), (tip_x, tip_y), 3)

    def _render_frame(self):
        self._handle_input()

        if self._surf is None:
            try:
                self._surf = pygame.Surface((self._win_w, self._win_h)).convert()
            except pygame.error as e:
                print(f"Warning: Could not create surface: {e}")
                return None

        self._surf.fill(self.G.water_primary)
        self._draw_water()

        if self.G.show_grid:
            self._draw_grid(self._surf)
        self._draw_rocks(self._surf)
        self._draw_goal(self._surf)
        self._draw_ai_ships(self._surf)
        self._draw_trail(self._surf)
        self._draw_rays(self._surf)
        self._draw_ship(self._surf)

        if self.render_mode == "human":
            if self._screen is None:
                self._screen = pygame.display.set_mode((self._win_w, self._win_h))
            self._screen.blit(self._surf, (0, 0))
            pygame.display.flip()
            if self._clock and self.G.fps_limit:
                self._clock.tick(self.G.fps_limit)
            return None

        if self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(self._surf)  # WxHx3
            return np.transpose(arr, (1, 0, 2)).copy()  # HxWx3

        return None

    # (optional) grid
    def _draw_grid(self, surf):
        L = self.P.world_size
        s = self.G.render_scale_px_per_m
        pad = self.G.render_padding_px
        rect = pygame.Rect(pad, pad, 2 * L * s, 2 * L * s)
        pygame.draw.rect(surf, (200, 200, 200), rect, 2)
        cx, cy = self._world_to_screen(0, 0)
        pygame.draw.line(surf, (220, 220, 220), (pad, cy), (pad + 2 * L * s, cy), 1)
        pygame.draw.line(surf, (220, 220, 220), (cx, pad), (cx, pad + 2 * L * s), 1)
        for i in range(-int(L), int(L) + 1):
            x1, _ = self._world_to_screen(i, 0)
            pygame.draw.line(surf, (235, 235, 235), (x1, pad), (x1, pad + 2 * L * s), 1)
            _, y1 = self._world_to_screen(0, i)
            pygame.draw.line(surf, (235, 235, 235), (pad, y1), (pad + 2 * L * s), 1)
