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
    dt: float = 0.1                 # seconds per step (will be overridden in fast mode)
    max_steps: int = 5000

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
    success_bonus: float = 10.0
    crash_penalty: float = -10.0     # penalty when hitting a rock

    # ---- Raycasting (does NOT affect physics) ----
    num_rays: int = 8                 # 0 keeps old 5-D observation
    include_goal_in_rays: bool = False  # treat goal as a hit (small radius) if True


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
    water_primary: Tuple[int, int, int] = (120, 170, 220)   # base color
    water_secondary: Tuple[int, int, int] = (90, 140, 200)  # darker streaks
    water_wave_px: int = 24
    water_scroll_px_per_step: int = 0   # keep static to avoid visual drift illusions

    # Trail
    trail_len: int = 100
    trail_color: Tuple[int, int, int] = (160, 160, 255)

    # FPS (human mode)
    fps_limit: int = 60

    # Optional sprite (None => try shipenv/assets/ship.png)
    sprite_path: Optional[str] = None
    sprite_meters_long: float = 0.9

    # Sprite alignment tweaks
    sprite_heading_deg_offset: float = 0.0           # rotate sprite to match physics heading
    sprite_px_offset: Tuple[int, int] = (0, 0)       # post-rotation pixel nudge (x,y)

    # Debug overlays
    show_velocity_vector: bool = False               # draw a small v arrow
    show_debug_triangle: bool = False                # draw fallback triangle under sprite for orientation comparison

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
    show_grid: bool = False  # off by default


class ShipEnv(gym.Env):
    """
    No lateral slip. One action per step; turning and moving are exclusive.

    Action space: Discrete(4)
        0: Turn left    (angular acceleration +torque_accel)  -> rotate in place this step
        1: Turn right   (angular acceleration -torque_accel)  -> rotate in place this step
        2: Accelerate   (linear acceleration +thrust_accel)   -> translate straight, no turning
        3: Brake        (linear acceleration -brake_accel)    -> translate straight, no turning; v>=0

    Kinematics (unchanged):
        * Turn steps (0/1): rotate direction unit (dx,dy) by ±torque*dt; drag on v; no x,y move.
        * Move steps (2/3): lock rotation; update v; move x+=v*dx*dt, y-=v*dy*dt.
        * v ∈ [0, v_max], no reverse. (dx,dy) is the ship's heading unit vector.

    Rendering controls (added while keeping 't' semantics):
        [ / ] : sprite heading offset ±5°
        Arrow keys : sprite pixel nudge
        V : toggle velocity vector
        T : toggle debug triangle AND rays together
        0 : reset sprite offsets
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        params: Optional[ShipParams] = None,
        graphics: Optional[GraphicsParams] = None,
        render_mode: Optional[str] = None,
        goal: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.P = params or ShipParams()
        self.G = graphics or GraphicsParams()
        self.render_mode = render_mode

        # Flags for rendering / headless
        self._render_on = render_mode in ("human", "rgb_array")
        self._fast_headless = not self._render_on  # True when render_mode is None

        # Discrete(4) actions (see docstring)
        self.action_space = spaces.Discrete(4)

        # ----- Observation space: [x, y, dx, dy, v] + (optional rays) -----
        L = self.P.world_size
        base_low  = [-L, -L, -1.0, -1.0, 0.0]
        base_high = [ L,  L,  1.0,  1.0, self.P.v_max]

        # Each ray distance ∈ [0, 2√2 L] (corner-to-corner upper bound)
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
        self.v = 0.0   # scalar speed (always >= 0)
        self.dx = 1.0  # direction unit vector x component
        self.dy = 0.0  # direction unit vector y component
        self.goal = np.array(goal if goal is not None else [0.0, 0.0], dtype=np.float32)
        self.steps = 0

        # Rocks
        self.rocks: List[Tuple[float, float]] = []

        # Rendering
        self._screen = None
        self._clock = None
        self._surf = None
        self._water_surf = None
        self._water_offset = 0
        self._sprite_img = None
        self._sprite_img_scaled = None
        self._trail: List[Tuple[float, float]] = []

        # Preallocated observation buffer
        self._obs_buf = np.zeros(5 + max(0, self.P.num_rays), dtype=np.float32)

        # Precompute ray directions (global bearings: 0..2π)
        self._ray_dirs: List[Tuple[float, float]] = []
        if self.P.num_rays > 0:
            two_pi = 2.0 * math.pi
            for k in range(self.P.num_rays):
                ang = two_pi * (k / self.P.num_rays)
                self._ray_dirs.append((math.cos(ang), math.sin(ang)))

        # Cache of last ray distances for rendering
        self._last_ray_ds = np.zeros(self.P.num_rays, dtype=np.float32) if self.P.num_rays > 0 else None

        # Pixel geometry
        s = self.G.render_scale_px_per_m
        pad = self.G.render_padding_px
        self._world_px = int(2 * L * s)
        self._win_w = self._world_px + 2 * pad
        self._win_h = self._world_px + 2 * pad

    # -------------------
    # Gym API
    # -------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        L = self.P.world_size

        # Start pose/speeds (PHYSICS UNCHANGED)
        self.x = float(-0.5 * L + rng.normal(0, self.P.start_noise))
        self.y = float(rng.normal(0, self.P.start_noise))
        self.v = max(0.0, float(rng.normal(0, 0.1)))  # non-negative
        self.dx = 1.0  # start facing +x direction
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

        # Rendering setup - only if rendering enabled
        if self._render_on:
            if not pygame.get_init():
                pygame.init()
            if self.render_mode == "human":
                if self._screen is None:
                    pygame.display.set_caption("ShipEnv")
                    self._screen = pygame.display.set_mode((self._win_w, self._win_h))
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

        dt = self.P.dt
        lin_drag = self.P.lin_drag
        ang_drag = self.P.ang_drag
        v_max = self.P.v_max
        w_max = self.P.w_max
        thrust = self.P.thrust_accel
        brake = self.P.brake_accel
        torque = self.P.torque_accel

        # Local refs for speed
        x = self.x
        y = self.y
        v = self.v
        dx = self.dx
        dy = self.dy

        # ---- PHYSICS UNCHANGED BELOW ----
        if action == 0:   # turn left
            # Rotate direction vector left by torque amount
            angle_change = torque * dt
            cos_angle = math.cos(angle_change)
            sin_angle = math.sin(angle_change)
            new_dx = dx * cos_angle + dy * sin_angle
            new_dy = -dx * sin_angle + dy * cos_angle
            dx, dy = new_dx, new_dy
            v -= (lin_drag * v) * dt
            if v < 0.0:
                v = 0.0

        elif action == 1:  # turn right
            # Rotate direction vector right by torque amount
            angle_change = -torque * dt
            cos_angle = math.cos(angle_change)
            sin_angle = math.sin(angle_change)
            new_dx = dx * cos_angle + dy * sin_angle
            new_dy = -dx * sin_angle + dy * cos_angle
            dx, dy = new_dx, new_dy
            v -= (lin_drag * v) * dt
            if v < 0.0:
                v = 0.0

        elif action == 2:  # accelerate in ship direction
            v += (thrust - lin_drag * v) * dt
            if v > v_max:
                v = v_max

        elif action == 3:  # brake in ship direction
            v += (-brake - lin_drag * v) * dt
            if v < 0.0:
                v = 0.0

        # Update position: apply velocity in ship's direction
        x += v * dx * dt
        y -= v * dy * dt  # screen-inverted Y convention
        # ---- END PHYSICS ----

        # Commit locals back to state
        self.x = x
        self.y = y
        self.v = v
        self.dx = dx
        self.dy = dy

        # Trail (rendering only)
        if self._render_on:
            self._trail.append((x, y))
            if len(self._trail) > self.G.trail_len:
                self._trail.pop(0)

        self.steps += 1

        # Distances / termination (use math.hypot for speed)
        dxg = x - float(self.goal[0])
        dyg = y - float(self.goal[1])
        dist_goal = math.hypot(dxg, dyg)
        destroyed = self._check_rock_collision(x, y)

        # Shaping + small action cost
        reward = -dist_goal - 0.01

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
        # Base state
        buf[0] = self.x
        buf[1] = self.y
        buf[2] = self.dx
        buf[3] = self.dy
        buf[4] = self.v

        # Rays appended after base (does not affect physics)
        if self.P.num_rays > 0:
            self._compute_rays_into(buf, offset=5)

        return buf

    @staticmethod
    def _wrap_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    # --- Raycasting (geometry only; no physics effects) ---
    def _compute_rays_into(self, out: np.ndarray, offset: int):
        """
        Cast num_rays global-bearing rays from (x,y) and write distances into out[offset:].
        Distances are to nearest rock circle; if none, distance to the world boundary.
        Optionally treat goal as a small circle if include_goal_in_rays=True.
        """
        if not self._ray_dirs:
            return
        x0 = self.x
        y0 = self.y
        L = self.P.world_size
        rr = self.P.rock_radius
        min_x = -L; max_x = +L
        min_y = -L; max_y = +L

        include_goal = self.P.include_goal_in_rays
        r2 = rr * rr

        for i, (dx, dy) in enumerate(self._ray_dirs):
            # Distance to box edges
            t_edge = self._ray_to_box(x0, y0, dx, dy, min_x, max_x, min_y, max_y)

            # Nearest rock hit
            t_obj = float('inf')
            for (cx, cy) in self.rocks:
                t = self._ray_circle_intersect(x0, y0, dx, dy, cx, cy, r2)
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
            if self._last_ray_ds is not None:
                self._last_ray_ds[i] = d_hit

    @staticmethod
    def _ray_circle_intersect(px: float, py: float, dx: float, dy: float,
                              cx: float, cy: float, r2: float) -> Optional[float]:
        """
        Ray from P=(px,py) along D=(dx,dy) (unit) vs circle center C, radius^2=r2.
        Returns the smallest non-negative t if hit, else None.
        """
        mx = px - cx
        my = py - cy
        b = mx * dx + my * dy            # dot(m, d)
        c = mx * mx + my * my - r2       # dot(m,m) - r^2

        if c <= 0.0:
            return 0.0                   # starting inside -> distance 0

        if b > 0.0:
            return None                  # pointing away

        disc = b * b - c
        if disc < 0.0:
            return None

        t = -b - math.sqrt(disc)
        return t if t >= 0.0 else None

    @staticmethod
    def _ray_to_box(px: float, py: float, dx: float, dy: float,
                    min_x: float, max_x: float, min_y: float, max_y: float) -> float:
        """
        Distance along ray from (px,py) direction (dx,dy) to the AABB edges.
        Returns the smallest positive t that lands on a valid boundary segment.
        """
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

        # Prefer the first candidate that actually hits the box extent
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

        # Original complex rock placement for rendering mode
        rocks: List[Tuple[float, float]] = []
        L = self.P.world_size
        rr = self.P.rock_radius
        clear = max(self.P.rock_clearance, rr + 0.1)
        sep = max(self.P.rock_min_separation, 0.0)
        max_tries = 2000

        # ensure we don't place outside the box considering radius & a tiny safety margin
        min_xy = -L + rr + 0.05
        max_xy = L - rr - 0.05

        def far_enough(ax, ay, bx, by, dmin):
            return (ax - bx) ** 2 + (ay - by) ** 2 >= dmin * dmin

        for _ in range(self.P.num_rocks):
            placed = False
            for _attempt in range(max_tries):
                rx = float(rng.uniform(min_xy, max_xy))
                ry = float(rng.uniform(min_xy, max_xy))
                # keep rocks away from start and goal
                if not far_enough(rx, ry, self.x, self.y, clear):
                    continue
                if not far_enough(rx, ry, float(self.goal[0]), float(self.goal[1]), clear):
                    continue
                # keep rocks separated
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
                # fallback: place anywhere inside
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

    # --- Rendering helpers
    def _world_to_screen(self, x, y):
        s = self.G.render_scale_px_per_m
        pad = self.G.render_padding_px
        L = self.P.world_size
        sx = pad + int((x + L) * s)
        sy = pad + int((L - y) * s)  # invert y for screen coords
        return sx, sy

    def _build_water(self, surf: pygame.Surface):
        """Pre-render simple wavy water texture onto `surf`."""
        w, h = surf.get_size()
        cell = self.G.water_wave_px
        base = self.G.water_primary
        dark = self.G.water_secondary
        surf.fill(base)

        # Dark sine streaks
        import math as _m
        for y in range(0, h, cell):
            amp = cell * 0.35
            for x in range(0, w, 2):
                y2 = int(y + amp * _m.sin((x / cell) * 2.2) * 0.6)
                if 0 <= y2 < h:
                    surf.set_at((x, y2), dark)

        # Light diagonal highlights
        highlight = (min(255, base[0] + 28), min(255, base[1] + 28), min(255, base[2] + 28))
        step = max(6, cell // 2)
        for d in range(-h, w, step):
            pygame.draw.aaline(surf, highlight, (d, 0), (d + h, h))

    def _handle_input(self):
        """Process window events and live calibration keys."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFTBRACKET:    # '['
                    self.G.sprite_heading_deg_offset -= 5.0
                    print(f"sprite_heading_deg_offset = {self.G.sprite_heading_deg_offset}")
                elif event.key == pygame.K_RIGHTBRACKET: # ']'
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
                    # Preserve original behavior AND add rays toggle
                    self.G.show_debug_triangle = not self.G.show_debug_triangle
                    self.G.show_rays = not self.G.show_rays
                    print(f"show_debug_triangle = {self.G.show_debug_triangle}, show_rays = {self.G.show_rays}")
                elif event.key == pygame.K_0:
                    self.G.sprite_heading_deg_offset = 0.0
                    self.G.sprite_px_offset = (0, 0)
                    print("sprite_heading_deg_offset=0, sprite_px_offset=(0,0)")

    def _draw_water(self):
        if not self._water_surf:
            return
        # static by default; enable scroll by raising water_scroll_px_per_step
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
        # Only in render mode, with rays enabled and cached distances available
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

            # ray line up to hit
            pygame.draw.aaline(surf, self.G.ray_color, (sx0, sy0), (sxe, sye))
            # hit marker
            pygame.draw.circle(surf, self.G.ray_hit_color, (sxe, sye), 3)

    def _draw_ship(self, surf):
        cx, cy = self._world_to_screen(self.x, self.y)

        if self._sprite_img_scaled is not None:
            # Match sprite rotation to physics direction vector (dx,dy)
            angle_deg = math.degrees(math.atan2(-self.dy, self.dx)) + self.G.sprite_heading_deg_offset
            img = pygame.transform.rotozoom(self._sprite_img_scaled, angle_deg, 1.0)
            rect = img.get_rect()
            rect.center = (cx + self.G.sprite_px_offset[0], cy + self.G.sprite_px_offset[1])
            surf.blit(img, rect)

            # Debug triangle under sprite for orientation comparison
            if self.G.show_debug_triangle:
                Lb = self.G.ship_length_m
                Wb = self.G.ship_width_m
                body = np.array([[+0.5 * Lb, 0.0], [-0.5 * Lb, +0.5 * Wb], [-0.5 * Lb, -0.5 * Wb]])
                # Rotation matrix from direction vector
                R = np.array([[self.dx, self.dy], [-self.dy, self.dx]])
                world = (R @ body.T).T + np.array([self.x, self.y])
                pts = [self._world_to_screen(px, py) for (px, py) in world]
                pygame.draw.polygon(surf, (255, 100, 100), pts)  # red triangle
                pygame.draw.polygon(surf, (180, 0, 0), pts, 2)
        else:
            # fallback triangle (nose points in direction vector)
            Lb = self.G.ship_length_m
            Wb = self.G.ship_width_m
            body = np.array([[+0.5 * Lb, 0.0], [-0.5 * Lb, +0.5 * Wb], [-0.5 * Lb, -0.5 * Wb]])
            R = np.array([[self.dx, self.dy], [-self.dy, self.dx]])
            world = (R @ body.T).T + np.array([self.x, self.y])
            pts = [self._world_to_screen(px, py) for (px, py) in world]
            pygame.draw.polygon(surf, (30, 144, 255), pts)
            pygame.draw.polygon(surf, (0, 60, 120), pts, 2)

        # Optional velocity vector for debugging alignment
        if self.G.show_velocity_vector and self.v > 1e-6:
            arrow_len = max(10, int(self.v * self.G.render_scale_px_per_m * 0.3))
            tip_x = cx + int(arrow_len * self.dx)
            tip_y = cy - int(arrow_len * self.dy)  # minus because screen y is inverted
            pygame.draw.line(surf, (0, 0, 0), (cx, cy), (tip_x, tip_y), 2)
            pygame.draw.circle(surf, (0, 0, 0), (tip_x, tip_y), 3)

    def _render_frame(self):
        # handle window + live calibration input
        self._handle_input()

        if self._surf is None:
            self._surf = pygame.Surface((self._win_w, self._win_h)).convert()

        # background
        self._surf.fill(self.G.water_primary)
        self._draw_water()

        # overlays
        if self.G.show_grid:
            self._draw_grid(self._surf)
        self._draw_rocks(self._surf)
        self._draw_goal(self._surf)
        self._draw_trail(self._surf)
        self._draw_rays(self._surf)   # draw rays to their hit points
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

    # (optional) grid if you ever want it back
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
