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
    max_steps: int = 5000

    # World bounds (square [-L, L] x [-L, L])
    world_size: float = 10.0

    # Control (accelerations, not velocities)
    thrust_accel: float = 0.15       # m/s^2 (forward/back along body x)
    torque_accel: float = 0.2       # rad/s^2 (left/right yaw)

    # Limits
    v_max: float = 5.0              # m/s
    w_max: float = 1.0              # rad/s

    # Drag
    lin_drag: float = 0.1           # linear drag coeff on v
    ang_drag: float = 0.1           # angular drag coeff on ω

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
    water_scroll_px_per_step: int = 2

    # Trail
    trail_len: int = 100
    trail_color: Tuple[int, int, int] = (160, 160, 255)

    # FPS (human mode)
    fps_limit: int = 60

    # Optional sprite (None => try shipenv/assets/ship.png)
    sprite_path: Optional[str] = None
    sprite_meters_long: float = 0.9

    # Rocks appearance
    rock_fill: Tuple[int, int, int] = (110, 110, 110)
    rock_stroke: Tuple[int, int, int] = (40, 40, 40)

    # Goal appearance
    goal_color: Tuple[int, int, int] = (0, 180, 0)

    # Optional overlays
    show_grid: bool = False  # left in, but off by default


class ShipEnv(gym.Env):
    """
    2D ship environment with discrete accelerations:
      - Action: 9 discrete combinations of (thrust, torque) in {-1,0,1}^2
      - State:  x, y, θ, v_forward, ω
      - Obs:    [x, y, cosθ, sinθ, v, ω]

    Kinematics:
      - Exact unicycle integration (no lateral/slip dof): ship moves only along its body x-axis.

    Hazards:
      - Static circular rocks (configurable count/radius). Touching a rock ends the episode with a penalty.

    Goal:
      - Random by default (within bounds & margin), or pass a goal in reset(options={'goal':[x, y]}).

    Render modes:
      - "human": pygame window
      - "rgb_array": returns HxWx3 image via info["frame"]
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

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

        # Action space: 3x3 grid of (thrust, torque) directions
        self._action_map = np.array(
            [(tx, rz) for tx in (-1, 0, 1) for rz in (-1, 0, 1)],
            dtype=np.int8,
        )
        self.action_space = spaces.Discrete(len(self._action_map))

        # Observation space
        high = np.array(
            [self.P.world_size, self.P.world_size, 1.0, 1.0, self.P.v_max, self.P.w_max],
            dtype=np.float32,
        )
        low = -high.copy()
        low[2:4] = -1.0  # cos, sin
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # State
        self.x = self.y = self.theta = 0.0
        self.v = self.w = 0.0
        self.goal = np.array(goal if goal is not None else [0.0, 0.0], dtype=np.float32)
        self.steps = 0

        # Hazards (rock centers in world coords)
        self.rocks: List[Tuple[float, float]] = []

        # Render internals
        self._screen = None
        self._clock = None
        self._surf = None
        self._water_surf = None
        self._water_offset = 0
        self._sprite_img = None
        self._sprite_img_scaled = None
        self._trail: List[Tuple[float, float]] = []

        # Precompute pixel sizes
        L = self.P.world_size
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

        # Start pose
        self.x = float(-0.5 * L + rng.normal(0, self.P.start_noise))
        self.y = float(rng.normal(0, self.P.start_noise))
        self.theta = float(rng.uniform(-math.pi, math.pi))
        self.v = float(rng.normal(0, 0.1))
        self.w = float(rng.normal(0, 0.1))
        self.steps = 0

        # Goal: options override; otherwise random if enabled
        if options and "goal" in options:
            self.goal = np.array(options["goal"], dtype=np.float32)
        elif self.P.random_goal:
            self.goal = self._sample_goal(rng)

        # Reset trail
        self._trail = [(self.x, self.y)]

        # Place rocks
        self.rocks = self._sample_rocks(rng)

        # Set up rendering on-demand
        if self.render_mode in ("human", "rgb_array"):
            if not pygame.get_init():
                pygame.init()

            # Window for human mode
            if self.render_mode == "human":
                if self._screen is None:
                    pygame.display.set_caption("ShipEnv")
                    self._screen = pygame.display.set_mode((self._win_w, self._win_h))
                if self._clock is None:
                    self._clock = pygame.time.Clock()

            # Drawing surface (always)
            self._surf = pygame.Surface((self._win_w, self._win_h)).convert()

            # Build water texture & reset scroll
            self._water_surf = pygame.Surface((self._win_w, self._win_h)).convert()
            self._build_water(self._water_surf)
            self._water_offset = 0

            # Load/scale sprite (explicit path or packaged default)
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
                    w, h = raw.get_size()
                    scale = target_len_px / max(w, h)
                    self._sprite_img_scaled = pygame.transform.smoothscale(
                        raw, (max(1, int(w * scale)), max(1, int(h * scale)))
                    )
                except Exception as e:
                    print(f"[ShipEnv] Sprite load failed ({sprite_path}): {e}")
                    self._sprite_img = self._sprite_img_scaled = None

        return self._obs(), {"goal": self.goal.copy(), "rocks": list(self.rocks)}

    def step(self, action: int):
        tx_dir, rz_dir = self._action_map[int(action)]
        a = tx_dir * self.P.thrust_accel
        alpha = rz_dir * self.P.torque_accel

        # Apply drag to accelerations (first-order viscous)
        a -= self.P.lin_drag * self.v
        alpha -= self.P.ang_drag * self.w

        # Integrate velocities
        self.v = float(np.clip(self.v + a * self.P.dt, -self.P.v_max, self.P.v_max))
        self.w = float(np.clip(self.w + alpha * self.P.dt, -self.P.w_max, self.P.w_max))

        # ---- Exact unicycle integration (no sideways motion) ----
        # x_dot = v cosθ,  y_dot = v sinθ,  θ_dot = w  (v, w constant over dt)
        v = self.v
        w = self.w
        dt = self.P.dt
        theta0 = self.theta
        if abs(w) < 1e-9:
            # Straight line
            self.x += v * math.cos(theta0) * dt
            self.y += v * math.sin(theta0) * dt
            self.theta = self._wrap_angle(theta0 + w * dt)
        else:
            # Constant turn rate arc
            self.x += (v / w) * (math.sin(theta0 + w * dt) - math.sin(theta0))
            self.y += -(v / w) * (math.cos(theta0 + w * dt) - math.cos(theta0))
            self.theta = self._wrap_angle(theta0 + w * dt)
        # ---------------------------------------------------------

        # Trail
        self._trail.append((self.x, self.y))
        if len(self._trail) > self.G.trail_len:
            self._trail.pop(0)

        self.steps += 1

        # Distances
        dist_goal = float(np.linalg.norm([self.x - self.goal[0], self.y - self.goal[1]]))

        # Check rock collision
        destroyed = self._check_rock_collision(self.x, self.y)

        # Reward / termination
        control_cost = 0.01 * (tx_dir * tx_dir + rz_dir * rz_dir)
        reward = -dist_goal - control_cost
        terminated = False

        if destroyed:
            reward += self.P.crash_penalty
            terminated = True  # ship destroyed

        reached = (dist_goal <= self.P.goal_radius) and not destroyed
        if reached:
            reward += self.P.success_bonus
            terminated = True

        out_of_bounds = (abs(self.x) > self.P.world_size or abs(self.y) > self.P.world_size)
        truncated = out_of_bounds or (self.steps >= self.P.max_steps)

        info = {
            "distance": dist_goal,
            "out_of_bounds": out_of_bounds,
            "goal": self.goal.copy(),
            "destroyed": destroyed,
            "reached_goal": reached,
        }

        # Render
        if self.render_mode:
            frame = self._render_frame()
            if self.render_mode == "rgb_array":
                info["frame"] = frame

        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode:
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
        return np.array(
            [self.x, self.y, math.cos(self.theta), math.sin(self.theta), self.v, self.w],
            dtype=np.float32,
        )

    @staticmethod
    def _wrap_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    # --- Random sampling helpers
    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        L = self.P.world_size
        m = self.P.goal_margin
        gx = float(rng.uniform(-L + m, L - m))
        gy = float(rng.uniform(-L + m, L - m))
        return np.array([gx, gy], dtype=np.float32)

    def _sample_rocks(self, rng: np.random.Generator) -> List[Tuple[float, float]]:
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
                # keep rocks from overlapping too much with each other
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
                # if we can't place respecting constraints, relax and place anywhere inside
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
        sy = pad + int((L - y) * s)  # invert y
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

    def _draw_water(self):
        if not self._water_surf:
            return
        self._water_offset = (self._water_offset + self.G.water_scroll_px_per_step) % self._win_h
        off = self._water_offset
        # two blits for seamless vertical wrap
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

    def _draw_ship(self, surf):
        if self._sprite_img_scaled is not None:
            angle_deg = -math.degrees(self.theta)  # pygame rotates clockwise
            img = pygame.transform.rotozoom(self._sprite_img_scaled, angle_deg, 1.0)
            rect = img.get_rect()
            cx, cy = self._world_to_screen(self.x, self.y)
            rect.center = (cx, cy)
            surf.blit(img, rect)
            return

        # fallback triangle
        Lb = self.G.ship_length_m
        Wb = self.G.ship_width_m
        body = np.array([[+0.5 * Lb, 0.0], [-0.5 * Lb, +0.5 * Wb], [-0.5 * Lb, -0.5 * Wb]])
        c, s = math.cos(self.theta), math.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        world = (R @ body.T).T + np.array([self.x, self.y])
        pts = [self._world_to_screen(px, py) for (px, py) in world]
        pygame.draw.polygon(surf, (30, 144, 255), pts)
        pygame.draw.polygon(surf, (0, 60, 120), pts, 2)

    def _render_frame(self):
        # keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        if self._surf is None:
            self._surf = pygame.Surface((self._win_w, self._win_h)).convert()

        # background
        self._surf.fill(self.G.water_primary)
        self._draw_water()

        # overlays: rocks first (so ship sits above)
        self._draw_rocks(self._surf)
        self._draw_goal(self._surf)
        self._draw_trail(self._surf)
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

    # (optional) if you ever toggle show_grid=True
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
            pygame.draw.line(surf, (235, 235, 235), (pad, y1), (pad + 2 * L * s, y1), 1)
