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
    max_steps: int = 500

    # World bounds (square [-L, L] x [-L, L])
    world_size: float = 10.0

    # Control (accelerations, not velocities)
    thrust_accel: float = 1.5       # m/s^2 (forward accel)
    brake_accel: float = 2.0        # m/s^2 (deceleration when braking)
    torque_accel: float = 2.5       # rad/s^2 (left/right yaw acceleration)

    # Limits
    v_max: float = 5.0              # m/s (non-negative; no reverse)
    w_max: float = 4.0              # rad/s

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

    Kinematics:
        * Turn steps (0/1): update w, then θ; apply linear drag on v; **do not move x,y**.
        * Move steps (2/3): lock w=0 (no rotation), update v; move x,y strictly along heading θ.
        * v ∈ [0, v_max], no reverse. θ increases CCW, θ=0 points +x.

    Rendering:
        * Optional sprite with live alignment controls:
            [ / ] : sprite_heading_deg_offset ±5°
            Arrow keys : sprite_px_offset x/y nudge
            V : toggle velocity vector
            0 : reset offsets
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

        # Discrete(4) actions (see docstring)
        self.action_space = spaces.Discrete(4)

        # Observation space: [x, y, cosθ, sinθ, v, w]
        high = np.array(
            [self.P.world_size, self.P.world_size, 1.0, 1.0, self.P.v_max, self.P.w_max],
            dtype=np.float32,
        )
        low = np.array(
            [-self.P.world_size, -self.P.world_size, -1.0, -1.0, 0.0, -self.P.w_max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # State
        self.x = self.y = self.theta = 0.0
        self.v = 0.0   # forward speed, >= 0
        self.w = 0.0   # yaw rate
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

        # Pixel geometry
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

        # Start pose/speeds
        self.x = float(-0.5 * L + rng.normal(0, self.P.start_noise))
        self.y = float(rng.normal(0, self.P.start_noise))
        self.theta = float(rng.uniform(-math.pi, math.pi))
        self.v = max(0.0, float(rng.normal(0, 0.1)))  # non-negative
        self.w = float(rng.normal(0, 0.1))
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

        # Rendering setup
        if self.render_mode in ("human", "rgb_array"):
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

        return self._obs(), {"goal": self.goal.copy(), "rocks": list(self.rocks)}

    def step(self, action: int):
        if action not in (0, 1, 2, 3):
            raise ValueError("Invalid action for Discrete(4).")

        dt = self.P.dt
        a = 0.0
        alpha = 0.0

        if action == 0:   # turn left
            alpha = +self.P.torque_accel
            # Turning step: rotate in place, apply drag to v (no translation)
            self.w = float(np.clip(self.w + (alpha - self.P.ang_drag * self.w) * dt,
                                   -self.P.w_max, self.P.w_max))
            self.theta = self._wrap_angle(self.theta + self.w * dt)
            self.v = float(np.clip(self.v + (-self.P.lin_drag * self.v) * dt,
                                   0.0, self.P.v_max))
            # no x,y update on this step

        elif action == 1:  # turn right
            alpha = -self.P.torque_accel
            self.w = float(np.clip(self.w + (alpha - self.P.ang_drag * self.w) * dt,
                                   -self.P.w_max, self.P.w_max))
            self.theta = self._wrap_angle(self.theta + self.w * dt)
            self.v = float(np.clip(self.v + (-self.P.lin_drag * self.v) * dt,
                                   0.0, self.P.v_max))
            # no x,y update on this step

        elif action == 2:  # accelerate straight
            a = +self.P.thrust_accel
            # Move step: heading locked, no turn
            self.w = 0.0
            self.v = float(np.clip(self.v + (a - self.P.lin_drag * self.v) * dt,
                                   0.0, self.P.v_max))
            self.x += self.v * math.cos(self.theta) * dt
            self.y += self.v * math.sin(self.theta) * dt

        elif action == 3:  # brake straight
            a = -self.P.brake_accel
            self.w = 0.0
            self.v = float(np.clip(self.v + (a - self.P.lin_drag * self.v) * dt,
                                   0.0, self.P.v_max))
            self.x += self.v * math.cos(self.theta) * dt
            self.y += self.v * math.sin(self.theta) * dt

        # Trail
        self._trail.append((self.x, self.y))
        if len(self._trail) > self.G.trail_len:
            self._trail.pop(0)

        self.steps += 1

        # Distances / termination
        dist_goal = float(np.linalg.norm([self.x - self.goal[0], self.y - self.goal[1]]))
        destroyed = self._check_rock_collision(self.x, self.y)

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

    def _draw_ship(self, surf):
        cx, cy = self._world_to_screen(self.x, self.y)

        if self._sprite_img_scaled is not None:
            # pygame rotates clockwise; our theta increases CCW
            angle_deg = -math.degrees(self.theta) + self.G.sprite_heading_deg_offset
            img = pygame.transform.rotozoom(self._sprite_img_scaled, angle_deg, 1.0)
            rect = img.get_rect()
            rect.center = (cx + self.G.sprite_px_offset[0], cy + self.G.sprite_px_offset[1])
            surf.blit(img, rect)
        else:
            # fallback triangle (nose points +x when theta=0)
            Lb = self.G.ship_length_m
            Wb = self.G.ship_width_m
            body = np.array([[+0.5 * Lb, 0.0], [-0.5 * Lb, +0.5 * Wb], [-0.5 * Lb, -0.5 * Wb]])
            c, s = math.cos(self.theta), math.sin(self.theta)
            R = np.array([[c, -s], [s, c]])
            world = (R @ body.T).T + np.array([self.x, self.y])
            pts = [self._world_to_screen(px, py) for (px, py) in world]
            pygame.draw.polygon(surf, (30, 144, 255), pts)
            pygame.draw.polygon(surf, (0, 60, 120), pts, 2)

        # Optional velocity vector for debugging alignment
        if self.G.show_velocity_vector and self.v > 1e-6:
            arrow_len = max(10, int(self.v * self.G.render_scale_px_per_m * 0.3))
            tip_x = cx + int(arrow_len * math.cos(self.theta))
            tip_y = cy - int(arrow_len * math.sin(self.theta))  # minus because screen y is inverted
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
