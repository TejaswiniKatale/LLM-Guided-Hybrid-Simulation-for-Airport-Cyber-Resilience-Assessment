"""
Airport check-in simulation: DES (Python) + JuPedSim (pedestrian movement only).

Flow: Door -> waiting queue -> ticket counter queue -> ticket counter service -> info desk queue -> info desk service -> security waiting -> security queue -> security service -> exit.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
import random
import sys
from collections import deque

import jupedsim as jps
import pygame

from new_layout import build_airport_checkin_layout
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import wkt



# ---------------------------------------------------------------------------
# Compatibility shim for jupedsim Agent wrapper
# ---------------------------------------------------------------------------
# Some versions of JuPedSim expect an `Agent.set_position()` method during
# `Simulation.switch_agent_journey()`. The installed `jupedsim` python wrapper
# may not expose it, which crashes the simulation with:
#   AttributeError: 'Agent' object has no attribute 'set_position'
try:
    from jupedsim.agent import Agent as _JpAgent

    if not hasattr(_JpAgent, "set_position"):

        def _set_position(self: _JpAgent, pos: tuple[float, float]) -> None:
            # Prefer direct assignment to the backing object's position.
            try:
                self._obj.position = pos  # type: ignore[attr-defined]
                return
            except Exception:
                pass

            # Fallback: some backends provide an explicit method.
            if hasattr(self._obj, "set_position"):
                self._obj.set_position(pos)  # type: ignore[attr-defined]
                return

            raise AttributeError("Backing agent has no way to set position")

        _JpAgent.set_position = _set_position  # type: ignore[assignment]
except Exception:
    # If anything goes wrong, we let the simulation fail later with the
    # original error; but in practice this shim should be harmless.
    pass

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# Flight display visit tracking
display_visit_count = 0
counted_display_agents: set[int] = set()

# Minimum simulated seconds between accepted spawns (safety; can throttle very fast TARGET_ARRIVALS_PER_HOUR).
MIN_SPAWN_GAP_SECONDS = 2.5
SPAWN_OFFSET_RADIUS = 0.35
SPAWN_MAX_ATTEMPTS = 30  # try this many positions to avoid "agent too close" error
SIMULATION_DT = 0.30
# Multiplier applied to SIMULATION_DT to slow/speed the simulation.
# 1.0 = normal speed, <1.0 slower, >1.0 faster.
SIM_SPEED_MULT = 2.0

# Demand: target passengers per simulated hour. Interarrival spacing drives spawn cadence.
TARGET_ARRIVALS_PER_HOUR = 25
# Simulated seconds advanced per `sim.iterate()` step (same as inner-loop `dt`).
SIMULATION_CLOCK_DT = SIMULATION_DT * SIM_SPEED_MULT
INTERARRIVAL_SIM_SECONDS = 3600.0 / TARGET_ARRIVALS_PER_HOUR
spawn_interval_iterations = max(1, round(INTERARRIVAL_SIM_SECONDS / SIMULATION_CLOCK_DT))
MAX_ITERATIONS = 10_000_000  # safety cap; stop when all passengers processed
SIMULATION_HORIZON = float("inf")
MAX_PASSENGERS = 500
# Cap the number of discrete overflow queue "slots" below the ticket waiting area.
MAX_OVERFLOW_SLOTS = 150
# Slightly smaller passengers (collision radius)
AGENT_RADIUS = 0.012
DESIRED_SPEED = 1.4
COUNTER_QUEUE_CAPACITY = 2  # must match number of counter queue positions in layout
COUNTER_DIST_AT_FRONT = 0.5
SECURITY_DIST_AT_FRONT = 0.15
NORMAL_CI_SERVICE_TIME = 300.0
DEGRADATION_START_TIME = 20000.0
DEGRADED_CI_SERVICE_TIME = 700.0


def get_current_ci_service_time(sim_time_now: float) -> float:
    return (
        NORMAL_CI_SERVICE_TIME
        if sim_time_now < DEGRADATION_START_TIME
        else DEGRADED_CI_SERVICE_TIME
    )


WAITING_FRONT_WAIT_TIME = 1.0
COUNTER_QUEUE_FRONT_WAIT_TIME = 2.0
WAYPOINT_RADIUS = 0.25
# Entrance area: same size as the blue door boxes (54px×12px → ~2.7×0.6 world units); radius covers the box
ENTRANCE_AREA_RADIUS = 1.4
# Y threshold for entrance zone: crossing this line means passenger is inside the airport (blue strip level)
ENTRANCE_ZONE_Y = 1.8

SECURITY_QUEUE_CAPACITY = 2  # same as ticket counter queue count
SECURITY_QUEUE_FRONT_WAIT_TIME = 2.0
SECURITY_WAITING_FRONT_WAIT_TIME = 1.5
SECURITY_SERVICE_TIME = 300.0

INFO_DESK_SERVICE_TIME = 120.0
INFO_DESK_QUEUE_CAPACITY = 2
INFO_DESK_FRONT_WAIT_TIME = 2.0
INFO_DESK_DIST_AT_FRONT = 0.15

# Fraction of passengers that visit the information desk after ticket counters
INFO_DESK_PROBABILITY = 0.2

# Post-security flight display behavior
DISPLAY_PROBABILITY = 0.80
DISPLAY_CAPACITY = 10
DISPLAY_WAIT_MIN = 30.0
DISPLAY_WAIT_MAX = 120.0
NUM_DISPLAYS = 4

# Flight display -> post-display routing (single weighted decision, includes direct gates)
PROB_DIRECT = 0.55
PROB_SHOPS = 0.20
PROB_FOOD = 0.15
PROB_REST = 0.10
COLOR_FOOD = (220, 50, 50)        # Red
COLOR_SHOPS = (50, 100, 255)      # Blue
COLOR_REST = (255, 105, 180)      # Pink
COLOR_DIRECT = (255, 215, 0)      # Yellow

# Post-entrance shopping behavior
SHOP_WAIT_MIN = 180.0
SHOP_WAIT_MAX = 480.0

# Food court behavior (after shops)
FOOD_WAIT_MIN = 300.0
FOOD_WAIT_MAX = 900.0

# Restrooms behavior (after food court)
REST_WAIT_MIN = 120.0
REST_WAIT_MAX = 300.0

# Gates behavior (after restrooms)
GATES_WAYPOINT_RADIUS = 0.25

STEPS_PER_FRAME = 4
# Experiment runtime: no window, no draw, no FPS cap (see main loop).
HEADLESS_MODE = False
# Larger SIM_SKIP and quieter logs for batch runs; when False, use interactive defaults.
FAST_MODE = True
# Time compression: simulation substeps per outer frame (batch size before optional draw).
SIM_SKIP = 20 if FAST_MODE else 3
# Iteration cadence for optional [TIME CHECK] logging (FAST_MODE uses a sparser default).
TIME_DEBUG_EVERY = 10_000 if FAST_MODE else 1_000
WORLD_W = 50.0
WORLD_H = 30.0
SCREEN_W = 1000
SCREEN_H = 600
# Hotspot option: focus occupancy on operational queue/waiting zones only.
HEATMAP_QUEUE_FOCUSED = False
HEATMAP_QUEUE_FOCUS_RADIUS = 0.6
# Lower FPS slows the simulation in real time (fewer steps per second)
FPS = 30

SAFE_MARGIN = 0.3
HALL_MIN_X, HALL_MAX_X = 3.0, 47.0
HALL_MIN_Y, HALL_MAX_Y = 0.0, 30.0

def world_to_screen(x: float, y: float) -> tuple[int, int]:
    px = int(x * (SCREEN_W / WORLD_W))
    py = int(SCREEN_H - y * (SCREEN_H / WORLD_H))
    return (px, py)

def sanitize_positions(positions, hall_polygon):
    safe_positions = []
    for (x, y) in positions:
        sx, sy = get_safe_position(x, y, hall_polygon)
        safe_positions.append((sx, sy))
    return safe_positions


def clamp_to_hall(x, y):
    x = max(HALL_MIN_X + SAFE_MARGIN, min(HALL_MAX_X - SAFE_MARGIN, x))
    y = max(HALL_MIN_Y + SAFE_MARGIN, min(HALL_MAX_Y - SAFE_MARGIN, y))
    return x, y


def project_inside_polygon(x, y, hall_polygon, step=0.05, max_steps=25):
    from shapely.geometry import Point
    import math

    point = Point(x, y)
    if hall_polygon.contains(point):
        return x, y

    centroid = hall_polygon.centroid
    cx, cy = centroid.x, centroid.y

    dx = cx - x
    dy = cy - y
    length = math.hypot(dx, dy)
    if length == 0:
        return None

    dx /= length
    dy /= length

    for _ in range(max_steps):
        x += dx * step
        y += dy * step
        if hall_polygon.contains(Point(x, y)):
            return x, y

    return None


def get_safe_position(x, y, hall_polygon):
    from shapely.geometry import Point
    import random

    # 1. Direct check
    if hall_polygon.contains(Point(x, y)):
        return x, y

    # 2. Try projection
    projected = project_inside_polygon(x, y, hall_polygon)
    if projected:
        return projected

    # 3. Small jitter retries (bounded, not infinite)
    for _ in range(10):
        px = x + random.uniform(-0.02, 0.02)
        py = y + random.uniform(-0.02, 0.02)
        if hall_polygon.contains(Point(px, py)):
            return px, py

    # 4. Final fallback (guaranteed safe)
    centroid = hall_polygon.centroid
    return centroid.x, centroid.y


def force_safe_point(x, y, hall_polygon):
    point = Point(x, y)
    if hall_polygon.contains(point):
        return x, y

    centroid = hall_polygon.centroid
    cx, cy = centroid.x, centroid.y
    px, py = x, y
    for _ in range(50):
        px = px + (cx - px) * 0.35
        py = py + (cy - py) * 0.35
        if hall_polygon.contains(Point(px, py)):
            return px, py

    return cx, cy


def safe_spawn_position(
    sim,
    hall_polygon,
    x_min,
    x_max,
    y_min,
    y_max,
    min_separation,
    *,
    attempts=100,
):
    centroid = hall_polygon.centroid
    fallback = force_safe_point(centroid.x, centroid.y, hall_polygon)

    for _ in range(attempts):
        sx = random.uniform(x_min, x_max)
        sy = random.uniform(y_min, y_max)
        sx, sy = get_safe_position(sx, sy, hall_polygon)
        sx, sy = force_safe_point(sx, sy, hall_polygon)

        safe = True
        for agent in sim.agents():
            ax, ay = agent.position
            if math.hypot(sx - ax, sy - ay) < min_separation:
                safe = False
                break
        if safe:
            return sx, sy

    return fallback


def is_safe_to_switch(agent, sim, min_dist=0.1):
    ax, ay = agent.position
    for other in sim.agents():
        if other.id == agent.id:
            continue
        ox, oy = other.position
        if (ax - ox) * (ax - ox) + (ay - oy) * (ay - oy) < min_dist * min_dist:
            return False
    return True


def safe_switch(sim, agent_id, journey_id, stage_id):
    try:
        sim.switch_agent_journey(agent_id, journey_id, stage_id)
    except RuntimeError:
        # Skip safely if switching fails
        return


def add_safe_waypoint(sim, x, y, radius, hall_polygon):
    from shapely.geometry import Point
    import random
    import math

    def project_inside_polygon(x, y):
        point = Point(x, y)
        if hall_polygon.contains(point):
            return x, y

        centroid = hall_polygon.centroid
        cx, cy = centroid.x, centroid.y

        dx = cx - x
        dy = cy - y
        length = math.hypot(dx, dy)
        if length == 0:
            return None

        dx /= length
        dy /= length

        for _ in range(25):
            x += dx * 0.05
            y += dy * 0.05
            if hall_polygon.contains(Point(x, y)):
                return x, y

        return None

    def get_safe_position(x, y):
        # 1. direct
        if hall_polygon.contains(Point(x, y)):
            return x, y

        # 2. projection
        p = project_inside_polygon(x, y)
        if p:
            return p

        # 3. jitter attempts
        for _ in range(10):
            px = x + random.uniform(-0.02, 0.02)
            py = y + random.uniform(-0.02, 0.02)
            if hall_polygon.contains(Point(px, py)):
                return px, py

        # 4. fallback (guaranteed safe)
        c = hall_polygon.centroid
        return c.x, c.y

    px, py = get_safe_position(x, y)
    px, py = force_safe_point(px, py, hall_polygon)
    px, py = clamp_to_hall(px, py)

    for _ in range(25):
        try:
            sid = sim.add_waypoint_stage((px, py), radius)
            return sid, (px, py)
        except RuntimeError:
            jx = px + random.uniform(-0.03, 0.03)
            jy = py + random.uniform(-0.03, 0.03)
            px, py = force_safe_point(jx, jy, hall_polygon)
            px, py = clamp_to_hall(px, py)

    c = hall_polygon.centroid
    cx, cy = force_safe_point(c.x, c.y, hall_polygon)
    cx, cy = clamp_to_hall(cx, cy)
    try:
        sid = sim.add_waypoint_stage((cx, cy), radius)
        return sid, (cx, cy)
    except RuntimeError:
        return None, (cx, cy)


def safe_add_waypoint(sim, pos, radius, hall_polygon, *, attempts=50, jitter=0.02, inward_shift=None):
    # Ensure all waypoint radii are large enough for stable switching/arrival.
    radius = max(radius, 0.08)
    x, y = pos
    if inward_shift is not None:
        sx, sy = inward_shift
        x += sx
        y += sy

    for _ in range(attempts):
        px = x + random.uniform(-jitter, jitter)
        py = y + random.uniform(-jitter, jitter)

        px, py = force_safe_point(px, py, hall_polygon)
        px, py = clamp_to_hall(px, py)

        sid, safe_pos = add_safe_waypoint(sim, px, py, radius, hall_polygon)
        if sid is not None:
            return sid, safe_pos

    c = hall_polygon.centroid
    cx, cy = force_safe_point(c.x, c.y, hall_polygon)
    sid, safe_pos = add_safe_waypoint(sim, cx, cy, radius, hall_polygon)
    return sid, safe_pos


def main() -> None:
    print(
        "Arrival process:",
        f"TARGET_ARRIVALS_PER_HOUR={TARGET_ARRIVALS_PER_HOUR}",
        f"interarrival_seconds={INTERARRIVAL_SIM_SECONDS:.6g}",
        f"spawn_interval_iterations={spawn_interval_iterations}",
        f"dt={SIMULATION_CLOCK_DT:.6g}",
        flush=True,
    )
    print("Running mode:", "HEADLESS" if HEADLESS_MODE else "VISUAL", flush=True)
    print("FAST_MODE:", "ON" if FAST_MODE else "OFF", flush=True)
    print("SIM_SKIP:", SIM_SKIP, flush=True)
    layout = build_airport_checkin_layout()
    sim = layout["simulation"]
    coordinates = layout["coordinates"]
    hall_polygon = wkt.loads(coordinates["hall"]["hall_wkt"])
    scenario = "CI_DEGRADED"
    global display_visit_count, counted_display_agents
    display_visit_count = 0
    counted_display_agents = set()

    # Preserve the intended serpentine queue order from the layout:
    # w1 is the "front" (rightmost slot of the top row), then w2, w3, ...
    waiting_positions_dict = coordinates["waiting_positions"]
    waiting_positions = [waiting_positions_dict[k] for k in sorted(waiting_positions_dict.keys(), key=lambda s: int(s[1:]))]
    num_waiting_slots = len(waiting_positions)
    # Waiting boundary and tail slot (used for overflow holding below the tail).
    LAST_WAITING_Y = min(y for (_x, y) in waiting_positions) if waiting_positions else 0.0
    last_waiting_slot = waiting_positions[-1] if waiting_positions else (0.0, 0.0)
    last_x, last_y = last_waiting_slot
    overflow_hold_points = [
        (last_x + 1.5, last_y - 1.5),
        (last_x + 2.5, last_y - 2.0),
        (last_x + 3.5, last_y - 1.8),
        (last_x + 2.0, last_y - 2.5),
        (last_x + 3.0, last_y - 2.8),
        (last_x + 1.8, last_y - 2.2),
    ]

    # Only two ticket counters (left side)
    num_counters = 2
    counter_queue_slot_positions: list[list[tuple[float, float]]] = []
    for j in range(num_counters):
        slots = [coordinates["counter_queue_slots"][f"q{j+1}-{s}"] for s in range(1, 3)]
        counter_queue_slot_positions.append(slots)

    counter_positions = [coordinates["counters"][f"Counter{i+1}"] for i in range(num_counters)]

    info_left_queue_slots = coordinates.get("information_left_queue_slots", [])
    info_right_queue_slots = coordinates.get("information_right_queue_slots", [])
    information_desks = coordinates.get("information_desks", {})

    num_info_desks = 2
    info_desk_queue_slot_positions: list[list[tuple[float, float]]] = [
        info_left_queue_slots,
        info_right_queue_slots,
    ]
    info_desk_service_positions = coordinates.get("info_desk_service_positions", [])
    if not info_desk_service_positions and information_desks:
        info_desk_service_positions = [information_desks["left"], information_desks["right"]]

    # Security positions from layout
    num_security = 2
    security_queue_slot_positions: list[list[tuple[float, float]]] = []
    for s in range(1, num_security + 1):
        slots = [coordinates["security_queue_slots"][f"sec_q{s}-{i}"] for i in range(1, 3)]
        security_queue_slot_positions.append(slots)
    security_point_names = ["Security1", "Security2"]
    security_positions = [coordinates["security_points"][name] for name in security_point_names]
    security_waiting_positions_dict = coordinates["security_waiting_positions"]
    security_waiting_positions = [security_waiting_positions_dict[k] for k in sorted(security_waiting_positions_dict.keys(), key=lambda s: int(s.split("_w")[-1]))]
    security_last_waiting_slot = (
        security_waiting_positions[-1] if security_waiting_positions else (0.0, 0.0)
    )
    security_last_x, security_last_y = security_last_waiting_slot
    security_overflow_hold_points = [
        (security_last_x - 0.8, security_last_y - 1.3),
        (security_last_x + 0.2, security_last_y - 1.7),
        (security_last_x + 1.0, security_last_y - 1.4),
        (security_last_x - 0.4, security_last_y - 2.1),
        (security_last_x + 0.7, security_last_y - 2.3),
        (security_last_x + 1.5, security_last_y - 2.0),
    ]

    # Security waiting queue (JuPedSim): passengers go here after info desk, then DES sends to security queue slots
    security_waiting_positions = sanitize_positions(security_waiting_positions, hall_polygon)
    security_waiting_stage_id = sim.add_queue_stage(security_waiting_positions)
    security_waiting_journey_id = sim.add_journey(jps.JourneyDescription([security_waiting_stage_id]))
    security_waiting_front = security_waiting_positions[0]

    # Queue-focused hotspot support: track only operational congestion areas when enabled.
    queue_focus_points: list[tuple[float, float]] = []
    if HEATMAP_QUEUE_FOCUSED:
        queue_focus_points.extend(waiting_positions)
        for slots in counter_queue_slot_positions:
            queue_focus_points.extend(slots)
        queue_focus_points.extend(security_waiting_positions)
        for slots in security_queue_slot_positions:
            queue_focus_points.extend(slots)
        for slots in info_desk_queue_slot_positions:
            queue_focus_points.extend(slots)

    # Entrance/exit block at the right side:
    # - "security_exit" is now treated as an internal entrance waypoint into the walking corridor.
    # - "final_exit" is a true exit polygon where agents are removed.
    security_exit_stage_id = layout["stage_ids"]["security_exit"]
    security_exit_journey_id = sim.add_journey(jps.JourneyDescription([security_exit_stage_id]))
    final_exit_stage_id = layout["stage_ids"]["final_exit"]
    final_exit_journey_id = sim.add_journey(jps.JourneyDescription([final_exit_stage_id]))

    # Flight display waiting waypoints (multiple spots inside each grey rectangle)
    display_wait_time_remaining: dict[int, float] = {}
    # Post-display spatial dwell (degraded mode only): per-agent timer and one-shot guard
    dwell_time_remaining: dict[int, float] = {}
    dwell_done: set[int] = set()
    # Map agent_id -> (display_index, spot_index)
    display_assignment: dict[int, tuple[int, int]] = {}
    # Final post-display/security choice: one activity max before gates.
    activity_choice: dict[int, str] = {}
    planned_activity: dict[int, str] = {}
    misroute_first_choice: dict[int, str] = {}
    misroute_done: dict[int, bool] = {}
    # Wrong leg only: walk the public zone, do not use service counters / wait spots
    misroute_roam_only: dict[int, bool] = {}
    food_misroute_roam_remaining: dict[int, float] = {}
    shops_misroute_roam_remaining: dict[int, float] = {}
    rest_misroute_roam_remaining: dict[int, float] = {}
    actual_path: dict[int, list[str]] = {}
    assigned_gate: dict[int, str] = {}
    passenger_type: dict[int, str] = {}
    passenger_color: dict[int, tuple[int, int, int]] = {}
    # Track per-agent phase: "to_entry", "to_spot", "waiting", "return_entry"
    display_phase: dict[int, str] = {}
    display_stage_ids: list[list[int]] = []
    display_journey_ids: list[list[int]] = []
    # Exact target coordinates for each spot: [ [ (x,y), ... ] per display ]
    display_spot_positions: list[list[tuple[float, float]]] = []
    # Entry waypoints (on the red edge) per display
    display_entry_stage_ids: list[int] = []
    display_entry_journey_ids: list[int] = []
    display_entry_positions: list[tuple[float, float]] = []
    grey_rects = coordinates.get("flight_display_grey", [])
    display_wait_lists: list[list[int]] = [[] for _ in range(len(grey_rects))]

    # -----------------------------------------------------------------------
    # Post-display spatial dwell zone (derived from layout rectangles)
    # -----------------------------------------------------------------------
    # These bounds are computed dynamically from:
    # - the flight display grey rectangles (upstream anchor)
    # - the shop rectangles (downstream boundary)
    DWELL_ZONE_X_MIN = 0.0
    DWELL_ZONE_X_MAX = 0.0
    DWELL_ZONE_Y_MIN = 0.0
    DWELL_ZONE_Y_MAX = 0.0
    try:
        grey_rects = coordinates.get("flight_display_grey", [])
        display_x_min = min(min(x for x, _ in rect) for rect in grey_rects)
        display_x_max = max(max(x for x, _ in rect) for rect in grey_rects)
        display_y_min = min(min(y for _, y in rect) for rect in grey_rects)

        shop_rects = [
            coordinates.get("cosmetics_rect"),
            coordinates.get("perfumes_rect"),
            coordinates.get("electronics_rect"),
            coordinates.get("books_rect"),
        ]
        shop_rects = [r for r in shop_rects if r is not None]
        shops_y_min = min(min(y for _, y in rect) for rect in shop_rects)

        DWELL_ZONE_X_MIN = display_x_min - 1.0
        DWELL_ZONE_X_MAX = display_x_max + 1.0
        DWELL_ZONE_Y_MIN = display_y_min - 2.0
        DWELL_ZONE_Y_MAX = shops_y_min + 1.0
    except Exception:
        # Fallback to hall bounds if layout rectangles are unavailable.
        try:
            DWELL_ZONE_X_MIN = 0.0
            DWELL_ZONE_X_MAX = float(coordinates.get("hall", {}).get("width_m", 50.0))
            DWELL_ZONE_Y_MIN = 0.0
            DWELL_ZONE_Y_MAX = float(coordinates.get("hall", {}).get("depth_m", 30.0))
        except Exception:
            DWELL_ZONE_X_MIN, DWELL_ZONE_X_MAX = 0.0, 50.0
            DWELL_ZONE_Y_MIN, DWELL_ZONE_Y_MAX = 0.0, 30.0

    def maybe_start_post_display_spatial_dwell(agent_id: int) -> bool:
        # CI degraded run: disabled degraded-only spatial dwell behavior.
        return False

    # Shops DES state: which shop and which phase each agent is in
    # Phases:
    #   "to_main_entrance"    -> main Entrance block (top green)
    #   "to_shops_enter_edge" -> along Shops Enter red edge
    #   "to_shop_entry_line"  -> along chosen shop's red line (entry)
    #   "to_shop_wait_spot"   -> inside chosen shop's grey area
    #   "shop_waiting"        -> waiting inside shop
    #   "to_shop_exit_line"   -> back to chosen shop's red line (exit)
    #   "to_shops_exit_edge"  -> along Shops Exit red edge, then final exit
    shops_phase: dict[int, str] = {}
    # Per-agent selected indices
    shops_choice: dict[int, int] = {}  # chosen shop index
    shops_choice_enter_edge_idx: dict[int, int] = {}
    shops_choice_shop_entry_idx: dict[int, int] = {}
    shops_choice_shop_wait_idx: dict[int, int] = {}
    shops_choice_shop_exit_idx: dict[int, int] = {}
    shops_choice_exit_edge_idx: dict[int, int] = {}
    shop_wait_time_remaining: dict[int, float] = {}

    # ---- Shops: enter/exit edges and per-shop entry/exit/wait waypoints ----
    shop_keys = ["cosmetics_rect", "perfumes_rect", "electronics_rect", "books_rect"]

    # Shops Enter red edge waypoints (multiple along left edge of block)
    shops_enter_stage_ids: list[int] = []
    shops_enter_journey_ids: list[int] = []
    shops_enter_positions: list[tuple[float, float]] = []

    if "shops_enter_block" in coordinates:
        rect = coordinates["shops_enter_block"]
        xs = [x for x, _ in rect]
        ys = [y for _, y in rect]
        left_x = min(xs)
        y_min = min(ys)
        y_max = max(ys)
        n_enter_points = 6
        if y_max > y_min:
            dy = (y_max - y_min) / n_enter_points
            for i in range(n_enter_points):
                cy = y_min + (i + 0.5) * dy
                base_pos = (left_x, cy)
                stage_id, safe_pos = safe_add_waypoint(sim, base_pos, 0.35, hall_polygon, inward_shift=(0.15, 0.0))
                if stage_id is None:
                    continue
                shops_enter_stage_ids.append(stage_id)
                shops_enter_positions.append(safe_pos)
                shops_enter_journey_ids.append(
                    sim.add_journey(jps.JourneyDescription([stage_id]))
                )

    # Shops Exit red edge waypoints (multiple along left edge of block)
    shops_exit_stage_ids: list[int] = []
    shops_exit_journey_ids: list[int] = []
    shops_exit_positions: list[tuple[float, float]] = []

    if "shops_exit_block" in coordinates:
        rect = coordinates["shops_exit_block"]
        xs = [x for x, _ in rect]
        ys = [y for _, y in rect]
        left_x = min(xs)
        y_min = min(ys)
        y_max = max(ys)
        n_exit_points = 6
        if y_max > y_min:
            dy = (y_max - y_min) / n_exit_points
            for i in range(n_exit_points):
                cy = y_min + (i + 0.5) * dy
                base_pos = (left_x, cy)
                stage_id, safe_pos = safe_add_waypoint(sim, base_pos, 0.35, hall_polygon, inward_shift=(0.15, 0.0))
                if stage_id is None:
                    continue
                shops_exit_stage_ids.append(stage_id)
                shops_exit_positions.append(safe_pos)
                shops_exit_journey_ids.append(
                    sim.add_journey(jps.JourneyDescription([stage_id]))
                )

    # For each shop, define multiple entry / wait / exit waypoints tied to its red line and grey area.
    num_shops = len(shop_keys)
    shop_entry_stage_ids: list[list[int]] = [[] for _ in range(num_shops)]
    shop_entry_journey_ids: list[list[int]] = [[] for _ in range(num_shops)]
    shop_entry_positions: list[list[tuple[float, float]]] = [[] for _ in range(num_shops)]
    shop_approach_below_stage_ids: list[int] = []
    shop_approach_below_journey_ids: list[int] = []
    shop_approach_below_positions: list[tuple[float, float]] = []

    shop_wait_stage_ids: list[list[int]] = [[] for _ in range(num_shops)]
    shop_wait_journey_ids: list[list[int]] = [[] for _ in range(num_shops)]
    shop_wait_positions: list[list[tuple[float, float]]] = [[] for _ in range(num_shops)]

    shop_exit_stage_ids: list[list[int]] = [[] for _ in range(num_shops)]
    shop_exit_journey_ids: list[list[int]] = [[] for _ in range(num_shops)]
    shop_exit_positions: list[list[tuple[float, float]]] = [[] for _ in range(num_shops)]

    for idx, key in enumerate(shop_keys):
        if key not in coordinates:
            continue
        rect = coordinates[key]
        if len(rect) < 4:
            continue
        xs = [x for x, _ in rect]
        ys = [y for _, y in rect]
        min_x = min(xs)
        max_x = max(xs)
        # bottom of shop in world coords (this maps to the TOP of the grey front in screen space)
        bottom_y = min(ys)
        # The grey front is drawn with a fixed pixel height (front_h = 40) and the red line is at its bottom edge.
        # Convert that pixel height into world units so the waypoint line_y matches the rendered red line exactly.
        grey_front_height_world = 40.0 * (WORLD_H / SCREEN_H)
        line_y = bottom_y - grey_front_height_world
        approach_pos = ((min_x + max_x) / 2.0, line_y - 0.30)
        aid, safe_pos = safe_add_waypoint(
            sim, approach_pos, 0.12, hall_polygon, inward_shift=None, jitter=0.0
        )
        if aid is not None:
            shop_approach_below_positions.append(safe_pos)
            shop_approach_below_stage_ids.append(aid)
            shop_approach_below_journey_ids.append(
                sim.add_journey(jps.JourneyDescription([aid]))
            )

        # Multiple entry/exit waypoints along the red line band (exactly where the red line is drawn).
        n_edge_points = 1
        if max_x > min_x:
            dx = (max_x - min_x) / n_edge_points
            for i in range(n_edge_points):
                ex_x = min_x + (i + 0.5) * dx
                entry_pos = (ex_x, line_y)
                # Keep exit points well below the service front so agents leave vertically, not sideways.
                exit_pos = (ex_x, line_y - 0.8)
                stage_id_entry, safe_entry = safe_add_waypoint(
                    sim, entry_pos, 0.35, hall_polygon, inward_shift=(0.0, 0.12), jitter=0.0
                )
                stage_id_exit, safe_exit = safe_add_waypoint(
                    sim, exit_pos, 0.14, hall_polygon, inward_shift=None, jitter=0.0
                )
                if stage_id_entry is None or stage_id_exit is None:
                    continue

                shop_entry_positions[idx].append(safe_entry)
                shop_entry_stage_ids[idx].append(stage_id_entry)
                shop_entry_journey_ids[idx].append(
                    sim.add_journey(jps.JourneyDescription([stage_id_entry]))
                )

                shop_exit_positions[idx].append(safe_exit)
                shop_exit_stage_ids[idx].append(stage_id_exit)
                shop_exit_journey_ids[idx].append(
                    sim.add_journey(jps.JourneyDescription([stage_id_exit]))
                )

        # Multiple waiting spots just INSIDE the grey area, above the red line
        n_wait_cols = 1
        n_wait_rows = 1
        depth = grey_front_height_world * 0.8  # use most of the grey depth in world units
        if max_x > min_x and depth > 0:
            dxw = (max_x - min_x) / n_wait_cols
            dyw = depth / max(n_wait_rows, 1)
            for r in range(n_wait_rows):
                # Start slightly above the red line so agents stand clearly inside the grey front
                wy = line_y + (dyw * 0.5) + r * dyw
                for c in range(n_wait_cols):
                    wx = min_x + (c + 0.5) * dxw
                    base_pos = (wx, wy)
                    stage_id, safe_pos = safe_add_waypoint(
                        sim, base_pos, 0.12, hall_polygon, inward_shift=None, jitter=0.0
                    )
                    if stage_id is None:
                        continue
                    shop_wait_positions[idx].append(safe_pos)
                    shop_wait_stage_ids[idx].append(stage_id)
                    shop_wait_journey_ids[idx].append(
                        sim.add_journey(jps.JourneyDescription([stage_id]))
                    )

    # Only allow routing to shops that have all required waypoint stages.
    # This prevents selecting an empty waypoint list later during the
    # dynamic shop flow.
    valid_shop_indices = [
        i
        for i in range(num_shops)
        if shop_entry_stage_ids[i] and shop_wait_stage_ids[i] and shop_exit_stage_ids[i]
    ]

    # ---- Food court: DES state and waypoints (after shops, same pattern as shops) ----
    food_phase: dict[int, str] = {}
    food_choice: dict[int, int] = {}
    food_choice_enter_edge_idx: dict[int, int] = {}
    food_choice_corridor_idx: dict[int, int] = {}
    food_choice_corridor_after_exit_idx: dict[int, int] = {}
    food_choice_entry_idx: dict[int, int] = {}
    food_choice_wait_idx: dict[int, int] = {}
    food_choice_exit_idx: dict[int, int] = {}
    food_choice_exit_edge_idx: dict[int, int] = {}
    food_wait_time_remaining: dict[int, float] = {}

    food_keys = ["food_burgers_rect", "food_pizza_rect", "food_coffee_rect", "food_desserts_rect"]
    num_food = len(food_keys)

    # Food Enter block waypoints (bottom edge of green block; passengers approach food counters only from below the vertical line)
    food_enter_stage_ids: list[int] = []
    food_enter_journey_ids: list[int] = []
    food_enter_positions: list[tuple[float, float]] = []

    if "food_enter_block" in coordinates:
        rect = coordinates["food_enter_block"]
        xs = [x for x, _ in rect]
        ys = [y for _, y in rect]
        x_min = min(xs)
        x_max = max(xs)
        bottom_y = min(ys)
        n_enter_points = 6
        if x_max > x_min:
            dx = (x_max - x_min) / n_enter_points
            for i in range(n_enter_points):
                cx = x_min + (i + 0.5) * dx
                base_pos = (cx, bottom_y)
                stage_id, safe_pos = safe_add_waypoint(sim, base_pos, 0.35, hall_polygon, inward_shift=(0.0, 0.12))
                if stage_id is None:
                    continue
                food_enter_stage_ids.append(stage_id)
                food_enter_positions.append(safe_pos)
                food_enter_journey_ids.append(
                    sim.add_journey(jps.JourneyDescription([stage_id]))
                )

    # Food Exit block waypoints (left edge of green block)
    food_exit_stage_ids: list[int] = []
    food_exit_journey_ids: list[int] = []
    food_exit_positions: list[tuple[float, float]] = []

    if "food_exit_block" in coordinates:
        rect = coordinates["food_exit_block"]
        xs = [x for x, _ in rect]
        ys = [y for _, y in rect]
        left_x = min(xs)
        y_min = min(ys)
        y_max = max(ys)
        n_exit_points = 6
        if y_max > y_min:
            dy = (y_max - y_min) / n_exit_points
            for i in range(n_exit_points):
                cy = y_min + (i + 0.5) * dy
                base_pos = (left_x, cy)
                stage_id, safe_pos = safe_add_waypoint(sim, base_pos, 0.35, hall_polygon, inward_shift=(0.15, 0.0))
                if stage_id is None:
                    continue
                food_exit_stage_ids.append(stage_id)
                food_exit_positions.append(safe_pos)
                food_exit_journey_ids.append(
                    sim.add_journey(jps.JourneyDescription([stage_id]))
                )

    # Food corridor waypoints: horizontal strip between food_horizontal_line and rest_horizontal_line only
    food_corridor_stage_ids: list[int] = []
    food_corridor_journey_ids: list[int] = []
    food_corridor_positions: list[tuple[float, float]] = []

    if "food_horizontal_line" in coordinates and "rest_horizontal_line" in coordinates:
        food_hline = coordinates["food_horizontal_line"]
        rest_hline = coordinates["rest_horizontal_line"]
        if len(food_hline) >= 1 and len(rest_hline) >= 1:
            food_strip_y_top = food_hline[0][1]
            food_strip_y_bottom = rest_hline[0][1]
            food_strip_y_center = (food_strip_y_top + food_strip_y_bottom) / 2.0
            x_left = food_hline[0][0]
            x_right = food_hline[1][0] if len(food_hline) > 1 else x_left + 10.0
            n_corridor_points = 8
            if x_right > x_left:
                dx = (x_right - x_left) / (n_corridor_points + 1)
                for i in range(n_corridor_points):
                    cx = x_left + (i + 1) * dx
                    base_pos = (cx, food_strip_y_center)
                    stage_id, safe_pos = safe_add_waypoint(sim, base_pos, 0.18, hall_polygon, inward_shift=None)
                    if stage_id is None:
                        continue
                    food_corridor_positions.append(safe_pos)
                    food_corridor_stage_ids.append(stage_id)
                    food_corridor_journey_ids.append(
                        sim.add_journey(jps.JourneyDescription([stage_id]))
                    )

    # Per-food-court: approach-from-below waypoint (below red line), then entry line, wait spots, exit line
    food_approach_below_stage_ids: list[int] = []
    food_approach_below_journey_ids: list[int] = []
    food_approach_below_positions: list[tuple[float, float]] = []

    food_entry_stage_ids: list[list[int]] = [[] for _ in range(num_food)]
    food_entry_journey_ids: list[list[int]] = [[] for _ in range(num_food)]
    food_entry_positions: list[list[tuple[float, float]]] = [[] for _ in range(num_food)]

    food_wait_stage_ids: list[list[int]] = [[] for _ in range(num_food)]
    food_wait_journey_ids: list[list[int]] = [[] for _ in range(num_food)]
    food_wait_positions: list[list[tuple[float, float]]] = [[] for _ in range(num_food)]

    food_exit_line_stage_ids: list[list[int]] = [[] for _ in range(num_food)]
    food_exit_line_journey_ids: list[list[int]] = [[] for _ in range(num_food)]
    food_exit_line_positions: list[list[tuple[float, float]]] = [[] for _ in range(num_food)]

    for idx, key in enumerate(food_keys):
        if key not in coordinates:
            continue
        rect = coordinates[key]
        if len(rect) < 4:
            continue
        xs = [x for x, _ in rect]
        ys = [y for _, y in rect]
        min_x = min(xs)
        max_x = max(xs)
        bottom_y = min(ys)
        grey_front_height_world = 40.0 * (WORLD_H / SCREEN_H)
        line_y = bottom_y - grey_front_height_world

        # One waypoint below the red line so passengers approach food counter only from below
        approach_x = (min_x + max_x) / 2.0
        approach_y = line_y - 0.25
        approach_pos = (approach_x, approach_y)
        aid, safe_pos = safe_add_waypoint(
            sim, approach_pos, 0.18, hall_polygon, inward_shift=None, jitter=0.0
        )
        if aid is not None:
            food_approach_below_positions.append(safe_pos)
            food_approach_below_stage_ids.append(aid)
            food_approach_below_journey_ids.append(
                sim.add_journey(jps.JourneyDescription([aid]))
            )

        n_edge_points = 1
        if max_x > min_x:
            inner_margin_x = 0.20
            usable_min_x = min_x + inner_margin_x
            usable_max_x = max_x - inner_margin_x
            if usable_max_x <= usable_min_x:
                usable_min_x, usable_max_x = min_x, max_x
            dx = (usable_max_x - usable_min_x) / n_edge_points
            for i in range(n_edge_points):
                ex_x = usable_min_x + (i + 0.5) * dx
                entry_pos = (ex_x, line_y)
                # Exit line sits well below the grey counter front to avoid cross-counter drift.
                exit_pos = (ex_x, line_y - 0.8)
                stage_id_entry, safe_entry = safe_add_waypoint(
                    sim, entry_pos, 0.14, hall_polygon, inward_shift=(0.0, 0.12), jitter=0.0
                )
                stage_id_exit, safe_exit = safe_add_waypoint(
                    sim, exit_pos, 0.14, hall_polygon, inward_shift=None, jitter=0.0
                )
                if stage_id_entry is None or stage_id_exit is None:
                    continue

                food_entry_positions[idx].append(safe_entry)
                food_entry_stage_ids[idx].append(stage_id_entry)
                food_entry_journey_ids[idx].append(
                    sim.add_journey(jps.JourneyDescription([stage_id_entry]))
                )

                food_exit_line_positions[idx].append(safe_exit)
                food_exit_line_stage_ids[idx].append(stage_id_exit)
                food_exit_line_journey_ids[idx].append(
                    sim.add_journey(jps.JourneyDescription([stage_id_exit]))
                )

        n_wait_cols = 1
        n_wait_rows = 1
        depth = grey_front_height_world * 0.8
        if max_x > min_x and depth > 0:
            dxw = (max_x - min_x) / n_wait_cols
            dyw = depth / max(n_wait_rows, 1)
            for r in range(n_wait_rows):
                inner_margin_y = 0.18
                wy = line_y + inner_margin_y + (dyw * 0.5) + r * dyw
                for c in range(n_wait_cols):
                    wx = min_x + (c + 0.5) * dxw
                    base_pos = (wx, wy)
                    stage_id, safe_pos = safe_add_waypoint(
                        sim, base_pos, 0.12, hall_polygon, inward_shift=None, jitter=0.0
                    )
                    if stage_id is None:
                        continue
                    food_wait_positions[idx].append(safe_pos)
                    food_wait_stage_ids[idx].append(stage_id)
                    food_wait_journey_ids[idx].append(
                        sim.add_journey(jps.JourneyDescription([stage_id]))
                    )

    # Only allow routing to food counters that have all required waypoint
    # stage lists (otherwise later phases would index into empty lists).
    valid_food_indices = [
        i
        for i in range(num_food)
        if food_entry_stage_ids[i] and food_wait_stage_ids[i] and food_exit_line_stage_ids[i]
    ]

    # ---- Restrooms: DES state and waypoints (after food, same pattern as food) ----
    rest_phase: dict[int, str] = {}
    rest_choice: dict[int, int] = {}
    rest_choice_enter_edge_idx: dict[int, int] = {}
    rest_choice_corridor_idx: dict[int, int] = {}
    rest_choice_corridor_after_exit_idx: dict[int, int] = {}
    rest_choice_entry_idx: dict[int, int] = {}
    rest_choice_wait_idx: dict[int, int] = {}
    rest_choice_exit_idx: dict[int, int] = {}
    rest_choice_exit_edge_idx: dict[int, int] = {}
    rest_wait_time_remaining: dict[int, float] = {}

    rest_keys = ["rest1_rect", "rest2_rect", "rest3_rect"]
    num_rest = len(rest_keys)

    # Rest Enter/Exit block waypoints (left edge; open red side)
    rest_enter_stage_ids: list[int] = []
    rest_enter_journey_ids: list[int] = []
    rest_enter_positions: list[tuple[float, float]] = []
    if "rest_enter_block" in coordinates:
        rect = coordinates["rest_enter_block"]
        xs = [x for x, _ in rect]
        ys = [y for _, y in rect]
        left_x = min(xs)
        y_min = min(ys)
        y_max = max(ys)
        n_points = 6
        if y_max > y_min:
            dy = (y_max - y_min) / n_points
            for i in range(n_points):
                cy = y_min + (i + 0.5) * dy
                base_pos = (left_x, cy)
                sid, safe_pos = safe_add_waypoint(sim, base_pos, 0.35, hall_polygon, inward_shift=(0.15, 0.0))
                if sid is None:
                    continue
                rest_enter_stage_ids.append(sid)
                rest_enter_positions.append(safe_pos)
                rest_enter_journey_ids.append(sim.add_journey(jps.JourneyDescription([sid])))

    rest_exit_stage_ids: list[int] = []
    rest_exit_journey_ids: list[int] = []
    rest_exit_positions: list[tuple[float, float]] = []
    if "rest_exit_block" in coordinates:
        rect = coordinates["rest_exit_block"]
        xs = [x for x, _ in rect]
        ys = [y for _, y in rect]
        left_x = min(xs)
        y_min = min(ys)
        y_max = max(ys)
        n_points = 6
        if y_max > y_min:
            dy = (y_max - y_min) / n_points
            for i in range(n_points):
                cy = y_min + (i + 0.5) * dy
                base_pos = (left_x, cy)
                sid, safe_pos = safe_add_waypoint(sim, base_pos, 0.35, hall_polygon, inward_shift=(0.15, 0.0))
                if sid is None:
                    continue
                rest_exit_stage_ids.append(sid)
                rest_exit_positions.append(safe_pos)
                rest_exit_journey_ids.append(sim.add_journey(jps.JourneyDescription([sid])))

    # Rest corridor waypoints: strip between rest_horizontal_line and gates_horizontal_line
    rest_corridor_stage_ids: list[int] = []
    rest_corridor_journey_ids: list[int] = []
    rest_corridor_positions: list[tuple[float, float]] = []
    if "rest_horizontal_line" in coordinates and "gates_horizontal_line" in coordinates:
        rest_hline = coordinates["rest_horizontal_line"]
        gates_hline = coordinates["gates_horizontal_line"]
        if len(rest_hline) >= 1 and len(gates_hline) >= 1:
            strip_y_top = rest_hline[0][1]
            strip_y_bottom = gates_hline[0][1]
            strip_y_center = (strip_y_top + strip_y_bottom) / 2.0
            x_left = rest_hline[0][0]
            x_right = rest_hline[1][0] if len(rest_hline) > 1 else x_left + 10.0
            n_corridor_points = 8
            if x_right > x_left:
                dx = (x_right - x_left) / (n_corridor_points + 1)
                for i in range(n_corridor_points):
                    cx = x_left + (i + 1) * dx
                    base_pos = (cx, strip_y_center)
                    sid, safe_pos = safe_add_waypoint(sim, base_pos, 0.18, hall_polygon, inward_shift=None)
                    if sid is None:
                        continue
                    rest_corridor_positions.append(safe_pos)
                    rest_corridor_stage_ids.append(sid)
                    rest_corridor_journey_ids.append(sim.add_journey(jps.JourneyDescription([sid])))

    # Per-restroom: entry line, wait spots, exit line (like food)
    rest_entry_stage_ids: list[list[int]] = [[] for _ in range(num_rest)]
    rest_entry_journey_ids: list[list[int]] = [[] for _ in range(num_rest)]
    rest_entry_positions: list[list[tuple[float, float]]] = [[] for _ in range(num_rest)]
    rest_approach_below_stage_ids: list[int] = []
    rest_approach_below_journey_ids: list[int] = []
    rest_approach_below_positions: list[tuple[float, float]] = []

    rest_wait_stage_ids: list[list[int]] = [[] for _ in range(num_rest)]
    rest_wait_journey_ids: list[list[int]] = [[] for _ in range(num_rest)]
    rest_wait_positions: list[list[tuple[float, float]]] = [[] for _ in range(num_rest)]

    rest_exit_line_stage_ids: list[list[int]] = [[] for _ in range(num_rest)]
    rest_exit_line_journey_ids: list[list[int]] = [[] for _ in range(num_rest)]
    rest_exit_line_positions: list[list[tuple[float, float]]] = [[] for _ in range(num_rest)]

    for idx, key in enumerate(rest_keys):
        if key not in coordinates:
            continue
        rect = coordinates[key]
        if len(rect) < 4:
            continue
        xs = [x for x, _ in rect]
        ys = [y for _, y in rect]
        min_x = min(xs)
        max_x = max(xs)
        bottom_y = min(ys)
        grey_front_height_world = 40.0 * (WORLD_H / SCREEN_H)
        line_y = bottom_y - grey_front_height_world
        approach_pos = ((min_x + max_x) / 2.0, line_y - 0.30)
        aid, safe_pos = safe_add_waypoint(
            sim, approach_pos, 0.12, hall_polygon, inward_shift=None, jitter=0.0
        )
        if aid is not None:
            rest_approach_below_positions.append(safe_pos)
            rest_approach_below_stage_ids.append(aid)
            rest_approach_below_journey_ids.append(
                sim.add_journey(jps.JourneyDescription([aid]))
            )

        n_edge_points = 4
        if max_x > min_x:
            dx = (max_x - min_x) / n_edge_points
            for i in range(n_edge_points):
                ex_x = min_x + (i + 0.5) * dx
                entry_pos = (ex_x, line_y)
                # Exit points are well below the counter front so agents head out directly.
                exit_pos = (ex_x, line_y - 0.8)
                sid_entry, safe_entry = safe_add_waypoint(
                    sim, entry_pos, 0.14, hall_polygon, inward_shift=(0.0, 0.12), jitter=0.0
                )
                if sid_entry is None:
                    continue
                sid_exit, safe_exit = safe_add_waypoint(
                    sim, exit_pos, 0.14, hall_polygon, inward_shift=None, jitter=0.0
                )
                if sid_exit is None:
                    continue
                rest_entry_positions[idx].append(safe_entry)
                rest_entry_stage_ids[idx].append(sid_entry)
                rest_entry_journey_ids[idx].append(sim.add_journey(jps.JourneyDescription([sid_entry])))

                rest_exit_line_positions[idx].append(safe_exit)
                rest_exit_line_stage_ids[idx].append(sid_exit)
                rest_exit_line_journey_ids[idx].append(sim.add_journey(jps.JourneyDescription([sid_exit])))

        n_wait_cols = 4
        n_wait_rows = 2
        depth = grey_front_height_world * 0.8
        if max_x > min_x and depth > 0:
            dxw = (max_x - min_x) / n_wait_cols
            dyw = depth / max(n_wait_rows, 1)
            for r in range(n_wait_rows):
                inner_margin_y = 0.18
                wy = line_y + inner_margin_y + (dyw * 0.5) + r * dyw
                for c in range(n_wait_cols):
                    wx = min_x + (c + 0.5) * dxw
                    base_pos = (wx, wy)
                    sid, safe_pos = safe_add_waypoint(
                        sim, base_pos, 0.12, hall_polygon, inward_shift=None, jitter=0.0
                    )
                    if sid is None:
                        continue
                    rest_wait_positions[idx].append(safe_pos)
                    rest_wait_stage_ids[idx].append(sid)
                    rest_wait_journey_ids[idx].append(sim.add_journey(jps.JourneyDescription([sid])))

    # Only allow routing to restroom counters that have all required
    # waypoint stage lists (entry/wait/exit lines), so later phases never
    # index into empty lists.
    valid_rest_indices = [
        i
        for i in range(num_rest)
        if rest_entry_stage_ids[i]
        and rest_wait_stage_ids[i]
        and rest_exit_line_stage_ids[i]
    ]

    # ---- Gates: DES state and gate exit stages (after restrooms) ----
    gates_phase: dict[int, str] = {}
    gates_choice_gate_name: dict[int, str] = {}
    gates_choice_enter_edge_idx: dict[int, int] = {}
    gates_choice_corridor_idx: dict[int, int] = {}

    gates_enter_stage_ids: list[int] = []
    gates_enter_journey_ids: list[int] = []
    gates_enter_positions: list[tuple[float, float]] = []
    if "gates_enter_block" in coordinates:
        rect = coordinates["gates_enter_block"]
        xs = [x for x, _ in rect]
        ys = [y for _, y in rect]
        left_x = min(xs)
        y_min = min(ys)
        y_max = max(ys)
        n_points = 6
        if y_max > y_min:
            dy = (y_max - y_min) / n_points
            for i in range(n_points):
                cy = y_min + (i + 0.5) * dy
                base_pos = (left_x, cy)
                sid, safe_pos = safe_add_waypoint(sim, base_pos, 0.35, hall_polygon, inward_shift=(0.15, 0.0))
                if sid is None:
                    continue
                gates_enter_stage_ids.append(sid)
                gates_enter_positions.append(safe_pos)
                gates_enter_journey_ids.append(sim.add_journey(jps.JourneyDescription([sid])))

    # Gates corridor waypoints: band below gates_horizontal_line down to hall bottom
    gates_corridor_stage_ids: list[int] = []
    gates_corridor_journey_ids: list[int] = []
    gates_corridor_positions: list[tuple[float, float]] = []
    hall_poly = coordinates["hall"]["polygon"]
    hall_bottom_y = min(y for _x, y in hall_poly)
    if "gates_horizontal_line" in coordinates and len(coordinates["gates_horizontal_line"]) >= 1:
        gates_hline = coordinates["gates_horizontal_line"]
        strip_y_top = gates_hline[0][1]
        strip_y_bottom = hall_bottom_y
        strip_y_center = (strip_y_top + strip_y_bottom) / 2.0
        x_left = gates_hline[0][0]
        x_right = gates_hline[1][0] if len(gates_hline) > 1 else x_left + 10.0
        n_corridor_points = 8
        if x_right > x_left:
            dx = (x_right - x_left) / (n_corridor_points + 1)
            for i in range(n_corridor_points):
                cx = x_left + (i + 1) * dx
                base_pos = (cx, strip_y_center)
                sid, safe_pos = safe_add_waypoint(sim, base_pos, 0.18, hall_polygon, inward_shift=None)
                if sid is None:
                    continue
                gates_corridor_positions.append(safe_pos)
                gates_corridor_stage_ids.append(sid)
                gates_corridor_journey_ids.append(sim.add_journey(jps.JourneyDescription([sid])))

    # Gate exit stages (entering gate red edge ends journey)
    gates_gate_exit_stage_ids: dict[str, int] = {}
    gates_gate_exit_journey_ids: dict[str, int] = {}
    gates_gate_exit_positions: dict[str, tuple[float, float]] = {}
    if "gates_gate_blocks" in coordinates:
        for name, rect in coordinates["gates_gate_blocks"].items():
            if len(rect) < 4:
                continue
            xs = [x for x, _ in rect]
            ys = [y for _, y in rect]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            # Gate "red line" in world coords:
            # A-row: bottom edge (min_y) -> approach from below (walkable below the obstacle)
            # B-row: top edge (max_y)   -> approach from above (walkable above the obstacle)
            if name.startswith("B"):
                line_y = max_y
                approach_offset = 0.12  # approach from above
            else:
                line_y = min_y
                approach_offset = -0.12  # approach from below

            # Target point must stay very close to the intended gate red line.
            base_pos = ((min_x + max_x) / 2.0, line_y + approach_offset)
            _sid, safe_pos = safe_add_waypoint(
                sim,
                base_pos,
                GATES_WAYPOINT_RADIUS,
                hall_polygon,
                attempts=40,
                jitter=0.02,
                inward_shift=None,
            )
            if _sid is None:
                continue
            if math.hypot(safe_pos[0] - base_pos[0], safe_pos[1] - base_pos[1]) > 0.20:
                # If the only "safe" point is far away, skip this gate (prevents accidental early exits).
                continue
            gx, gy = safe_pos
            half = 0.18
            exit_poly = [
                (gx - half, gy - half),
                (gx + half, gy - half),
                (gx + half, gy + half),
                (gx - half, gy + half),
            ]
            try:
                exit_stage_id = sim.add_exit_stage(exit_poly)
            except RuntimeError:
                continue
            gates_gate_exit_positions[name] = safe_pos
            gates_gate_exit_stage_ids[name] = exit_stage_id
            gates_gate_exit_journey_ids[name] = sim.add_journey(
                jps.JourneyDescription([exit_stage_id])
            )
    GATE_EXIT_STAGE_IDS = set(gates_gate_exit_stage_ids.values())
    # A-lane only (B row removed from layout): 13 gates, divide MAX_PASSENGERS as evenly as possible
    gate_names_ordered = [f"A{i}" for i in range(1, 14)]
    gate_allocation_list: list[str] = []
    n_gates = len(gate_names_ordered)
    if n_gates > 0:
        base_quota, remainder = divmod(MAX_PASSENGERS, n_gates)
        for idx, gate_name in enumerate(gate_names_ordered):
            gate_quota = base_quota + (1 if idx < remainder else 0)
            gate_allocation_list.extend([gate_name] * gate_quota)
    random.shuffle(gate_allocation_list)
    next_gate_allocation_idx = 0

    for rect in grey_rects:
        if len(rect) >= 4:
            xs = [x for x, _ in rect]
            ys = [y for _, y in rect]

            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

            # 3 columns × 2 rows grid inside the grey rectangle
            cols = 3
            rows = 2
            dx = (max_x - min_x) / (cols + 1)
            dy = (max_y - min_y) / (rows + 1)

            stages_for_display: list[int] = []
            journeys_for_display: list[int] = []
            spots_for_display: list[tuple[float, float]] = []

            for r in range(rows):
                for c in range(cols):
                    cx = min_x + (c + 1) * dx
                    cy = min_y + (r + 1) * dy - 0.15
                    base_pos = (cx, cy)
                    stage_id, safe_pos = safe_add_waypoint(sim, base_pos, 0.30, hall_polygon, inward_shift=None)
                    if stage_id is None:
                        continue
                    stages_for_display.append(stage_id)
                    journeys_for_display.append(
                        sim.add_journey(jps.JourneyDescription([stage_id]))
                    )
                    spots_for_display.append(safe_pos)

            display_stage_ids.append(stages_for_display)
            display_journey_ids.append(journeys_for_display)
            display_spot_positions.append(spots_for_display)

            # Entry waypoint on the red (right) edge mid‑height; offset and larger radius to reduce clustering
            entry_x = max_x
            entry_y = (min_y + max_y) / 2.0 - 0.15
            base_pos = (entry_x, entry_y)
            entry_stage_id, safe_entry = safe_add_waypoint(sim, base_pos, 0.40, hall_polygon, inward_shift=(-0.15, 0.0))
            if entry_stage_id is None:
                continue
            display_entry_positions.append(safe_entry)
            display_entry_stage_ids.append(entry_stage_id)
            display_entry_journey_ids.append(
                sim.add_journey(jps.JourneyDescription([entry_stage_id]))
            )

    # Single JuPedSim queue stage for waiting area
    waiting_positions = sanitize_positions(waiting_positions, hall_polygon)
    waiting_queue_stage_id = sim.add_queue_stage(waiting_positions)

    counter_stage_ids: list[int] = []
    counter_journey_ids: list[int] = []
    for (x, y) in counter_positions:
        sid, _safe_pos = safe_add_waypoint(sim, (x, y), WAYPOINT_RADIUS, hall_polygon)
        if sid is None:
            continue
        counter_stage_ids.append(sid)
        counter_journey_ids.append(sim.add_journey(jps.JourneyDescription([sid])))

    # Counter queue slot waypoints and single-target journeys (hybrid DES + JuPedSim like security)
    counter_queue_slot_stage_ids: list[list[int]] = []
    counter_queue_slot_journey_ids: list[list[int]] = []
    for j in range(num_counters):
        stage_ids_j = []
        journey_ids_j = []
        for (x, y) in counter_queue_slot_positions[j]:
            sid, _safe_pos = safe_add_waypoint(sim, (x, y), WAYPOINT_RADIUS, hall_polygon)
            if sid is None:
                continue
            stage_ids_j.append(sid)
            journey_ids_j.append(sim.add_journey(jps.JourneyDescription([sid])))
        counter_queue_slot_stage_ids.append(stage_ids_j)
        counter_queue_slot_journey_ids.append(journey_ids_j)

    # Counter exit waypoints: after service, go here then to info desk waiting
    counter_exit_stage_ids: list[int] = []
    counter_exit_journey_ids: list[int] = []
    counter_exit_positions: list[tuple[float, float]] = []
    for j, (cx, cy) in enumerate(counter_positions):
        release_dx = -1.0 if j in (0, 2) else 1.0
        release_pos = (cx + release_dx, cy + 1.2)
        sid, safe_pos = safe_add_waypoint(sim, release_pos, WAYPOINT_RADIUS, hall_polygon, inward_shift=None)
        if sid is None:
            continue
        counter_exit_positions.append(safe_pos)
        counter_exit_stage_ids.append(sid)
        counter_exit_journey_ids.append(sim.add_journey(jps.JourneyDescription([sid])))

    # Info desk queue slot waypoints and service waypoints (same pattern as ticket counters)
    info_desk_queue_slot_stage_ids: list[list[int]] = []
    info_desk_queue_slot_journey_ids: list[list[int]] = []
    for d in range(num_info_desks):
        stage_ids_d = []
        journey_ids_d = []
        for (x, y) in info_desk_queue_slot_positions[d]:
            # Use safe clamp/jitter so shifted layouts never place waypoints
            # outside the walkable area.
            sid, _safe_pos = safe_add_waypoint(sim, (x, y), WAYPOINT_RADIUS, hall_polygon)
            if sid is None:
                continue
            stage_ids_d.append(sid)
            journey_ids_d.append(sim.add_journey(jps.JourneyDescription([sid])))
        info_desk_queue_slot_stage_ids.append(stage_ids_d)
        info_desk_queue_slot_journey_ids.append(journey_ids_d)
    info_desk_stage_ids: list[int] = []
    info_desk_journey_ids: list[int] = []
    for (x, y) in info_desk_service_positions:
        sid, _safe_pos = safe_add_waypoint(sim, (x, y), WAYPOINT_RADIUS, hall_polygon)
        if sid is None:
            continue
        info_desk_stage_ids.append(sid)
        info_desk_journey_ids.append(sim.add_journey(jps.JourneyDescription([sid])))

    # Waiting: journey to waiting queue (used after agent reaches entrance)
    journey_waiting_id = sim.add_journey(jps.JourneyDescription([waiting_queue_stage_id]))

    # Fixed overflow holding waypoints (not a queue stage).
    safe_overflow_points: list[tuple[float, float]] = []
    for (x, y) in overflow_hold_points:
        x, y = get_safe_position(x, y, hall_polygon)
        x, y = force_safe_point(x, y, hall_polygon)
        x, y = clamp_to_hall(x, y)
        if y > LAST_WAITING_Y - 0.3:
            y = LAST_WAITING_Y - 0.3
            x, y = force_safe_point(x, y, hall_polygon)
            x, y = clamp_to_hall(x, y)
        safe_overflow_points.append((x, y))

    overflow_stage_ids: list[int] = []
    overflow_journey_ids: list[int] = []
    for pos in safe_overflow_points:
        sid, _safe_pos = safe_add_waypoint(sim, pos, 0.35, hall_polygon)
        if sid is None:
            continue
        overflow_stage_ids.append(sid)
        overflow_journey_ids.append(
            sim.add_journey(jps.JourneyDescription([sid]))
        )

    # Ordered overflow tracking (FIFO promotion + round-robin hold assignment).
    overflow_passengers: list[int] = []
    overflow_index = 0

    entrance_names = list(coordinates["entrances"].keys())
    entrance_positions = [coordinates["entrances"][name] for name in entrance_names]

    # Parking area bounds (world coords) for random spawn
    # Use a narrower region on the left side of the hall
    hall_poly = coordinates["hall"]["polygon"]
    door_ys = [dy for (_dx, dy) in coordinates["entrances"].values()]
    door_y_world = min(door_ys)
    park_y_top = door_y_world - 0.3
    park_y_bottom = 0.0
    park_x_left = min(x for x, _ in hall_poly)
    hall_width = max(x for x, _ in hall_poly) - park_x_left
    park_x_right = park_x_left + hall_width / 3.0
    park_margin = 0.15  # inset from parking edges for spawn

    waiting_front = waiting_positions[0]
    counter_front_slots = [counter_queue_slot_positions[j][0] for j in range(num_counters)]

    # DES: counter queues and service (hybrid DES + JuPedSim like security)
    des_counter_queue_lists: list[list[int]] = [[] for _ in range(num_counters)]
    waiting_front_wait_remaining: float | None = None
    prev_waiting_count = num_waiting_slots
    counter_queue_front_wait_remaining: list[float | None] = [None] * num_counters
    counter_serving_agent: list[int | None] = [None] * num_counters
    counter_service_remaining: list[float] = [0.0] * num_counters
    des_agents_at_counter: set[int] = set()

    # Security queue slot waypoints and journeys (same pattern as counter)
    security_queue_slot_stage_ids: list[list[int]] = []
    security_queue_slot_journey_ids: list[list[int]] = []
    for s in range(num_security):
        stage_ids_s = []
        journey_ids_s = []
        for (x, y) in security_queue_slot_positions[s]:
            sid, _safe_pos = safe_add_waypoint(sim, (x, y), WAYPOINT_RADIUS, hall_polygon)
            if sid is None:
                continue
            stage_ids_s.append(sid)
            journey_ids_s.append(sim.add_journey(jps.JourneyDescription([sid])))
        security_queue_slot_stage_ids.append(stage_ids_s)
        security_queue_slot_journey_ids.append(journey_ids_s)

    security_stage_ids: list[int] = []
    security_journey_ids: list[int] = []
    for (x, y) in security_positions:
        sid, _safe_pos = safe_add_waypoint(sim, (x, y), WAYPOINT_RADIUS, hall_polygon)
        if sid is None:
            continue
        security_stage_ids.append(sid)
        security_journey_ids.append(sim.add_journey(jps.JourneyDescription([sid])))

    security_front_slots = [security_queue_slot_positions[s][0] for s in range(num_security)]
    info_desk_front_slots = [info_desk_queue_slot_positions[d][0] for d in range(num_info_desks)]
    num_security_waiting_slots = len(security_waiting_positions)
    SECURITY_LAST_WAITING_Y = (
        min(y for (_x, y) in security_waiting_positions)
        if security_waiting_positions
        else 0.0
    )

    # DES: info desk queues and service (after ticket counter)
    des_info_queue_lists: list[list[int]] = [[] for _ in range(num_info_desks)]
    info_desk_queue_front_wait_remaining: list[float | None] = [None] * num_info_desks
    info_desk_serving_agent: list[int | None] = [None] * num_info_desks
    info_desk_service_remaining: list[float] = [0.0] * num_info_desks

    # DES: security queues and service
    des_security_queue_lists: list[list[int]] = [[] for _ in range(num_security)]
    security_queue_front_wait_remaining: list[float | None] = [None] * num_security
    security_serving_agent: list[int | None] = [None] * num_security
    security_service_remaining: list[float] = [0.0] * num_security

    # DES: first in security waiting queue waits at front before going to security queue
    security_waiting_front_wait_remaining: float | None = None
    prev_security_waiting_count = num_security_waiting_slots

    # Fixed security overflow holding waypoints below security waiting area.
    safe_security_overflow_points: list[tuple[float, float]] = []
    for (x, y) in security_overflow_hold_points:
        x, y = get_safe_position(x, y, hall_polygon)
        x, y = force_safe_point(x, y, hall_polygon)
        x, y = clamp_to_hall(x, y)
        if y > SECURITY_LAST_WAITING_Y - 0.3:
            y = SECURITY_LAST_WAITING_Y - 0.3
            x, y = force_safe_point(x, y, hall_polygon)
            x, y = clamp_to_hall(x, y)
        safe_security_overflow_points.append((x, y))

    security_overflow_stage_ids: list[int] = []
    security_overflow_journey_ids: list[int] = []
    for pos in safe_security_overflow_points:
        sid, _safe_pos = safe_add_waypoint(sim, pos, 0.35, hall_polygon)
        if sid is None:
            continue
        security_overflow_stage_ids.append(sid)
        security_overflow_journey_ids.append(
            sim.add_journey(jps.JourneyDescription([sid]))
        )

    # Ordered overflow tracking for security waiting queue.
    security_overflow_passengers: list[int] = []
    security_overflow_index = 0

    if not HEADLESS_MODE:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Airport check-in (JuPedSim waiting queue, DES counters)")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 22)
        lane_font = pygame.font.Font(None, 14)
        info_desk_label_font = pygame.font.Font(None, 10)
        # Dedicated font for the flight display label to keep it visually smaller.
        display_label_font = pygame.font.Font(None, 10)
        lane_label_font = pygame.font.Font(None, 10)
        counter_number_font = pygame.font.Font(None, 20)
        counter_label_font = pygame.font.Font(None, 14)
    else:
        screen = None
        clock = None
        font = lane_font = info_desk_label_font = display_label_font = None
        lane_label_font = counter_number_font = counter_label_font = None

    hall_points = coordinates["hall"]["polygon"]
    iteration = 0
    total_spawned = 0

    # Performance metrics
    completed_passengers = 0
    completed_security_passengers = 0
    total_passengers_generated = 0
    total_passengers_exited = 0
    exited_agent_ids: set[int] = set()

    # Time-series logs for analysis
    time_steps: list[float] = []
    exited_counts: list[int] = []
    ci_queue_lengths: list[int] = []
    sc_queue_lengths: list[int] = []
    overflow_counts: list[int] = []
    time_series = []
    Y_series = []
    passengers_in_system_series: list[int] = []
    checkin_utilization_series: list[float] = []
    security_utilization_series: list[float] = []
    # Zone-based occupancy aggregation for hotspot heatmaps.
    ZONE_COLS = 50
    ZONE_ROWS = 30
    zone_sum_occupancy: dict[tuple[int, int], float] = {}
    zone_sample_steps = 0

    # Per-passenger waiting-time timestamps
    entry_times: dict[int, float] = {}
    ci_join_times: dict[int, float] = {}
    ci_service_start_times: dict[int, float] = {}
    sc_join_times: dict[int, float] = {}
    sc_service_start_times: dict[int, float] = {}
    system_time_samples: list[float] = []

    stop_due_to_horizon = False
    # Gate: passengers must complete ticket-counter service before any other services
    completed_ticket_counter: set[int] = set()
    last_spawn_sim_time: float = -1e9
    entered_airport: dict[int, bool] = {}

    ready_to_switch: dict[int, float] = {}
    pending_switch: dict[int, tuple[int, int]] = {}  # agent_id -> (journey_id, stage_id)

    original_switch_agent_journey = sim.switch_agent_journey

    def _switch_agent_journey_safe(agent_id, journey_id, stage_id):
        # If we are in the "not ready yet" window, defer the switch.
        if agent_id in ready_to_switch:
            pending_switch[agent_id] = (journey_id, stage_id)
            return

        # Collision-safe switching: skip if too close to another agent.
        try:
            agent = sim.agent(agent_id)
        except (KeyError, RuntimeError):
            return
        if not is_safe_to_switch(agent, sim):
            return

        try:
            original_switch_agent_journey(agent_id, journey_id, stage_id)
        except RuntimeError:
            return
        if stage_id in GATE_EXIT_STAGE_IDS:
            mark_passenger_exited(agent_id)

    try:
        sim.switch_agent_journey = _switch_agent_journey_safe
    except Exception:
        pass

    def mark_passenger_exited(agent_id: int) -> None:
        nonlocal completed_passengers, total_passengers_exited
        if agent_id in exited_agent_ids:
            return
        exited_agent_ids.add(agent_id)
        completed_passengers += 1
        total_passengers_exited += 1
        if agent_id in entry_times:
            system_time_samples.append(max(0.0, sim.elapsed_time() - entry_times[agent_id]))
        plan_activity = planned_activity.get(agent_id, activity_choice.get(agent_id, "unknown"))
        if plan_activity == "gates":
            plan_activity = "direct"
        gate_name = assigned_gate.get(agent_id, "unknown")
        raw_steps = list(actual_path.get(agent_id, []))
        deduped: list[str] = []
        for s in raw_steps:
            if not deduped or deduped[-1] != s:
                deduped.append(s)
        if gate_name != "unknown":
            while deduped and deduped[-1] in gate_names_ordered:
                deduped.pop()
            deduped.append(gate_name)
        actual = " → ".join(deduped) if deduped else gate_name
        if not FAST_MODE:
            print(f"{agent_id} → PLAN: {plan_activity} → {gate_name} | ACTUAL: {actual}")

    def choose_intermediate_dest() -> str:
        """
        Choose exactly one post-display destination using a single weighted decision.
        Fallback order is: gates first, then any other valid available destination.
        """
        chosen = random.choices(
            ["gates", "shops", "food", "rest"],
            weights=[PROB_DIRECT, PROB_SHOPS, PROB_FOOD, PROB_REST],
            k=1,
        )[0]

        # Use the same availability checks as the original routing blocks.
        gates_available = bool(gates_enter_stage_ids and gates_gate_exit_journey_ids)
        shops_available = bool(valid_shop_indices and shops_enter_stage_ids and shops_exit_stage_ids)
        food_available = bool(valid_food_indices and food_enter_stage_ids and food_exit_stage_ids)
        rest_available = bool(
            rest_enter_stage_ids
            and rest_exit_stage_ids
            and rest_corridor_stage_ids
            and valid_rest_indices
        )

        if chosen == "gates" and gates_available:
            return "gates"
        if (
            chosen == "shops"
            and shops_available
        ):
            return "shops"
        if (
            chosen == "food"
            and food_available
        ):
            return "food"
        if (
            chosen == "rest"
            and rest_available
        ):
            return "rest"

        # Fallback order: gates first, then any other valid available destination.
        if gates_available:
            return "gates"
        if shops_available:
            return "shops"
        if food_available:
            return "food"
        if rest_available:
            return "rest"
        # Final safety fallback if no area-specific destination is available.
        return "gates"

    def append_path_segment(aid: int, seg: str) -> None:
        if not actual_path.get(aid):
            actual_path[aid] = []
        if not actual_path[aid] or actual_path[aid][-1] != seg:
            actual_path[aid].append(seg)

    def dispatch_post_security_activity(aid: int, dest: str) -> None:
        """Start shops / food / rest / gates flow for this agent (post-security or post-display)."""
        if dest == "shops":
            append_path_segment(aid, "shops")
            food_phase.pop(aid, None)
            rest_phase.pop(aid, None)
            sim.switch_agent_journey(aid, security_exit_journey_id, security_exit_stage_id)
            shop_idx = random.choice(valid_shop_indices)
            shops_choice[aid] = shop_idx
            shops_choice_shop_wait_idx[aid] = random.randrange(
                len(shop_wait_stage_ids[shop_idx])
            )
            shops_phase[aid] = "to_main_entrance"
        elif dest == "food":
            append_path_segment(aid, "food")
            shops_phase.pop(aid, None)
            rest_phase.pop(aid, None)
            food_idx = random.choice(valid_food_indices)
            food_choice[aid] = food_idx
            food_choice_entry_idx[aid] = 0
            food_choice_wait_idx[aid] = 0
            food_choice_exit_idx[aid] = 0
            food_phase[aid] = "to_food_enter"
            enter_idx = 0
            food_choice_enter_edge_idx[aid] = enter_idx
            sim.switch_agent_journey(aid, food_enter_journey_ids[enter_idx], food_enter_stage_ids[enter_idx])
        elif dest == "rest":
            append_path_segment(aid, "rest")
            shops_phase.pop(aid, None)
            food_phase.pop(aid, None)
            rest_idx = random.choice(valid_rest_indices)
            rest_choice[aid] = rest_idx
            rest_phase[aid] = "to_rest_enter"
            enter_edge_idx = 0
            rest_choice_enter_edge_idx[aid] = enter_edge_idx
            rest_choice_corridor_idx[aid] = 0
            rest_choice_entry_idx[aid] = 0
            rest_choice_wait_idx[aid] = 0
            rest_choice_exit_idx[aid] = 0
            rest_choice_corridor_after_exit_idx[aid] = 0
            sim.switch_agent_journey(aid, rest_enter_journey_ids[enter_edge_idx], rest_enter_stage_ids[enter_edge_idx])
        else:
            # Direct to gates (or fallback if chosen destination unavailable)
            if gates_enter_stage_ids and gates_gate_exit_journey_ids:
                append_path_segment(aid, "direct")
                shops_phase.pop(aid, None)
                shops_choice.pop(aid, None)
                shops_choice_enter_edge_idx.pop(aid, None)
                shops_choice_shop_entry_idx.pop(aid, None)
                shops_choice_shop_wait_idx.pop(aid, None)
                shops_choice_shop_exit_idx.pop(aid, None)
                shops_choice_exit_edge_idx.pop(aid, None)
                shop_wait_time_remaining.pop(aid, None)

                food_phase.pop(aid, None)
                food_choice.pop(aid, None)
                food_choice_enter_edge_idx.pop(aid, None)
                food_choice_corridor_idx.pop(aid, None)
                food_choice_corridor_after_exit_idx.pop(aid, None)
                food_choice_entry_idx.pop(aid, None)
                food_choice_wait_idx.pop(aid, None)
                food_choice_exit_idx.pop(aid, None)
                food_choice_exit_edge_idx.pop(aid, None)
                food_wait_time_remaining.pop(aid, None)

                rest_phase.pop(aid, None)
                rest_choice.pop(aid, None)
                rest_choice_enter_edge_idx.pop(aid, None)
                rest_choice_corridor_idx.pop(aid, None)
                rest_choice_corridor_after_exit_idx.pop(aid, None)
                rest_choice_entry_idx.pop(aid, None)
                rest_choice_wait_idx.pop(aid, None)
                rest_choice_exit_idx.pop(aid, None)
                rest_choice_exit_edge_idx.pop(aid, None)
                rest_wait_time_remaining.pop(aid, None)

                display_assignment.pop(aid, None)
                display_phase.pop(aid, None)
                display_wait_time_remaining.pop(aid, None)
                for q in display_wait_lists:
                    if aid in q:
                        q.remove(aid)

                enter_edge_idx = random.randrange(len(gates_enter_stage_ids))
                gates_choice_enter_edge_idx[aid] = enter_edge_idx
                gate_name = assigned_gate.get(aid, "A1")
                if gate_name not in gates_gate_exit_journey_ids and gates_gate_exit_journey_ids:
                    gate_name = next(iter(gates_gate_exit_journey_ids))
                gates_choice_gate_name[aid] = gate_name
                gates_phase[aid] = "to_gates_enter"
                append_path_segment(aid, gate_name)
                sim.switch_agent_journey(aid, gates_enter_journey_ids[enter_edge_idx], gates_enter_stage_ids[enter_edge_idx])
            else:
                append_path_segment(aid, "direct")
                sim.switch_agent_journey(aid, final_exit_journey_id, final_exit_stage_id)

    def compute_activity_decision(aid: int) -> None:
        planned = choose_intermediate_dest()

        planned_activity[aid] = planned

        if planned == "shops":
            passenger_type[aid] = "shops_gate"
            passenger_color[aid] = COLOR_SHOPS
        elif planned == "food":
            passenger_type[aid] = "food_gate"
            passenger_color[aid] = COLOR_FOOD
        elif planned == "rest":
            passenger_type[aid] = "rest_gate"
            passenger_color[aid] = COLOR_REST
        else:
            passenger_type[aid] = "direct_gate"
            passenger_color[aid] = COLOR_DIRECT

        misroute_first_choice.pop(aid, None)
        misroute_done.pop(aid, None)
        misroute_roam_only.pop(aid, None)

        if planned == "gates":
            activity_choice[aid] = "gates"
            actual_path.pop(aid, None)
            return

        activity_choice[aid] = planned
        actual_path.pop(aid, None)

    def decide_and_route_after_display_or_security(aid: int) -> None:
        compute_activity_decision(aid)
        dispatch_post_security_activity(aid, activity_choice[aid])

    def try_misroute_redirect_to_planned(aid: int) -> bool:
        if aid not in misroute_first_choice or misroute_done.get(aid, False):
            return False
        planned = planned_activity.get(aid)
        if planned is None or planned == "gates":
            return False
        misroute_done[aid] = True
        misroute_roam_only.pop(aid, None)
        food_misroute_roam_remaining.pop(aid, None)
        shops_misroute_roam_remaining.pop(aid, None)
        rest_misroute_roam_remaining.pop(aid, None)
        activity_choice[aid] = planned
        dispatch_post_security_activity(aid, planned)
        return True

    simulation_complete = False

    while True:
        if not HEADLESS_MODE:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)

        for _ in range(STEPS_PER_FRAME * SIM_SKIP):
            dt = SIMULATION_DT * SIM_SPEED_MULT
            sim_time_now = iteration * SIMULATION_DT * SIM_SPEED_MULT
            agents_list = list(sim.agents())
            if iteration % TIME_DEBUG_EVERY == 0:
                print(f"[TIME CHECK] sim_time_now={sim_time_now}", flush=True)
            if sim_time_now >= SIMULATION_HORIZON:
                stop_due_to_horizon = True
                break

            # Defer switching after distance targets are reached.
            for aid in list(ready_to_switch.keys()):
                ready_to_switch[aid] -= dt
                if ready_to_switch[aid] > 0:
                    continue

                ready_to_switch.pop(aid, None)
                js = pending_switch.pop(aid, None)
                if js is None:
                    continue

                journey_id, stage_id = js
                safe_switch(sim, aid, journey_id, stage_id)

                # If switching was skipped due to collision safety, retry soon.
                try:
                    agent = sim.agent(aid)
                    if agent.stage_id != stage_id:
                        ready_to_switch[aid] = 0.2
                        pending_switch[aid] = (journey_id, stage_id)
                except (KeyError, RuntimeError):
                    pass

            # Track which agents have entered the airport (crossed the entrance zone)
            for agent in agents_list:
                aid = agent.id
                if aid not in entered_airport:
                    entered_airport[aid] = False
                if not entered_airport[aid] and agent.position[1] >= ENTRANCE_ZONE_Y:
                    entered_airport[aid] = True

            # ---- 1. Spawn: in parking area, then journey to waiting or overflow ----
            queue_stage = sim.get_stage(waiting_queue_stage_id)
            if (
                iteration > 0
                and iteration % spawn_interval_iterations == 0
                and total_passengers_generated < MAX_PASSENGERS
                and sim.agent_count() < MAX_PASSENGERS
                and (sim_time_now - last_spawn_sim_time) >= MIN_SPAWN_GAP_SECONDS
            ):
                # Use a larger minimum separation at spawn to satisfy JuPedSim's
                # collision-free model constraints and avoid "too close" errors.
                min_separation = max(2.5 * AGENT_RADIUS, 0.4)
                # Spawn only from bottom border of parking area
                spawn_pos = safe_spawn_position(
                    sim,
                    hall_polygon,
                    park_x_left + park_margin,
                    park_x_right - park_margin,
                    park_y_bottom + 0.1,
                    park_y_bottom + 0.4,
                    min_separation,
                    attempts=100,
                )
                spawn_pos = force_safe_point(spawn_pos[0], spawn_pos[1], hall_polygon)
                waiting_count = queue_stage.count_enqueued()
                waiting_has_space = waiting_count < num_waiting_slots
                use_overflow = (
                    (not waiting_has_space)
                    and bool(overflow_stage_ids)
                )
                spawn_journey_id = journey_waiting_id
                spawn_stage_id = waiting_queue_stage_id
                if use_overflow:
                    idx = overflow_index % len(overflow_stage_ids)
                    overflow_index += 1
                    spawn_journey_id = overflow_journey_ids[idx]
                    spawn_stage_id = overflow_stage_ids[idx]
                agent_params = jps.CollisionFreeSpeedModelAgentParameters(
                    journey_id=spawn_journey_id,
                    stage_id=spawn_stage_id,
                    position=spawn_pos,
                    radius=AGENT_RADIUS,
                    desired_speed=DESIRED_SPEED,
                    time_gap=0.12,
                )
                agent_id = sim.add_agent(agent_params)
                entry_times[agent_id] = sim_time_now
                assigned_gate[agent_id] = gate_allocation_list[next_gate_allocation_idx % len(gate_allocation_list)]
                next_gate_allocation_idx += 1
                # Activity + planned path assigned only after security/display (not at spawn)
                passenger_color[agent_id] = (128, 128, 128)
                if use_overflow:
                    overflow_passengers.append(agent_id)
                total_spawned += 1
                total_passengers_generated += 1
                last_spawn_sim_time = sim_time_now

            # Keep overflow agents below waiting area to avoid queue interference.
            agents_list = list(sim.agents())
            for agent in agents_list:
                if agent.id in overflow_passengers:
                    x, y = agent.position
                    if y > LAST_WAITING_Y - 0.3:
                        y = LAST_WAITING_Y - 0.3
                        x, y = force_safe_point(x, y, hall_polygon)
                        x, y = clamp_to_hall(x, y)
                        agent.set_position((x, y))
                if agent.id in security_overflow_passengers:
                    x, y = agent.position
                    if y > SECURITY_LAST_WAITING_Y - 0.3:
                        y = SECURITY_LAST_WAITING_Y - 0.3
                        x, y = force_safe_point(x, y, hall_polygon)
                        x, y = clamp_to_hall(x, y)
                        agent.set_position((x, y))

            # ---- 2. DES: Counter service timers (when done, optionally send to info desk, otherwise to security waiting) ----
            for j in range(num_counters):
                if counter_serving_agent[j] is not None:
                    counter_service_remaining[j] -= dt
                    if counter_service_remaining[j] <= 0:
                        served_id = counter_serving_agent[j]
                        counter_serving_agent[j] = None
                        counter_service_remaining[j] = 0.0
                        completed_ticket_counter.add(served_id)
                        # Branch: some passengers visit the info desk, others go directly to security waiting
                        if random.random() < INFO_DESK_PROBABILITY:
                            # Assign to shortest info desk queue; when equal length, choose randomly
                            lengths = [len(q) for q in des_info_queue_lists]
                            min_len = min(lengths)
                            candidates = [i for i, L in enumerate(lengths) if L == min_len]
                            best_d = random.choice(candidates)
                            slot_in_queue = len(des_info_queue_lists[best_d])
                            des_info_queue_lists[best_d].append(served_id)
                            slot_index = min(slot_in_queue, INFO_DESK_QUEUE_CAPACITY - 1)
                            if served_id in completed_ticket_counter:
                                sim.switch_agent_journey(
                                    served_id,
                                    info_desk_queue_slot_journey_ids[best_d][slot_index],
                                    info_desk_queue_slot_stage_ids[best_d][slot_index],
                                )
                            for i, aid in enumerate(des_info_queue_lists[best_d]):
                                slot_index = min(i, INFO_DESK_QUEUE_CAPACITY - 1)
                                if aid in completed_ticket_counter:
                                    sim.switch_agent_journey(
                                        aid,
                                        info_desk_queue_slot_journey_ids[best_d][slot_index],
                                        info_desk_queue_slot_stage_ids[best_d][slot_index],
                                    )
                        else:
                            # Skip info desk: send directly to security waiting
                            sec_wait_has_space = (
                                sim.get_stage(security_waiting_stage_id).count_enqueued()
                                < num_security_waiting_slots
                            )
                            use_security_overflow = (
                                (not sec_wait_has_space)
                                and bool(security_overflow_stage_ids)
                            )
                            if served_id in completed_ticket_counter and not use_security_overflow:
                                sim.switch_agent_journey(
                                    served_id,
                                    security_waiting_journey_id,
                                    security_waiting_stage_id,
                                )
                            elif served_id in completed_ticket_counter and use_security_overflow:
                                idx = security_overflow_index % len(security_overflow_stage_ids)
                                security_overflow_index += 1
                                sim.switch_agent_journey(
                                    served_id,
                                    security_overflow_journey_ids[idx],
                                    security_overflow_stage_ids[idx],
                                )
                                security_overflow_passengers.append(served_id)

            # ---- 2b. DES: Info desk service timers (when done, send to security waiting) ----
            for d in range(num_info_desks):
                if info_desk_serving_agent[d] is not None:
                    info_desk_service_remaining[d] -= dt
                    if info_desk_service_remaining[d] <= 0:
                        served_id = info_desk_serving_agent[d]
                        info_desk_serving_agent[d] = None
                        info_desk_service_remaining[d] = 0.0
                        sec_wait_has_space = (
                            sim.get_stage(security_waiting_stage_id).count_enqueued()
                            < num_security_waiting_slots
                        )
                        use_security_overflow = (
                            (not sec_wait_has_space)
                            and bool(security_overflow_stage_ids)
                        )
                        if use_security_overflow:
                            idx = security_overflow_index % len(security_overflow_stage_ids)
                            security_overflow_index += 1
                            sim.switch_agent_journey(
                                served_id,
                                security_overflow_journey_ids[idx],
                                security_overflow_stage_ids[idx],
                            )
                            security_overflow_passengers.append(served_id)
                        else:
                            sim.switch_agent_journey(
                                served_id,
                                security_waiting_journey_id,
                                security_waiting_stage_id,
                            )

            # ---- 3. DES: When first in waiting queue leaves (wait at front, then send to shortest counter queue) ----
            queue_stage = sim.get_stage(waiting_queue_stage_id)
            waiting_count = queue_stage.count_enqueued()
            # Detect NEW vacancy (waiting count just decreased).
            if waiting_count < prev_waiting_count:
                if overflow_passengers:
                    # Select nearest overflow passenger.
                    def dist_to_waiting(agent_id):
                        agent = sim.agent(agent_id)
                        ax, ay = agent.position
                        wx, wy = waiting_front
                        return (ax - wx) ** 2 + (ay - wy) ** 2

                    next_id = min(overflow_passengers, key=dist_to_waiting)
                    overflow_passengers.remove(next_id)
                    sim.switch_agent_journey(
                        next_id,
                        journey_waiting_id,
                        waiting_queue_stage_id,
                    )
            prev_waiting_count = waiting_count
            enqueued = queue_stage.enqueued()
            if enqueued:
                first_agent_id = enqueued[0]
                agent = sim.agent(first_agent_id)
                ax, ay = agent.position
                fx, fy = waiting_front
                dist = math.hypot(ax - fx, ay - fy)
                if dist >= COUNTER_DIST_AT_FRONT:
                    waiting_front_wait_remaining = None
                else:
                    if waiting_front_wait_remaining is None:
                        waiting_front_wait_remaining = WAITING_FRONT_WAIT_TIME
                    waiting_front_wait_remaining -= dt
                    if waiting_front_wait_remaining <= 0:
                        waiting_front_wait_remaining = None
                        best_j = None
                        best_len = COUNTER_QUEUE_CAPACITY
                        # Prefer rightmost counter (TC1) when choosing among shortest queues
                        for j in range(num_counters - 1, -1, -1):
                            L = len(des_counter_queue_lists[j])
                            if L < COUNTER_QUEUE_CAPACITY and L < best_len:
                                best_len = L
                                best_j = j

                        # Only pop when there is a counter queue available (same as security logic)
                        if best_j is not None:
                            queue_stage.pop(1)

                            des_counter_queue_lists[best_j].append(first_agent_id)
                            ci_join_times.setdefault(first_agent_id, sim_time_now)

                            slot_index = min(
                                len(des_counter_queue_lists[best_j]) - 1,
                                COUNTER_QUEUE_CAPACITY - 1,
                            )
                            sim.switch_agent_journey(
                                first_agent_id,
                                counter_queue_slot_journey_ids[best_j][slot_index],
                                counter_queue_slot_stage_ids[best_j][slot_index],
                            )

            # ---- 3a. DES: When first in security waiting queue leaves (wait at front, then send to shortest security queue) ----
            sec_wait_stage = sim.get_stage(security_waiting_stage_id)
            sec_wait_count = sec_wait_stage.count_enqueued()
            if sec_wait_count < prev_security_waiting_count:
                if security_overflow_passengers:
                    def dist_to_sec_waiting(agent_id):
                        agent = sim.agent(agent_id)
                        ax, ay = agent.position
                        wx, wy = security_waiting_front
                        return (ax - wx) ** 2 + (ay - wy) ** 2

                    next_id = min(security_overflow_passengers, key=dist_to_sec_waiting)
                    security_overflow_passengers.remove(next_id)
                    sim.switch_agent_journey(
                        next_id,
                        security_waiting_journey_id,
                        security_waiting_stage_id,
                    )
            prev_security_waiting_count = sec_wait_count
            sec_enqueued = sec_wait_stage.enqueued()
            if sec_enqueued:
                first_agent_id = sec_enqueued[0]
                agent = sim.agent(first_agent_id)
                ax, ay = agent.position
                fx, fy = security_waiting_front
                dist = math.hypot(ax - fx, ay - fy)
                if dist >= SECURITY_DIST_AT_FRONT:
                    security_waiting_front_wait_remaining = None
                else:
                    if security_waiting_front_wait_remaining is None:
                        security_waiting_front_wait_remaining = SECURITY_WAITING_FRONT_WAIT_TIME
                    security_waiting_front_wait_remaining -= dt
                    if security_waiting_front_wait_remaining <= 0:
                        security_waiting_front_wait_remaining = None
                        best_s = None
                        best_len = SECURITY_QUEUE_CAPACITY
                        # Prefer rightmost security counter (West SC) when choosing among shortest queues
                        for s in range(num_security - 1, -1, -1):
                            L = len(des_security_queue_lists[s])
                            if L < SECURITY_QUEUE_CAPACITY and L < best_len:
                                best_len = L
                                best_s = s
                        if best_s is not None:
                            sec_wait_stage.pop(1)
                            slot_in_queue = len(des_security_queue_lists[best_s])
                            des_security_queue_lists[best_s].append(first_agent_id)
                            sc_join_times.setdefault(first_agent_id, sim_time_now)
                            slot_index = min(slot_in_queue, SECURITY_QUEUE_CAPACITY - 1)
                            sim.switch_agent_journey(
                                first_agent_id,
                                security_queue_slot_journey_ids[best_s][slot_index],
                                security_queue_slot_stage_ids[best_s][slot_index],
                            )
                            for i, aid in enumerate(des_security_queue_lists[best_s]):
                                slot_index = min(i, SECURITY_QUEUE_CAPACITY - 1)
                                sim.switch_agent_journey(
                                    aid,
                                    security_queue_slot_journey_ids[best_s][slot_index],
                                    security_queue_slot_stage_ids[best_s][slot_index],
                                )

            # ---- Maintain counter queue slot targets every step ----
            for j in range(num_counters):
                for i, aid in enumerate(des_counter_queue_lists[j]):
                    slot_index = min(i, COUNTER_QUEUE_CAPACITY - 1)
                    sim.switch_agent_journey(
                        aid,
                        counter_queue_slot_journey_ids[j][slot_index],
                        counter_queue_slot_stage_ids[j][slot_index],
                    )

            # ---- Maintain info desk queue slot targets every step ----
            for d in range(num_info_desks):
                for i, aid in enumerate(des_info_queue_lists[d]):
                    slot_index = min(i, INFO_DESK_QUEUE_CAPACITY - 1)
                    sim.switch_agent_journey(
                        aid,
                        info_desk_queue_slot_journey_ids[d][slot_index],
                        info_desk_queue_slot_stage_ids[d][slot_index],
                    )

            # ---- Maintain security queue slot targets every step ----
            for s in range(num_security):
                for i, aid in enumerate(des_security_queue_lists[s]):
                    slot_index = min(i, SECURITY_QUEUE_CAPACITY - 1)
                    sim.switch_agent_journey(
                        aid,
                        security_queue_slot_journey_ids[s][slot_index],
                        security_queue_slot_stage_ids[s][slot_index],
                    )

            # ---- 4. DES: Counter queue -> counter (front waits at front slot, then counter free) ----
            for j in range(num_counters):
                if counter_serving_agent[j] is not None:
                    counter_queue_front_wait_remaining[j] = None
                    continue
                qlist = des_counter_queue_lists[j]
                if not qlist:
                    counter_queue_front_wait_remaining[j] = None
                    continue
                front_id = qlist[0]
                agent = sim.agent(front_id)
                ax, ay = agent.position
                fx, fy = counter_front_slots[j]
                dist = math.hypot(ax - fx, ay - fy)
                if dist >= COUNTER_DIST_AT_FRONT:
                    counter_queue_front_wait_remaining[j] = None
                else:
                    if counter_queue_front_wait_remaining[j] is None:
                        counter_queue_front_wait_remaining[j] = COUNTER_QUEUE_FRONT_WAIT_TIME
                    counter_queue_front_wait_remaining[j] -= dt
                    if counter_queue_front_wait_remaining[j] <= 0:
                        counter_queue_front_wait_remaining[j] = None
                        des_counter_queue_lists[j].pop(0)
                        counter_serving_agent[j] = front_id
                        ci_service_start_times.setdefault(front_id, sim_time_now)
                        counter_service_remaining[j] = get_current_ci_service_time(
                            sim_time_now
                        )
                        sim.switch_agent_journey(
                            front_id,
                            counter_journey_ids[j],
                            counter_stage_ids[j],
                        )

            # ---- 4b. DES: Info desk queue -> info desk (front waits at front slot, then service 30s) ----
            for d in range(num_info_desks):
                if info_desk_serving_agent[d] is not None:
                    info_desk_queue_front_wait_remaining[d] = None
                    continue
                qlist = des_info_queue_lists[d]
                if not qlist:
                    info_desk_queue_front_wait_remaining[d] = None
                    continue
                front_id = qlist[0]
                agent = sim.agent(front_id)
                ax, ay = agent.position
                fx, fy = info_desk_front_slots[d]
                dist = math.hypot(ax - fx, ay - fy)
                if dist >= INFO_DESK_DIST_AT_FRONT:
                    info_desk_queue_front_wait_remaining[d] = None
                    continue
                if info_desk_queue_front_wait_remaining[d] is None:
                    info_desk_queue_front_wait_remaining[d] = INFO_DESK_FRONT_WAIT_TIME
                info_desk_queue_front_wait_remaining[d] -= dt
                if info_desk_queue_front_wait_remaining[d] <= 0:
                    info_desk_queue_front_wait_remaining[d] = None
                    qlist.pop(0)
                    for i, aid in enumerate(qlist):
                        slot_index = min(i, INFO_DESK_QUEUE_CAPACITY - 1)
                        sim.switch_agent_journey(
                            aid,
                            info_desk_queue_slot_journey_ids[d][slot_index],
                            info_desk_queue_slot_stage_ids[d][slot_index],
                        )
                    info_desk_serving_agent[d] = front_id
                    info_desk_service_remaining[d] = INFO_DESK_SERVICE_TIME
                    sim.switch_agent_journey(
                        front_id, info_desk_journey_ids[d], info_desk_stage_ids[d]
                    )

            # ---- 5. DES: Security service timers (when done, send to display or exit) ----
            for s in range(num_security):
                if security_serving_agent[s] is not None:
                    security_service_remaining[s] -= dt
                    if security_service_remaining[s] <= 0:
                        served_id = security_serving_agent[s]
                        security_serving_agent[s] = None
                        security_service_remaining[s] = 0.0
                        completed_security_passengers += 1
                        # Decide whether this passenger goes to a display area or exits directly
                        send_to_display = False
                        if display_stage_ids and random.random() < DISPLAY_PROBABILITY:
                            # Choose least crowded display; allow clustering (no capacity limit)
                            # Only consider displays that have valid entry waypoints and at least
                            # one spot waypoint stage (otherwise later indexing would be invalid).
                            available_d_indices = [
                                i
                                for i in range(len(display_wait_lists))
                                if i < len(display_stage_ids)
                                and i < len(display_entry_journey_ids)
                                and i < len(display_entry_stage_ids)
                                and i < len(display_entry_positions)
                                and display_stage_ids[i]
                            ]
                            if available_d_indices:
                                lengths = [len(display_wait_lists[i]) for i in available_d_indices]
                                min_len = min(lengths) if lengths else 0
                                candidates = [
                                    available_d_indices[k]
                                    for k, L in enumerate(lengths)
                                    if L == min_len
                                ]
                                if candidates:
                                    d_idx = random.choice(candidates)
                                    display_wait_lists[d_idx].append(served_id)
                                    # Randomly choose a spot within this display
                                    num_spots = len(display_stage_ids[d_idx])
                                    spot_idx = random.randrange(num_spots)
                                    # Remember which display and spot this passenger was assigned to; initial phase: to_entry
                                    display_assignment[served_id] = (d_idx, spot_idx)
                                    # Count each agent's first visit to the flight display area once.
                                    if served_id not in counted_display_agents:
                                        counted_display_agents.add(served_id)
                                        display_visit_count += 1
                                    display_phase[served_id] = "to_entry"
                                    # First send to entry point on the red edge
                                    sim.switch_agent_journey(
                                        served_id,
                                        display_entry_journey_ids[d_idx],
                                        display_entry_stage_ids[d_idx],
                                    )
                                    send_to_display = True
                        if not send_to_display:
                            decide_and_route_after_display_or_security(served_id)

            # ---- 6. DES: Display waiting timers (arrival-based; when done, send to exit) ----
            if display_stage_ids:
                # First, handle arrival phases: to_entry -> to_spot -> waiting -> return_entry
                for agent_id, (d_idx, spot_idx) in list(display_assignment.items()):
                    # FIX: never process display logic for agents that are already in gates (final phase)
                    if agent_id in gates_phase:
                        continue
                    phase = display_phase.get(agent_id, "to_entry")
                    try:
                        agent = sim.agent(agent_id)
                    except (KeyError, RuntimeError):
                        # Agent might have been removed; clean up
                        display_assignment.pop(agent_id, None)
                        display_phase.pop(agent_id, None)
                        for q in display_wait_lists:
                            if agent_id in q:
                                q.remove(agent_id)
                        continue
                    ax, ay = agent.position

                    # Entry and exit target on red edge
                    ex, ey = display_entry_positions[d_idx]

                    if phase == "to_entry":
                        # Heading to entry waypoint on red edge
                        dist = math.hypot(ax - ex, ay - ey)
                        if dist <= 0.25:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            # Now send to the assigned spot inside the display
                            display_phase[agent_id] = "to_spot"
                            sim.switch_agent_journey(
                                agent_id,
                                display_journey_ids[d_idx][spot_idx],
                                display_stage_ids[d_idx][spot_idx],
                            )
                    elif phase == "to_spot":
                        # Heading to assigned spot
                        tx, ty = display_spot_positions[d_idx][spot_idx]
                        dist = math.hypot(ax - tx, ay - ty)
                        # IMPORTANT: dwell no longer happens inside the display rectangles.
                        # Once the agent reaches the assigned display spot, immediately route back to
                        # the entry waypoint, and then apply post-display corridor dwell (degraded-only).
                        if dist <= 0.2:
                            display_phase[agent_id] = "return_entry"
                            sim.switch_agent_journey(
                                agent_id,
                                display_entry_journey_ids[d_idx],
                                display_entry_stage_ids[d_idx],
                            )
                    elif phase == "return_entry":
                        # Heading back to entry before exiting
                        dist = math.hypot(ax - ex, ay - ey)
                        if dist <= 0.25:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            # Finished return; now send to entrance for shopping area
                            display_assignment.pop(agent_id, None)
                            display_phase.pop(agent_id, None)
                            for q in display_wait_lists:
                                if agent_id in q:
                                    q.remove(agent_id)
                            # Post-display spatial dwell (degraded-only) happens here, before routing.
                            if not maybe_start_post_display_spatial_dwell(agent_id):
                                decide_and_route_after_display_or_security(agent_id)

                # Display-spot dwell has been removed; clear any leftover timers defensively.
                if display_wait_time_remaining:
                    display_wait_time_remaining.clear()

            # ---- 6b. Post-display dwell timers (degraded-only; resume original routing) ----
            for agent_id in list(dwell_time_remaining.keys()):
                # If the agent disappeared, clean up safely.
                try:
                    _ = sim.agent(agent_id)
                except (KeyError, RuntimeError):
                    dwell_time_remaining.pop(agent_id, None)
                    continue

                remaining = dwell_time_remaining[agent_id] - dt
                if remaining <= 0:
                    dwell_time_remaining.pop(agent_id, None)
                    # Resume the original intended routing logic.
                    decide_and_route_after_display_or_security(agent_id)
                else:
                    dwell_time_remaining[agent_id] = remaining

            # ---- 7. DES: Shopping area flow (Security/Display -> main Entrance block -> Shops Enter red edge ->
            #                chosen shop red line -> grey area wait -> same shop red line exit ->
            #                Shops Exit red edge -> final exit) ----
            if any(shop_entry_stage_ids) and shops_enter_stage_ids and shops_exit_stage_ids:
                for agent_id, phase in list(shops_phase.items()):
                    # FIX: never process shops logic for agents that are already in gates (final phase)
                    if agent_id in gates_phase:
                        continue
                    try:
                        agent = sim.agent(agent_id)
                    except (KeyError, RuntimeError):
                        # Agent might have been removed; clean up
                        shops_phase.pop(agent_id, None)
                        shops_choice.pop(agent_id, None)
                        shops_choice_enter_edge_idx.pop(agent_id, None)
                        shops_choice_shop_entry_idx.pop(agent_id, None)
                        shops_choice_shop_wait_idx.pop(agent_id, None)
                        shops_choice_shop_exit_idx.pop(agent_id, None)
                        shops_choice_exit_edge_idx.pop(agent_id, None)
                        shop_wait_time_remaining.pop(agent_id, None)
                        shops_misroute_roam_remaining.pop(agent_id, None)
                        continue

                    ax, ay = agent.position
                    shop_idx = shops_choice.get(agent_id, 0)
                    shop_idx = max(0, min(shop_idx, len(shop_entry_stage_ids) - 1))

                    # Main entrance (top green Entrance block center)
                    # Main entrance (top green Entrance block)
                    if "exit" in coordinates:
                        ent_cx, ent_cy = coordinates["exit"].get("center", (25.0, 29.0))
                    else:
                        ent_cx, ent_cy = 0.0, 0.0

                    if phase == "to_main_entrance":
                        # From security/display exit toward the main Entrance block
                        dist = math.hypot(ax - ent_cx, ay - ent_cy)
                        if dist <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            # Once at Entrance, choose a random Shops Enter waypoint along its red edge
                            entry_idx = random.randrange(len(shops_enter_stage_ids))
                            shops_choice_enter_edge_idx[agent_id] = entry_idx
                            shops_phase[agent_id] = "to_shops_enter_edge"
                            sim.switch_agent_journey(
                                agent_id,
                                shops_enter_journey_ids[entry_idx],
                                shops_enter_stage_ids[entry_idx],
                            )

                    elif phase == "to_shops_enter_edge":
                        # From main Entrance towards the chosen Shops Enter red-edge waypoint
                        enter_edge_idx = shops_choice_enter_edge_idx.get(agent_id, 0)
                        enter_edge_idx = max(0, min(enter_edge_idx, len(shops_enter_positions) - 1))
                        tx, ty = shops_enter_positions[enter_edge_idx]
                        dist = math.hypot(ax - tx, ay - ty)
                        if dist <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            # Now select specific shop entry/wait/exit waypoints
                            if shop_entry_stage_ids[shop_idx]:
                                entry_list_len = len(shop_entry_stage_ids[shop_idx])
                                if agent_id not in shops_choice_shop_entry_idx:
                                    shops_choice_shop_entry_idx[agent_id] = random.randrange(
                                        entry_list_len
                                    )
                            if shop_wait_stage_ids[shop_idx]:
                                wait_list_len = len(shop_wait_stage_ids[shop_idx])
                                if agent_id not in shops_choice_shop_wait_idx:
                                    shops_choice_shop_wait_idx[agent_id] = random.randrange(
                                        wait_list_len
                                    )
                            if shop_exit_stage_ids[shop_idx]:
                                exit_list_len = len(shop_exit_stage_ids[shop_idx])
                                if agent_id not in shops_choice_shop_exit_idx:
                                    shops_choice_shop_exit_idx[agent_id] = random.randrange(
                                        exit_list_len
                                    )

                            # Misroute wrong area: enter shops zone only, short roam, exit — no shop counters
                            if misroute_roam_only.get(agent_id):
                                if (
                                    shop_approach_below_stage_ids
                                    and shop_idx < len(shop_approach_below_stage_ids)
                                ):
                                    shops_phase[agent_id] = "to_shop_approach_below"
                                    safe_switch(
                                        sim,
                                        agent_id,
                                        shop_approach_below_journey_ids[shop_idx],
                                        shop_approach_below_stage_ids[shop_idx],
                                    )
                                else:
                                    shops_phase[agent_id] = "shops_misroute_roam"
                                    shops_misroute_roam_remaining[agent_id] = random.uniform(
                                        SHOP_WAIT_MIN * 0.2, SHOP_WAIT_MAX * 0.5
                                    )
                            elif (
                                shop_approach_below_stage_ids
                                and shop_idx < len(shop_approach_below_stage_ids)
                            ):
                                shops_phase[agent_id] = "to_shop_approach_below"
                                safe_switch(
                                    sim,
                                    agent_id,
                                    shop_approach_below_journey_ids[shop_idx],
                                    shop_approach_below_stage_ids[shop_idx],
                                )
                            else:
                                entry_i = shops_choice_shop_entry_idx.get(agent_id, 0)
                                entry_i = max(
                                    0, min(entry_i, len(shop_entry_stage_ids[shop_idx]) - 1)
                                )
                                shops_phase[agent_id] = "to_shop_entry_line"
                                safe_switch(
                                    sim,
                                    agent_id,
                                    shop_entry_journey_ids[shop_idx][entry_i],
                                    shop_entry_stage_ids[shop_idx][entry_i],
                                )

                    elif phase == "to_shop_approach_below":
                        if (
                            shop_approach_below_positions
                            and shop_idx < len(shop_approach_below_positions)
                        ):
                            tx, ty = shop_approach_below_positions[shop_idx]
                            dist = math.hypot(ax - tx, ay - ty)
                            if dist <= 0.20:
                                if agent_id not in ready_to_switch:
                                    ready_to_switch[agent_id] = 0.2
                                if misroute_roam_only.get(agent_id):
                                    shops_phase[agent_id] = "shops_misroute_roam"
                                    shops_misroute_roam_remaining[agent_id] = random.uniform(
                                        SHOP_WAIT_MIN * 0.2, SHOP_WAIT_MAX * 0.5
                                    )
                                else:
                                    entry_i = shops_choice_shop_entry_idx.get(agent_id, 0)
                                    entry_i = max(
                                        0, min(entry_i, len(shop_entry_stage_ids[shop_idx]) - 1)
                                    )
                                    shops_phase[agent_id] = "to_shop_entry_line"
                                    safe_switch(
                                        sim,
                                        agent_id,
                                        shop_entry_journey_ids[shop_idx][entry_i],
                                        shop_entry_stage_ids[shop_idx][entry_i],
                                    )

                    elif phase == "shops_misroute_roam":
                        if agent_id not in shops_misroute_roam_remaining:
                            shops_misroute_roam_remaining[agent_id] = random.uniform(
                                SHOP_WAIT_MIN * 0.2, SHOP_WAIT_MAX * 0.5
                            )
                        remaining = shops_misroute_roam_remaining[agent_id] - dt
                        if remaining > 0:
                            shops_misroute_roam_remaining[agent_id] = remaining
                            continue
                        shops_misroute_roam_remaining.pop(agent_id, None)
                        exit_edge_idx = random.randrange(len(shops_exit_stage_ids))
                        shops_choice_exit_edge_idx[agent_id] = exit_edge_idx
                        shops_phase[agent_id] = "to_shops_exit_edge"
                        sim.switch_agent_journey(
                            agent_id,
                            shops_exit_journey_ids[exit_edge_idx],
                            shops_exit_stage_ids[exit_edge_idx],
                        )

                    elif phase == "to_shop_entry_line":
                        # Heading to chosen shop's red entry line
                        entry_i = shops_choice_shop_entry_idx.get(agent_id, 0)
                        entry_i = max(0, min(entry_i, len(shop_entry_positions[shop_idx]) - 1))
                        sx, sy = shop_entry_positions[shop_idx][entry_i]
                        dist = math.hypot(ax - sx, ay - sy)
                        if dist <= 0.25:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            # Proceed to waiting spot inside shop
                            wait_i = shops_choice_shop_wait_idx.get(agent_id, 0)
                            wait_i = max(0, min(wait_i, len(shop_wait_stage_ids[shop_idx]) - 1))
                            shops_phase[agent_id] = "to_shop_wait_spot"
                            sim.switch_agent_journey(
                                agent_id,
                                shop_wait_journey_ids[shop_idx][wait_i],
                                shop_wait_stage_ids[shop_idx][wait_i],
                            )

                    elif phase == "to_shop_wait_spot":
                        # Heading to chosen waiting spot inside shop grey area
                        wait_i = shops_choice_shop_wait_idx.get(agent_id, 0)
                        wait_i = max(0, min(wait_i, len(shop_wait_positions[shop_idx]) - 1))
                        wx, wy = shop_wait_positions[shop_idx][wait_i]
                        dist = math.hypot(ax - wx, ay - wy)
                        if dist <= 0.12 and agent_id not in shop_wait_time_remaining:
                            shops_phase[agent_id] = "shop_waiting"
                            shop_wait_time_remaining[agent_id] = random.uniform(SHOP_WAIT_MIN, SHOP_WAIT_MAX)

                    elif phase == "shop_waiting":
                        # HOLD in service spot: no switching/movement until service time is done.
                        if agent_id not in shop_wait_time_remaining:
                            shop_wait_time_remaining[agent_id] = random.uniform(
                                SHOP_WAIT_MIN, SHOP_WAIT_MAX
                            )
                        remaining = shop_wait_time_remaining[agent_id] - dt
                        if remaining > 0:
                            shop_wait_time_remaining[agent_id] = remaining
                            continue
                        shop_wait_time_remaining.pop(agent_id, None)
                        if agent_id in shops_choice:
                            shop_idx = shops_choice[agent_id]
                            shop_idx = max(0, min(shop_idx, len(shop_exit_stage_ids) - 1))
                            exit_i = shops_choice_shop_exit_idx.get(agent_id, 0)
                            exit_i = max(0, min(exit_i, len(shop_exit_stage_ids[shop_idx]) - 1))
                            shops_phase[agent_id] = "to_shop_exit_line"
                            safe_switch(
                                sim,
                                agent_id,
                                shop_exit_journey_ids[shop_idx][exit_i],
                                shop_exit_stage_ids[shop_idx][exit_i],
                            )
                            continue

                    elif phase == "to_shop_exit_line":
                        # Leaving the grey area via chosen shop's red exit line
                        exit_i = shops_choice_shop_exit_idx.get(agent_id, 0)
                        exit_i = max(0, min(exit_i, len(shop_exit_positions[shop_idx]) - 1))
                        lx, ly = shop_exit_positions[shop_idx][exit_i]
                        dist = math.hypot(ax - lx, ay - ly)
                        if dist <= 0.25:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            # Now go to Shops Exit red edge
                            exit_edge_idx = random.randrange(len(shops_exit_stage_ids))
                            shops_choice_exit_edge_idx[agent_id] = exit_edge_idx
                            shops_phase[agent_id] = "to_shops_exit_edge"
                            sim.switch_agent_journey(
                                agent_id,
                                shops_exit_journey_ids[exit_edge_idx],
                                shops_exit_stage_ids[exit_edge_idx],
                            )

                    elif phase == "to_shops_exit_edge":
                        # From Shops Exit red edge: activity is complete -> proceed directly to gates.
                        exit_edge_idx = shops_choice_exit_edge_idx.get(agent_id, 0)
                        exit_edge_idx = max(0, min(exit_edge_idx, len(shops_exit_positions) - 1))
                        exx, exy = shops_exit_positions[exit_edge_idx]
                        dist = math.hypot(ax - exx, ay - exy)
                        if dist <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            shops_phase.pop(agent_id, None)
                            shops_choice.pop(agent_id, None)
                            shops_choice_enter_edge_idx.pop(agent_id, None)
                            shops_choice_shop_entry_idx.pop(agent_id, None)
                            shops_choice_shop_wait_idx.pop(agent_id, None)
                            shops_choice_shop_exit_idx.pop(agent_id, None)
                            shops_choice_exit_edge_idx.pop(agent_id, None)
                            shop_wait_time_remaining.pop(agent_id, None)
                            shops_misroute_roam_remaining.pop(agent_id, None)

                            if try_misroute_redirect_to_planned(agent_id):
                                continue

                            # Always proceed to GATES after shops.
                            if gates_enter_stage_ids and gates_gate_exit_journey_ids:
                                shops_phase.pop(agent_id, None)
                                shops_choice.pop(agent_id, None)
                                shops_choice_enter_edge_idx.pop(agent_id, None)
                                shops_choice_shop_entry_idx.pop(agent_id, None)
                                shops_choice_shop_wait_idx.pop(agent_id, None)
                                shops_choice_shop_exit_idx.pop(agent_id, None)
                                shops_choice_exit_edge_idx.pop(agent_id, None)
                                shop_wait_time_remaining.pop(agent_id, None)
                                shops_misroute_roam_remaining.pop(agent_id, None)

                                food_phase.pop(agent_id, None)
                                food_choice.pop(agent_id, None)
                                food_choice_enter_edge_idx.pop(agent_id, None)
                                food_choice_corridor_idx.pop(agent_id, None)
                                food_choice_corridor_after_exit_idx.pop(agent_id, None)
                                food_choice_entry_idx.pop(agent_id, None)
                                food_choice_wait_idx.pop(agent_id, None)
                                food_choice_exit_idx.pop(agent_id, None)
                                food_choice_exit_edge_idx.pop(agent_id, None)
                                food_wait_time_remaining.pop(agent_id, None)
                                food_misroute_roam_remaining.pop(agent_id, None)

                                rest_phase.pop(agent_id, None)
                                rest_choice.pop(agent_id, None)
                                rest_choice_enter_edge_idx.pop(agent_id, None)
                                rest_choice_corridor_idx.pop(agent_id, None)
                                rest_choice_corridor_after_exit_idx.pop(agent_id, None)
                                rest_choice_entry_idx.pop(agent_id, None)
                                rest_choice_wait_idx.pop(agent_id, None)
                                rest_choice_exit_idx.pop(agent_id, None)
                                rest_choice_exit_edge_idx.pop(agent_id, None)
                                rest_wait_time_remaining.pop(agent_id, None)
                                rest_misroute_roam_remaining.pop(agent_id, None)

                                display_assignment.pop(agent_id, None)
                                display_phase.pop(agent_id, None)
                                display_wait_time_remaining.pop(agent_id, None)
                                for q in display_wait_lists:
                                    if agent_id in q:
                                        q.remove(agent_id)

                                enter_edge_idx = random.randrange(len(gates_enter_stage_ids))
                                gates_choice_enter_edge_idx[agent_id] = enter_edge_idx
                                gate_name = assigned_gate.get(agent_id, "A1")
                                if gate_name not in gates_gate_exit_journey_ids and gates_gate_exit_journey_ids:
                                    gate_name = next(iter(gates_gate_exit_journey_ids))
                                gates_choice_gate_name[agent_id] = gate_name
                                gates_phase[agent_id] = "to_gates_enter"
                                actual_path[agent_id].append(gate_name)
                                sim.switch_agent_journey(agent_id, gates_enter_journey_ids[enter_edge_idx], gates_enter_stage_ids[enter_edge_idx])
                            else:
                                sim.switch_agent_journey(agent_id, final_exit_journey_id, final_exit_stage_id)

            # ---- 6b. DES: Food court flow (shops_exit_edge -> corridor -> food_enter_block -> food court -> food_exit_block -> restrooms) ----
            if food_enter_stage_ids and food_exit_stage_ids and valid_food_indices:
                for agent_id, phase in list(food_phase.items()):
                    # FIX: never process food logic for agents that are already in gates (final phase)
                    if agent_id in gates_phase:
                        continue
                    try:
                        agent = sim.agent(agent_id)
                    except (KeyError, RuntimeError):
                        food_phase.pop(agent_id, None)
                        food_choice.pop(agent_id, None)
                        food_wait_time_remaining.pop(agent_id, None)
                        food_misroute_roam_remaining.pop(agent_id, None)
                        food_choice_enter_edge_idx.pop(agent_id, None)
                        food_choice_corridor_idx.pop(agent_id, None)
                        food_choice_corridor_after_exit_idx.pop(agent_id, None)
                        food_choice_entry_idx.pop(agent_id, None)
                        food_choice_wait_idx.pop(agent_id, None)
                        food_choice_exit_idx.pop(agent_id, None)
                        food_choice_exit_edge_idx.pop(agent_id, None)
                        continue

                    ax, ay = agent.position
                    food_idx = food_choice.get(agent_id, 0)
                    food_idx = max(0, min(food_idx, num_food - 1))

                    if phase == "to_food_enter":
                        enter_edge_idx = food_choice_enter_edge_idx.get(agent_id, 0)
                        enter_edge_idx = max(0, min(enter_edge_idx, len(food_enter_positions) - 1))
                        tx, ty = food_enter_positions[enter_edge_idx]
                        dist = math.hypot(ax - tx, ay - ty)
                        if dist <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            food_choice_entry_idx[agent_id] = 0
                            food_choice_wait_idx[agent_id] = 0
                            food_choice_exit_idx[agent_id] = 0

                            # Misroute wrong area: enter food zone only (corridor/approach), roam, exit — no counters
                            if misroute_roam_only.get(agent_id):
                                if food_corridor_stage_ids:
                                    corridor_idx = 0
                                    food_choice_corridor_idx[agent_id] = corridor_idx
                                    food_phase[agent_id] = "to_food_corridor"
                                    sim.switch_agent_journey(
                                        agent_id,
                                        food_corridor_journey_ids[corridor_idx],
                                        food_corridor_stage_ids[corridor_idx],
                                    )
                                elif food_approach_below_stage_ids and food_idx < len(food_approach_below_stage_ids):
                                    food_phase[agent_id] = "to_food_approach_below"
                                    sim.switch_agent_journey(
                                        agent_id,
                                        food_approach_below_journey_ids[food_idx],
                                        food_approach_below_stage_ids[food_idx],
                                    )
                                else:
                                    exit_edge_idx = 0
                                    food_choice_exit_edge_idx[agent_id] = exit_edge_idx
                                    food_phase[agent_id] = "to_food_exit_block"
                                    sim.switch_agent_journey(
                                        agent_id,
                                        food_exit_journey_ids[exit_edge_idx],
                                        food_exit_stage_ids[exit_edge_idx],
                                    )
                            # Mandatory: go to food corridor waypoint inside the food strip first
                            elif food_corridor_stage_ids:
                                corridor_idx = 0
                                food_choice_corridor_idx[agent_id] = corridor_idx
                                food_phase[agent_id] = "to_food_corridor"
                                sim.switch_agent_journey(
                                    agent_id,
                                    food_corridor_journey_ids[corridor_idx],
                                    food_corridor_stage_ids[corridor_idx],
                                )
                            elif food_approach_below_stage_ids and food_idx < len(food_approach_below_stage_ids):
                                food_phase[agent_id] = "to_food_approach_below"
                                sim.switch_agent_journey(
                                    agent_id,
                                    food_approach_below_journey_ids[food_idx],
                                    food_approach_below_stage_ids[food_idx],
                                )
                            else:
                                entry_i = food_choice_entry_idx.get(agent_id, 0)
                                entry_i = max(0, min(entry_i, len(food_entry_stage_ids[food_idx]) - 1))
                                food_phase[agent_id] = "to_food_entry_line"
                                sim.switch_agent_journey(
                                    agent_id,
                                    food_entry_journey_ids[food_idx][entry_i],
                                    food_entry_stage_ids[food_idx][entry_i],
                                )

                    elif phase == "to_food_corridor":
                        corridor_idx = food_choice_corridor_idx.get(agent_id, 0)
                        corridor_idx = max(0, min(corridor_idx, len(food_corridor_positions) - 1))
                        tx, ty = food_corridor_positions[corridor_idx]
                        dist = math.hypot(ax - tx, ay - ty)
                        if dist <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            if misroute_roam_only.get(agent_id):
                                food_phase[agent_id] = "food_misroute_roam"
                                food_misroute_roam_remaining[agent_id] = random.uniform(
                                    FOOD_WAIT_MIN * 0.2, FOOD_WAIT_MAX * 0.5
                                )
                            elif food_approach_below_stage_ids and food_idx < len(food_approach_below_stage_ids):
                                food_phase[agent_id] = "to_food_approach_below"
                                sim.switch_agent_journey(
                                    agent_id,
                                    food_approach_below_journey_ids[food_idx],
                                    food_approach_below_stage_ids[food_idx],
                                )
                            else:
                                entry_i = food_choice_entry_idx.get(agent_id, 0)
                                entry_i = max(0, min(entry_i, len(food_entry_stage_ids[food_idx]) - 1))
                                food_phase[agent_id] = "to_food_entry_line"
                                sim.switch_agent_journey(
                                    agent_id,
                                    food_entry_journey_ids[food_idx][entry_i],
                                    food_entry_stage_ids[food_idx][entry_i],
                                )

                    elif phase == "to_food_approach_below":
                        tx, ty = food_approach_below_positions[food_idx]
                        dist = math.hypot(ax - tx, ay - ty)
                        if dist <= 0.25:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            if misroute_roam_only.get(agent_id):
                                food_phase[agent_id] = "food_misroute_roam"
                                food_misroute_roam_remaining[agent_id] = random.uniform(
                                    FOOD_WAIT_MIN * 0.2, FOOD_WAIT_MAX * 0.5
                                )
                            else:
                                entry_i = food_choice_entry_idx.get(agent_id, 0)
                                entry_i = max(0, min(entry_i, len(food_entry_stage_ids[food_idx]) - 1))
                                food_phase[agent_id] = "to_food_entry_line"
                                sim.switch_agent_journey(
                                    agent_id,
                                    food_entry_journey_ids[food_idx][entry_i],
                                    food_entry_stage_ids[food_idx][entry_i],
                                )

                    elif phase == "food_misroute_roam":
                        if agent_id not in food_misroute_roam_remaining:
                            food_misroute_roam_remaining[agent_id] = random.uniform(
                                FOOD_WAIT_MIN * 0.2, FOOD_WAIT_MAX * 0.5
                            )
                        remaining = food_misroute_roam_remaining[agent_id] - dt
                        if remaining > 0:
                            food_misroute_roam_remaining[agent_id] = remaining
                            continue
                        food_misroute_roam_remaining.pop(agent_id, None)
                        exit_edge_idx = 0
                        food_choice_exit_edge_idx[agent_id] = exit_edge_idx
                        food_phase[agent_id] = "to_food_exit_block"
                        sim.switch_agent_journey(
                            agent_id,
                            food_exit_journey_ids[exit_edge_idx],
                            food_exit_stage_ids[exit_edge_idx],
                        )

                    elif phase == "to_food_entry_line":
                        entry_i = food_choice_entry_idx.get(agent_id, 0)
                        entry_i = max(0, min(entry_i, len(food_entry_positions[food_idx]) - 1))
                        sx, sy = food_entry_positions[food_idx][entry_i]
                        dist = math.hypot(ax - sx, ay - sy)
                        if dist <= 0.25:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            wait_i = food_choice_wait_idx.get(agent_id, 0)
                            wait_i = max(0, min(wait_i, len(food_wait_stage_ids[food_idx]) - 1))
                            food_phase[agent_id] = "to_food_wait_spot"
                            sim.switch_agent_journey(
                                agent_id,
                                food_wait_journey_ids[food_idx][wait_i],
                                food_wait_stage_ids[food_idx][wait_i],
                            )

                    elif phase == "to_food_wait_spot":
                        wait_i = food_choice_wait_idx.get(agent_id, 0)
                        wait_i = max(0, min(wait_i, len(food_wait_positions[food_idx]) - 1))
                        wx, wy = food_wait_positions[food_idx][wait_i]
                        dist = math.hypot(ax - wx, ay - wy)
                        if dist <= 0.12 and agent_id not in food_wait_time_remaining:
                            food_phase[agent_id] = "food_waiting"
                            food_wait_time_remaining[agent_id] = random.uniform(FOOD_WAIT_MIN, FOOD_WAIT_MAX)

                    elif phase == "food_waiting":
                        # HOLD in service spot: no switching/movement until service time is done.
                        if agent_id not in food_wait_time_remaining:
                            food_wait_time_remaining[agent_id] = random.uniform(
                                FOOD_WAIT_MIN, FOOD_WAIT_MAX
                            )
                        remaining = food_wait_time_remaining[agent_id] - dt
                        if remaining > 0:
                            food_wait_time_remaining[agent_id] = remaining
                            continue
                        food_wait_time_remaining.pop(agent_id, None)
                        if agent_id in food_choice:
                            food_idx = food_choice[agent_id]
                            food_idx = max(0, min(food_idx, num_food - 1))
                            exit_i = food_choice_exit_idx.get(agent_id, 0)
                            exit_i = max(
                                0, min(exit_i, len(food_exit_line_stage_ids[food_idx]) - 1)
                            )
                            food_phase[agent_id] = "to_food_exit_line"
                            safe_switch(
                                sim,
                                agent_id,
                                food_exit_line_journey_ids[food_idx][exit_i],
                                food_exit_line_stage_ids[food_idx][exit_i],
                            )
                            continue

                    elif phase == "to_food_exit_line":
                        exit_i = food_choice_exit_idx.get(agent_id, 0)
                        exit_i = max(0, min(exit_i, len(food_exit_line_positions[food_idx]) - 1))
                        lx, ly = food_exit_line_positions[food_idx][exit_i]
                        dist = math.hypot(ax - lx, ay - ly)
                        if dist <= 0.25:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            if food_corridor_stage_ids:
                                corridor_idx = 0
                                food_choice_corridor_after_exit_idx[agent_id] = corridor_idx
                                food_phase[agent_id] = "to_food_corridor_after_exit"
                                sim.switch_agent_journey(
                                    agent_id,
                                    food_corridor_journey_ids[corridor_idx],
                                    food_corridor_stage_ids[corridor_idx],
                                )
                            else:
                                exit_edge_idx = 0
                                food_choice_exit_edge_idx[agent_id] = exit_edge_idx
                                food_phase[agent_id] = "to_food_exit_block"
                                sim.switch_agent_journey(
                                    agent_id,
                                    food_exit_journey_ids[exit_edge_idx],
                                    food_exit_stage_ids[exit_edge_idx],
                                )

                    elif phase == "to_food_corridor_after_exit":
                        corridor_idx = food_choice_corridor_after_exit_idx.get(agent_id, 0)
                        corridor_idx = max(0, min(corridor_idx, len(food_corridor_positions) - 1))
                        tx, ty = food_corridor_positions[corridor_idx]
                        dist = math.hypot(ax - tx, ay - ty)
                        if dist <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            exit_edge_idx = 0
                            food_choice_exit_edge_idx[agent_id] = exit_edge_idx
                            food_phase[agent_id] = "to_food_exit_block"
                            sim.switch_agent_journey(
                                agent_id,
                                food_exit_journey_ids[exit_edge_idx],
                                food_exit_stage_ids[exit_edge_idx],
                            )

                    elif phase == "to_food_exit_block":
                        exit_edge_idx = food_choice_exit_edge_idx.get(agent_id, 0)
                        exit_edge_idx = max(0, min(exit_edge_idx, len(food_exit_positions) - 1))
                        exx, exy = food_exit_positions[exit_edge_idx]
                        dist = math.hypot(ax - exx, ay - exy)
                        if dist <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2

                            if try_misroute_redirect_to_planned(agent_id):
                                continue

                            # Always proceed to GATES after food.
                            if gates_enter_stage_ids and gates_gate_exit_journey_ids:
                                shops_phase.pop(agent_id, None)
                                shops_choice.pop(agent_id, None)
                                shops_choice_enter_edge_idx.pop(agent_id, None)
                                shops_choice_shop_entry_idx.pop(agent_id, None)
                                shops_choice_shop_wait_idx.pop(agent_id, None)
                                shops_choice_shop_exit_idx.pop(agent_id, None)
                                shops_choice_exit_edge_idx.pop(agent_id, None)
                                shop_wait_time_remaining.pop(agent_id, None)
                                shops_misroute_roam_remaining.pop(agent_id, None)

                                food_phase.pop(agent_id, None)
                                food_choice.pop(agent_id, None)
                                food_choice_enter_edge_idx.pop(agent_id, None)
                                food_choice_corridor_idx.pop(agent_id, None)
                                food_choice_corridor_after_exit_idx.pop(agent_id, None)
                                food_choice_entry_idx.pop(agent_id, None)
                                food_choice_wait_idx.pop(agent_id, None)
                                food_choice_exit_idx.pop(agent_id, None)
                                food_choice_exit_edge_idx.pop(agent_id, None)
                                food_wait_time_remaining.pop(agent_id, None)
                                food_misroute_roam_remaining.pop(agent_id, None)

                                rest_phase.pop(agent_id, None)
                                rest_choice.pop(agent_id, None)
                                rest_choice_enter_edge_idx.pop(agent_id, None)
                                rest_choice_corridor_idx.pop(agent_id, None)
                                rest_choice_corridor_after_exit_idx.pop(agent_id, None)
                                rest_choice_entry_idx.pop(agent_id, None)
                                rest_choice_wait_idx.pop(agent_id, None)
                                rest_choice_exit_idx.pop(agent_id, None)
                                rest_choice_exit_edge_idx.pop(agent_id, None)
                                rest_wait_time_remaining.pop(agent_id, None)
                                rest_misroute_roam_remaining.pop(agent_id, None)

                                display_assignment.pop(agent_id, None)
                                display_phase.pop(agent_id, None)
                                display_wait_time_remaining.pop(agent_id, None)
                                for q in display_wait_lists:
                                    if agent_id in q:
                                        q.remove(agent_id)

                                enter_edge_idx = random.randrange(len(gates_enter_stage_ids))
                                gates_choice_enter_edge_idx[agent_id] = enter_edge_idx
                                gate_name = assigned_gate.get(agent_id, "A1")
                                if gate_name not in gates_gate_exit_journey_ids and gates_gate_exit_journey_ids:
                                    gate_name = next(iter(gates_gate_exit_journey_ids))
                                gates_choice_gate_name[agent_id] = gate_name
                                gates_phase[agent_id] = "to_gates_enter"
                                actual_path[agent_id].append(gate_name)
                                sim.switch_agent_journey(agent_id, gates_enter_journey_ids[enter_edge_idx], gates_enter_stage_ids[enter_edge_idx])
                            else:
                                sim.switch_agent_journey(agent_id, final_exit_journey_id, final_exit_stage_id)
                            food_phase.pop(agent_id, None)
                            food_choice.pop(agent_id, None)
                            food_choice_enter_edge_idx.pop(agent_id, None)
                            food_choice_corridor_idx.pop(agent_id, None)
                            food_choice_corridor_after_exit_idx.pop(agent_id, None)
                            food_choice_entry_idx.pop(agent_id, None)
                            food_choice_wait_idx.pop(agent_id, None)
                            food_choice_exit_idx.pop(agent_id, None)
                            food_choice_exit_edge_idx.pop(agent_id, None)
                            food_wait_time_remaining.pop(agent_id, None)
                            food_misroute_roam_remaining.pop(agent_id, None)
                            # Do not count here; counting is done only at final exit stage.

            # ---- 6c. DES: Restrooms flow (food_exit_block -> rest_enter_block -> restroom -> rest_exit_block -> exit) ----
            if rest_enter_stage_ids and rest_exit_stage_ids and rest_corridor_stage_ids and valid_rest_indices:
                for agent_id, phase in list(rest_phase.items()):
                    # FIX: never process rest logic for agents that are already in gates (final phase)
                    if agent_id in gates_phase:
                        continue
                    try:
                        agent = sim.agent(agent_id)
                    except (KeyError, RuntimeError):
                        rest_phase.pop(agent_id, None)
                        rest_choice.pop(agent_id, None)
                        rest_wait_time_remaining.pop(agent_id, None)
                        rest_choice_enter_edge_idx.pop(agent_id, None)
                        rest_choice_corridor_idx.pop(agent_id, None)
                        rest_choice_corridor_after_exit_idx.pop(agent_id, None)
                        rest_choice_entry_idx.pop(agent_id, None)
                        rest_choice_wait_idx.pop(agent_id, None)
                        rest_choice_exit_idx.pop(agent_id, None)
                        rest_choice_exit_edge_idx.pop(agent_id, None)
                        rest_misroute_roam_remaining.pop(agent_id, None)
                        continue

                    ax, ay = agent.position
                    if phase in (
                        "to_rest_enter",
                    ):
                        enter_edge_idx = rest_choice_enter_edge_idx.get(agent_id, 0)
                        enter_edge_idx = max(0, min(enter_edge_idx, len(rest_enter_positions) - 1))
                        tx, ty = rest_enter_positions[enter_edge_idx]
                        if math.hypot(ax - tx, ay - ty) <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            corridor_idx = 0
                            rest_choice_corridor_idx[agent_id] = 0
                            rest_phase[agent_id] = "to_rest_corridor"
                            sim.switch_agent_journey(
                                agent_id,
                                rest_corridor_journey_ids[corridor_idx],
                                rest_corridor_stage_ids[corridor_idx],
                            )

                    elif phase == "to_rest_corridor":
                        corridor_idx = rest_choice_corridor_idx.get(agent_id, 0)
                        corridor_idx = max(0, min(corridor_idx, len(rest_corridor_positions) - 1))
                        tx, ty = rest_corridor_positions[corridor_idx]
                        if math.hypot(ax - tx, ay - ty) <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            rest_idx = rest_choice.get(agent_id, valid_rest_indices[0])
                            rest_idx = max(0, min(rest_idx, num_rest - 1))
                            # Misroute wrong area: restroom block public corridor only, roam, exit — no stalls
                            if misroute_roam_only.get(agent_id):
                                rest_phase[agent_id] = "rest_misroute_roam"
                                rest_misroute_roam_remaining[agent_id] = random.uniform(
                                    REST_WAIT_MIN * 0.2, REST_WAIT_MAX * 0.5
                                )
                            elif (
                                rest_approach_below_stage_ids
                                and rest_idx < len(rest_approach_below_stage_ids)
                            ):
                                rest_phase[agent_id] = "to_rest_approach_below"
                                safe_switch(
                                    sim,
                                    agent_id,
                                    rest_approach_below_journey_ids[rest_idx],
                                    rest_approach_below_stage_ids[rest_idx],
                                )
                            else:
                                entry_i = 0
                                rest_choice_entry_idx[agent_id] = 0
                                rest_phase[agent_id] = "to_rest_entry_line"
                                safe_switch(
                                    sim,
                                    agent_id,
                                    rest_entry_journey_ids[rest_idx][entry_i],
                                    rest_entry_stage_ids[rest_idx][entry_i],
                                )

                    elif phase == "rest_misroute_roam":
                        if agent_id not in rest_misroute_roam_remaining:
                            rest_misroute_roam_remaining[agent_id] = random.uniform(
                                REST_WAIT_MIN * 0.2, REST_WAIT_MAX * 0.5
                            )
                        remaining = rest_misroute_roam_remaining[agent_id] - dt
                        if remaining > 0:
                            rest_misroute_roam_remaining[agent_id] = remaining
                            continue
                        rest_misroute_roam_remaining.pop(agent_id, None)
                        exit_edge_idx = 0
                        rest_choice_exit_edge_idx[agent_id] = exit_edge_idx
                        rest_phase[agent_id] = "to_rest_exit_block"
                        sim.switch_agent_journey(
                            agent_id,
                            rest_exit_journey_ids[exit_edge_idx],
                            rest_exit_stage_ids[exit_edge_idx],
                        )

                    elif phase == "to_rest_approach_below":
                        rest_idx = rest_choice.get(agent_id, valid_rest_indices[0])
                        rest_idx = max(0, min(rest_idx, num_rest - 1))
                        if (
                            rest_approach_below_positions
                            and rest_idx < len(rest_approach_below_positions)
                        ):
                            tx, ty = rest_approach_below_positions[rest_idx]
                            if math.hypot(ax - tx, ay - ty) <= 0.20:
                                if agent_id not in ready_to_switch:
                                    ready_to_switch[agent_id] = 0.2
                                entry_i = rest_choice_entry_idx.get(agent_id, 0)
                                entry_i = 0
                                rest_choice_entry_idx[agent_id] = 0
                                rest_phase[agent_id] = "to_rest_entry_line"
                                safe_switch(
                                    sim,
                                    agent_id,
                                    rest_entry_journey_ids[rest_idx][entry_i],
                                    rest_entry_stage_ids[rest_idx][entry_i],
                                )

                    elif phase == "to_rest_entry_line":
                        rest_idx = rest_choice.get(agent_id, valid_rest_indices[0])
                        rest_idx = max(0, min(rest_idx, num_rest - 1))
                        entry_i = rest_choice_entry_idx.get(agent_id, 0)
                        entry_i = max(0, min(entry_i, len(rest_entry_positions[rest_idx]) - 1))
                        tx, ty = rest_entry_positions[rest_idx][entry_i]
                        if math.hypot(ax - tx, ay - ty) <= 0.25:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            wait_i = 0
                            rest_choice_wait_idx[agent_id] = 0
                            rest_phase[agent_id] = "to_rest_wait_spot"
                            sim.switch_agent_journey(
                                agent_id,
                                rest_wait_journey_ids[rest_idx][wait_i],
                                rest_wait_stage_ids[rest_idx][wait_i],
                            )

                    elif phase == "to_rest_wait_spot":
                        rest_idx = rest_choice.get(agent_id, valid_rest_indices[0])
                        rest_idx = max(0, min(rest_idx, num_rest - 1))
                        wait_i = rest_choice_wait_idx.get(agent_id, 0)
                        wait_i = max(0, min(wait_i, len(rest_wait_positions[rest_idx]) - 1))
                        tx, ty = rest_wait_positions[rest_idx][wait_i]
                        if math.hypot(ax - tx, ay - ty) <= 0.12 and agent_id not in rest_wait_time_remaining:
                            rest_phase[agent_id] = "rest_waiting"
                            rest_wait_time_remaining[agent_id] = random.uniform(REST_WAIT_MIN, REST_WAIT_MAX)

                    elif phase == "rest_waiting":
                        # HOLD in service spot: no switching/movement until service time is done.
                        if agent_id not in rest_wait_time_remaining:
                            rest_wait_time_remaining[agent_id] = random.uniform(
                                REST_WAIT_MIN, REST_WAIT_MAX
                            )
                        remaining = rest_wait_time_remaining[agent_id] - dt
                        if remaining > 0:
                            rest_wait_time_remaining[agent_id] = remaining
                            continue
                        rest_wait_time_remaining.pop(agent_id, None)
                        if agent_id in rest_choice:
                            rest_idx = rest_choice[agent_id]
                            rest_idx = max(0, min(rest_idx, num_rest - 1))
                            exit_i = 0
                            rest_choice_exit_idx[agent_id] = 0
                            rest_phase[agent_id] = "to_rest_exit_line"
                            safe_switch(
                                sim,
                                agent_id,
                                rest_exit_line_journey_ids[rest_idx][exit_i],
                                rest_exit_line_stage_ids[rest_idx][exit_i],
                            )
                            continue

                    elif phase == "to_rest_exit_line":
                        rest_idx = rest_choice.get(agent_id, valid_rest_indices[0])
                        rest_idx = max(0, min(rest_idx, num_rest - 1))
                        exit_i = rest_choice_exit_idx.get(agent_id, 0)
                        exit_i = max(0, min(exit_i, len(rest_exit_line_positions[rest_idx]) - 1))
                        tx, ty = rest_exit_line_positions[rest_idx][exit_i]
                        if math.hypot(ax - tx, ay - ty) <= 0.25:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            corridor_idx = 0
                            rest_choice_corridor_after_exit_idx[agent_id] = 0
                            rest_phase[agent_id] = "to_rest_corridor_after_exit"
                            sim.switch_agent_journey(
                                agent_id,
                                rest_corridor_journey_ids[corridor_idx],
                                rest_corridor_stage_ids[corridor_idx],
                            )

                    elif phase == "to_rest_corridor_after_exit":
                        corridor_idx = rest_choice_corridor_after_exit_idx.get(agent_id, 0)
                        corridor_idx = max(0, min(corridor_idx, len(rest_corridor_positions) - 1))
                        tx, ty = rest_corridor_positions[corridor_idx]
                        if math.hypot(ax - tx, ay - ty) <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            exit_edge_idx = 0
                            rest_choice_exit_edge_idx[agent_id] = exit_edge_idx
                            rest_phase[agent_id] = "to_rest_exit_block"
                            sim.switch_agent_journey(
                                agent_id,
                                rest_exit_journey_ids[exit_edge_idx],
                                rest_exit_stage_ids[exit_edge_idx],
                            )

                    elif phase == "to_rest_exit_block":
                        exit_edge_idx = rest_choice_exit_edge_idx.get(agent_id, 0)
                        exit_edge_idx = max(0, min(exit_edge_idx, len(rest_exit_positions) - 1))
                        tx, ty = rest_exit_positions[exit_edge_idx]
                        if math.hypot(ax - tx, ay - ty) <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            if try_misroute_redirect_to_planned(agent_id):
                                continue
                            # After restrooms, enter the All Gates area (if available).
                            if gates_enter_stage_ids and gates_gate_exit_journey_ids:
                                # FIX: entering gates is final. Purge from ALL other phases/tracking so
                                # they can never be reassigned backward (shops/food/rest/display).
                                #
                                # Shops
                                shops_phase.pop(agent_id, None)
                                shops_choice.pop(agent_id, None)
                                shops_choice_enter_edge_idx.pop(agent_id, None)
                                shops_choice_shop_entry_idx.pop(agent_id, None)
                                shops_choice_shop_wait_idx.pop(agent_id, None)
                                shops_choice_shop_exit_idx.pop(agent_id, None)
                                shops_choice_exit_edge_idx.pop(agent_id, None)
                                shop_wait_time_remaining.pop(agent_id, None)
                                shops_misroute_roam_remaining.pop(agent_id, None)
                                # Food
                                food_phase.pop(agent_id, None)
                                food_choice.pop(agent_id, None)
                                food_choice_enter_edge_idx.pop(agent_id, None)
                                food_choice_corridor_idx.pop(agent_id, None)
                                food_choice_corridor_after_exit_idx.pop(agent_id, None)
                                food_choice_entry_idx.pop(agent_id, None)
                                food_choice_wait_idx.pop(agent_id, None)
                                food_choice_exit_idx.pop(agent_id, None)
                                food_choice_exit_edge_idx.pop(agent_id, None)
                                food_wait_time_remaining.pop(agent_id, None)
                                food_misroute_roam_remaining.pop(agent_id, None)
                                # Rest (we're leaving it right now, but clear everything anyway)
                                rest_phase.pop(agent_id, None)
                                rest_choice.pop(agent_id, None)
                                rest_choice_enter_edge_idx.pop(agent_id, None)
                                rest_choice_corridor_idx.pop(agent_id, None)
                                rest_choice_corridor_after_exit_idx.pop(agent_id, None)
                                rest_choice_entry_idx.pop(agent_id, None)
                                rest_choice_wait_idx.pop(agent_id, None)
                                rest_choice_exit_idx.pop(agent_id, None)
                                rest_choice_exit_edge_idx.pop(agent_id, None)
                                rest_wait_time_remaining.pop(agent_id, None)
                                rest_misroute_roam_remaining.pop(agent_id, None)
                                # Displays
                                display_assignment.pop(agent_id, None)
                                display_phase.pop(agent_id, None)
                                display_wait_time_remaining.pop(agent_id, None)
                                for q in display_wait_lists:
                                    if agent_id in q:
                                        q.remove(agent_id)

                                enter_edge_idx = random.randrange(len(gates_enter_stage_ids))
                                gates_choice_enter_edge_idx[agent_id] = enter_edge_idx
                                gate_name = assigned_gate.get(agent_id, "A1")
                                if gate_name not in gates_gate_exit_journey_ids and gates_gate_exit_journey_ids:
                                    gate_name = next(iter(gates_gate_exit_journey_ids))
                                gates_choice_gate_name[agent_id] = gate_name
                                gates_phase[agent_id] = "to_gates_enter"
                                actual_path[agent_id].append(gate_name)
                                sim.switch_agent_journey(
                                    agent_id,
                                    gates_enter_journey_ids[enter_edge_idx],
                                    gates_enter_stage_ids[enter_edge_idx],
                                )
                            else:
                                sim.switch_agent_journey(
                                    agent_id,
                                    final_exit_journey_id,
                                    final_exit_stage_id,
                                )

                            # FIX: rest cleanup is already handled above (and must also run for the
                            # final-exit fallback), so keep the block empty here.

            # ---- 6d. DES: Gates flow (rest_exit_block -> gates_enter_block -> random gate -> exit) ----
            if gates_enter_stage_ids and gates_gate_exit_journey_ids:
                for agent_id, phase in list(gates_phase.items()):
                    try:
                        agent = sim.agent(agent_id)
                    except (KeyError, RuntimeError):
                        # Agent removed by a gate exit stage; do not count here
                        # because only the true final exit stage is counted.
                        gates_phase.pop(agent_id, None)
                        gates_choice_gate_name.pop(agent_id, None)
                        gates_choice_enter_edge_idx.pop(agent_id, None)
                        gates_choice_corridor_idx.pop(agent_id, None)
                        continue

                    ax, ay = agent.position
                    # FIX: once assigned to the gate exit journey (exit_stage), never switch again.
                    if phase == "to_gate_exit":
                        continue
                    if phase == "to_gates_enter":
                        enter_edge_idx = gates_choice_enter_edge_idx.get(agent_id, 0)
                        enter_edge_idx = max(0, min(enter_edge_idx, len(gates_enter_positions) - 1))
                        tx, ty = gates_enter_positions[enter_edge_idx]
                        if math.hypot(ax - tx, ay - ty) <= 0.4:
                            if agent_id not in ready_to_switch:
                                ready_to_switch[agent_id] = 0.2
                            gate_name = gates_choice_gate_name.get(agent_id)
                            if gate_name in gates_gate_exit_journey_ids:
                                gates_phase[agent_id] = "to_gate_exit"
                                sim.switch_agent_journey(
                                    agent_id,
                                    gates_gate_exit_journey_ids[gate_name],
                                    gates_gate_exit_stage_ids[gate_name],
                                )
                            else:
                                # Fallback: just exit
                                sim.switch_agent_journey(
                                    agent_id,
                                    final_exit_journey_id,
                                    final_exit_stage_id,
                                )
                                gates_phase.pop(agent_id, None)
                                gates_choice_gate_name.pop(agent_id, None)
                                gates_choice_enter_edge_idx.pop(agent_id, None)
                                gates_choice_corridor_idx.pop(agent_id, None)

            # ---- 7. DES: Security queue -> scanner (front waits 2s at front slot, then to scanner) ----
            for s in range(num_security):
                if security_serving_agent[s] is not None:
                    security_queue_front_wait_remaining[s] = None
                    continue
                qlist = des_security_queue_lists[s]
                if not qlist:
                    security_queue_front_wait_remaining[s] = None
                    continue
                front_id = qlist[0]
                agent = sim.agent(front_id)
                ax, ay = agent.position
                fx, fy = security_front_slots[s]
                dist = math.hypot(ax - fx, ay - fy)
                if dist >= SECURITY_DIST_AT_FRONT:
                    security_queue_front_wait_remaining[s] = None
                    continue
                if security_queue_front_wait_remaining[s] is None:
                    security_queue_front_wait_remaining[s] = SECURITY_QUEUE_FRONT_WAIT_TIME
                security_queue_front_wait_remaining[s] -= dt
                if security_queue_front_wait_remaining[s] <= 0:
                    security_queue_front_wait_remaining[s] = None
                    qlist.pop(0)
                    for i, aid in enumerate(qlist):
                        slot_index = min(i, SECURITY_QUEUE_CAPACITY - 1)
                        sim.switch_agent_journey(
                            aid,
                            security_queue_slot_journey_ids[s][slot_index],
                            security_queue_slot_stage_ids[s][slot_index],
                        )
                    security_serving_agent[s] = front_id
                    sc_service_start_times.setdefault(front_id, sim_time_now)
                    security_service_remaining[s] = SECURITY_SERVICE_TIME
                    sim.switch_agent_journey(
                        front_id, security_journey_ids[s], security_stage_ids[s]
                    )

            # ---- 7. Track agents at counter ----
            des_agents_at_counter.clear()
            agents_list = list(sim.agents())
            for agent in agents_list:
                if agent.stage_id in counter_stage_ids:
                    des_agents_at_counter.add(agent.id)

            # Queue-only metrics:
            # ci_queue_lengths = check-in queue spots only
            # sc_queue_lengths = security queue spots only
            # valid range is 0 to 4 for each total system queue metric
            ci_queue_only = sum(len(q) for q in des_counter_queue_lists)
            sc_queue_only = sum(len(q) for q in des_security_queue_lists)
            overflow_count = len(overflow_passengers) + len(security_overflow_passengers)
            time_steps.append(sim_time_now)
            exited_counts.append(total_passengers_exited)
            ci_queue_lengths.append(ci_queue_only)
            sc_queue_lengths.append(sc_queue_only)
            overflow_counts.append(overflow_count)
            passengers_in_system_series.append(sim.agent_count())
            counter_busy_now = sum(1 for a in counter_serving_agent if a is not None)
            security_busy_now = sum(1 for a in security_serving_agent if a is not None)
            checkin_utilization_series.append(counter_busy_now / num_counters if num_counters > 0 else 0.0)
            security_utilization_series.append(security_busy_now / num_security if num_security > 0 else 0.0)
            # Aggregate occupancy per defined zone for this timestep.
            zone_step_occupancy: dict[tuple[int, int], float] = {}
            hall_w = HALL_MAX_X - HALL_MIN_X
            hall_h = HALL_MAX_Y - HALL_MIN_Y
            for _agent in agents_list:
                x, y = _agent.position
                if x < HALL_MIN_X or x > HALL_MAX_X or y < HALL_MIN_Y or y > HALL_MAX_Y:
                    continue
                if HEATMAP_QUEUE_FOCUSED:
                    near_operational_zone = False
                    for (qx, qy) in queue_focus_points:
                        if math.hypot(x - qx, y - qy) <= HEATMAP_QUEUE_FOCUS_RADIUS:
                            near_operational_zone = True
                            break
                    if not near_operational_zone:
                        continue
                zc = min(
                    ZONE_COLS - 1,
                    max(0, int(((x - HALL_MIN_X) / hall_w) * ZONE_COLS)),
                )
                zr = min(
                    ZONE_ROWS - 1,
                    max(0, int(((y - HALL_MIN_Y) / hall_h) * ZONE_ROWS)),
                )
                zone_id = (zr, zc)
                zone_step_occupancy[zone_id] = zone_step_occupancy.get(zone_id, 0.0) + 1.0
            for zone_id, occ in zone_step_occupancy.items():
                zone_sum_occupancy[zone_id] = zone_sum_occupancy.get(zone_id, 0.0) + occ
            zone_sample_steps += 1
            time_series.append(sim_time_now)
            Y_series.append(completed_security_passengers)

            sim.iterate()
            iteration += 1
            if total_passengers_generated >= MAX_PASSENGERS and sim.agent_count() == 0:
                simulation_complete = True
                break

        if simulation_complete:
            break

        if HEADLESS_MODE:
            if stop_due_to_horizon:
                break
            continue

        # ---- Draw (VISUAL mode only) ----
        # Uniform light yellow background for entire scene
        background_color = (245, 240, 230)
        screen.fill(background_color)
        if hall_points:
            pts = [world_to_screen(x, y) for x, y in hall_points]
            # Hall interior (same color as background so everything looks uniform)
            pygame.draw.polygon(screen, background_color, pts)

        # (Removed outer yellow page border to keep only the hall border)

        # Background arrows from doors into the hall (drawn before waiting spots so they appear behind)
        for name, (dx, dy) in coordinates["entrances"].items():
            px, py = world_to_screen(dx, dy)
            door_w, door_h = 54, 20
            arrow_color = (190, 230, 255)
            # Shaft is a vertical line centered at px
            arrow_center_x = px
            start_y = py - door_h // 2
            end_y = start_y - 40   # even shorter arrow
            start = (arrow_center_x, start_y)
            end = (arrow_center_x, end_y)
            shaft_width = 26
            pygame.draw.line(screen, arrow_color, start, end, shaft_width)

            # Perfectly centered triangular arrow head
            head_width = 32
            head_height = 8    # smaller head
            arrow_tip_y = end_y - head_height
            triangle_points = [
                (arrow_center_x, arrow_tip_y),  # tip
                (arrow_center_x - head_width / 2, end_y),  # left base
                (arrow_center_x + head_width / 2, end_y),  # right base
            ]
            pygame.draw.polygon(screen, arrow_color, triangle_points)

        waypoint_fill_color = (70, 140, 230)
        waypoint_border_color = (30, 90, 180)

        for (wx, wy) in waiting_positions:
            px, py = world_to_screen(wx, wy)
            # Same visual size as passengers (r_px = 1)
            pygame.draw.circle(screen, waypoint_fill_color, (px, py), 1)

        # Ticket waiting-area left wall (visual debug)
        # Mirrors the geometry-only side wall created in `new_layout.py`
        # for the ticket waiting area.
        if waiting_positions:
            xs = [x for x, _ in waiting_positions]
            ys = [y for _, y in waiting_positions]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            wall_gap = 0.15
            wall_thickness = 0.05
            wall_y_pad = 0.15
            left_wall_x = x_min - wall_gap - wall_thickness
            p0 = world_to_screen(left_wall_x, y_min - wall_y_pad)
            p1 = world_to_screen(left_wall_x, y_max + wall_y_pad)
            pygame.draw.line(screen, (0, 0, 0), p0, p1, 1)
        # Lane labels for check-in waiting (row-wise, on the left)
        checkin_row_ys = sorted(set(wy for _, wy in waiting_positions), reverse=True)
        if waiting_positions:
            first_col_x = min(wx for wx, _ in waiting_positions)
            last_col_x = max(wx for wx, _ in waiting_positions)
        else:
            first_col_x = 0.0
            last_col_x = 0.0
        # Place lane labels to the right of the last column
        label_x_world = last_col_x + 0.6
        if checkin_row_ys and waiting_positions:
            # Horizontal heading centered above the waiting area and below the counters
            min_x = min(wx for wx, _ in waiting_positions)
            max_x = max(wx for wx, _ in waiting_positions)
            heading_x_world = (min_x + max_x) / 2.0
            heading_y_world = checkin_row_ys[0] + 0.5
            px_h, py_h = world_to_screen(heading_x_world, heading_y_world)
            wait_heading = lane_font.render("Waiting area", True, (0, 0, 0))
            screen.blit(wait_heading, (px_h - wait_heading.get_width() // 2, py_h - wait_heading.get_height() // 2))
        if overflow_hold_points:
            # Center a label in the overflow holding region below ticket waiting.
            ov_min_x = min(x for x, _ in overflow_hold_points)
            ov_max_x = max(x for x, _ in overflow_hold_points)
            ov_min_y = min(y for _, y in overflow_hold_points)
            ov_max_y = max(y for _, y in overflow_hold_points)
            ov_center_x = (ov_min_x + ov_max_x) / 2.0
            ov_center_y = (ov_min_y + ov_max_y) / 2.0
            px_ov, py_ov = world_to_screen(ov_center_x, ov_center_y)
            overflow_heading = lane_font.render("Overflow waiting area", True, (0, 0, 0))
            screen.blit(
                overflow_heading,
                (
                    px_ov - overflow_heading.get_width() // 2,
                    py_ov - overflow_heading.get_height() // 2,
                ),
            )
        for lane_num, row_y in enumerate(checkin_row_ys, start=1):
            px, py = world_to_screen(label_x_world, row_y)
            lane_label = lane_label_font.render(f"L{lane_num}", True, (0, 0, 0))
            screen.blit(lane_label, (px, py - lane_label.get_height() // 2))
        for j in range(num_counters):
            for (qx, qy) in counter_queue_slot_positions[j]:
                px, py = world_to_screen(qx, qy)
                pygame.draw.circle(screen, waypoint_fill_color, (px, py), 1)
                pygame.draw.circle(screen, waypoint_border_color, (px, py), 1, 1)
        for j, (cx, cy) in enumerate(counter_positions):
            px, py = world_to_screen(cx, cy)

            # Ticket counter stack:
            # [yellow number bar]
            # [grey label bar]
            # [grey service box]  <-- where passenger stands (queues drawn below)
            # Smaller ticket counter size
            header_w = 80
            number_h = 16
            label_h = 14
            service_w, service_h = header_w, 14

            # Place the service box (blue box) exactly at the counter position (cx, cy)
            service_rect_x = px - service_w // 2
            service_rect_y = py - service_h // 2

            label_rect_y = service_rect_y - label_h
            number_rect_y = label_rect_y - number_h

            number_rect = (px - header_w // 2, number_rect_y, header_w, number_h)
            label_rect = (px - header_w // 2, label_rect_y, header_w, label_h)
            service_rect = (service_rect_x, service_rect_y, service_w, service_h)

            # Outer border around the whole stack
            outer_rect = (px - header_w // 2, number_rect_y, header_w, number_h + label_h + service_h)
            pygame.draw.rect(screen, (0, 0, 0), outer_rect, 2)

            # Number bar (yellow/orange)
            pygame.draw.rect(screen, (245, 195, 80), number_rect)

            # Label bar (grey)
            pygame.draw.rect(screen, (210, 210, 210), label_rect)

            # Service box (grey) – where passenger stands for service
            pygame.draw.rect(screen, (220, 220, 220), service_rect)
            # Blue border around the service box cell (visual emphasis)
            pygame.draw.rect(screen, (70, 120, 200), service_rect, 2)

            # Counter numbers: rightmost counter is 1, then 2, 3, 4 to the left
            tc_number = num_counters - j
            num_surf = counter_number_font.render(str(tc_number), True, (0, 0, 0))
            number_center_y = number_rect_y + number_h // 2
            screen.blit(
                num_surf,
                (
                    px - num_surf.get_width() // 2,
                    number_center_y - num_surf.get_height() // 2,
                ),
            )

            # "Ticket Counters" label centered in the grey bar
            label_surf = counter_label_font.render("Ticket Counters", True, (0, 0, 0))
            label_center_y = label_rect_y + label_h // 2
            screen.blit(
                label_surf,
                (
                    px - label_surf.get_width() // 2,
                    label_center_y - label_surf.get_height() // 2,
                ),
            )

        # Information desks: label box + vertical service box in front of each (2 vertical boxes, one per desk)
        if coordinates.get("information_desks"):
            # Make the green info-desk boxes visually thinner.
            info_box_w, info_box_h = 18, 90  # slim label box (vertical title)
            service_w, service_h = 10, 40    # vertical service box (narrow, tall) in front of each desk
            for key in ("left", "right"):
                ix, iy = coordinates["information_desks"][key]
                px_info, py_info = world_to_screen(ix, iy)
                box_rect = pygame.Rect(
                    px_info - info_box_w // 2,
                    py_info - info_box_h // 2,
                    info_box_w,
                    info_box_h,
                )
                pygame.draw.rect(screen, (200, 255, 200), box_rect)  # light green, no border
                info_surf = info_desk_label_font.render("Information desks", True, (0, 0, 0))
                info_rotated = pygame.transform.rotate(info_surf, 90)
                text_rect = info_rotated.get_rect(center=(box_rect.centerx, box_rect.centery))
                screen.blit(info_rotated, text_rect)
                # Vertical service box touching the label (left desk: box to the right; right desk: box to the left)
                # Draw service rect at the real service waypoint x (visual-only; does not affect simulation).
                if "info_desk_service_positions" in coordinates and len(coordinates["info_desk_service_positions"]) >= 2:
                    service_idx = 0 if key == "left" else 1
                    sx_world, sy_world = coordinates["info_desk_service_positions"][service_idx]
                    psx, psy = world_to_screen(sx_world, sy_world)
                    service_rect = pygame.Rect(
                        psx - service_w // 2,
                        psy - service_h // 2,
                        service_w,
                        service_h,
                    )
                else:
                    # Fallback to previous behavior if service coords are missing.
                    if key == "left":
                        service_center_x = box_rect.right + service_w // 2
                    else:
                        service_center_x = box_rect.left - service_w // 2
                    service_rect = pygame.Rect(
                        service_center_x - service_w // 2,
                        py_info - service_h // 2,
                        service_w,
                        service_h,
                    )
                pygame.draw.rect(screen, (200, 255, 200), service_rect)
                pygame.draw.rect(screen, (70, 120, 200), service_rect, 2)
        # Horizontal queues for info desks (2 positions each, same spacing as ticket counter queues)
        for (qx, qy) in info_left_queue_slots:
            px, py = world_to_screen(qx, qy)
            pygame.draw.circle(screen, waypoint_fill_color, (px, py), 1)
            pygame.draw.circle(screen, waypoint_border_color, (px, py), 1, 1)
        for (qx, qy) in info_right_queue_slots:
            px, py = world_to_screen(qx, qy)
            pygame.draw.circle(screen, waypoint_fill_color, (px, py), 1)
            pygame.draw.circle(screen, waypoint_border_color, (px, py), 1, 1)
        # Security: waiting lanes (below), then queue slots, then counters (at top)
        for (wx, wy) in security_waiting_positions:
            px, py = world_to_screen(wx, wy)
            # Same visual size as passengers (r_px = 1)
            pygame.draw.circle(screen, waypoint_fill_color, (px, py), 1)
            pygame.draw.circle(screen, waypoint_border_color, (px, py), 1, 1)
        # Lane labels for security waiting (same as check-in: L1, L2, ... on the right)
        security_row_ys = sorted(set(wy for _, wy in security_waiting_positions), reverse=True)
        if security_waiting_positions:
            sec_first_col_x = min(wx for wx, _ in security_waiting_positions)
            sec_last_col_x = max(wx for wx, _ in security_waiting_positions)
        else:
            sec_first_col_x = 0.0
            sec_last_col_x = 0.0
        # Place lane labels to the right of the last column
        sec_label_x_world = sec_last_col_x + 0.6
        if security_row_ys and security_waiting_positions:
            # Horizontal heading centered above the security waiting area and below the scanners
            sec_min_x = min(wx for wx, _ in security_waiting_positions)
            sec_max_x = max(wx for wx, _ in security_waiting_positions)
            sec_heading_x_world = (sec_min_x + sec_max_x) / 2.0
            sec_heading_y_world = security_row_ys[0] + 0.5
            px_sh, py_sh = world_to_screen(sec_heading_x_world, sec_heading_y_world)
            sec_wait_heading = lane_font.render("Waiting area", True, (0, 0, 0))
            screen.blit(sec_wait_heading, (px_sh - sec_wait_heading.get_width() // 2, py_sh - sec_wait_heading.get_height() // 2))
        for lane_num, row_y in enumerate(security_row_ys, start=1):
            px, py = world_to_screen(sec_label_x_world, row_y)
            lane_label = lane_label_font.render(f"L{lane_num}", True, (0, 0, 0))
            screen.blit(lane_label, (px, py - lane_label.get_height() // 2))
        # Security counter numbers on the yellow bar (reversed: 2, 1)
        security_headings = ["2", "1"]
        for s in range(num_security):
            # Queue slot circles (same visual size as passengers)
            for (qx, qy) in security_queue_slot_positions[s]:
                px, py = world_to_screen(qx, qy)
                pygame.draw.circle(screen, waypoint_fill_color, (px, py), 1)
                pygame.draw.circle(screen, waypoint_border_color, (px, py), 1, 1)
            # Security counter stack: blue service box at counter position (service spot inside blue border box)
            # [yellow number bar] [grey label bar] [blue-bordered service box] with queue slots below
            (sx, sy) = security_positions[s]
            px, py = world_to_screen(sx, sy)
            header_w = 80
            number_h = 16
            label_h = 14
            service_w, service_h = header_w, 14
            # Place the service box (blue box) exactly at the counter position so service happens inside it
            service_rect_x = px - service_w // 2
            service_rect_y = py - service_h // 2
            label_rect_y = service_rect_y - label_h
            number_rect_y = label_rect_y - number_h
            number_rect = (px - header_w // 2, number_rect_y, header_w, number_h)
            label_rect = (px - header_w // 2, label_rect_y, header_w, label_h)
            service_rect = (service_rect_x, service_rect_y, service_w, service_h)
            outer_rect = (px - header_w // 2, number_rect_y, header_w, number_h + label_h + service_h)
            pygame.draw.rect(screen, (0, 0, 0), outer_rect, 2)
            # East/West block: blue (same as doors)
            pygame.draw.rect(screen, (190, 230, 255), number_rect)
            pygame.draw.rect(screen, (210, 210, 210), label_rect)
            pygame.draw.rect(screen, (220, 220, 220), service_rect)
            pygame.draw.rect(screen, (70, 120, 200), service_rect, 2)
            num_surf = counter_number_font.render(security_headings[s], True, (0, 0, 0))
            number_center_y = number_rect_y + number_h // 2
            screen.blit(
                num_surf,
                (px - num_surf.get_width() // 2, number_center_y - num_surf.get_height() // 2),
            )
            label_surf = counter_label_font.render("Security", True, (0, 0, 0))
            label_center_y = label_rect_y + label_h // 2
            screen.blit(
                label_surf,
                (px - label_surf.get_width() // 2, label_center_y - label_surf.get_height() // 2),
            )
        # Parking area rectangle just below the doors (world: y=0 to y=2.0, left third of hall)
        if hall_points and coordinates.get("entrances"):
            door_ys = [dy for (_dx, dy) in coordinates["entrances"].values()]
            door_y_world = min(door_ys)  # door center y in world
            # Door half-height in world: 12px -> 0.3 world units (scale WORLD_H/SCREEN_H)
            park_y_top = door_y_world - 0.3   # parking top = door bottom (they touch)
            park_y_bottom = 0.0
            hall_xs = [x for x, _ in hall_points]
            park_x_left = min(hall_xs)
            hall_width = max(hall_xs) - park_x_left
            park_x_right = park_x_left + hall_width / 3.0
            park_pts = [
                world_to_screen(park_x_left, park_y_bottom),
                world_to_screen(park_x_right, park_y_bottom),
                world_to_screen(park_x_right, park_y_top),
                world_to_screen(park_x_left, park_y_top),
            ]
            pygame.draw.polygon(screen, (180, 185, 190), park_pts)
            pygame.draw.polygon(screen, (140, 145, 150), park_pts, 2)
            # "Parking area" label centered in the grey box
            px_min, py_min = world_to_screen(park_x_left, park_y_bottom)
            px_max, py_top = world_to_screen(park_x_right, park_y_top)
            label_x = (px_min + px_max) // 2
            label_y = (py_min + py_top) // 2
            parking_surf = lane_font.render("Parking area", True, (0, 0, 0))
            screen.blit(
                parking_surf,
                (
                    label_x - parking_surf.get_width() // 2,
                    label_y - parking_surf.get_height() // 2,
                ),
            )
        # Vertical line 0.2 units from right edge of parking (full height, draw-only)
        if "parking_wall_line" in coordinates and len(coordinates["parking_wall_line"]) >= 2:
            (x0, y0), (x1, y1) = coordinates["parking_wall_line"][:2]
            p0 = world_to_screen(x0, y0)
            p1 = world_to_screen(x1, y1)
            pygame.draw.line(screen, (60, 65, 75), p0, p1, 2)
        # Second vertical line 3 units to the right of the first (full height, draw-only)
        if "parking_wall_line_2" in coordinates and len(coordinates["parking_wall_line_2"]) >= 2:
            (x0, y0), (x1, y1) = coordinates["parking_wall_line_2"][:2]
            p0 = world_to_screen(x0, y0)
            p1 = world_to_screen(x1, y1)
            pygame.draw.line(screen, (60, 65, 75), p0, p1, 2)
        # Horizontal line above the food court, starting from the second vertical line
        if "food_horizontal_line" in coordinates and len(coordinates["food_horizontal_line"]) >= 2:
            (hx0, hy0), (hx1, hy1) = coordinates["food_horizontal_line"][:2]
            ph0 = world_to_screen(hx0, hy0)
            ph1 = world_to_screen(hx1, hy1)
            pygame.draw.line(screen, (60, 65, 75), ph0, ph1, 2)
        # Horizontal line between food court and restrooms, same style
        if "rest_horizontal_line" in coordinates and len(coordinates["rest_horizontal_line"]) >= 2:
            (rx0, ry0), (rx1, ry1) = coordinates["rest_horizontal_line"][:2]
            pr0 = world_to_screen(rx0, ry0)
            pr1 = world_to_screen(rx1, ry1)
            pygame.draw.line(screen, (60, 65, 75), pr0, pr1, 2)
        # Horizontal line between restrooms and All Gates, same style
        if "gates_horizontal_line" in coordinates and len(coordinates["gates_horizontal_line"]) >= 2:
            (gx0, gy0), (gx1, gy1) = coordinates["gates_horizontal_line"][:2]
            pg0 = world_to_screen(gx0, gy0)
            pg1 = world_to_screen(gx1, gy1)
            pygame.draw.line(screen, (60, 65, 75), pg0, pg1, 2)

        # Vertical "Walking area" labels between the two black vertical lines:
        # first aligned with shops heading, second 5 units from bottom (both clamped inside corridor)
        if "walking_corridor_rect" in coordinates:
            corridor_pts = coordinates["walking_corridor_rect"]
            if len(corridor_pts) >= 4:
                xs = [x for x, _ in corridor_pts]
                ys = [y for _, y in corridor_pts]
                cx_world = (min(xs) + max(xs)) / 2.0
                y_min = min(ys)
                y_max = max(ys)
                # Desired positions in world coords
                # First label: vertically aligned with the row of shop boxes (e.g., Cosmetics) if available;
                # otherwise fall back to 5 units from top
                if "cosmetics_rect" in coordinates:
                    shops_rect = coordinates["cosmetics_rect"]
                    s_ys = [sy for _, sy in shops_rect]
                    desired_top = (min(s_ys) + max(s_ys)) / 2.0
                else:
                    desired_top = WORLD_H - 5.0   # fallback: 5 units from top
                # Clamp desired_top inside corridor band with small padding
                cy = desired_top
                if cy < y_min + 0.5:
                    cy = y_min + 0.5
                if cy > y_max - 0.5:
                    cy = y_max - 0.5
                cy_world = cy
                px, py = world_to_screen(cx_world, cy_world)
                walk_surf = lane_font.render("Walking area", True, (0, 0, 0))
                walk_rot = pygame.transform.rotate(walk_surf, 90)
                screen.blit(
                    walk_rot,
                    (
                        px - walk_rot.get_width() // 2,
                        py - walk_rot.get_height() // 2,
                    ),
                )

        # Entrance doors at bottom
        for name, (dx, dy) in coordinates["entrances"].items():
            px, py = world_to_screen(dx, dy)
            # Draw door as a short rectangle (reduced height)
            door_w, door_h = 54, 12
            # Match door color to arrow color (soft blue)
            door_color = (190, 230, 255)
            border_color = (150, 200, 235)
            pygame.draw.rect(screen, door_color, (px - door_w // 2, py - door_h // 2, door_w, door_h))
            pygame.draw.rect(screen, border_color, (px - door_w // 2, py - door_h // 2, door_w, door_h), 2)
            # Labels: left = Entrance 1, right = Entrance 2 (drawn above the doors)
            door_label = "Entrance 1" if dx < 25.0 else "Entrance 2"
            label_surf = lane_font.render(door_label, True, (0, 0, 0))
            screen.blit(label_surf, (px - label_surf.get_width() // 2, py - door_h // 2 - label_surf.get_height() - 2))
        # Green block at top: corridor to shops (passengers go here after security, then to Shops Enter)
        if "exit" in coordinates and "polygon" in coordinates["exit"]:
            exit_pts = [world_to_screen(x, y) for x, y in coordinates["exit"]["polygon"]]
            if len(exit_pts) >= 3:
                pygame.draw.polygon(screen, (80, 200, 120), exit_pts)
                pygame.draw.polygon(screen, (50, 160, 90), exit_pts, 2)
            exit_center = coordinates["exit"].get("center", (25.0, 29.0))
            ex, ey = exit_center
            px, py = world_to_screen(ex, ey)
            exit_label = lane_font.render("MainEntrance", True, (255, 255, 255))
            exit_rot = pygame.transform.rotate(exit_label, 90)
            screen.blit(
                exit_rot,
                (
                    px - exit_rot.get_width() // 2,
                    py - exit_rot.get_height() // 2,
                ),
            )
        # Flight information displays (blue screens with grey box in front, x increased)
        if "flight_displays" in coordinates:
            grey_rects = coordinates.get("flight_display_grey", [])
            for i, rect in enumerate(coordinates["flight_displays"]):
                pts = [world_to_screen(x, y) for x, y in rect]
                if len(pts) >= 3:
                    # Blue screen drawn first (behind)
                    pygame.draw.polygon(screen, (190, 230, 255), pts)
                    pygame.draw.polygon(screen, (150, 200, 235), pts, 2)
                    # Label "Display" centered inside the blue screen
                    xs = [px for px, _ in pts]
                    ys = [py for _, py in pts]
                    center_x = (min(xs) + max(xs)) // 2
                    center_y = (min(ys) + max(ys)) // 2
                    disp_surf = display_label_font.render("Display", True, (0, 0, 0))
                    disp_surf = pygame.transform.rotate(disp_surf, 90)
                    screen.blit(
                        disp_surf,
                        (
                            center_x - disp_surf.get_width() // 2,
                            center_y - disp_surf.get_height() // 2,
                        ),
                    )
                # Grey box in front (same size, x increased) drawn on top
                if i < len(grey_rects):
                    grey_pts_world = grey_rects[i]
                    grey_pts = [world_to_screen(x, y) for x, y in grey_pts_world]
                    if len(grey_pts) >= 3:
                        # Fill and general border
                        pygame.draw.polygon(screen, (210, 210, 210), grey_pts)
                        pygame.draw.polygon(screen, (160, 160, 160), grey_pts, 1)
                        # Highlight the right edge in red
                        # Find the two points with the maximum x in world coords (right edge)
                        xs = [x for x, _ in grey_pts_world]
                        max_x = max(xs)
                        right_edge_points = [(x, y) for x, y in grey_pts_world if abs(x - max_x) < 1e-6]
                        if len(right_edge_points) >= 2:
                            # Sort by y to get bottom and top
                            right_edge_points.sort(key=lambda p: p[1])
                            (rx1, ry1), (rx2, ry2) = right_edge_points[0], right_edge_points[-1]
                            pr1 = world_to_screen(rx1, ry1)
                            pr2 = world_to_screen(rx2, ry2)
                            pygame.draw.line(screen, (220, 60, 60), pr1, pr2, 3)
        # Shops heading box (vertical, to the right of the black line)
        if "shops_label_rect" in coordinates:
            shops_rect_pts = [world_to_screen(x, y) for x, y in coordinates["shops_label_rect"]]
            if len(shops_rect_pts) >= 3:
                pygame.draw.polygon(screen, (210, 210, 210), shops_rect_pts)
                pygame.draw.polygon(screen, (160, 160, 160), shops_rect_pts, 1)
                # Vertical "Shops" label centered in the box
                xs = [px for px, _ in shops_rect_pts]
                ys = [py for _, py in shops_rect_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                shops_surf = lane_font.render("Shops", True, (0, 0, 0))
                shops_rotated = pygame.transform.rotate(shops_surf, 90)
                screen.blit(
                    shops_rotated,
                    (
                        center_x - shops_rotated.get_width() // 2,
                        center_y - shops_rotated.get_height() // 2,
                    ),
                )
        # Green vertical blocks in front of Shops heading: Enter (left) and Exit (right)
        if "shops_enter_block" in coordinates:
            enter_pts = [world_to_screen(x, y) for x, y in coordinates["shops_enter_block"]]
            if len(enter_pts) >= 3:
                pygame.draw.polygon(screen, (180, 235, 180), enter_pts)
                pygame.draw.polygon(screen, (60, 140, 60), enter_pts, 1)
                exs = [px for px, _ in enter_pts]
                eys = [py for _, py in enter_pts]
                ecx = (min(exs) + max(exs)) // 2
                ecy = (min(eys) + max(eys)) // 2
                enter_surf = lane_font.render("Enter", True, (0, 0, 0))
                enter_rot = pygame.transform.rotate(enter_surf, 90)
                screen.blit(
                    enter_rot,
                    (
                        ecx - enter_rot.get_width() // 2,
                        ecy - enter_rot.get_height() // 2,
                    ),
                )
                left_x = min(exs)
                left_ys = [py for (px, py) in enter_pts if px == left_x]
                if len(left_ys) >= 2:
                    pygame.draw.line(screen, (220, 60, 60), (left_x, min(left_ys)), (left_x, max(left_ys)), 3)
        if "shops_exit_block" in coordinates:
            exit_pts = [world_to_screen(x, y) for x, y in coordinates["shops_exit_block"]]
            if len(exit_pts) >= 3:
                pygame.draw.polygon(screen, (180, 235, 180), exit_pts)
                pygame.draw.polygon(screen, (60, 140, 60), exit_pts, 1)
                xs2 = [px for px, _ in exit_pts]
                ys2 = [py for _, py in exit_pts]
                xc2 = (min(xs2) + max(xs2)) // 2
                yc2 = (min(ys2) + max(ys2)) // 2
                exit_surf = lane_font.render("Exit", True, (0, 0, 0))
                exit_rot = pygame.transform.rotate(exit_surf, 90)
                screen.blit(
                    exit_rot,
                    (
                        xc2 - exit_rot.get_width() // 2,
                        yc2 - exit_rot.get_height() // 2,
                    ),
                )
                left_x = min(xs2)
                left_ys = [py for (px, py) in exit_pts if px == left_x]
                if len(left_ys) >= 2:
                    pygame.draw.line(screen, (220, 60, 60), (left_x, min(left_ys)), (left_x, max(left_ys)), 3)
        # Shop "front" grey areas (below each shop/food/restroom box; increase height slightly more for extra area)
        front_h = 40  # pixels
        front_margin_x = 8  # pixels (extend left/right)

        # Track overall shopping area bounds to draw a border around all shops
        shopping_min_x = None
        shopping_max_x = None
        shopping_min_y = None
        shopping_max_y = None

        def _update_shopping_bounds(pts):
            nonlocal shopping_min_x, shopping_max_x, shopping_min_y, shopping_max_y
            xs = [px for px, _ in pts]
            ys = [py for _, py in pts]
            if not xs or not ys:
                return
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            shopping_min_x = min(min_x, shopping_min_x) if shopping_min_x is not None else min_x
            shopping_max_x = max(max_x, shopping_max_x) if shopping_max_x is not None else max_x
            shopping_min_y = min(min_y, shopping_min_y) if shopping_min_y is not None else min_y
            shopping_max_y = max(max_y, shopping_max_y) if shopping_max_y is not None else max_y

        # Single shop: "Cosmetics" box to the right of Shops area enter
        if "cosmetics_rect" in coordinates:
            cosmetics_pts = [world_to_screen(x, y) for x, y in coordinates["cosmetics_rect"]]
            if len(cosmetics_pts) >= 3:
                pygame.draw.polygon(screen, (230, 230, 240), cosmetics_pts)
                pygame.draw.polygon(screen, (160, 160, 170), cosmetics_pts, 1)
                _update_shopping_bounds(cosmetics_pts)
                # Grey area below Cosmetics
                min_x = min(px for px, _ in cosmetics_pts)
                max_x = max(px for px, _ in cosmetics_pts)
                max_y = max(py for _, py in cosmetics_pts)
                cosmetics_front_rect = pygame.Rect(
                    min_x - front_margin_x,
                    max_y,
                    (max_x - min_x) + 2 * front_margin_x,
                    front_h,
                )
                pygame.draw.rect(screen, (210, 210, 210), cosmetics_front_rect)
                pygame.draw.rect(screen, (160, 160, 160), cosmetics_front_rect, 1)
                # Red line along bottom edge of grey front
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (cosmetics_front_rect.left, cosmetics_front_rect.bottom - 1),
                    (cosmetics_front_rect.right, cosmetics_front_rect.bottom - 1),
                    3,
                )
                xs = [px for px, _ in cosmetics_pts]
                ys = [py for _, py in cosmetics_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                shop_surf = lane_font.render("Cosmetics", True, (0, 0, 0))
                screen.blit(
                    shop_surf,
                    (
                        center_x - shop_surf.get_width() // 2,
                        center_y - shop_surf.get_height() // 2,
                    ),
                )
        # Second shop: "Perfumes" box, same level as Cosmetics, further right
        if "perfumes_rect" in coordinates:
            perfumes_pts = [world_to_screen(x, y) for x, y in coordinates["perfumes_rect"]]
            if len(perfumes_pts) >= 3:
                pygame.draw.polygon(screen, (230, 230, 240), perfumes_pts)
                pygame.draw.polygon(screen, (160, 160, 170), perfumes_pts, 1)
                _update_shopping_bounds(perfumes_pts)
                # Grey area below Perfumes
                min_x = min(px for px, _ in perfumes_pts)
                max_x = max(px for px, _ in perfumes_pts)
                max_y = max(py for _, py in perfumes_pts)
                perfumes_front_rect = pygame.Rect(
                    min_x - front_margin_x,
                    max_y,
                    (max_x - min_x) + 2 * front_margin_x,
                    front_h,
                )
                pygame.draw.rect(screen, (210, 210, 210), perfumes_front_rect)
                pygame.draw.rect(screen, (160, 160, 160), perfumes_front_rect, 1)
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (perfumes_front_rect.left, perfumes_front_rect.bottom - 1),
                    (perfumes_front_rect.right, perfumes_front_rect.bottom - 1),
                    3,
                )
                xs = [px for px, _ in perfumes_pts]
                ys = [py for _, py in perfumes_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                perfumes_surf = lane_font.render("Perfumes", True, (0, 0, 0))
                screen.blit(
                    perfumes_surf,
                    (
                        center_x - perfumes_surf.get_width() // 2,
                        center_y - perfumes_surf.get_height() // 2,
                    ),
                )
        # Third shop: "Electronics" box, same level as Cosmetics, 5 units to the right of Perfumes
        if "electronics_rect" in coordinates:
            electronics_pts = [world_to_screen(x, y) for x, y in coordinates["electronics_rect"]]
            if len(electronics_pts) >= 3:
                pygame.draw.polygon(screen, (230, 230, 240), electronics_pts)
                pygame.draw.polygon(screen, (160, 160, 170), electronics_pts, 1)
                _update_shopping_bounds(electronics_pts)
                # Grey area below Electronics
                min_x = min(px for px, _ in electronics_pts)
                max_x = max(px for px, _ in electronics_pts)
                max_y = max(py for _, py in electronics_pts)
                electronics_front_rect = pygame.Rect(
                    min_x - front_margin_x,
                    max_y,
                    (max_x - min_x) + 2 * front_margin_x,
                    front_h,
                )
                pygame.draw.rect(screen, (210, 210, 210), electronics_front_rect)
                pygame.draw.rect(screen, (160, 160, 160), electronics_front_rect, 1)
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (electronics_front_rect.left, electronics_front_rect.bottom - 1),
                    (electronics_front_rect.right, electronics_front_rect.bottom - 1),
                    3,
                )
                xs = [px for px, _ in electronics_pts]
                ys = [py for _, py in electronics_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                electronics_surf = lane_font.render("Electronics", True, (0, 0, 0))
                screen.blit(
                    electronics_surf,
                    (
                        center_x - electronics_surf.get_width() // 2,
                        center_y - electronics_surf.get_height() // 2,
                    ),
                )
        # Fourth shop: "Books" box, same level, 5 units to the right of Electronics
        if "books_rect" in coordinates:
            books_pts = [world_to_screen(x, y) for x, y in coordinates["books_rect"]]
            if len(books_pts) >= 3:
                pygame.draw.polygon(screen, (230, 230, 240), books_pts)
                pygame.draw.polygon(screen, (160, 160, 170), books_pts, 1)
                # Grey area below Books
                min_x = min(px for px, _ in books_pts)
                max_x = max(px for px, _ in books_pts)
                max_y = max(py for _, py in books_pts)
                books_front_rect = pygame.Rect(
                    min_x - front_margin_x,
                    max_y,
                    (max_x - min_x) + 2 * front_margin_x,
                    front_h,
                )
                pygame.draw.rect(screen, (210, 210, 210), books_front_rect)
                pygame.draw.rect(screen, (160, 160, 160), books_front_rect, 1)
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (books_front_rect.left, books_front_rect.bottom - 1),
                    (books_front_rect.right, books_front_rect.bottom - 1),
                    3,
                )
                xs = [px for px, _ in books_pts]
                ys = [py for _, py in books_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                books_surf = lane_font.render("Books", True, (0, 0, 0))
                screen.blit(
                    books_surf,
                    (
                        center_x - books_surf.get_width() // 2,
                        center_y - books_surf.get_height() // 2,
                    ),
                )

        # ---- Food court section (mirrors shopping area, below it) ----
        # Food heading box
        if "food_label_rect" in coordinates:
            food_label_pts = [world_to_screen(x, y) for x, y in coordinates["food_label_rect"]]
            if len(food_label_pts) >= 3:
                pygame.draw.polygon(screen, (210, 210, 210), food_label_pts)
                pygame.draw.polygon(screen, (160, 160, 160), food_label_pts, 1)
                xs = [px for px, _ in food_label_pts]
                ys = [py for _, py in food_label_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                food_surf = lane_font.render("Food court", True, (0, 0, 0))
                food_rotated = pygame.transform.rotate(food_surf, 90)
                screen.blit(
                    food_rotated,
                    (
                        center_x - food_rotated.get_width() // 2,
                        center_y - food_rotated.get_height() // 2,
                    ),
                )
        # Food Enter/Exit green blocks (stacked on left)
        if "food_enter_block" in coordinates:
            enter_pts = [world_to_screen(x, y) for x, y in coordinates["food_enter_block"]]
            if len(enter_pts) >= 3:
                pygame.draw.polygon(screen, (180, 235, 180), enter_pts)
                pygame.draw.polygon(screen, (60, 140, 60), enter_pts, 1)
                exs = [px for px, _ in enter_pts]
                eys = [py for _, py in enter_pts]
                ecx = (min(exs) + max(exs)) // 2
                ecy = (min(eys) + max(eys)) // 2
                enter_surf = lane_font.render("Enter", True, (0, 0, 0))
                enter_rot = pygame.transform.rotate(enter_surf, 90)
                screen.blit(
                    enter_rot,
                    (
                        ecx - enter_rot.get_width() // 2,
                        ecy - enter_rot.get_height() // 2,
                    ),
                )
                left_x = min(exs)
                left_ys = [py for (px, py) in enter_pts if px == left_x]
                if len(left_ys) >= 2:
                    pygame.draw.line(screen, (220, 60, 60), (left_x, min(left_ys)), (left_x, max(left_ys)), 3)
        if "food_exit_block" in coordinates:
            exit_pts = [world_to_screen(x, y) for x, y in coordinates["food_exit_block"]]
            if len(exit_pts) >= 3:
                pygame.draw.polygon(screen, (180, 235, 180), exit_pts)
                pygame.draw.polygon(screen, (60, 140, 60), exit_pts, 1)
                xs2 = [px for px, _ in exit_pts]
                ys2 = [py for _, py in exit_pts]
                xc2 = (min(xs2) + max(xs2)) // 2
                yc2 = (min(ys2) + max(ys2)) // 2
                exit_surf = lane_font.render("Exit", True, (0, 0, 0))
                exit_rot = pygame.transform.rotate(exit_surf, 90)
                screen.blit(
                    exit_rot,
                    (
                        xc2 - exit_rot.get_width() // 2,
                        yc2 - exit_rot.get_height() // 2,
                    ),
                )
                left_x = min(xs2)
                left_ys = [py for (px, py) in exit_pts if px == left_x]
                if len(left_ys) >= 2:
                    pygame.draw.line(screen, (220, 60, 60), (left_x, min(left_ys)), (left_x, max(left_ys)), 3)

        # Food court shop boxes and grey fronts (Burgers, Pizza, Coffee, Desserts)
        if "food_burgers_rect" in coordinates:
            burgers_pts = [world_to_screen(x, y) for x, y in coordinates["food_burgers_rect"]]
            if len(burgers_pts) >= 3:
                pygame.draw.polygon(screen, (230, 230, 240), burgers_pts)
                pygame.draw.polygon(screen, (160, 160, 170), burgers_pts, 1)
                min_x = min(px for px, _ in burgers_pts)
                max_x = max(px for px, _ in burgers_pts)
                max_y = max(py for _, py in burgers_pts)
                burgers_front_rect = pygame.Rect(
                    min_x - front_margin_x,
                    max_y,
                    (max_x - min_x) + 2 * front_margin_x,
                    front_h,
                )
                pygame.draw.rect(screen, (210, 210, 210), burgers_front_rect)
                pygame.draw.rect(screen, (160, 160, 160), burgers_front_rect, 1)
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (burgers_front_rect.left, burgers_front_rect.bottom - 1),
                    (burgers_front_rect.right, burgers_front_rect.bottom - 1),
                    3,
                )
                # Additional horizontal line at the same level as the food red line
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (burgers_front_rect.left, burgers_front_rect.bottom - 1),
                    (burgers_front_rect.right, burgers_front_rect.bottom - 1),
                    1,
                )
                xs = [px for px, _ in burgers_pts]
                ys = [py for _, py in burgers_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                b_surf = lane_font.render("Burgers", True, (0, 0, 0))
                screen.blit(
                    b_surf,
                    (
                        center_x - b_surf.get_width() // 2,
                        center_y - b_surf.get_height() // 2,
                    ),
                )

        if "food_pizza_rect" in coordinates:
            pizza_pts = [world_to_screen(x, y) for x, y in coordinates["food_pizza_rect"]]
            if len(pizza_pts) >= 3:
                pygame.draw.polygon(screen, (230, 230, 240), pizza_pts)
                pygame.draw.polygon(screen, (160, 160, 170), pizza_pts, 1)
                min_x = min(px for px, _ in pizza_pts)
                max_x = max(px for px, _ in pizza_pts)
                max_y = max(py for _, py in pizza_pts)
                pizza_front_rect = pygame.Rect(
                    min_x - front_margin_x,
                    max_y,
                    (max_x - min_x) + 2 * front_margin_x,
                    front_h,
                )
                pygame.draw.rect(screen, (210, 210, 210), pizza_front_rect)
                pygame.draw.rect(screen, (160, 160, 160), pizza_front_rect, 1)
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (pizza_front_rect.left, pizza_front_rect.bottom - 1),
                    (pizza_front_rect.right, pizza_front_rect.bottom - 1),
                    3,
                )
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (pizza_front_rect.left, pizza_front_rect.bottom - 1),
                    (pizza_front_rect.right, pizza_front_rect.bottom - 1),
                    1,
                )
                xs = [px for px, _ in pizza_pts]
                ys = [py for _, py in pizza_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                p_surf = lane_font.render("Pizza", True, (0, 0, 0))
                screen.blit(
                    p_surf,
                    (
                        center_x - p_surf.get_width() // 2,
                        center_y - p_surf.get_height() // 2,
                    ),
                )

        if "food_coffee_rect" in coordinates:
            coffee_pts = [world_to_screen(x, y) for x, y in coordinates["food_coffee_rect"]]
            if len(coffee_pts) >= 3:
                pygame.draw.polygon(screen, (230, 230, 240), coffee_pts)
                pygame.draw.polygon(screen, (160, 160, 170), coffee_pts, 1)
                min_x = min(px for px, _ in coffee_pts)
                max_x = max(px for px, _ in coffee_pts)
                max_y = max(py for _, py in coffee_pts)
                coffee_front_rect = pygame.Rect(
                    min_x - front_margin_x,
                    max_y,
                    (max_x - min_x) + 2 * front_margin_x,
                    front_h,
                )
                pygame.draw.rect(screen, (210, 210, 210), coffee_front_rect)
                pygame.draw.rect(screen, (160, 160, 160), coffee_front_rect, 1)
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (coffee_front_rect.left, coffee_front_rect.bottom - 1),
                    (coffee_front_rect.right, coffee_front_rect.bottom - 1),
                    3,
                )
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (coffee_front_rect.left, coffee_front_rect.bottom - 1),
                    (coffee_front_rect.right, coffee_front_rect.bottom - 1),
                    1,
                )
                xs = [px for px, _ in coffee_pts]
                ys = [py for _, py in coffee_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                c_surf = lane_font.render("Coffee", True, (0, 0, 0))
                screen.blit(
                    c_surf,
                    (
                        center_x - c_surf.get_width() // 2,
                        center_y - c_surf.get_height() // 2,
                    ),
                )

        if "food_desserts_rect" in coordinates:
            desserts_pts = [world_to_screen(x, y) for x, y in coordinates["food_desserts_rect"]]
            if len(desserts_pts) >= 3:
                pygame.draw.polygon(screen, (230, 230, 240), desserts_pts)
                pygame.draw.polygon(screen, (160, 160, 170), desserts_pts, 1)
                min_x = min(px for px, _ in desserts_pts)
                max_x = max(px for px, _ in desserts_pts)
                max_y = max(py for _, py in desserts_pts)
                desserts_front_rect = pygame.Rect(
                    min_x - front_margin_x,
                    max_y,
                    (max_x - min_x) + 2 * front_margin_x,
                    front_h,
                )
                pygame.draw.rect(screen, (210, 210, 210), desserts_front_rect)
                pygame.draw.rect(screen, (160, 160, 160), desserts_front_rect, 1)
                pygame.draw.line(
                    screen,
                    (220, 60, 60),
                    (desserts_front_rect.left, desserts_front_rect.bottom - 1),
                    (desserts_front_rect.right, desserts_front_rect.bottom - 1),
                    3,
                )
                xs = [px for px, _ in desserts_pts]
                ys = [py for _, py in desserts_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                d_surf = lane_font.render("Desserts", True, (0, 0, 0))
                screen.blit(
                    d_surf,
                    (
                        center_x - d_surf.get_width() // 2,
                        center_y - d_surf.get_height() // 2,
                    ),
                )

        # (Removed) continuous red line across all food counters

        # ---- Restrooms section (same style, below food court; 3 restrooms) ----
        if "rest_label_rect" in coordinates:
            rest_label_pts = [world_to_screen(x, y) for x, y in coordinates["rest_label_rect"]]
            if len(rest_label_pts) >= 3:
                pygame.draw.polygon(screen, (210, 210, 210), rest_label_pts)
                pygame.draw.polygon(screen, (160, 160, 160), rest_label_pts, 1)
                xs = [px for px, _ in rest_label_pts]
                ys = [py for _, py in rest_label_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                rest_surf = lane_font.render("Restrooms", True, (0, 0, 0))
                rest_rotated = pygame.transform.rotate(rest_surf, 90)
                screen.blit(
                    rest_rotated,
                    (
                        center_x - rest_rotated.get_width() // 2,
                        center_y - rest_rotated.get_height() // 2,
                    ),
                )

        if "rest_enter_block" in coordinates:
            enter_pts = [world_to_screen(x, y) for x, y in coordinates["rest_enter_block"]]
            if len(enter_pts) >= 3:
                pygame.draw.polygon(screen, (180, 235, 180), enter_pts)
                pygame.draw.polygon(screen, (60, 140, 60), enter_pts, 1)
                exs = [px for px, _ in enter_pts]
                eys = [py for _, py in enter_pts]
                ecx = (min(exs) + max(exs)) // 2
                ecy = (min(eys) + max(eys)) // 2
                enter_surf = lane_font.render("Enter", True, (0, 0, 0))
                enter_rot = pygame.transform.rotate(enter_surf, 90)
                screen.blit(
                    enter_rot,
                    (
                        ecx - enter_rot.get_width() // 2,
                        ecy - enter_rot.get_height() // 2,
                    ),
                )
                left_x = min(exs)
                left_ys = [py for (px, py) in enter_pts if px == left_x]
                if len(left_ys) >= 2:
                    pygame.draw.line(screen, (220, 60, 60), (left_x, min(left_ys)), (left_x, max(left_ys)), 3)

        if "rest_exit_block" in coordinates:
            exit_pts = [world_to_screen(x, y) for x, y in coordinates["rest_exit_block"]]
            if len(exit_pts) >= 3:
                pygame.draw.polygon(screen, (180, 235, 180), exit_pts)
                pygame.draw.polygon(screen, (60, 140, 60), exit_pts, 1)
                xs2 = [px for px, _ in exit_pts]
                ys2 = [py for _, py in exit_pts]
                xc2 = (min(xs2) + max(xs2)) // 2
                yc2 = (min(ys2) + max(ys2)) // 2
                exit_surf = lane_font.render("Exit", True, (0, 0, 0))
                exit_rot = pygame.transform.rotate(exit_surf, 90)
                screen.blit(
                    exit_rot,
                    (
                        xc2 - exit_rot.get_width() // 2,
                        yc2 - exit_rot.get_height() // 2,
                    ),
                )
                left_x = min(xs2)
                left_ys = [py for (px, py) in exit_pts if px == left_x]
                if len(left_ys) >= 2:
                    pygame.draw.line(screen, (220, 60, 60), (left_x, min(left_ys)), (left_x, max(left_ys)), 3)

        # 3 restroom blocks with grey fronts (below)
        def _draw_restroom(key: str, label: str) -> None:
            if key not in coordinates:
                return
            pts = [world_to_screen(x, y) for x, y in coordinates[key]]
            if len(pts) < 3:
                return
            pygame.draw.polygon(screen, (230, 230, 240), pts)
            pygame.draw.polygon(screen, (160, 160, 170), pts, 1)
            min_x = min(px for px, _ in pts)
            max_x = max(px for px, _ in pts)
            max_y = max(py for _, py in pts)
            front_rect = pygame.Rect(
                min_x - front_margin_x,
                max_y,
                (max_x - min_x) + 2 * front_margin_x,
                front_h,
            )
            pygame.draw.rect(screen, (210, 210, 210), front_rect)
            pygame.draw.rect(screen, (160, 160, 160), front_rect, 1)
            pygame.draw.line(
                screen,
                (220, 60, 60),
                (front_rect.left, front_rect.bottom - 1),
                (front_rect.right, front_rect.bottom - 1),
                3,
            )
            xs = [px for px, _ in pts]
            ys = [py for _, py in pts]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            surf = lane_font.render(label, True, (0, 0, 0))
            screen.blit(surf, (cx - surf.get_width() // 2, cy - surf.get_height() // 2))

        _draw_restroom("rest1_rect", "Restroom 1")
        _draw_restroom("rest2_rect", "Restroom 2")
        _draw_restroom("rest3_rect", "Restroom 3")

        # ---- All Gates section (heading + enter/exit blocks, below restrooms) ----
        if "gates_label_rect" in coordinates:
            gates_label_pts = [world_to_screen(x, y) for x, y in coordinates["gates_label_rect"]]
            if len(gates_label_pts) >= 3:
                pygame.draw.polygon(screen, (210, 210, 210), gates_label_pts)
                pygame.draw.polygon(screen, (160, 160, 160), gates_label_pts, 1)
                xs = [px for px, _ in gates_label_pts]
                ys = [py for _, py in gates_label_pts]
                center_x = (min(xs) + max(xs)) // 2
                center_y = (min(ys) + max(ys)) // 2
                gates_surf = lane_font.render("All Gates", True, (0, 0, 0))
                gates_rotated = pygame.transform.rotate(gates_surf, 90)
                screen.blit(
                    gates_rotated,
                    (
                        center_x - gates_rotated.get_width() // 2,
                        center_y - gates_rotated.get_height() // 2,
                    ),
                )

        if "gates_enter_block" in coordinates:
            enter_pts = [world_to_screen(x, y) for x, y in coordinates["gates_enter_block"]]
            if len(enter_pts) >= 3:
                pygame.draw.polygon(screen, (180, 235, 180), enter_pts)
                pygame.draw.polygon(screen, (60, 140, 60), enter_pts, 1)
                exs = [px for px, _ in enter_pts]
                eys = [py for _, py in enter_pts]
                ecx = (min(exs) + max(exs)) // 2
                ecy = (min(eys) + max(eys)) // 2
                enter_surf = lane_font.render("Enter", True, (0, 0, 0))
                enter_rot = pygame.transform.rotate(enter_surf, 90)
                screen.blit(
                    enter_rot,
                    (
                        ecx - enter_rot.get_width() // 2,
                        ecy - enter_rot.get_height() // 2,
                    ),
                )
                left_x = min(exs)
                left_ys = [py for (px, py) in enter_pts if px == left_x]
                if len(left_ys) >= 2:
                    pygame.draw.line(screen, (220, 60, 60), (left_x, min(left_ys)), (left_x, max(left_ys)), 3)

        # No "gates_exit_block": All Gates area is entry-only

        # Gate blocks A1..A13 (small horizontal pink blocks) with red edge toward approach
        if "gates_gate_blocks" in coordinates:
            for name, rect in coordinates["gates_gate_blocks"].items():
                pts = [world_to_screen(x, y) for x, y in rect]
                if len(pts) >= 3:
                    pygame.draw.polygon(screen, (255, 180, 210), pts)
                    pygame.draw.polygon(screen, (200, 90, 140), pts, 1)
                    xs = [px for px, _ in pts]
                    ys = [py for _, py in pts]
                    center_x = (min(xs) + max(xs)) // 2
                    center_y = (min(ys) + max(ys)) // 2
                    surf = lane_font.render(name, True, (0, 0, 0))
                    screen.blit(
                        surf,
                        (
                            center_x - surf.get_width() // 2,
                            center_y - surf.get_height() // 2,
                        ),
                    )
                    # Red line directly under each pink gate box (no grey box)
                    min_x = min(xs)
                    max_x = max(xs)
                    min_y = min(ys)
                    max_y = max(ys)
                    # A-lane: red line on bottom edge (approach from below)
                    y_line = max_y + 1
                    pygame.draw.line(
                        screen,
                        (220, 60, 60),
                        (min_x, y_line),
                        (max_x, y_line),
                        3,
                    )

        for agent in sim.agents():
            ax, ay = agent.position
            px, py = world_to_screen(ax, ay)
            r_px = 1
            color = passenger_color.get(agent.id, (0, 0, 0))
            pygame.draw.circle(screen, color, (px, py), r_px)
            pygame.draw.circle(screen, (160, 30, 30), (px, py), r_px, 1)
        queue_stage = sim.get_stage(waiting_queue_stage_id)
        waiting_count = queue_stage.count_enqueued()
        sim_time_now = iteration * SIMULATION_DT * SIM_SPEED_MULT
        counter_busy = sum(1 for a in counter_serving_agent if a is not None)
        security_busy = sum(1 for a in security_serving_agent if a is not None)
        hud = font.render(
            f"t={sim_time_now:.1f}s  n={sim.agent_count()}  wait={waiting_count}  counter={counter_busy}  security={security_busy}  [ESC]",
            True,
            (220, 220, 220),
        )
        screen.blit(hud, (10, 70))

        time_text = font.render(f"Time: {sim_time_now:.1f}s", True, (0, 0, 0))
        screen.blit(time_text, (10, 10))
        exited_text = font.render(
            f"Passengers Exited: {completed_passengers}",
            True,
            (0, 0, 0),
        )
        screen.blit(exited_text, (10, 30))

        pygame.display.flip()
        clock.tick(FPS)

        # Stop only at fixed simulated-time horizon.
        if stop_due_to_horizon:
            break

    sim_time_now = iteration * SIMULATION_DT * SIM_SPEED_MULT
    final_sim_time = min(sim_time_now, SIMULATION_HORIZON)

    # Summary metrics
    passengers_remaining = max(
        0, total_passengers_generated - total_passengers_exited
    )
    completion_rate = (
        total_passengers_exited / total_passengers_generated
        if total_passengers_generated > 0
        else 0.0
    )

    avg_ci_queue_len = (
        sum(ci_queue_lengths) / len(ci_queue_lengths) if ci_queue_lengths else 0.0
    )
    max_ci_queue_len = max(ci_queue_lengths) if ci_queue_lengths else 0
    avg_sc_queue_len = (
        sum(sc_queue_lengths) / len(sc_queue_lengths) if sc_queue_lengths else 0.0
    )
    max_sc_queue_len = max(sc_queue_lengths) if sc_queue_lengths else 0

    ci_wait_samples = [
        ci_service_start_times[aid] - t_join
        for aid, t_join in ci_join_times.items()
        if aid in ci_service_start_times
    ]
    sc_wait_samples = [
        sc_service_start_times[aid] - t_join
        for aid, t_join in sc_join_times.items()
        if aid in sc_service_start_times
    ]
    avg_ci_wait = sum(ci_wait_samples) / len(ci_wait_samples) if ci_wait_samples else 0.0
    avg_sc_wait = sum(sc_wait_samples) / len(sc_wait_samples) if sc_wait_samples else 0.0

    max_overflow_count = max(overflow_counts) if overflow_counts else 0
    avg_overflow_count = (
        sum(overflow_counts) / len(overflow_counts) if overflow_counts else 0.0
    )

    DELTA_T = 300.0

    def compute_throughput(time_series, Y_series, delta_t):
        throughput = []
        for i in range(len(time_series)):
            j = i
            while j < len(time_series) and time_series[j] < time_series[i] + delta_t:
                j += 1
            if j < len(time_series):
                dt = time_series[j] - time_series[i]
                if dt > 0:
                    x = (Y_series[j] - Y_series[i]) / dt
                else:
                    x = 0
            else:
                x = 0
            throughput.append(x)
        return throughput

    throughput_series = compute_throughput(time_series, Y_series, DELTA_T)

    throughput: list[float] = []
    for i in range(1, len(time_steps)):
        dt = time_steps[i] - time_steps[i - 1]
        if dt > 0:
            flow = (exited_counts[i] - exited_counts[i - 1]) / dt
        else:
            flow = 0.0
        throughput.append(flow)

    avg_throughput = sum(throughput) / len(throughput) if throughput else 0.0
    avg_throughput_per_hour = avg_throughput * 3600.0
    peak_throughput_per_hour = (max(throughput) * 3600.0) if throughput else 0.0
    avg_ci_queue = (
        sum(ci_queue_lengths) / len(ci_queue_lengths) if ci_queue_lengths else 0.0
    )
    avg_sc_queue = (
        sum(sc_queue_lengths) / len(sc_queue_lengths) if sc_queue_lengths else 0.0
    )
    peak_ci_queue = max(ci_queue_lengths) if ci_queue_lengths else 0
    peak_sc_queue = max(sc_queue_lengths) if sc_queue_lengths else 0
    total_completion_time = time_steps[-1] if time_steps else 0.0

    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        rank = (len(s) - 1) * (p / 100.0)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return float(s[lo])
        frac = rank - lo
        return float(s[lo] * (1.0 - frac) + s[hi] * frac)

    p95_ci_wait = percentile(ci_wait_samples, 95.0)
    p95_sc_wait = percentile(sc_wait_samples, 95.0)
    avg_system_time = (
        sum(system_time_samples) / len(system_time_samples) if system_time_samples else 0.0
    )
    p95_system_time = percentile(system_time_samples, 95.0)

    shops_count = 0
    food_count = 0
    restroom_count = 0
    direct_gate_count = 0
    for aid in exited_agent_ids:
        act = planned_activity.get(aid, activity_choice.get(aid, "unknown"))
        if act == "shops":
            shops_count += 1
        elif act == "food":
            food_count += 1
        elif act == "rest":
            restroom_count += 1
        elif act == "gates":
            direct_gate_count += 1

    completion_rate_percent = completion_rate * 100.0
    throughput_per_hour_series = [x * 3600.0 for x in throughput]
    throughput_per_hour_by_step = [0.0] + throughput_per_hour_series

    # Save JSON/CSV artifacts next to this script for analysis and plotting.
    _out_dir = os.path.dirname(os.path.abspath(__file__))
    scenario_prefix = "2_CI_degarded_"

    def prefixed_name(filename: str) -> str:
        return f"{scenario_prefix}{filename}"

    metrics_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("ci_degarded_metrics.json")))
    timeseries_json_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("ci_degarded_timeseries.json")))
    timeseries_csv_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("ci_degarded_timeseries.csv")))
    plotdata_json_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("ci_degarded_plotdata.json")))

    summary_metrics = {
        "total_passengers_generated": total_passengers_generated,
        "total_passengers_exited": total_passengers_exited,
        "passengers_remaining": passengers_remaining,
        "completion_rate_percent": completion_rate_percent,
        "total_completion_time_s": total_completion_time,
        "avg_throughput_per_hour": avg_throughput_per_hour,
        "peak_throughput_per_hour": peak_throughput_per_hour,
        "avg_ci_queue_length": avg_ci_queue,
        "peak_ci_queue_length": peak_ci_queue,
        "avg_sc_queue_length": avg_sc_queue,
        "peak_sc_queue_length": peak_sc_queue,
        "avg_ci_wait_s": avg_ci_wait,
        "p95_ci_wait_s": p95_ci_wait,
        "avg_sc_wait_s": avg_sc_wait,
        "p95_sc_wait_s": p95_sc_wait,
        "avg_system_time_s": avg_system_time,
        "p95_system_time_s": p95_system_time,
        "display_visit_count": display_visit_count,
        "shops_count": shops_count,
        "food_count": food_count,
        "restroom_count": restroom_count,
        "direct_gate_count": direct_gate_count,
    }

    timeseries_rows = []
    for i, t in enumerate(time_steps):
        timeseries_rows.append(
            {
                "time_s": float(t),
                "time_h": float(t / 3600.0),
                "exited_cumulative": int(exited_counts[i]) if i < len(exited_counts) else 0,
                "throughput_per_hour": float(throughput_per_hour_by_step[i]) if i < len(throughput_per_hour_by_step) else 0.0,
                "ci_queue_length": int(ci_queue_lengths[i]) if i < len(ci_queue_lengths) else 0,
                "sc_queue_length": int(sc_queue_lengths[i]) if i < len(sc_queue_lengths) else 0,
                "overflow_count": int(overflow_counts[i]) if i < len(overflow_counts) else 0,
                "passengers_in_system": int(passengers_in_system_series[i]) if i < len(passengers_in_system_series) else 0,
                "checkin_utilization": float(checkin_utilization_series[i]) if i < len(checkin_utilization_series) else 0.0,
                "security_utilization": float(security_utilization_series[i]) if i < len(security_utilization_series) else 0.0,
            }
        )

    plotdata = {
        "time_steps": [float(x) for x in time_steps],
        "exited_counts": [int(x) for x in exited_counts],
        "ci_queue_lengths": [int(x) for x in ci_queue_lengths],
        "sc_queue_lengths": [int(x) for x in sc_queue_lengths],
        "throughput_time_hours": [float(t / 3600.0) for t in time_steps[1:]],
        "throughput_per_hour": [float(x) for x in throughput_per_hour_series],
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    with open(timeseries_json_path, "w", encoding="utf-8") as f:
        json.dump(timeseries_rows, f, indent=2)
    with open(timeseries_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "time_s",
            "time_h",
            "exited_cumulative",
            "throughput_per_hour",
            "ci_queue_length",
            "sc_queue_length",
            "overflow_count",
            "passengers_in_system",
            "checkin_utilization",
            "security_utilization",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeseries_rows)
    with open(plotdata_json_path, "w", encoding="utf-8") as f:
        json.dump(plotdata, f, indent=2)

    throughput_png_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("throughput_vs_time.png")))
    throughput_pdf_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("throughput_vs_time.pdf")))
    queue_png_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("queue_lengths_vs_time_hours.png")))
    queue_pdf_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("queue_lengths_vs_time_hours.pdf")))
    cumulative_png_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("cumulative_exits_vs_time_hours.png")))
    cumulative_pdf_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("cumulative_exits_vs_time_hours.pdf")))
    # Keep canonical hotspot filename for the layout-overlay heatmap.
    heatmap_png_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("pedestrian_hotspot_heatmap.png")))
    heatmap_pdf_path = os.path.abspath(os.path.join(_out_dir, prefixed_name("pedestrian_hotspot_heatmap.pdf")))

    print("\n--- FINAL METRICS ---")
    print("Avg Throughput (passengers/hour):", avg_throughput_per_hour)

    print("\n--- QUEUE METRICS ---")
    print("Avg CI Queue Length:", avg_ci_queue)
    print("Avg SC Queue Length:", avg_sc_queue)
    print("Peak CI Queue Length:", peak_ci_queue)
    print("Peak SC Queue Length:", peak_sc_queue)

    print("\n--- SYSTEM METRICS ---")
    print("Total Completion Time:", total_completion_time)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter

        plt.rcParams.update(
            {
                "font.size": 13,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
            }
        )

        # Save PNGs next to this script (same folder as run_simulation.py) for easy local access.
        sec_per_hour = 3600.0

        # 1) Throughput vs time — presentation plot only: fixed-window aggregate from
        #    time_steps + exited_counts (does not replace per-step `throughput` for metrics).
        #    Use 600.0 here if 300 s is still too noisy.
        throughput_agg_window_sec = 5000.0
        if time_steps and exited_counts and len(time_steps) == len(exited_counts):
            t_h_plot: list[float] = []
            th_plot: list[float] = []
            for j in range(len(time_steps)):
                t_end = time_steps[j]
                t_win_start = t_end - throughput_agg_window_sec
                if t_win_start < time_steps[0]:
                    continue
                i0 = bisect.bisect_right(time_steps, t_win_start) - 1
                if i0 < 0:
                    continue
                d_exits = exited_counts[j] - exited_counts[i0]
                t_h_plot.append(t_end / sec_per_hour)
                th_plot.append((d_exits / throughput_agg_window_sec) * 3600.0)
            if t_h_plot:
                fig1, ax1 = plt.subplots()
                ax1.plot(t_h_plot, th_plot, color="C0")
                ax1.set_xlabel("Simulation Time (hours)")
                ax1.set_ylabel("Passenger Throughput (passengers/hour)")
                ax1.set_title("Passenger Throughput Over Time (Check in Degarded)")
                fig1.tight_layout()
                fig1.savefig(throughput_png_path, dpi=150)
                fig1.savefig(throughput_pdf_path)
                plt.close(fig1)
                print("SMOOTHED THROUGHPUT FIGURE SAVED:", throughput_png_path, flush=True)
                print("SMOOTHED THROUGHPUT FIGURE SAVED:", throughput_pdf_path, flush=True)
            else:
                print(
                    "\n(smoothed throughput plot skipped: no samples after "
                    f"{throughput_agg_window_sec:g}s window — is the run very short?)",
                    flush=True,
                )
        else:
            print(
                "\n(smoothed throughput plot skipped: time_steps/exited_counts missing or length mismatch)",
                flush=True,
            )

        # 2) Queue lengths vs time (same time_steps as ci_queue_lengths / sc_queue_lengths)
        if (
            time_steps
            and ci_queue_lengths
            and sc_queue_lengths
            and len(time_steps) == len(ci_queue_lengths) == len(sc_queue_lengths)
        ):
            t_hours = [t / sec_per_hour for t in time_steps]
            fig2, ax2 = plt.subplots()
            ax2.plot(t_hours, ci_queue_lengths, label="Check-in queue length", color="C0")
            ax2.plot(t_hours, sc_queue_lengths, label="Security queue length", color="C1")
            ax2.set_xlabel("Time (hours)")
            ax2.set_ylabel("Queue Length (passengers)")
            ax2.set_title("Check in Degarded Queue Lengths vs Time")
            ax2.legend(loc="best")
            fig2.tight_layout()
            fig2.savefig(queue_png_path, dpi=150)
            fig2.savefig(queue_pdf_path)
            plt.close(fig2)
            print(">>> QUEUE LENGTH FIGURE SAVED:", queue_png_path, flush=True)
            print(">>> QUEUE LENGTH FIGURE SAVED:", queue_pdf_path, flush=True)

        # 3) Cumulative exited passengers vs time (exited_counts aligned with time_steps)
        if time_steps and exited_counts and len(time_steps) == len(exited_counts):
            t_hours = [t / sec_per_hour for t in time_steps]
            fig3, ax3 = plt.subplots()
            ax3.plot(t_hours, exited_counts, color="C2")
            ax3.set_xlabel("Time (hours)")
            ax3.set_ylabel("Cumulative Exited Passengers")
            ax3.set_title("Check in Degarded Cumulative Exited Passengers vs Time")
            ax3.legend(loc="best")
            fig3.tight_layout()
            fig3.savefig(cumulative_png_path, dpi=150)
            fig3.savefig(cumulative_pdf_path)
            plt.close(fig3)
            print(">>> CUMULATIVE EXITS FIGURE SAVED:", cumulative_png_path, flush=True)
            print(">>> CUMULATIVE EXITS FIGURE SAVED:", cumulative_pdf_path, flush=True)

        # 4) Pedestrian hotspot heatmap (layout-overlay publication style)
        if zone_sample_steps > 0:
            # Average occupancy per defined airport zone.
            zone_avg_occupancy: dict[tuple[int, int], float] = {
                zone_id: occ_sum / float(zone_sample_steps)
                for zone_id, occ_sum in zone_sum_occupancy.items()
            }
            hotspot_matrix = [
                [float(zone_avg_occupancy.get((r, c), 0.0)) for c in range(ZONE_COLS)]
                for r in range(ZONE_ROWS)
            ]

            # A) Publication-style overlay heatmap with visible axes and outside colorbar.
            # Fixed vmin/vmax for cross-scenario comparison; constrained_layout avoids label clipping.
            fig4, ax4 = plt.subplots(figsize=(12, 6), constrained_layout=True)
            zero_bg_color = plt.cm.inferno(0.0)
            fig4.patch.set_facecolor(zero_bg_color)
            ax4.set_facecolor(zero_bg_color)
            im_overlay = ax4.imshow(
                hotspot_matrix,
                origin="lower",
                extent=[HALL_MIN_X, HALL_MAX_X, HALL_MIN_Y, HALL_MAX_Y],
                cmap="inferno",
                interpolation="nearest",
                aspect="equal",
                alpha=0.82,
                zorder=1,
                vmin=0.0,
                vmax=5.0,
            )
            cbar_overlay = fig4.colorbar(im_overlay, ax=ax4, pad=0.06)
            cbar_overlay.set_label(
                "Average Passenger Occupancy", color="white", fontsize=11
            )
            cbar_overlay.ax.yaxis.set_tick_params(color="white", labelsize=9)
            plt.setp(
                cbar_overlay.ax.get_yticklabels(), color="white", fontsize=9
            )
            cbar_overlay.outline.set_edgecolor("white")
            cbar_overlay.formatter = FormatStrFormatter("%.1f")
            cbar_overlay.update_ticks()
            # Show full simulation axes (0..50, 0..30); hall-aligned extent creates dark side padding.
            ax4.set_xlim(0.0, WORLD_W)
            ax4.set_ylim(0.0, WORLD_H)
            ax4.set_xlabel("Airport Width Position (m)", color="white", fontsize=11)
            ax4.set_ylabel("Airport Length Position (m)", color="white", fontsize=11)
            ax4.xaxis.set_major_locator(MultipleLocator(5.0))
            ax4.yaxis.set_major_locator(MultipleLocator(5.0))
            ax4.xaxis.set_major_formatter(FormatStrFormatter("%d"))
            ax4.yaxis.set_major_formatter(FormatStrFormatter("%d"))
            ax4.tick_params(axis="x", colors="white", labelsize=9)
            ax4.tick_params(axis="y", colors="white", labelsize=9)
            for spine in ax4.spines.values():
                spine.set_color("white")
            ax4.set_title(
                "Spatial Congestion Heatmap Over Airport Layout - Check in Degarded Scenario",
                fontsize=16,
                color="white",
            )
            fig4.savefig(heatmap_png_path, dpi=180, bbox_inches="tight", pad_inches=0.15)
            fig4.savefig(heatmap_pdf_path, bbox_inches="tight", pad_inches=0.15)
            plt.close(fig4)
            print("HOTSPOT LAYOUT OVERLAY SAVED:", heatmap_png_path, flush=True)
            print("HOTSPOT LAYOUT OVERLAY SAVED:", heatmap_pdf_path, flush=True)
    except ImportError:
        print("\n(matplotlib not installed; skipping research plots)")

    print(
        f"Running {scenario} scenario "
        f"(check-in service {NORMAL_CI_SERVICE_TIME:.0f}s before "
        f"{DEGRADATION_START_TIME:.0f}s, "
        f"{DEGRADED_CI_SERVICE_TIME:.0f}s after)",
        flush=True,
    )
    print(f"Current simulation time: {final_sim_time:.1f} sec")
    print(
        "Passenger stats - generated: "
        f"{total_passengers_generated}, exited: {total_passengers_exited}, remaining: {passengers_remaining}"
    )
    print("Total passengers who visited display:", display_visit_count)
    print("\n--- RUN METRICS ---")
    print("Total completed (security):", completed_security_passengers)
    print("Final throughput (last value):", throughput_series[-1] if throughput_series else 0)
    print("\nSaved output files:")
    print("JSON:", metrics_path)
    print("JSON:", timeseries_json_path)
    print("CSV:", timeseries_csv_path)
    print("JSON:", plotdata_json_path)
    print("PNG:", throughput_png_path)
    print("PDF:", throughput_pdf_path)
    print("PNG:", queue_png_path)
    print("PDF:", queue_pdf_path)
    print("PNG:", cumulative_png_path)
    print("PDF:", cumulative_pdf_path)
    print("PNG:", heatmap_png_path)
    print("PDF:", heatmap_pdf_path)

    # Keep window open until user presses ESC or closes the window (visual runs only).
    if not HEADLESS_MODE:
        if sim.agent_count() == 0:
            hud_done = font.render(
                "Simulation complete. Press ESC or close window to exit.", True, (220, 220, 220)
            )
        else:
            hud_done = font.render(
                f"Stopped at iteration cap. n={sim.agent_count()} remaining. Press ESC to exit.",
                True,
                (220, 220, 220),
            )
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
            screen.fill((28, 30, 38))
            if hall_points:
                pts = [world_to_screen(x, y) for x, y in hall_points]
                pygame.draw.polygon(screen, (60, 65, 85), pts)
                pygame.draw.polygon(screen, (100, 110, 130), pts, 2)
            screen.blit(hud_done, (10, 10))
            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    main()


