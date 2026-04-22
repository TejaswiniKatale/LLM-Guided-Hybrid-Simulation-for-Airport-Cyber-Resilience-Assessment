"""
Airport check-in layout builder using JuPedSim.

Layout-only module:
- Builds geometry
- Builds stage coordinates
- No passengers spawned here
"""

from __future__ import annotations

import jupedsim as jps
from shapely.geometry import Polygon

# Vertical spacing between queue slots (counter and security); reduced so people stand closer
QUEUE_SLOT_SPACING = 0.20
# Gap below counter (blue service box) so first queue slot is clearly below it
COUNTER_QUEUE_TOP_GAP = 0.5

# Waiting-area spacing (serpentine lanes)
WAITING_SLOT_SPACING = 0.40  # distance between spots along a lane
WAITING_ROW_SPACING = 0.35   # distance between lanes (rows)

# ------------------------------------------------
# Serpentine waiting area (snake queue) – row-wise, below counter queues
# ------------------------------------------------
def build_serpentine_waiting_positions(
    *,
    top_row_y: float,
) -> list[tuple[float, float]]:
    waiting_positions: list[tuple[float, float]] = []

    slot_spacing = WAITING_SLOT_SPACING  # distance between waiting spots along a lane
    # Increase waiting spots: +4 on each side (total +8)
    num_slots_per_row = 34
    num_rows = 4
    # Center lanes horizontally in a narrower (left-shifted) hall region and nudge further left
    hall_width = 20.0
    total_width = (num_slots_per_row - 1) * slot_spacing
    base_x = (hall_width - total_width) / 2.0 - 1.0
    # Distance between waiting lanes (rows)
    row_spacing = WAITING_ROW_SPACING

    x_positions = [base_x + slot_spacing * i for i in range(num_slots_per_row)]
    # Ensure waiting positions stay inside the hall (hall left x ≈ 3.0)
    min_x = min(x_positions)
    hall_left_x = 2.99
    desired_min_x = hall_left_x + 0.2
    if min_x < desired_min_x:
        shift = desired_min_x - min_x
        x_positions = [x + shift for x in x_positions]
    y_positions = [top_row_y - row_spacing * j for j in range(num_rows)]

    # First row = front (closest to counters, highest y); last row = queue entry (lowest y)
    # Start serpentine from the leftmost column so the first cell is the left-column first slot.
    for row_idx, y in enumerate(y_positions):
        x_order = x_positions if row_idx % 2 == 0 else list(reversed(x_positions))
        for x in x_order:
            waiting_positions.append((x, y))

    return waiting_positions

# ------------------------------------------------
# Security waiting area (below security queue – lanes like ticket counters)
# Order from top: security counter -> security queue -> security waiting lanes
# ------------------------------------------------
def build_security_waiting_positions(base_y: float | None = None) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    # Same as check-in: use the same waiting-area spacings
    slot_spacing = WAITING_SLOT_SPACING
    num_slots_per_row = 34
    num_rows = 4
    hall_width = 20.0
    total_width = (num_slots_per_row - 1) * slot_spacing
    base_x = (hall_width - total_width) / 2.0 - 1.0
    row_spacing = WAITING_ROW_SPACING
    if base_y is None:
        last_queue_slot_y_ref = 22.0 - 4 * QUEUE_SLOT_SPACING
        base_y = last_queue_slot_y_ref - 4.0 - 1.0
    x_positions = [base_x + slot_spacing * i for i in range(num_slots_per_row)]
    # Ensure security waiting positions stay inside the hall (hall left x ≈ 3.0)
    min_x = min(x_positions)
    hall_left_x = 2.99
    desired_min_x = hall_left_x + 0.2
    if min_x < desired_min_x:
        shift = desired_min_x - min_x
        x_positions = [x + shift for x in x_positions]
    y_positions = [base_y - row_spacing * j for j in range(num_rows)]
    # Match check-in: start serpentine from the leftmost column
    for row_idx, y in enumerate(y_positions):
        x_order = x_positions if row_idx % 2 == 0 else list(reversed(x_positions))
        for x in x_order:
            positions.append((x, y))
    return positions

# ------------------------------------------------
# Build layout
# ------------------------------------------------
def build_airport_checkin_layout() -> dict:

    # Hall geometry (inset so boundary surrounds the layout, not the whole world)
    HALL_TOP_Y = 30.0
    hall_outer = Polygon(
        [
            (2.99, 0.0),
            (47.0, 0.0),
            (47.0, HALL_TOP_Y),
            (2.99, HALL_TOP_Y),
        ]
    )

    # ------------------------------------------------
    # Check-in Counters (above waiting lanes; shifted toward left, only 2 counters)
    # ------------------------------------------------
    # Place ticket counters just above the waiting area (lower than default 13.0).
    # Their x-positions will be aligned with the entrances defined below.
    # We'll initialize with temporary values and overwrite once entrances are defined.
    counters = {
        "Counter1": (8.0, 6.0),
        "Counter2": (16.0, 6.0),
    }

    # ------------------------------------------------
    # Entrances (doors at bottom wall)
    # ------------------------------------------------
    # Doors higher, closer to the waiting area but still below the first waiting row
    door_y = 1.8  # 0.5 units below 2.3 (0.7 below original 2.5)
    # Align entrances with the parking segment used for spawning:
    # parking spans the left third of the hall: [hall_left_x, park_x_right]
    hall_left_x = 2.99
    hall_right_x = 47.0
    hall_width = hall_right_x - hall_left_x
    park_x_left = hall_left_x
    park_x_right = hall_left_x + hall_width / 3.0
    # Two entrances at the centers of the left and right halves of the parking segment,
    # so distance(left edge -> left entrance) == distance(right entrance -> right edge)
    door1_x = park_x_left + (park_x_right - park_x_left) / 4.0
    door2_x = park_x_left + 3.0 * (park_x_right - park_x_left) / 4.0
    entrances = {
        "Door1": (door1_x, door_y),
        "Door2": (door2_x, door_y),
    }

    # Now align ticket counters horizontally above the entrances
    # Shift ticket counters up by 5 units total (includes +2 per latest request)
    ticket_counters_shift_up = 5.0
    counters = {
        "Counter1": (door1_x, 6.0 + ticket_counters_shift_up),
        "Counter2": (door2_x, 6.0 + ticket_counters_shift_up),
    }

    # Barrier between parking and hall: only pass through the two door openings
    park_y_top = door_y - 0.3
    barrier_y_lo = park_y_top - 0.1
    barrier_y_hi = park_y_top + 0.1
    door_gap_half = 0.3  # half-width of opening at each door
    barrier_left = Polygon([
        (hall_left_x, barrier_y_lo),
        (door1_x - door_gap_half, barrier_y_lo),
        (door1_x - door_gap_half, barrier_y_hi),
        (hall_left_x, barrier_y_hi),
    ])
    barrier_mid = Polygon([
        (door1_x + door_gap_half, barrier_y_lo),
        (door2_x - door_gap_half, barrier_y_lo),
        (door2_x - door_gap_half, barrier_y_hi),
        (door1_x + door_gap_half, barrier_y_hi),
    ])
    barrier_right = Polygon([
        (door2_x + door_gap_half, barrier_y_lo),
        (47.0, barrier_y_lo),
        (47.0, barrier_y_hi),
        (door2_x + door_gap_half, barrier_y_hi),
    ])
    hall_polygon = hall_outer.difference(barrier_left).difference(barrier_mid).difference(barrier_right)

    corridor_left = Polygon([
        (door1_x - 0.5, door_y + 0.2),
        (door1_x + 0.5, door_y + 0.2),
        (door1_x + 0.5, door_y + 2.0),
        (door1_x - 0.5, door_y + 2.0),
    ])

    corridor_right = Polygon([
        (door2_x - 0.5, door_y + 0.2),
        (door2_x + 0.5, door_y + 0.2),
        (door2_x + 0.5, door_y + 2.0),
        (door2_x - 0.5, door_y + 2.0),
    ])

    hall_polygon = hall_polygon.difference(corridor_left).difference(corridor_right)

    # Vertical line 1 unit from right edge of parking (draw-only, full height)
    wall_line_x = park_x_right + 1.0
    parking_wall_line = [(wall_line_x, 0.0), (wall_line_x, HALL_TOP_Y)]
    # Second vertical line 5 units to the right of the first (full height)
    wall_line2_x = wall_line_x + 5.0
    parking_wall_line_2 = [(wall_line2_x, 0.0), (wall_line2_x, HALL_TOP_Y)]

    # ------------------------------------------------
    # Counter vertical queues
    # ------------------------------------------------
    counter_queue_lists = []
    counter_queue_slots = {}

    for idx, counter_name in enumerate(counters, start=1):

        cx, cy = counters[counter_name]
        one_queue = []

        # 4 queue positions per counter, in a straight vertical line below the service box
        for slot in range(1, 5):  # q{idx}-1 .. q{idx}-4
            point = (cx, cy - COUNTER_QUEUE_TOP_GAP - (slot - 1) * QUEUE_SLOT_SPACING)

            counter_queue_slots[f"q{idx}-{slot}"] = point
            one_queue.append(point)

        counter_queue_lists.append(one_queue)

    # ------------------------------------------------
    # Waiting area
    # ------------------------------------------------
    # Ensure waiting spots are ONLY below the last counter-queue slot (never above it).
    last_counter_queue_y = min(y for (_x, y) in counter_queue_slots.values())
    gap_below_last_queue = 0.8
    waiting_top_row_y = last_counter_queue_y - gap_below_last_queue
    waiting_positions = build_serpentine_waiting_positions(top_row_y=waiting_top_row_y)
    waiting_map = {f"w{i+1}": p for i, p in enumerate(waiting_positions)}
    # Entry point: agents go here first and stand in the waiting area, then join the serpentine queue
    waiting_area_entry = (7.5, waiting_top_row_y)

    # ------------------------------------------------
    # Information desks (above ticket counters; left at parking left edge, right at parking right edge)
    # ------------------------------------------------
    # Shift info desks upward (and keep queues/waiting aligned)
    INFO_DESK_SHIFT_UP = 4.0
    info_desk_y = 12.5 + INFO_DESK_SHIFT_UP  # 0.5 units down from 13.0, then shifted up

    # Visual alignment tweak: shift ONLY the LEFT info desk further left.
    # This keeps the left label box, left service box, and left queue slots together.
    #
    # Clamp so we never place the LEFT info-desk service/queue waypoints outside
    # the hall walkable area (JuPedSim rejects those points).
    INFO_DESK_LEFT_SHIFT_DX_DESIRED = -1.5
    # Margin inside hall boundary for safety.
    INFO_DESK_LEFT_SHIFT_MARGIN = 0.05
    # Service waypoint x is `info_desk_left_x + 1.05`. Keep it inside hall_left_x + margin.
    info_desk_left_shift_dx_min = (hall_left_x + INFO_DESK_LEFT_SHIFT_MARGIN - 1.05) - park_x_left
    INFO_DESK_LEFT_SHIFT_DX = max(INFO_DESK_LEFT_SHIFT_DX_DESIRED, info_desk_left_shift_dx_min)
    info_desk_left_x = park_x_left + INFO_DESK_LEFT_SHIFT_DX

    information_desks = {
        "left": (info_desk_left_x, info_desk_y),
        "right": (park_x_right, info_desk_y),
    }

    # Service spot = center of the vertical border box (where passenger stands during info desk service)
    info_left_service_center_x = info_desk_left_x + 1.05   # center of left border box
    info_right_service_center_x = park_x_right - 1.05
    info_desk_service_positions = [
        (info_left_service_center_x, info_desk_y),
        (info_right_service_center_x, info_desk_y),
    ]

    # Two horizontal queue positions for the left info desk: start from the border (service) box, not the desk
    # Right edge of left service box ≈ park_x_left + (label half-width + service width) in world
    info_left_service_right_x = info_desk_left_x + 1.4  # queue starts here (at the border box)
    info_left_queue_y = info_desk_y - 0.55 + 1.0 - 0.5   # 1 unit up then 0.5 down
    information_left_queue_slots = [
        (info_left_service_right_x, info_left_queue_y),
        (info_left_service_right_x + QUEUE_SLOT_SPACING, info_left_queue_y),
    ]  # horizontal: slot 1 at border box, slot 2 one spacing to the right

    # Two horizontal queue positions for the right info desk: same logic, queue starts from border box (left edge)
    info_right_service_left_x = park_x_right - 1.4  # left edge of right service box
    info_right_queue_y = info_left_queue_y  # same y as left desk queue
    information_right_queue_slots = [
        (info_right_service_left_x, info_right_queue_y),
        (info_right_service_left_x - QUEUE_SLOT_SPACING, info_right_queue_y),
    ]  # horizontal: slot 1 at border box, slot 2 one spacing to the left

    # Waiting positions in front of info desks (before being assigned to left/right queue)
    info_desk_waiting_y = 11.0 + INFO_DESK_SHIFT_UP
    info_desk_waiting_positions = [
        ((park_x_left + park_x_right) / 2.0 - 0.5, info_desk_waiting_y),
        ((park_x_left + park_x_right) / 2.0 - 0.15, info_desk_waiting_y),
        ((park_x_left + park_x_right) / 2.0 + 0.15, info_desk_waiting_y),
        ((park_x_left + park_x_right) / 2.0 + 0.5, info_desk_waiting_y),
    ]

    # ------------------------------------------------
    # SECURITY AREA (counter at top, queue below, same distance counter–waiting as ticket counters)
    # ------------------------------------------------
    # Ticket: counter y=6.0, waiting first row ~4.6 → distance 1.4. Match that for security.
    SECURITY_COUNTER_Y = 18.0  # base y for display placement
    # Shift only the security counters + their waiting area up (do not move displays).
    # +1.0 extra unit per latest request (was 4.0).
    SECURITY_SHIFT_UP = 5.0
    security_counter_y = SECURITY_COUNTER_Y + SECURITY_SHIFT_UP
    ticket_counter_y = 6.0 + ticket_counters_shift_up
    # Keep security waiting placement behavior as it was before (do not couple it
    # to the ticket waiting-area generation logic, which may change independently).
    ticket_waiting_top_y = 4.6  # from prior build_serpentine_waiting_positions placement
    counter_to_waiting_distance = ticket_counter_y - ticket_waiting_top_y  # 1.4
    # Base security waiting top aligned like ticket; then shift security waiting area up by 4 units.
    security_waiting_top_y = security_counter_y - counter_to_waiting_distance + 4.0

    # Align security displays horizontally with ticket counters (same x positions)
    security_points_coords = {
        "Security1": (counters["Counter1"][0], security_counter_y),
        "Security2": (counters["Counter2"][0], security_counter_y),
    }

    NUM_SECURITY_SLOTS = 2  # same count as ticket counter queues

    security_queue_lists = []
    security_queue_slots = {}

    for idx, (name, (sx, sy)) in enumerate(security_points_coords.items(), start=1):
        one_queue = []
        # First queue slot below the blue border box (gap like ticket counters), then spacing between slots
        for slot in range(1, NUM_SECURITY_SLOTS + 1):
            point = (sx, sy - COUNTER_QUEUE_TOP_GAP - (slot - 1) * QUEUE_SLOT_SPACING)
            security_queue_slots[f"sec_q{idx}-{slot}"] = point
            one_queue.append(point)
        security_queue_lists.append(one_queue)

    # ------------------------------------------------
    # SECURITY WAITING AREA (same distance below security queues as ticket waiting below ticket queues)
    # ------------------------------------------------
    security_waiting_positions = build_security_waiting_positions(base_y=security_waiting_top_y)
    security_waiting_map = {f"sec_w{i+1}": p for i, p in enumerate(security_waiting_positions)}

    # ------------------------------------------------
    # EXIT after security (vertical block near top-left, shifted 3 units right,
    # then 0.3 units further left per latest request)
    # ------------------------------------------------
    exit_top_y = HALL_TOP_Y
    exit_bottom_y = HALL_TOP_Y - 3.0
    # Place SecurityExit block so it STRADDLES the first vertical wall line (wall_line_x),
    # creating an actual opening through that wall while keeping the block near the corridor.
    exit_left_x = wall_line_x - 0.5
    exit_right_x = exit_left_x + 1.0  # narrow vertical block
    exit_center_x = (exit_left_x + exit_right_x) / 2.0
    exit_polygon = Polygon(
        [
            (exit_left_x, exit_bottom_y),
            (exit_right_x, exit_bottom_y),
            (exit_right_x, exit_top_y),
            (exit_left_x, exit_top_y),
        ]
    )

    # With the vertical wall at wall_line_x, avoid placing points exactly on the wall line.
    # Shift slightly into the corridor (to the right).
    exit_queue_positions = [(exit_center_x + 0.35, exit_bottom_y - 0.5)]
    security_exit_center = (exit_center_x + 0.35, (exit_bottom_y + exit_top_y) / 2.0)

    # ------------------------------------------------
    # Flight information displays (vertical rectangles above security, below exit)
    # ------------------------------------------------
    # Make displays vertical, anchored near the very top of the hall
    # Reduce total display column height by 1.0 unit and move the pair up
    display_height = 3.0
    display_y_top = HALL_TOP_Y - 0.5
    display_y_bottom = display_y_top - display_height
    # Single vertical flight information display.
    # Keep the grey "front" boxes (used for agent display spots) fixed, but
    # shift the blue screens left by 1.5 units for visual alignment.
    display_center_base = park_x_left + 0.6
    display_shift_blue_dx = -1.5
    display_centers_grey = [display_center_base]
    display_centers_blue = [display_center_base + display_shift_blue_dx]
    # Make the displays thinner horizontally
    display_half_width = 0.4
    # Blue screens (rendering only)
    flight_display_rects: list[list[tuple[float, float]]] = []
    # Grey fronts (used for display waypoints / agent spots)
    flight_display_rects_grey: list[list[tuple[float, float]]] = []
    # Two screens per side (top & bottom) with a clear vertical gap between them
    gap = 1.0  # distance between top and bottom screens (kept the same)
    span = display_y_top - display_y_bottom
    panel_height = (span - gap) / 2.0
    bottom_top_y = display_y_bottom + panel_height
    top_bottom_y = display_y_top - panel_height

    for cx in display_centers_grey:
        # Top screen
        flight_display_rects_grey.append(
            [
                (cx - display_half_width, top_bottom_y),
                (cx + display_half_width, top_bottom_y),
                (cx + display_half_width, display_y_top),
                (cx - display_half_width, display_y_top),
            ]
        )
        # Bottom screen
        flight_display_rects_grey.append(
            [
                (cx - display_half_width, display_y_bottom),
                (cx + display_half_width, display_y_bottom),
                (cx + display_half_width, bottom_top_y),
                (cx - display_half_width, bottom_top_y),
            ]
        )

    # Rebuild for blue screens separately so we can shift them without moving grey fronts.
    flight_display_rects = []
    for cx in display_centers_blue:
        # Top screen
        flight_display_rects.append(
            [
                (cx - display_half_width, top_bottom_y),
                (cx + display_half_width, top_bottom_y),
                (cx + display_half_width, display_y_top),
                (cx - display_half_width, display_y_top),
            ]
        )
        # Bottom screen
        flight_display_rects.append(
            [
                (cx - display_half_width, display_y_bottom),
                (cx + display_half_width, display_y_bottom),
                (cx + display_half_width, bottom_top_y),
                (cx - display_half_width, bottom_top_y),
            ]
        )

    # Grey box in front of each blue screen (wider, x shifted so it sits in front)
    # For the left column: grey boxes sit slightly in front of the blue screens.
    flight_display_grey_rects: list[list[tuple[float, float]]] = []
    grey_extra_width = 0.5  # extend 0.5 units on each side
    # Build grey fronts from the fixed (non-shifted) rectangles.
    flight_display_rects_grey_fronts: list[list[tuple[float, float]]] = []
    for rect in flight_display_rects_grey:
        # Shift x to the right just a little so the grey box remains in front.
        shifted = [(x + 0.2, y) for x, y in rect]
        xs = [x for x, _ in shifted]
        ys = [y for _, y in shifted]
        min_x = min(xs) - grey_extra_width
        max_x = max(xs) + grey_extra_width
        min_y = min(ys)
        max_y = max(ys)
        wider_rect = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ]
        flight_display_rects_grey_fronts.append(wider_rect)

    flight_display_grey_rects = flight_display_rects_grey_fronts

    # ------------------------------------------------
    # Shops heading (vertical box to the right of the black line)
    # ------------------------------------------------
    shops_offset_x = 6.0  # shifted 1 unit further right from the black vertical line
    shops_center_x = wall_line_x + shops_offset_x
    # Place shops band at the extreme top of the hall
    shops_width = 1.0
    shops_height = 4.5
    shops_y_top = HALL_TOP_Y
    shops_y_bottom = shops_y_top - shops_height
    shops_label_y_bottom = shops_y_bottom - 1.0
    shops_label_y_top = shops_label_y_bottom + shops_height
    shops_label_rect = [
        (shops_center_x - shops_width / 2.0, shops_label_y_bottom),
        (shops_center_x + shops_width / 2.0, shops_label_y_bottom),
        (shops_center_x + shops_width / 2.0, shops_label_y_top),
        (shops_center_x - shops_width / 2.0, shops_label_y_top),
    ]

    # Two small vertical blocks in front of "Shops area enter": Enter (top) and Exit (bottom), same x (left side)
    entrance_block_offset_y = 1.2  # move blocks slightly down
    entrance_block_height = 2.0
    entrance_block_width = 0.6
    # ------------------------------------------------
    # Common grey title-box alignment (all sections)
    # ------------------------------------------------
    # Align the left and right edges of ALL grey title boxes (Shops/Food/Restrooms/All Gates)
    # to the same x-range (matching the All Gates title box placement).
    title_box_width = 0.60
    title_box_left_x = wall_line2_x + 0.15 + entrance_block_width + 0.20
    title_box_right_x = title_box_left_x + title_box_width

    # Override Shops title box to use the common x-range
    shops_label_rect = [
        (title_box_left_x, shops_label_y_bottom),
        (title_box_right_x, shops_label_y_bottom),
        (title_box_right_x, shops_label_y_top),
        (title_box_left_x, shops_label_y_top),
    ]
    entrance_center_y = (shops_label_y_bottom + shops_label_y_top) / 2.0 + entrance_block_offset_y
    # Place all Enter/Exit blocks slightly to the RIGHT of the second corridor wall line,
    # so they act as openings through that wall.
    enter_center_x = wall_line2_x + 0.15 + entrance_block_width / 2.0
    # Keep Shops Enter/Exit as two separate stacked green blocks, matching other areas.
    vertical_gap = 0.4
    exit_center_y = entrance_center_y - (entrance_block_height + vertical_gap)
    enter_block_rect = [
        (enter_center_x - entrance_block_width / 2.0, entrance_center_y - entrance_block_height / 2.0),
        (enter_center_x + entrance_block_width / 2.0, entrance_center_y - entrance_block_height / 2.0),
        (enter_center_x + entrance_block_width / 2.0, entrance_center_y + entrance_block_height / 2.0),
        (enter_center_x - entrance_block_width / 2.0, entrance_center_y + entrance_block_height / 2.0),
    ]
    exit_block_rect = [
        (enter_center_x - entrance_block_width / 2.0, exit_center_y - entrance_block_height / 2.0),
        (enter_center_x + entrance_block_width / 2.0, exit_center_y - entrance_block_height / 2.0),
        (enter_center_x + entrance_block_width / 2.0, exit_center_y + entrance_block_height / 2.0),
        (enter_center_x - entrance_block_width / 2.0, exit_center_y + entrance_block_height / 2.0),
    ]

    # One shop box (Cosmetics) to the right of "Shops area enter"
    cosmetics_offset_x = 5.0  # distance to the right of shops center (shifted further right)
    cosmetics_center_x = shops_center_x + cosmetics_offset_x
    # Make cosmetics box horizontal: wider and not as tall
    cosmetics_width = 3.0
    cosmetics_height = 1.2
    # Place cosmetics so its top edge touches the top of the hall (very top of the screen)
    cosmetics_y_top = HALL_TOP_Y
    cosmetics_y_bottom = cosmetics_y_top - cosmetics_height
    cosmetics_y_center = (cosmetics_y_top + cosmetics_y_bottom) / 2.0
    cosmetics_rect = [
        (cosmetics_center_x - cosmetics_width / 2.0, cosmetics_y_bottom),
        (cosmetics_center_x + cosmetics_width / 2.0, cosmetics_y_bottom),
        (cosmetics_center_x + cosmetics_width / 2.0, cosmetics_y_top),
        (cosmetics_center_x - cosmetics_width / 2.0, cosmetics_y_top),
    ]

    # Second shop box (Perfumes) 5 units to the right of Cosmetics, same level and size
    perfumes_center_x = cosmetics_center_x + 5.0
    perfumes_width = cosmetics_width
    perfumes_height = cosmetics_height
    perfumes_y_bottom = cosmetics_y_bottom
    perfumes_y_top = cosmetics_y_top
    perfumes_rect = [
        (perfumes_center_x - perfumes_width / 2.0, perfumes_y_bottom),
        (perfumes_center_x + perfumes_width / 2.0, perfumes_y_bottom),
        (perfumes_center_x + perfumes_width / 2.0, perfumes_y_top),
        (perfumes_center_x - perfumes_width / 2.0, perfumes_y_top),
    ]

    # Third shop box (Electronics) 5 units to the right of Perfumes, same level and size
    electronics_center_x = perfumes_center_x + 5.0
    electronics_width = cosmetics_width
    electronics_height = cosmetics_height
    electronics_y_bottom = cosmetics_y_bottom
    electronics_y_top = cosmetics_y_top
    electronics_rect = [
        (electronics_center_x - electronics_width / 2.0, electronics_y_bottom),
        (electronics_center_x + electronics_width / 2.0, electronics_y_bottom),
        (electronics_center_x + electronics_width / 2.0, electronics_y_top),
        (electronics_center_x - electronics_width / 2.0, electronics_y_top),
    ]

    # Fourth shop box (Books) 5 units to the right of Electronics, same level and size
    books_center_x = electronics_center_x + 5.0
    books_width = cosmetics_width
    books_height = cosmetics_height
    books_y_bottom = cosmetics_y_bottom
    books_y_top = cosmetics_y_top
    books_rect = [
        (books_center_x - books_width / 2.0, books_y_bottom),
        (books_center_x + books_width / 2.0, books_y_bottom),
        (books_center_x + books_width / 2.0, books_y_top),
        (books_center_x - books_width / 2.0, books_y_top),
    ]

    # ------------------------------------------------
    # Food court section (mirrors shopping area, placed below)
    # ------------------------------------------------
    food_offset_y = -6.0  # shift food court band below shopping band
    food_center_x = shops_center_x
    food_width = shops_width
    # Reduce the overall Food Court area height (pull the Shops→Food divider down)
    food_height = max(2.5, shops_height - 3.0)
    food_band_y_bottom = shops_y_bottom + food_offset_y
    food_band_y_top = food_band_y_bottom + food_height
    food_label_y_bottom = food_band_y_bottom - 1.0
    food_label_y_top = food_label_y_bottom + food_height
    # Shift the Food Court heading box + its enter/exit blocks down
    food_heading_shift_down = 2.0
    food_label_y_bottom -= food_heading_shift_down
    food_label_y_top -= food_heading_shift_down
    # Keep Food title box aligned with other grey title boxes
    food_label_rect = [
        (title_box_left_x, food_label_y_bottom),
        (title_box_right_x, food_label_y_bottom),
        (title_box_right_x, food_label_y_top),
        (title_box_left_x, food_label_y_top),
    ]

    # Two vertical blocks in front of Food heading: Enter (top) and Exit (bottom) on left side
    food_block_offset_y = entrance_block_offset_y  # reuse same relative offset
    food_block_height = entrance_block_height
    food_block_width = entrance_block_width
    food_center_y = (food_label_y_bottom + food_label_y_top) / 2.0 + food_block_offset_y
    food_enter_center_x = wall_line2_x + 0.15 + food_block_width / 2.0
    food_vertical_gap = vertical_gap
    food_exit_center_y = food_center_y - (food_block_height + food_vertical_gap)

    food_enter_block_rect = [
        (food_enter_center_x - food_block_width / 2.0, food_center_y - food_block_height / 2.0),
        (food_enter_center_x + food_block_width / 2.0, food_center_y - food_block_height / 2.0),
        (food_enter_center_x + food_block_width / 2.0, food_center_y + food_block_height / 2.0),
        (food_enter_center_x - food_block_width / 2.0, food_center_y + food_block_height / 2.0),
    ]

    food_exit_block_rect = [
        (food_enter_center_x - food_block_width / 2.0, food_exit_center_y - food_block_height / 2.0),
        (food_enter_center_x + food_block_width / 2.0, food_exit_center_y - food_block_height / 2.0),
        (food_enter_center_x + food_block_width / 2.0, food_exit_center_y + food_block_height / 2.0),
        (food_enter_center_x - food_block_width / 2.0, food_exit_center_y + food_block_height / 2.0),
    ]

    # Four horizontal food court units (mirroring shops): Burgers, Pizza, Coffee, Desserts
    food_unit_width = cosmetics_width
    food_unit_height = cosmetics_height
    # Place food counters so their top edge touches the Shops/Food horizontal divider
    food_y_center = food_band_y_top - food_unit_height / 2.0
    food_y_bottom = food_y_center - food_unit_height / 2.0
    food_y_top = food_y_center + food_unit_height / 2.0

    burgers_center_x = cosmetics_center_x
    burgers_rect = [
        (burgers_center_x - food_unit_width / 2.0, food_y_bottom),
        (burgers_center_x + food_unit_width / 2.0, food_y_bottom),
        (burgers_center_x + food_unit_width / 2.0, food_y_top),
        (burgers_center_x - food_unit_width / 2.0, food_y_top),
    ]

    pizza_center_x = perfumes_center_x
    pizza_rect = [
        (pizza_center_x - food_unit_width / 2.0, food_y_bottom),
        (pizza_center_x + food_unit_width / 2.0, food_y_bottom),
        (pizza_center_x + food_unit_width / 2.0, food_y_top),
        (pizza_center_x - food_unit_width / 2.0, food_y_top),
    ]

    coffee_center_x = electronics_center_x
    coffee_rect = [
        (coffee_center_x - food_unit_width / 2.0, food_y_bottom),
        (coffee_center_x + food_unit_width / 2.0, food_y_bottom),
        (coffee_center_x + food_unit_width / 2.0, food_y_top),
        (coffee_center_x - food_unit_width / 2.0, food_y_top),
    ]

    desserts_center_x = books_center_x
    desserts_rect = [
        (desserts_center_x - food_unit_width / 2.0, food_y_bottom),
        (desserts_center_x + food_unit_width / 2.0, food_y_bottom),
        (desserts_center_x + food_unit_width / 2.0, food_y_top),
        (desserts_center_x - food_unit_width / 2.0, food_y_top),
    ]

    # ------------------------------------------------
    # Restrooms section (same style as food court, placed below; only 3 restrooms)
    # ------------------------------------------------
    # Shift the entire Restrooms area further down
    rest_offset_y = -9.0  # below food court band (3 units lower)
    rest_center_x = shops_center_x
    rest_width = shops_width
    # Reduce the overall Restrooms area height (this also pulls the Food→Rest divider down)
    rest_height = max(2.5, shops_height - 2.0)
    rest_band_y_bottom = food_band_y_bottom + rest_offset_y
    rest_band_y_top = rest_band_y_bottom + rest_height
    # Move the Restrooms grey title box down
    rest_label_y_bottom = rest_band_y_bottom - 1.0 - 2.0
    rest_label_y_top = rest_label_y_bottom + rest_height
    # Keep Restrooms title box aligned with other grey title boxes
    rest_label_rect = [
        (title_box_left_x, rest_label_y_bottom),
        (title_box_right_x, rest_label_y_bottom),
        (title_box_right_x, rest_label_y_top),
        (title_box_left_x, rest_label_y_top),
    ]

    # Three restroom blocks (purple). Make their TOP edge touch the horizontal divider above restrooms.
    rest_unit_width = food_unit_width
    rest_unit_height = food_unit_height
    rest_y_center = rest_band_y_top - rest_unit_height / 2.0
    rest_unit_y_bottom = rest_y_center - rest_unit_height / 2.0
    rest_unit_y_top = rest_y_center + rest_unit_height / 2.0

    # Restrooms Enter/Exit blocks (stacked on left, same as shops/food)
    rest_block_offset_y = entrance_block_offset_y
    rest_block_height = entrance_block_height
    rest_block_width = entrance_block_width
    # Pull the blocks slightly down so the restroom section looks even after top-aligning the purple boxes.
    # Pull both Restrooms Enter/Exit blocks down by 2 units
    rest_center_y = rest_y_center - 0.2 - 2.0
    rest_enter_center_x = wall_line2_x + 0.15 + rest_block_width / 2.0
    rest_exit_center_y = rest_center_y - (rest_block_height + vertical_gap)
    rest_enter_block_rect = [
        (rest_enter_center_x - rest_block_width / 2.0, rest_center_y - rest_block_height / 2.0),
        (rest_enter_center_x + rest_block_width / 2.0, rest_center_y - rest_block_height / 2.0),
        (rest_enter_center_x + rest_block_width / 2.0, rest_center_y + rest_block_height / 2.0),
        (rest_enter_center_x - rest_block_width / 2.0, rest_center_y + rest_block_height / 2.0),
    ]
    rest_exit_block_rect = [
        (rest_enter_center_x - rest_block_width / 2.0, rest_exit_center_y - rest_block_height / 2.0),
        (rest_enter_center_x + rest_block_width / 2.0, rest_exit_center_y - rest_block_height / 2.0),
        (rest_enter_center_x + rest_block_width / 2.0, rest_exit_center_y + rest_block_height / 2.0),
        (rest_enter_center_x - rest_block_width / 2.0, rest_exit_center_y + rest_block_height / 2.0),
    ]

    # Three restroom blocks (evenly spaced across the row)

    # Spread the 3 restrooms evenly across the row (left -> right)
    rest_left_x = burgers_center_x
    rest_right_x = desserts_center_x
    rest1_center_x = rest_left_x
    rest2_center_x = (rest_left_x + rest_right_x) / 2.0
    rest3_center_x = rest_right_x
    rest1_rect = [
        (rest1_center_x - rest_unit_width / 2.0, rest_unit_y_bottom),
        (rest1_center_x + rest_unit_width / 2.0, rest_unit_y_bottom),
        (rest1_center_x + rest_unit_width / 2.0, rest_unit_y_top),
        (rest1_center_x - rest_unit_width / 2.0, rest_unit_y_top),
    ]
    rest2_rect = [
        (rest2_center_x - rest_unit_width / 2.0, rest_unit_y_bottom),
        (rest2_center_x + rest_unit_width / 2.0, rest_unit_y_bottom),
        (rest2_center_x + rest_unit_width / 2.0, rest_unit_y_top),
        (rest2_center_x - rest_unit_width / 2.0, rest_unit_y_top),
    ]
    rest3_rect = [
        (rest3_center_x - rest_unit_width / 2.0, rest_unit_y_bottom),
        (rest3_center_x + rest_unit_width / 2.0, rest_unit_y_bottom),
        (rest3_center_x + rest_unit_width / 2.0, rest_unit_y_top),
        (rest3_center_x - rest_unit_width / 2.0, rest_unit_y_top),
    ]

    # ------------------------------------------------
    # Gates section (same style, placed below restrooms)
    # ------------------------------------------------
    # Shift the entire All Gates area downward (toward y=0)
    gates_shift_down = 3.0
    gates_offset_y = -6.0 - gates_shift_down  # below restrooms band
    gates_center_x = shops_center_x
    gates_width = shops_width
    gates_height = shops_height + 2.0
    gates_band_y_bottom = rest_band_y_bottom + gates_offset_y
    gates_band_y_top = gates_band_y_bottom + gates_height

    gates_block_offset_y = entrance_block_offset_y
    gates_block_height = entrance_block_height
    gates_block_width = entrance_block_width
    gates_enter_center_x = wall_line2_x + 0.15 + gates_block_width / 2.0

    # Make the "All Gates" grey box thinner and ~half height, and place it directly
    # to the RIGHT of the Gates Enter block (same style as other area title boxes).
    gates_label_width = 0.60
    gates_label_height = gates_height * 0.50
    gates_label_y_bottom = gates_band_y_bottom - 1.0 - 2.4
    gates_label_y_top = gates_label_y_bottom + gates_label_height
    # Move the title box upward a bit
    gates_label_shift_up = 3.2
    gates_label_y_bottom += gates_label_shift_up
    gates_label_y_top += gates_label_shift_up

    gates_label_left_x = (gates_enter_center_x + gates_block_width / 2.0) + 0.20
    gates_label_right_x = gates_label_left_x + gates_label_width
    gates_label_rect = [
        (gates_label_left_x, gates_label_y_bottom),
        (gates_label_right_x, gates_label_y_bottom),
        (gates_label_right_x, gates_label_y_top),
        (gates_label_left_x, gates_label_y_top),
    ]

    # Shift the Gates Enter block downward (workflow unchanged; waypoints follow layout coords)
    gates_center_y = (gates_label_y_bottom + gates_label_y_top) / 2.0 + gates_block_offset_y - 1.5
    gates_enter_block_rect = [
        (gates_enter_center_x - gates_block_width / 2.0, gates_center_y - gates_block_height / 2.0),
        (gates_enter_center_x + gates_block_width / 2.0, gates_center_y - gates_block_height / 2.0),
        (gates_enter_center_x + gates_block_width / 2.0, gates_center_y + gates_block_height / 2.0),
        (gates_enter_center_x - gates_block_width / 2.0, gates_center_y + gates_block_height / 2.0),
    ]

    # Walking corridor to the right of the black line, up to the vertical heading blocks
    corridor_x_left = wall_line_x
    corridor_x_right = shops_center_x - shops_width / 2.0
    # From just above parking barrier up to top of All Gates band
    corridor_y_bottom = barrier_y_hi
    corridor_y_top = gates_band_y_top
    walking_corridor_rect = [
        (corridor_x_left, corridor_y_bottom),
        (corridor_x_right, corridor_y_bottom),
        (corridor_x_right, corridor_y_top),
        (corridor_x_left, corridor_y_top),
    ]

    # Small horizontal gate blocks (A1..A13 only) in pink, to the right of the "All Gates" heading
    gate_block_width = 1.2
    gate_block_height = 0.6
    gate_block_gap_x = 0.7
    # Horizontal line between restrooms and All Gates, same style (shifted 2 units further down)
    # (also used to align the top A-gate lane)
    gates_horizontal_y = rest_band_y_bottom - 2.0 - gates_shift_down

    # Place A-gate lane so its TOP edge touches the horizontal black line above it.
    # Top edge = gates_blocks_y_center + gate_block_height/2 == gates_horizontal_y
    gates_blocks_y_center = gates_horizontal_y - gate_block_height / 2.0
    gates_blocks_start_x = (gates_center_x + gates_width / 2.0) + 1.0
    gate_block_centers_x = [
        gates_blocks_start_x + i * (gate_block_width + gate_block_gap_x)
        for i in range(13)
    ]
    gates_gate_blocks: dict[str, list[tuple[float, float]]] = {}
    # Row A (top)
    for i, cx in enumerate(gate_block_centers_x, start=1):
        name = f"A{i}"
        gates_gate_blocks[name] = [
            (cx - gate_block_width / 2.0, gates_blocks_y_center - gate_block_height / 2.0),
            (cx + gate_block_width / 2.0, gates_blocks_y_center - gate_block_height / 2.0),
            (cx + gate_block_width / 2.0, gates_blocks_y_center + gate_block_height / 2.0),
            (cx - gate_block_width / 2.0, gates_blocks_y_center + gate_block_height / 2.0),
        ]

    # Horizontal line above the food court, starting from the second vertical line to the right hall boundary
    food_horizontal_y = food_band_y_top
    food_horizontal_line = [
        (wall_line2_x, food_horizontal_y),
        (47.0, food_horizontal_y),
    ]

    # Horizontal line between food court and restrooms, same style
    rest_horizontal_y = rest_band_y_top
    rest_horizontal_line = [
        (wall_line2_x, rest_horizontal_y),
        (47.0, rest_horizontal_y),
    ]

    # Horizontal line between restrooms and All Gates, same style
    gates_horizontal_line = [
        (wall_line2_x, gates_horizontal_y),
        (47.0, gates_horizontal_y),
    ]

    # ------------------------------------------------
    # Corridor vertical walls as real geometry (not just drawing)
    # ------------------------------------------------
    # Enforce: passengers can only move between areas using the corridor between the
    # two vertical black lines. They must not cross those lines except through the
    # designated Enter/Exit blocks (and the SecurityExit block near the top).
    def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda t: t[0])
        merged: list[tuple[float, float]] = [intervals[0]]
        for a, b in intervals[1:]:
            la, lb = merged[-1]
            if a <= lb:
                merged[-1] = (la, max(lb, b))
            else:
                merged.append((a, b))
        return merged

    def _subtract_vertical_wall_with_gaps(
        *,
        x: float,
        thickness: float,
        y0: float,
        y1: float,
        gaps: list[tuple[float, float]],
    ) -> None:
        nonlocal hall_polygon
        gaps = [(max(y0, a), min(y1, b)) for (a, b) in gaps if b > y0 and a < y1]
        gaps = _merge_intervals(gaps)
        cursor = y0
        for a, b in gaps:
            if a > cursor:
                seg = Polygon([
                    (x - thickness / 2.0, cursor),
                    (x + thickness / 2.0, cursor),
                    (x + thickness / 2.0, a),
                    (x - thickness / 2.0, a),
                ])
                hall_polygon = hall_polygon.difference(seg)
            cursor = max(cursor, b)
        if cursor < y1:
            seg = Polygon([
                (x - thickness / 2.0, cursor),
                (x + thickness / 2.0, cursor),
                (x + thickness / 2.0, y1),
                (x - thickness / 2.0, y1),
            ])
            hall_polygon = hall_polygon.difference(seg)

    corridor_wall_thickness = 0.12
    gap_pad_y = 0.18  # small vertical pad around openings
    gap_pad_x = 0.25  # allow openings when wall passes through a block

    # Collect openings for each wall based on which rectangles intersect the wall x.
    opening_rects = [
        enter_block_rect,           # shops_enter_block
        exit_block_rect,            # shops_exit_block
        food_enter_block_rect,
        food_exit_block_rect,
        rest_enter_block_rect,
        rest_exit_block_rect,
        gates_enter_block_rect,
        list(exit_polygon.exterior.coords)[:-1],  # SecurityExit block polygon
    ]

    def _gaps_for_wall_x(xw: float) -> list[tuple[float, float]]:
        gaps: list[tuple[float, float]] = []
        for rect in opening_rects:
            if len(rect) < 4:
                continue
            xs = [px for px, _ in rect]
            ys = [py for _, py in rect]
            min_x = min(xs)
            max_x = max(xs)
            if (min_x - gap_pad_x) <= xw <= (max_x + gap_pad_x):
                gaps.append((min(ys) - gap_pad_y, max(ys) + gap_pad_y))
        return gaps

    # Enforce BOTH vertical corridor lines as real blocking walls (full height).
    # Openings are created only where rectangles intersect each wall x:
    # - wall_line_x: SecurityExit polygon provides the only passage
    # - wall_line2_x: green area Enter/Exit blocks provide the passages
    _subtract_vertical_wall_with_gaps(
        x=wall_line_x,
        thickness=corridor_wall_thickness,
        y0=0.0,
        y1=HALL_TOP_Y,
        gaps=_gaps_for_wall_x(wall_line_x),
    )
    _subtract_vertical_wall_with_gaps(
        x=wall_line2_x,
        thickness=corridor_wall_thickness,
        y0=0.0,
        y1=HALL_TOP_Y,
        gaps=_gaps_for_wall_x(wall_line2_x),
    )

    # ------------------------------------------------
    # Shops as solid obstacles (subtract from walkable area)
    # ------------------------------------------------
    # bottom_y = top of grey front, red_line_y = bottom of grey front.
    # Grey front spans vertically from red_line_y up to bottom_y.
    # We subtract: (1) solid purple shop body, (2) left side wall of grey front,
    # (3) right side wall of grey front. Bottom of grey front stays open (red entry line).
    shop_rects = [cosmetics_rect, perfumes_rect, electronics_rect, books_rect]
    front_h_px = 40.0
    screen_h_px = 600.0
    grey_front_height_world = front_h_px * (HALL_TOP_Y / screen_h_px)
    side_wall_thickness = 0.12

    for rect in shop_rects:
        if len(rect) < 4:
            continue

        (x0, y0), (x1, _), (_, y1), _ = rect
        left_x = min(x0, x1)
        right_x = max(x0, x1)
        bottom_y = y0
        top_y = y1
        red_line_y = bottom_y - grey_front_height_world

        # solid purple shop body
        shop_body = Polygon([
            (left_x, bottom_y),
            (right_x, bottom_y),
            (right_x, top_y),
            (left_x, top_y),
        ])

        # left side wall of grey front
        grey_left_wall = Polygon([
            (left_x, red_line_y),
            (left_x + side_wall_thickness, red_line_y),
            (left_x + side_wall_thickness, bottom_y),
            (left_x, bottom_y),
        ])

        # right side wall of grey front
        grey_right_wall = Polygon([
            (right_x - side_wall_thickness, red_line_y),
            (right_x, red_line_y),
            (right_x, bottom_y),
            (right_x - side_wall_thickness, bottom_y),
        ])

        hall_polygon = hall_polygon.difference(shop_body)
        hall_polygon = hall_polygon.difference(grey_left_wall)
        hall_polygon = hall_polygon.difference(grey_right_wall)

    # ------------------------------------------------
    # Food strip horizontal walls (geometry constraints, not just drawing)
    # ------------------------------------------------
    # Constrain agents to the food band (between Shops/Food and Food/Rest horizontal lines)
    # on the right-hand side of the corridor (to the right of the second vertical wall).
    food_wall_thickness = 0.12
    hall_right_x = 47.0

    # Upper food boundary: just above/below food_horizontal_y
    food_upper_wall = Polygon([
        (wall_line2_x, food_horizontal_y - food_wall_thickness / 2.0),
        (hall_right_x, food_horizontal_y - food_wall_thickness / 2.0),
        (hall_right_x, food_horizontal_y + food_wall_thickness / 2.0),
        (wall_line2_x, food_horizontal_y + food_wall_thickness / 2.0),
    ])

    # Lower food boundary: just above/below rest_horizontal_y
    food_lower_wall = Polygon([
        (wall_line2_x, rest_horizontal_y - food_wall_thickness / 2.0),
        (hall_right_x, rest_horizontal_y - food_wall_thickness / 2.0),
        (hall_right_x, rest_horizontal_y + food_wall_thickness / 2.0),
        (wall_line2_x, rest_horizontal_y + food_wall_thickness / 2.0),
    ])

    # Subtract these thin walls from the walkable hall polygon.
    # This prevents vertical movement across the band boundaries to the right of wall_line2_x,
    # while leaving the main corridor and the food band itself walkable.
    hall_polygon = hall_polygon.difference(food_upper_wall)
    hall_polygon = hall_polygon.difference(food_lower_wall)

    # ------------------------------------------------
    # Food court solid obstacles (purple food boxes and grey fronts, with central entry gap)
    # ------------------------------------------------
    # Block the purple food counter rectangles and most of the grey fronts while leaving
    # a narrow central gap aligned with the DES entry/exit line positions.
    food_rects = [burgers_rect, pizza_rect, coffee_rect, desserts_rect]
    # Use the same world-per-pixel scale as for grey_front_height_world to convert the
    # front_margin_x used in drawing into world units.
    pixel_to_world = grey_front_height_world / front_h_px
    front_margin_x_px = 8.0
    front_margin_world = front_margin_x_px * pixel_to_world
    inner_margin_x_food = 0.20

    for rect in food_rects:
        if len(rect) < 4:
            continue

        # Solid purple food body
        food_body = Polygon(rect)
        hall_polygon = hall_polygon.difference(food_body)

        # Grey front geometry directly in front of this food block
        (fx0, fy0), (fx1, _), (_, fy1), _ = rect
        left_x = min(fx0, fx1)
        right_x = max(fx0, fx1)
        bottom_y = fy0
        grey_top_y = bottom_y
        grey_bottom_y = bottom_y - grey_front_height_world

        # Extend grey front left/right by same margin as in drawing
        front_left_x = left_x - front_margin_world
        front_right_x = right_x + front_margin_world

        # Match DES entry band horizontally: keep central band (usable_min_x..usable_max_x) open,
        # and block the left and right parts of the grey front.
        usable_min_x = left_x + inner_margin_x_food
        usable_max_x = right_x - inner_margin_x_food
        if usable_max_x <= usable_min_x:
            usable_min_x, usable_max_x = left_x, right_x

        # Left grey-front obstacle
        grey_front_left = Polygon([
            (front_left_x, grey_bottom_y),
            (usable_min_x, grey_bottom_y),
            (usable_min_x, grey_top_y),
            (front_left_x, grey_top_y),
        ])

        # Right grey-front obstacle
        grey_front_right = Polygon([
            (usable_max_x, grey_bottom_y),
            (front_right_x, grey_bottom_y),
            (front_right_x, grey_top_y),
            (usable_max_x, grey_top_y),
        ])

        hall_polygon = hall_polygon.difference(grey_front_left)
        hall_polygon = hall_polygon.difference(grey_front_right)

    # ------------------------------------------------
    # Restrooms strip horizontal walls (geometry constraints)
    # ------------------------------------------------
    # Constrain agents to the restrooms band between:
    #   - rest_horizontal_y (Food/Rest boundary)
    #   - gates_horizontal_y (Rest/Gates boundary)
    rest_wall_thickness = 0.12

    rest_upper_wall = Polygon([
        (wall_line2_x, rest_horizontal_y - rest_wall_thickness / 2.0),
        (hall_right_x, rest_horizontal_y - rest_wall_thickness / 2.0),
        (hall_right_x, rest_horizontal_y + rest_wall_thickness / 2.0),
        (wall_line2_x, rest_horizontal_y + rest_wall_thickness / 2.0),
    ])

    rest_lower_wall = Polygon([
        (wall_line2_x, gates_horizontal_y - rest_wall_thickness / 2.0),
        (hall_right_x, gates_horizontal_y - rest_wall_thickness / 2.0),
        (hall_right_x, gates_horizontal_y + rest_wall_thickness / 2.0),
        (wall_line2_x, gates_horizontal_y + rest_wall_thickness / 2.0),
    ])

    hall_polygon = hall_polygon.difference(rest_upper_wall)
    hall_polygon = hall_polygon.difference(rest_lower_wall)

    # ------------------------------------------------
    # Restrooms solid obstacles (purple boxes and grey fronts, with central entry gap)
    # ------------------------------------------------
    rest_rects = [rest1_rect, rest2_rect, rest3_rect]
    inner_margin_x_rest = 0.20

    for rect in rest_rects:
        if len(rect) < 4:
            continue

        rest_body = Polygon(rect)
        hall_polygon = hall_polygon.difference(rest_body)

        (rx0, ry0), (rx1, _), (_, ry1), _ = rect
        left_x = min(rx0, rx1)
        right_x = max(rx0, rx1)
        bottom_y = ry0
        grey_top_y = bottom_y
        grey_bottom_y = bottom_y - grey_front_height_world

        front_left_x = left_x - front_margin_world
        front_right_x = right_x + front_margin_world

        usable_min_x = left_x + inner_margin_x_rest
        usable_max_x = right_x - inner_margin_x_rest
        if usable_max_x <= usable_min_x:
            usable_min_x, usable_max_x = left_x, right_x

        grey_front_left = Polygon([
            (front_left_x, grey_bottom_y),
            (usable_min_x, grey_bottom_y),
            (usable_min_x, grey_top_y),
            (front_left_x, grey_top_y),
        ])
        grey_front_right = Polygon([
            (usable_max_x, grey_bottom_y),
            (front_right_x, grey_bottom_y),
            (front_right_x, grey_top_y),
            (usable_max_x, grey_top_y),
        ])

        hall_polygon = hall_polygon.difference(grey_front_left)
        hall_polygon = hall_polygon.difference(grey_front_right)

    # ------------------------------------------------
    # Gate blocks as solid obstacles (pink rectangles)
    # ------------------------------------------------
    # Prevent agents from cutting through the small gate boxes in the All Gates area.
    for _name, rect in gates_gate_blocks.items():
        if len(rect) < 4:
            continue
        try:
            gate_body = Polygon(rect)
        except Exception:
            continue
        hall_polygon = hall_polygon.difference(gate_body)

    # ------------------------------------------------
    # Build JuPedSim geometry
    # ------------------------------------------------
    # JuPedSim requires the accessible (walkable) area to be connected.
    # Some combinations of thin walls + obstacles can produce small disconnected islands.
    # Keep only the largest connected component to avoid:
    #   RuntimeError: accessible area not connected
    try:
        if getattr(hall_polygon, "geom_type", "") in ("MultiPolygon", "GeometryCollection"):
            geoms = [g for g in getattr(hall_polygon, "geoms", []) if getattr(g, "area", 0.0) > 1e-6]
            if geoms:
                hall_polygon = max(geoms, key=lambda g: g.area)
    except Exception:
        pass

    # ------------------------------------------------
    # Side barriers around ticket waiting area (geometry-only)
    # ------------------------------------------------
    # Your JuPedSim build doesn't expose Simulation.add_obstacle(), so we add these
    # as obstacles by subtracting them from the walkable hall polygon before building geometry.
    if waiting_positions:
        xs = [x for x, _ in waiting_positions]
        ys = [y for _, y in waiting_positions]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # Small gap from the waiting spots + thin wall thickness (same on both sides)
        wall_gap = 0.15
        wall_thickness = 0.05
        wall_y_pad = 0.15

        left_wall = Polygon(
            [
                (x_min - wall_gap - wall_thickness, y_min - wall_y_pad),
                (x_min - wall_gap - wall_thickness, y_max + wall_y_pad),
                (x_min - wall_gap, y_max + wall_y_pad),
                (x_min - wall_gap, y_min - wall_y_pad),
            ]
        )
        right_wall = Polygon(
            [
                (x_max + wall_gap, y_min - wall_y_pad),
                (x_max + wall_gap, y_max + wall_y_pad),
                (x_max + wall_gap + wall_thickness, y_max + wall_y_pad),
                (x_max + wall_gap + wall_thickness, y_min - wall_y_pad),
            ]
        )
        hall_polygon = hall_polygon.difference(left_wall).difference(right_wall)

    try:
        geometry_builder = jps.GeometryBuilder()
        geometry_builder.add_polygon(hall_polygon)
        geometry = geometry_builder.build()
    except AttributeError:
        geometry = hall_polygon

    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=geometry,
    )

    # ------------------------------------------------
    # STAGES
    # ------------------------------------------------

    entrance_stage_ids = {
        name: simulation.add_waypoint_stage(coords, 0.6)
        for name, coords in entrances.items()
    }

    counter_stage_ids = {
        name: simulation.add_waypoint_stage(coords, 0.6)
        for name, coords in counters.items()
    }

    # Explicit waypoint stages for each counter queue slot (no JuPedSim queue stages)
    counter_queue_stage_ids: dict[str, list[int]] = {}
    for i, queue in enumerate(counter_queue_lists):
        slot_stage_ids: list[int] = []
        for slot_pos in queue:
            sid = simulation.add_waypoint_stage(slot_pos, 0.6)
            slot_stage_ids.append(sid)
        counter_queue_stage_ids[f"QueueCounter{i+1}"] = slot_stage_ids

    central_waiting_queue_stage_id = simulation.add_queue_stage(waiting_positions)
    waiting_area_entry_stage_id = simulation.add_waypoint_stage(
        waiting_area_entry, 0.6
    )

    # Security waiting queue (serpentine at top) – can use JuPedSim queue stage
    security_waiting_queue_stage_id = simulation.add_queue_stage(security_waiting_positions)

    # SECURITY queues: explicit waypoint stages per slot (no JuPedSim queue stages)
    security_queue_stage_ids: dict[str, list[int]] = {}
    for name, points in zip(security_points_coords.keys(), security_queue_lists):
        slot_stage_ids: list[int] = []
        for slot_pos in points:
            sid = simulation.add_waypoint_stage(slot_pos, 0.6)
            slot_stage_ids.append(sid)
        security_queue_stage_ids[name] = slot_stage_ids

    security_point_stage_ids = {
        name: simulation.add_waypoint_stage(coords, 0.6)
        for name, coords in security_points_coords.items()
    }

    # exit queue (just below the vertical entrance/exit block)
    exit_queue_stage_id = simulation.add_queue_stage(exit_queue_positions)

    # Treat the green vertical block as an internal entrance waypoint (not an auto-exit),
    # and create a separate final exit polygon that removes agents.
    security_exit_stage_id = simulation.add_waypoint_stage(security_exit_center, 0.6)
    final_exit_stage_id = simulation.add_exit_stage(
        list(exit_polygon.exterior.coords)[:-1]
    )

    # ------------------------------------------------
    # Layout coordinate storage
    # ------------------------------------------------
    layout_coordinates = {

        "hall": {
            "width_m": 50.0,
            "depth_m": HALL_TOP_Y,
            "polygon": list(hall_polygon.exterior.coords)[:-1],
            "hall_wkt": hall_polygon.wkt,
        },

        "entrances": entrances,

        "counters": counters,

        "counter_queue_slots": counter_queue_slots,

        "waiting_positions": waiting_map,

        "waiting_area_entry": waiting_area_entry,

        "information_desks": information_desks,
        "info_desk_service_positions": info_desk_service_positions,
        "information_left_queue_slots": information_left_queue_slots,
        "information_right_queue_slots": information_right_queue_slots,
        "info_desk_waiting_positions": info_desk_waiting_positions,

        "security_points": security_points_coords,

        "security_queue_slots": security_queue_slots,

        "security_waiting_positions": security_waiting_map,

        "exit": {
            "name": "SecurityExit",
            "center": security_exit_center,
            "polygon": list(exit_polygon.exterior.coords)[:-1],
        },

        "parking_wall_line": parking_wall_line,
        "parking_wall_line_2": parking_wall_line_2,
        # Parking barrier y-range (used to prevent agents from walking back out through doors)
        "parking_barrier_y_hi": barrier_y_hi,
        "flight_displays": flight_display_rects,
        "flight_display_grey": flight_display_grey_rects,
        "shops_label_rect": shops_label_rect,
        "shops_enter_block": enter_block_rect,
        "shops_exit_block": exit_block_rect,
        "cosmetics_rect": cosmetics_rect,
        "perfumes_rect": perfumes_rect,
        "electronics_rect": electronics_rect,
        "books_rect": books_rect,

        "food_label_rect": food_label_rect,
        "food_enter_block": food_enter_block_rect,
        "food_exit_block": food_exit_block_rect,
        "food_burgers_rect": burgers_rect,
        "food_pizza_rect": pizza_rect,
        "food_coffee_rect": coffee_rect,
        "food_desserts_rect": desserts_rect,

        "rest_label_rect": rest_label_rect,
        "rest_enter_block": rest_enter_block_rect,
        "rest_exit_block": rest_exit_block_rect,
        "rest1_rect": rest1_rect,
        "rest2_rect": rest2_rect,
        "rest3_rect": rest3_rect,

        "gates_label_rect": gates_label_rect,
        "gates_enter_block": gates_enter_block_rect,
        "gates_gate_blocks": gates_gate_blocks,
        "walking_corridor_rect": walking_corridor_rect,
        "food_horizontal_line": food_horizontal_line,
        "rest_horizontal_line": rest_horizontal_line,
        "gates_horizontal_line": gates_horizontal_line,
    }

    stage_ids = {

        "entrances": entrance_stage_ids,

        "counters": counter_stage_ids,

        "counter_queues": counter_queue_stage_ids,

        "central_waiting_queue": central_waiting_queue_stage_id,

        "waiting_area_entry": waiting_area_entry_stage_id,

        "security_waiting_queue": security_waiting_queue_stage_id,

        "security_queues": security_queue_stage_ids,

        "security_points": security_point_stage_ids,

        "exit_queue": exit_queue_stage_id,

        "security_exit": security_exit_stage_id,
        "final_exit": final_exit_stage_id,
    }

    return {

        "simulation": simulation,

        "geometry": hall_polygon,

        "coordinates": layout_coordinates,

        "stage_ids": stage_ids,
    }








