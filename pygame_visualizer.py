"""Pygame-based GUI for the Epidemic flu-spread simulation.

A fully local Pygame window for interactive exploration of the model. Layout:

    +------------+--------------------------+------------------+
    |            |                          |                  |
    |  sidebar   |       agent map          |  live stats      |
    | (controls, |                          |                  |
    |  params,   +--------------------------+------------------+
    |  presets,  |   SEIR plot       |  New exposures plot   |
    |  export)   +-------------------+------------------------+
    |            |   Cumulative plot |  Infectious by type    |
    +------------+--------------------------+------------------+

Matplotlib is used (via the Agg backend) to render the four small plots, and
those rendered images are blitted into the Pygame window. Custom lightweight
widgets handle buttons, sliders, checkboxes and the map-preset dropdown.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pygame  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from agent_types import AgentType  # noqa: E402
from analytics import export_live_snapshot  # noqa: E402
from map_presets import PREDEFINED_CITY_MAPS  # noqa: E402
from model import EpidemicModel  # noqa: E402
from states import CellType, HealthState  # noqa: E402


# ---------------------------------------------------------------------------
# Color palette (dark theme, easy on the eyes for long runs)
# ---------------------------------------------------------------------------

BG = (24, 24, 30)
PANEL = (35, 35, 42)
PANEL_LIGHT = (50, 50, 60)
BORDER = (70, 70, 82)
TEXT = (230, 230, 235)
TEXT_DIM = (160, 160, 170)
ACCENT = (60, 145, 230)
ACCENT_HOVER = (90, 170, 245)
ACCENT_PRESSED = (40, 110, 180)
DANGER = (220, 80, 70)
DANGER_HOVER = (240, 110, 100)
SUCCESS = (90, 180, 110)
ERROR = (240, 90, 90)

HEALTH_COLORS = {
    HealthState.SUSCEPTIBLE: (76, 175, 80),
    HealthState.EXPOSED: (255, 213, 79),
    HealthState.INFECTIOUS: (244, 67, 54),
    HealthState.RECOVERED: (158, 158, 158),
}

CELL_COLORS = {
    CellType.DEFAULT: (44, 44, 54),       # streets / unused
    CellType.HOUSEHOLD: (70, 110, 80),    # muted green
    CellType.WORKPLACE: (150, 110, 70),   # warm brown
    CellType.PUBLIC_SPACE: (60, 105, 145),  # blue
    CellType.UNIVERSITY: (130, 95, 175),  # purple (students / academia)
    CellType.SCHOOL: (210, 175, 70),      # gold (school buses)
}

AGENT_TYPE_COLORS = {
    AgentType.STUDENT: "tab:blue",
    AgentType.WORKER: "tab:orange",
    AgentType.SENIOR: "tab:red",
    AgentType.HEALTHCARE: "tab:green",
    AgentType.CHILDREN: "tab:purple",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def fig_to_surface(fig: plt.Figure) -> pygame.Surface:
    """Render a matplotlib figure to an RGBA pygame surface."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    buf = bytes(canvas.buffer_rgba())
    return pygame.image.frombuffer(buf, (width, height), "RGBA")


# ---------------------------------------------------------------------------
# Lightweight widgets
# ---------------------------------------------------------------------------

class Widget:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)

    def draw(self, surface: pygame.Surface, fonts: dict) -> None:
        pass

    def handle_event(self, event: pygame.event.Event) -> bool:
        return False


class Button(Widget):
    def __init__(self, rect, label, on_click, *, color=ACCENT, hover_color=ACCENT_HOVER,
                 pressed_color=ACCENT_PRESSED, text_color=TEXT):
        super().__init__(rect)
        self.label = label
        self.on_click = on_click
        self.color = color
        self.hover_color = hover_color
        self.pressed_color = pressed_color
        self.text_color = text_color
        self.hover = False
        self.pressed = False

    def draw(self, surface, fonts):
        if self.pressed:
            color = self.pressed_color
        elif self.hover:
            color = self.hover_color
        else:
            color = self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, BORDER, self.rect, 1, border_radius=4)
        font = fonts["body"]
        text = font.render(self.label, True, self.text_color)
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed
            self.pressed = False
            if was_pressed and self.rect.collidepoint(event.pos):
                self.on_click()
                return True
        return False


class Slider(Widget):
    def __init__(self, rect, label, min_v, max_v, step, value, on_change,
                 value_format="{:.2f}"):
        super().__init__(rect)
        self.label = label
        self.min_v = float(min_v)
        self.max_v = float(max_v)
        self.step = float(step)
        self.value = float(value)
        self.on_change = on_change
        self.value_format = value_format
        self.dragging = False

    def _track_rect(self):
        margin = 4
        return pygame.Rect(
            self.rect.x + margin,
            self.rect.y + self.rect.height - 10,
            self.rect.width - 2 * margin,
            6,
        )

    def _thumb_x(self):
        track = self._track_rect()
        span = self.max_v - self.min_v
        ratio = (self.value - self.min_v) / span if span > 0 else 0.0
        return track.x + int(ratio * track.width)

    def draw(self, surface, fonts):
        font = fonts["small"]
        label = font.render(self.label, True, TEXT_DIM)
        value_text = font.render(self.value_format.format(self.value), True, TEXT)
        surface.blit(label, (self.rect.x, self.rect.y))
        value_rect = value_text.get_rect(topright=(self.rect.right, self.rect.y))
        surface.blit(value_text, value_rect)
        track = self._track_rect()
        pygame.draw.rect(surface, PANEL_LIGHT, track, border_radius=3)
        filled = track.copy()
        filled.width = self._thumb_x() - track.x
        if filled.width > 0:
            pygame.draw.rect(surface, ACCENT, filled, border_radius=3)
        thumb = pygame.Rect(0, 0, 12, 18)
        thumb.center = (self._thumb_x(), track.centery)
        pygame.draw.rect(surface, TEXT, thumb, border_radius=3)
        pygame.draw.rect(surface, BORDER, thumb, 1, border_radius=3)

    def _set_from_x(self, x):
        track = self._track_rect()
        ratio = (x - track.x) / max(track.width, 1)
        ratio = max(0.0, min(1.0, ratio))
        raw = self.min_v + ratio * (self.max_v - self.min_v)
        steps = round((raw - self.min_v) / self.step) if self.step > 0 else 0
        new = self.min_v + steps * self.step
        new = max(self.min_v, min(self.max_v, new))
        if abs(new - self.value) > 1e-9:
            self.value = new
            self.on_change(new)

    def _hit_rect(self):
        # Make the entire widget area clickable, not just the thin track.
        return self.rect.inflate(0, 0)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._hit_rect().collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        return False


class Checkbox(Widget):
    def __init__(self, rect, label, value, on_change):
        super().__init__(rect)
        self.label = label
        self.value = bool(value)
        self.on_change = on_change

    def draw(self, surface, fonts):
        font = fonts["small"]
        box_size = 18
        box = pygame.Rect(self.rect.x, self.rect.centery - box_size // 2, box_size, box_size)
        pygame.draw.rect(surface, PANEL_LIGHT, box, border_radius=3)
        pygame.draw.rect(surface, BORDER, box, 1, border_radius=3)
        if self.value:
            pygame.draw.line(surface, ACCENT, (box.x + 3, box.centery),
                             (box.centerx, box.bottom - 4), 3)
            pygame.draw.line(surface, ACCENT, (box.centerx, box.bottom - 4),
                             (box.right - 3, box.y + 3), 3)
        label = font.render(self.label, True, TEXT)
        surface.blit(label, (box.right + 8, box.centery - label.get_height() // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.value = not self.value
                self.on_change(self.value)
                return True
        return False


class Dropdown(Widget):
    """A simple click-to-open dropdown drawn with a popup list."""

    OPTION_HEIGHT = 24

    def __init__(self, rect, label, options, selected_index, on_change):
        super().__init__(rect)
        self.label = label
        self.options = list(options)
        self.selected_index = selected_index
        self.on_change = on_change
        self.open = False

    def _box_rect(self):
        return pygame.Rect(self.rect.x, self.rect.y + 16, self.rect.width, self.rect.height - 16)

    def _options_rect(self):
        box = self._box_rect()
        return pygame.Rect(box.x, box.bottom, box.width, len(self.options) * self.OPTION_HEIGHT)

    def draw(self, surface, fonts):
        font_small = fonts["small"]
        font_body = fonts["body"]
        label = font_small.render(self.label, True, TEXT_DIM)
        surface.blit(label, (self.rect.x, self.rect.y))
        box = self._box_rect()
        pygame.draw.rect(surface, PANEL_LIGHT, box, border_radius=4)
        pygame.draw.rect(surface, BORDER, box, 1, border_radius=4)
        current = font_body.render(self.options[self.selected_index], True, TEXT)
        surface.blit(current, (box.x + 8, box.centery - current.get_height() // 2))
        ax, ay = box.right - 14, box.centery
        pygame.draw.polygon(surface, TEXT, [(ax - 5, ay - 3), (ax + 5, ay - 3), (ax, ay + 4)])

    def draw_popup(self, surface, fonts):
        if not self.open:
            return
        font = fonts["small"]
        opts = self._options_rect()
        pygame.draw.rect(surface, PANEL, opts)
        pygame.draw.rect(surface, BORDER, opts, 1)
        mx, my = pygame.mouse.get_pos()
        for i, opt in enumerate(self.options):
            row = pygame.Rect(opts.x, opts.y + i * self.OPTION_HEIGHT, opts.width, self.OPTION_HEIGHT)
            if i == self.selected_index:
                pygame.draw.rect(surface, ACCENT_PRESSED, row)
            elif row.collidepoint(mx, my):
                pygame.draw.rect(surface, ACCENT_HOVER, row)
            text = font.render(opt, True, TEXT)
            surface.blit(text, (row.x + 8, row.centery - text.get_height() // 2))

    def handle_event(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        box = self._box_rect()
        if self.open:
            opts = self._options_rect()
            if opts.collidepoint(event.pos):
                idx = (event.pos[1] - opts.y) // self.OPTION_HEIGHT
                idx = max(0, min(len(self.options) - 1, int(idx)))
                if idx != self.selected_index:
                    self.selected_index = idx
                    self.on_change(idx)
                self.open = False
                return True
            self.open = False
            if box.collidepoint(event.pos):
                self.open = True
            return True
        if box.collidepoint(event.pos):
            self.open = True
            return True
        return False


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

class PygameVisualizer:
    SIDEBAR_WIDTH = 300
    BOTTOM_PANEL_HEIGHT = 360
    INNER_PADDING = 10
    PLOT_DPI = 90

    def __init__(
        self,
        model: EpidemicModel,
        figsize: tuple[int, int] = (12, 9),
        agent_size: int = 6,
        window_size: tuple[int, int] = (1600, 900),
    ):
        self.model = model
        self.figsize = figsize  # kept for API parity; not directly used
        self.agent_radius = max(2, int(agent_size) // 2)
        self.window_size = window_size

        # Simulation control state
        self.playing = False
        self.play_interval_ms = 100
        self.render_interval = 1
        self._steps_since_render = 0
        self._last_step_at = 0

        # Pending parameters - applied to a new model on Reset
        self.pending_population = len(model.agents)
        self.pending_avg_household_size = int(model.avg_household_size)
        self.pending_avg_family_size = int(model.avg_family_size)
        self.pending_avg_friend_group_size = int(model.avg_friend_group_size)
        self.pending_time_of_day = float(model.time_of_day)
        self.pending_timestep = float(model.timestep)
        self.pending_verbose = bool(model.verbose)
        self.pending_preset_key = model.city_map_preset

        # Cached render surfaces
        self._cell_surface: pygame.Surface | None = None
        self._cell_surface_rect: pygame.Rect | None = None
        self._plot_surfaces: list[pygame.Surface | None] = [None, None, None, None]
        self._plots_dirty = True
        self._last_plot_step = -1

        # Status feedback (export)
        self._status_text = ""
        self._status_is_error = False
        self._status_until_ms = 0

        # Widget handles populated in _build_widgets
        self.widgets: list[Widget] = []
        self.dropdowns: list[Dropdown] = []
        self.play_button: Button | None = None
        self.fonts: dict[str, pygame.font.Font] = {}

    # ----- Public entry point -------------------------------------------

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Flu Spread Simulation - Pygame")
        flags = pygame.RESIZABLE
        self.screen = pygame.display.set_mode(self.window_size, flags)
        self.clock = pygame.time.Clock()
        self._init_fonts()
        self._build_widgets()
        self._next_step_at = pygame.time.get_ticks()

        running = True
        while running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    new_size = (max(900, event.w), max(600, event.h))
                    self.screen = pygame.display.set_mode(new_size, flags)
                    self.window_size = new_size
                    self._invalidate_cached_surfaces()
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event)
                else:
                    self._dispatch_event(event)

            self._tick_simulation(now)
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    # ----- Initialization helpers ---------------------------------------

    def _init_fonts(self) -> None:
        self.fonts = {
            "title": pygame.font.SysFont("Segoe UI", 18, bold=True),
            "subtitle": pygame.font.SysFont("Segoe UI", 14, bold=True),
            "body": pygame.font.SysFont("Segoe UI", 13),
            "small": pygame.font.SysFont("Segoe UI", 12),
            "tiny": pygame.font.SysFont("Segoe UI", 11),
        }

    def _invalidate_cached_surfaces(self) -> None:
        self._cell_surface = None
        self._cell_surface_rect = None
        self._plots_dirty = True

    def _build_widgets(self) -> None:
        self.widgets = []
        self.dropdowns = []

        pad = 12
        width = self.SIDEBAR_WIDTH - 2 * pad
        slider_h = 36
        button_h = 30
        section_gap = 26  # vertical room reserved for a section header

        # Title is drawn at y=12 with ~22px height, leave a small margin then
        # start the first section header below it.
        y = 44

        # --- Controls section -----------------------------------------
        self._section_controls_y = y
        y += section_gap
        btn_w = (width - 2 * pad) // 3
        self.play_button = Button(
            (pad, y, btn_w, button_h),
            "Play",
            self._toggle_play,
        )
        self.widgets.append(self.play_button)
        self.widgets.append(Button(
            (pad + btn_w + pad, y, btn_w, button_h),
            "Step",
            self._do_step,
        ))
        self.widgets.append(Button(
            (pad + 2 * (btn_w + pad), y, btn_w, button_h),
            "Reset",
            self._do_reset,
            color=DANGER,
            hover_color=DANGER_HOVER,
            pressed_color=(170, 60, 50),
        ))
        y += button_h + 12

        self.widgets.append(Slider(
            (pad, y, width, slider_h),
            "Play interval (ms)", 1, 500, 1, self.play_interval_ms,
            on_change=lambda v: setattr(self, "play_interval_ms", int(v)),
            value_format="{:.0f}",
        ))
        y += slider_h + 6
        self.widgets.append(Slider(
            (pad, y, width, slider_h),
            "Render every N steps", 1, 100, 1, self.render_interval,
            on_change=lambda v: setattr(self, "render_interval", int(v)),
            value_format="{:.0f}",
        ))
        y += slider_h + 12

        # --- Model parameters section --------------------------------
        self._section_params_y = y
        y += section_gap
        self.widgets.append(Slider(
            (pad, y, width, slider_h),
            "Population", 200, 20000, 200, self.pending_population,
            on_change=lambda v: setattr(self, "pending_population", int(v)),
            value_format="{:.0f}",
        ))
        y += slider_h + 6
        self.widgets.append(Slider(
            (pad, y, width, slider_h),
            "Avg household size", 1, 8, 1, self.pending_avg_household_size,
            on_change=lambda v: setattr(self, "pending_avg_household_size", int(v)),
            value_format="{:.0f}",
        ))
        y += slider_h + 6
        self.widgets.append(Slider(
            (pad, y, width, slider_h),
            "Avg family size", 1, 8, 1, self.pending_avg_family_size,
            on_change=lambda v: setattr(self, "pending_avg_family_size", int(v)),
            value_format="{:.0f}",
        ))
        y += slider_h + 6
        self.widgets.append(Slider(
            (pad, y, width, slider_h),
            "Avg friend group size", 1, 10, 1, self.pending_avg_friend_group_size,
            on_change=lambda v: setattr(self, "pending_avg_friend_group_size", int(v)),
            value_format="{:.0f}",
        ))
        y += slider_h + 6
        self.widgets.append(Slider(
            (pad, y, width, slider_h),
            "Start time (h)", 0.0, 23.5, 0.5, self.pending_time_of_day,
            on_change=lambda v: setattr(self, "pending_time_of_day", float(v)),
            value_format="{:.1f}",
        ))
        y += slider_h + 6
        self.widgets.append(Slider(
            (pad, y, width, slider_h),
            "Timestep (h)", 0.05, 2.0, 0.05, self.pending_timestep,
            on_change=lambda v: setattr(self, "pending_timestep", float(v)),
            value_format="{:.2f}",
        ))
        y += slider_h + 6
        self.widgets.append(Checkbox(
            (pad, y, width, 22),
            "Verbose logs", self.pending_verbose,
            on_change=lambda v: setattr(self, "pending_verbose", bool(v)),
        ))
        y += 22 + 12

        # --- Map preset section --------------------------------------
        self._section_map_y = y
        y += section_gap
        preset_keys = list(PREDEFINED_CITY_MAPS.keys())
        preset_labels = [PREDEFINED_CITY_MAPS[k]["label"] for k in preset_keys]
        try:
            current_idx = preset_keys.index(self.pending_preset_key)
        except (ValueError, TypeError):
            current_idx = 0
        self.map_dropdown = Dropdown(
            (pad, y, width, 16 + 28),
            "Map preset", preset_labels, current_idx,
            on_change=lambda i: self._select_preset(preset_keys[i]),
        )
        self.widgets.append(self.map_dropdown)
        self.dropdowns.append(self.map_dropdown)
        y += 16 + 28 + 12

        # --- Export section ------------------------------------------
        self._section_export_y = y
        y += section_gap
        self.widgets.append(Button(
            (pad, y, width, button_h),
            "Generate visualization",
            self._do_export,
            color=(95, 155, 90),
            hover_color=(120, 185, 110),
            pressed_color=(70, 125, 70),
        ))
        y += button_h + 8
        self._status_area_y = y

    # ----- Event dispatch -----------------------------------------------

    def _dispatch_event(self, event):
        # Route mouse clicks to open dropdowns first so they always close /
        # select even when overlapping other widgets.
        for dd in self.dropdowns:
            if dd.open and event.type == pygame.MOUSEBUTTONDOWN:
                dd.handle_event(event)
                return
        for widget in self.widgets:
            if widget.handle_event(event):
                return

    def _handle_key(self, event):
        if event.key == pygame.K_SPACE:
            self._toggle_play()
        elif event.key in (pygame.K_RIGHT, pygame.K_n):
            self._do_step()
        elif event.key == pygame.K_r:
            self._do_reset()
        elif event.key == pygame.K_e:
            self._do_export()

    # ----- Simulation control -------------------------------------------

    def _toggle_play(self):
        self.playing = not self.playing
        if self.play_button is not None:
            self.play_button.label = "Pause" if self.playing else "Play"
        # Start the next step right away after pressing Play.
        self._last_step_at = pygame.time.get_ticks() - self.play_interval_ms

    def _do_step(self):
        self.model.step()
        self._steps_since_render += 1
        self._plots_dirty = True

    def _do_reset(self):
        self.playing = False
        if self.play_button is not None:
            self.play_button.label = "Play"
        try:
            self.model = EpidemicModel(
                city_map_path=self.model.city_map_path,
                city_map_preset=self.pending_preset_key,
                population=int(self.pending_population),
                avg_household_size=int(self.pending_avg_household_size),
                avg_family_size=int(self.pending_avg_family_size),
                avg_friend_group_size=int(self.pending_avg_friend_group_size),
                time_of_day=float(self.pending_time_of_day),
                timestep=float(self.pending_timestep),
                verbose=bool(self.pending_verbose),
            )
            self._set_status(
                f"Model reset with population={int(self.pending_population)}",
                is_error=False,
            )
        except Exception as exc:  # defensive, surfaces config issues
            self._set_status(f"Reset failed: {exc}", is_error=True)
        self._invalidate_cached_surfaces()
        self._steps_since_render = 0

    def _select_preset(self, preset_key: str):
        self.pending_preset_key = preset_key
        label = PREDEFINED_CITY_MAPS[preset_key]["label"]
        self._set_status(
            f"Preset '{label}' selected - click Reset to apply",
            is_error=False,
        )

    def _do_export(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = Path("output/live_export") / f"step_{self.model.current_step:04d}_{timestamp}"
            artifacts = export_live_snapshot(
                self.model,
                str(target),
                config={
                    "cityMapPath": self.model.city_map_path,
                    "cityMapPreset": self.model.city_map_preset,
                    "mapName": self.model.map_name,
                },
            )
            self._set_status(
                f"Exported to {os.path.basename(artifacts['output_dir'])}",
                is_error=False,
            )
        except Exception as exc:
            self._set_status(f"Export failed: {exc}", is_error=True)

    def _set_status(self, text: str, *, is_error: bool):
        self._status_text = text
        self._status_is_error = is_error
        self._status_until_ms = pygame.time.get_ticks() + 6000

    def _tick_simulation(self, now_ms: int):
        if not self.playing:
            return
        # Step at most once per frame. We never try to "catch up" if rendering
        # took longer than play_interval_ms - otherwise multiple steps would be
        # squeezed into a single frame and the user would see the simulation
        # jump several steps at a time. Effective step rate is therefore
        #   min(frame_rate, 1000 / play_interval_ms)  steps per second.
        if now_ms - self._last_step_at < self.play_interval_ms:
            return
        self.model.step()
        self._last_step_at = now_ms
        self._steps_since_render += 1
        if self._steps_since_render >= self.render_interval:
            self._plots_dirty = True
            self._steps_since_render = 0

    # ----- Drawing -------------------------------------------------------

    def _draw(self):
        self.screen.fill(BG)
        self._draw_sidebar()
        self._draw_main_area()
        # Popups overlay everything
        for dd in self.dropdowns:
            dd.draw_popup(self.screen, self.fonts)

    def _draw_sidebar(self):
        sidebar_rect = pygame.Rect(0, 0, self.SIDEBAR_WIDTH, self.window_size[1])
        pygame.draw.rect(self.screen, PANEL, sidebar_rect)
        pygame.draw.line(self.screen, BORDER,
                         (self.SIDEBAR_WIDTH, 0),
                         (self.SIDEBAR_WIDTH, self.window_size[1]), 1)
        title = self.fonts["title"].render("Flu Spread Sim", True, TEXT)
        self.screen.blit(title, (12, 12))

        self._draw_section_header("Controls", self._section_controls_y)
        self._draw_section_header("Model parameters", self._section_params_y)
        self._draw_section_header("Map preset", self._section_map_y)
        self._draw_section_header("Export", self._section_export_y)

        for widget in self.widgets:
            widget.draw(self.screen, self.fonts)

        self._draw_status_message()

    def _draw_section_header(self, text: str, y: int):
        font = self.fonts["subtitle"]
        label = font.render(text, True, TEXT_DIM)
        self.screen.blit(label, (12, y))
        pygame.draw.line(
            self.screen, BORDER,
            (12 + label.get_width() + 8, y + label.get_height() // 2),
            (self.SIDEBAR_WIDTH - 12, y + label.get_height() // 2),
            1,
        )

    def _draw_status_message(self):
        if not self._status_text:
            return
        now = pygame.time.get_ticks()
        if now > self._status_until_ms:
            self._status_text = ""
            return
        font = self.fonts["small"]
        color = ERROR if self._status_is_error else SUCCESS
        bg_color = (60, 30, 30) if self._status_is_error else (30, 60, 35)
        pad = 12
        max_width = self.SIDEBAR_WIDTH - 2 * pad
        lines = self._wrap_text(self._status_text, font, max_width - 16)
        line_h = font.get_linesize()
        box_h = len(lines) * line_h + 12
        box = pygame.Rect(pad, self._status_area_y, max_width, box_h)
        pygame.draw.rect(self.screen, bg_color, box, border_radius=4)
        pygame.draw.rect(self.screen, color, box, 1, border_radius=4)
        for i, line in enumerate(lines):
            text = font.render(line, True, color)
            self.screen.blit(text, (box.x + 8, box.y + 6 + i * line_h))

    @staticmethod
    def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        words = text.split(" ")
        lines = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    # ----- Main area: map + stats + plots --------------------------------

    def _draw_main_area(self):
        w, h = self.window_size
        bottom_h = min(self.BOTTOM_PANEL_HEIGHT, max(220, h // 2 - 40))
        top_h = h - bottom_h - 2 * self.INNER_PADDING
        top_y = self.INNER_PADDING
        bottom_y = top_y + top_h + self.INNER_PADDING

        # Top: map + stats
        main_x = self.SIDEBAR_WIDTH + self.INNER_PADDING
        main_w = w - main_x - self.INNER_PADDING
        # Map gets 60% of the top region, stats get 40%
        map_w = int(main_w * 0.6) - self.INNER_PADDING // 2
        stats_x = main_x + map_w + self.INNER_PADDING
        stats_w = main_w - map_w - self.INNER_PADDING

        map_rect = pygame.Rect(main_x, top_y, map_w, top_h)
        stats_rect = pygame.Rect(stats_x, top_y, stats_w, top_h)
        self._draw_map(map_rect)
        self._draw_stats(stats_rect)

        # Bottom: 4 plots in 2x2
        plot_w = (main_w - self.INNER_PADDING) // 2
        plot_h = (bottom_h - self.INNER_PADDING) // 2
        plot_rects = [
            pygame.Rect(main_x, bottom_y, plot_w, plot_h),
            pygame.Rect(main_x + plot_w + self.INNER_PADDING, bottom_y, plot_w, plot_h),
            pygame.Rect(main_x, bottom_y + plot_h + self.INNER_PADDING, plot_w, plot_h),
            pygame.Rect(main_x + plot_w + self.INNER_PADDING,
                        bottom_y + plot_h + self.INNER_PADDING, plot_w, plot_h),
        ]
        self._draw_plots(plot_rects)

    # ----- Map ----------------------------------------------------------

    def _draw_map(self, rect: pygame.Rect):
        self._draw_panel(rect, title=f"Agent map ({self.model.map_name})")
        # Reserve room at the top for the panel title and at the bottom for
        # the two-row legend.
        legend_reserve = 36
        inner = pygame.Rect(
            rect.x + 8,
            rect.y + 36,
            rect.width - 16,
            rect.height - 36 - legend_reserve,
        )

        grid_w = self.model.grid.width
        grid_h = self.model.grid.height
        if grid_w <= 0 or grid_h <= 0:
            return

        cell_size = max(1, min(inner.width // grid_w, inner.height // grid_h))
        map_pixel_w = cell_size * grid_w
        map_pixel_h = cell_size * grid_h
        offset_x = inner.x + (inner.width - map_pixel_w) // 2
        offset_y = inner.y + (inner.height - map_pixel_h) // 2

        # Build / reuse the static cell layer.
        if (self._cell_surface is None
                or self._cell_surface_rect is None
                or self._cell_surface_rect.size != (map_pixel_w, map_pixel_h)):
            self._cell_surface = self._build_cell_surface(grid_w, grid_h, cell_size)
            self._cell_surface_rect = pygame.Rect(0, 0, map_pixel_w, map_pixel_h)

        self.screen.blit(self._cell_surface, (offset_x, offset_y))

        # Draw agents on top.
        radius = max(2, min(cell_size // 2, self.agent_radius + cell_size // 12))
        for agent in self.model.agents:
            pos = agent.pos
            if pos is None:
                continue
            x, y = pos
            cx = offset_x + x * cell_size + cell_size // 2
            cy = offset_y + y * cell_size + cell_size // 2
            color = HEALTH_COLORS.get(agent.health_state, (255, 255, 255))
            pygame.draw.circle(self.screen, color, (cx, cy), radius)

        self._draw_map_legend(rect)
        self._draw_time_overlay(rect)

    def _build_cell_surface(self, grid_w: int, grid_h: int, cell_size: int) -> pygame.Surface:
        surface = pygame.Surface((grid_w * cell_size, grid_h * cell_size))
        surface.fill(CELL_COLORS[CellType.DEFAULT])
        for pos, (cell_type, _) in self.model.location_data.items():
            color = CELL_COLORS.get(cell_type, CELL_COLORS[CellType.DEFAULT])
            x, y = pos
            pygame.draw.rect(
                surface, color,
                pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size),
            )
        # Subtle grid lines only when cells are wide enough
        if cell_size >= 8:
            grid_color = (0, 0, 0)
            for gx in range(grid_w + 1):
                pygame.draw.line(surface, grid_color,
                                 (gx * cell_size, 0),
                                 (gx * cell_size, grid_h * cell_size), 1)
            for gy in range(grid_h + 1):
                pygame.draw.line(surface, grid_color,
                                 (0, gy * cell_size),
                                 (grid_w * cell_size, gy * cell_size), 1)
        return surface

    def _draw_map_legend(self, rect: pygame.Rect):
        font = self.fonts["tiny"]
        # Two rows so it never overflows the panel width
        rows: list[list[tuple[str, tuple[int, int, int]]]] = [
            [
                ("Susceptible", HEALTH_COLORS[HealthState.SUSCEPTIBLE]),
                ("Exposed", HEALTH_COLORS[HealthState.EXPOSED]),
                ("Infectious", HEALTH_COLORS[HealthState.INFECTIOUS]),
                ("Recovered", HEALTH_COLORS[HealthState.RECOVERED]),
            ],
            [
                ("Street / open space (default)", CELL_COLORS[CellType.DEFAULT]),
                ("Household", CELL_COLORS[CellType.HOUSEHOLD]),
                ("Public space", CELL_COLORS[CellType.PUBLIC_SPACE]),
            ],
            [
                ("Workplace", CELL_COLORS[CellType.WORKPLACE]),
                ("University", CELL_COLORS[CellType.UNIVERSITY]),
                ("School", CELL_COLORS[CellType.SCHOOL]),
            ],
        ]
        line_h = 14
        max_x = rect.right - 8
        base_y = rect.bottom - line_h * len(rows) - 4
        for row_idx, items in enumerate(rows):
            x = rect.x + 12
            y = base_y + row_idx * line_h
            for label, color in items:
                text = font.render(label, True, TEXT_DIM)
                box = pygame.Rect(x, y + 2, 10, 10)
                # If the next item would overflow, stop drawing this row
                if box.x + 10 + 4 + text.get_width() > max_x:
                    break
                pygame.draw.rect(self.screen, color, box)
                self.screen.blit(text, (box.right + 4, y))
                x = box.right + 4 + text.get_width() + 12

    def _draw_time_overlay(self, rect: pygame.Rect):
        font = self.fonts["small"]
        text = f"Step {self.model.current_step}  |  Time {self.model.time_of_day:.2f}h"
        rendered = font.render(text, True, TEXT)
        bg = pygame.Surface((rendered.get_width() + 14, rendered.get_height() + 6),
                            pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        self.screen.blit(bg, (rect.right - bg.get_width() - 10, rect.y + 28))
        self.screen.blit(rendered, (rect.right - rendered.get_width() - 17, rect.y + 31))

    # ----- Stats panel --------------------------------------------------

    def _draw_stats(self, rect: pygame.Rect):
        self._draw_panel(rect, title="Live statistics")
        stats = self.model.get_metrics_snapshot()
        try:
            summary = self.model.get_summary_metrics()
        except (ValueError, IndexError):
            summary = {"peak_infectious": 0, "peak_infectious_step": 0}
        by_type = self.model.get_health_counts_by_type()

        population = max(stats["population"], 1)
        font_body = self.fonts["body"]
        font_small = self.fonts["small"]
        x = rect.x + 14
        y = rect.y + 36
        col_w = (rect.width - 28) // 2
        line_h = font_body.get_linesize() + 2

        def line(surface_x: int, surface_y: int, label: str, value: str,
                 color: tuple = TEXT, label_color: tuple = TEXT_DIM):
            lbl = font_small.render(label, True, label_color)
            self.screen.blit(lbl, (surface_x, surface_y))
            val = font_body.render(value, True, color)
            self.screen.blit(val, (surface_x, surface_y + 14))

        # Two-column top stats
        line(x, y, "Step", str(stats["step"]))
        line(x + col_w, y, "Time of day", f"{stats['time_of_day']:.2f} h")
        y += 36
        line(x, y, "Population", _format_int(stats["population"]))
        line(x + col_w, y, "New exposures (last step)", _format_int(stats["new_exposures"]))
        y += 36
        line(x, y, "Peak infectious",
             f"{_format_int(summary['peak_infectious'])}  (step {summary['peak_infectious_step']})")
        line(x + col_w, y, "Cumulative infected",
             f"{_format_int(stats['cumulative_infected'])}  ({stats['cumulative_ratio']:.1%})")
        y += 36

        # Divider
        pygame.draw.line(self.screen, BORDER, (rect.x + 14, y),
                         (rect.right - 14, y), 1)
        y += 10

        # SEIR bar chart
        seir_items = [
            ("Susceptible", "susceptible", HEALTH_COLORS[HealthState.SUSCEPTIBLE]),
            ("Exposed", "exposed", HEALTH_COLORS[HealthState.EXPOSED]),
            ("Infectious", "infectious", HEALTH_COLORS[HealthState.INFECTIOUS]),
            ("Recovered", "recovered", HEALTH_COLORS[HealthState.RECOVERED]),
        ]
        bar_label_w = 90
        bar_value_w = 110
        bar_area_w = rect.width - 28 - bar_label_w - bar_value_w
        bar_h = 16
        for label, key, color in seir_items:
            count = stats[key]
            ratio = count / population
            label_surf = font_body.render(label, True, TEXT)
            self.screen.blit(label_surf, (x, y))
            bar_x = x + bar_label_w
            track = pygame.Rect(bar_x, y + 2, max(40, bar_area_w), bar_h)
            pygame.draw.rect(self.screen, PANEL_LIGHT, track, border_radius=3)
            fill = track.copy()
            fill.width = int(track.width * ratio)
            if fill.width > 0:
                pygame.draw.rect(self.screen, color, fill, border_radius=3)
            value_text = font_small.render(
                f"{_format_int(count)} ({ratio:.1%})", True, TEXT,
            )
            self.screen.blit(value_text, (track.right + 10, y + 1))
            y += line_h + 6

        y += 6
        pygame.draw.line(self.screen, BORDER, (rect.x + 14, y),
                         (rect.right - 14, y), 1)
        y += 10

        # Infectious by agent type
        header = font_body.render("Infectious by agent type", True, TEXT)
        self.screen.blit(header, (x, y))
        y += line_h + 2
        for agent_type in AgentType:
            counts = by_type.get(agent_type.value, {"infectious": 0,
                                                    "susceptible": 0,
                                                    "exposed": 0,
                                                    "recovered": 0})
            total = sum(counts.values())
            inf = counts["infectious"]
            ratio = (inf / total) if total else 0.0
            text = font_small.render(
                f"- {agent_type.value.capitalize()}: "
                f"{_format_int(inf)} infectious / {_format_int(total)}  ({ratio:.1%})",
                True, TEXT_DIM,
            )
            self.screen.blit(text, (x, y))
            y += line_h
            if y > rect.bottom - 16:
                break

    # ----- Plots --------------------------------------------------------

    def _draw_plots(self, rects: Sequence[pygame.Rect]):
        # Re-render plots only when something changed and the current model
        # has at least one collected sample.
        steps_collected = len(self.model.metrics_history)
        need_rerender = (
            self._plots_dirty
            or any(s is None for s in self._plot_surfaces)
            or (self._plot_surfaces[0] is not None
                and (self._plot_surfaces[0].get_width() != rects[0].width - 8
                     or self._plot_surfaces[0].get_height() != rects[0].height - 8))
        )
        if need_rerender and steps_collected > 0:
            self._render_plot_surfaces([r.inflate(-8, -8) for r in rects])
            self._plots_dirty = False
            self._last_plot_step = self.model.current_step

        for rect, surface in zip(rects, self._plot_surfaces):
            self._draw_panel(rect, title=None)
            if surface is not None:
                self.screen.blit(surface, (rect.x + 4, rect.y + 4))
            else:
                font = self.fonts["small"]
                msg = font.render("Run at least one step to see plots",
                                  True, TEXT_DIM)
                self.screen.blit(
                    msg, (rect.centerx - msg.get_width() // 2,
                          rect.centery - msg.get_height() // 2),
                )

    def _render_plot_surfaces(self, target_rects: Sequence[pygame.Rect]):
        data = self.model.datacollector.model_vars
        steps = list(range(len(next(iter(data.values()), []))))
        if not steps:
            return

        configs = [
            {
                "title": "Health states (S/E/I/R)",
                "ylabel": "Agents",
                "series": [
                    ("Susceptible", "#4caf50"),
                    ("Exposed", "#ffb300"),
                    ("Infectious", "#e53935"),
                    ("Recovered", "#9e9e9e"),
                ],
            },
            {
                "title": "New exposures per step (S->E)",
                "ylabel": "Agents",
                "series": [("New_Exposures", "#7e57c2")],
            },
            {
                "title": "Cumulative infected",
                "ylabel": "Agents",
                "series": [("Cumulative_Infected", "#e53935")],
            },
            {
                "title": "Infectious by agent type",
                "ylabel": "Infectious agents",
                "series": [
                    (f"Infectious_{t.value}", AGENT_TYPE_COLORS[t])
                    for t in AgentType
                ],
            },
        ]

        for idx, (rect, cfg) in enumerate(zip(target_rects, configs)):
            width_in = max(2.0, rect.width / self.PLOT_DPI)
            height_in = max(1.5, rect.height / self.PLOT_DPI)
            fig = plt.figure(figsize=(width_in, height_in), dpi=self.PLOT_DPI,
                             facecolor="#2a2a35")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#1e1e26")
            for key, color in cfg["series"]:
                if key not in data:
                    continue
                ax.plot(steps, data[key], label=key.replace("Infectious_", ""),
                        color=color, linewidth=1.5)
            ax.set_title(cfg["title"], color="#e6e6eb", fontsize=10)
            ax.set_xlabel("Step", color="#9999a3", fontsize=8)
            ax.set_ylabel(cfg["ylabel"], color="#9999a3", fontsize=8)
            ax.tick_params(colors="#9999a3", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#55555f")
            ax.grid(alpha=0.2, color="#888892")
            if len(cfg["series"]) > 1:
                legend = ax.legend(loc="upper left", fontsize=7, framealpha=0.7,
                                   facecolor="#2a2a35", edgecolor="#55555f",
                                   labelcolor="#e6e6eb")
                if legend is not None:
                    for text in legend.get_texts():
                        text.set_color("#e6e6eb")
            fig.tight_layout(pad=0.6)
            self._plot_surfaces[idx] = fig_to_surface(fig)
            plt.close(fig)

    # ----- Panel chrome -------------------------------------------------

    def _draw_panel(self, rect: pygame.Rect, *, title: str | None):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=6)
        pygame.draw.rect(self.screen, BORDER, rect, 1, border_radius=6)
        if title:
            font = self.fonts["subtitle"]
            text = font.render(title, True, TEXT)
            self.screen.blit(text, (rect.x + 12, rect.y + 8))
            pygame.draw.line(
                self.screen, BORDER,
                (rect.x + 12, rect.y + 8 + text.get_height() + 4),
                (rect.right - 12, rect.y + 8 + text.get_height() + 4),
                1,
            )
