"""
control_fsm.py — Assistive exoskeleton finite-state machine (FSM).

8 States with hysteresis (N=3 consecutive same predictions before switching).

ASCII State Diagram:
═══════════════════════════════════════════════════════════════════════════
                            ┌──────────────┐
                   ┌────────►     IDLE     ◄────────────────────────┐
                   │        └──────┬───────┘                        │
                   │ stop/error    │ activation                     │
                   │               ▼                                │
                   │        ┌──────────────┐                        │
         ┌─────────┤◄───────┤   STANDING   │◄───────────────────┐  │
         │         │        │  torque=0.0  │                    │  │
         │ sit_pred│        └──┬───────┬───┘                    │  │
         │         │    walk   │       │                         │  │
         │         │    pred   │  sit_to                         │  │
         │         │           ▼  stand_pred                     │  │
         │         │        ┌──────────────────┐                 │  │
         │         │        │    SIT_TO_STAND   │─────────────►──┘  │
         │         │        │   torque=2.0      │ (after burst)     │
         │         │        └──────────────────┘                    │
         │         │                                                 │
         │         │           ▼                                     │
         │         │        ┌──────────────────┐                    │
         │         └───────►│  LEVEL_WALKING   │                    │
         │                  │   torque=1.0     │                    │
         │                  └──┬────┬──────┬───┘                    │
         │         stair_asc   │    │      │ ramp_asc               │
         │           ┌─────────┘    │      └──────────────────┐     │
         │           ▼              │ stair/ramp_desc          ▼     │
         │   ┌──────────────┐       │             ┌─────────────────┐│
         │   │ STAIR_ASCENT │       │             │  RAMP_ASCENT    ││
         │   │ torque=1.5   │       │             │  torque=1.2     ││
         │   └──────────────┘       │             └─────────────────┘│
         │                          ▼                                 │
         │              ┌───────────────────────┐                    │
         │              │   STAIR/RAMP DESCENT  │                    │
         │              │   torque=0.8 (brake)  │                    │
         │              └───────────────────────┘                    │
         │                                                            │
         └───────────────────────── stop ────────────────────────────┘

All states can transition directly to IDLE on stop/error command.
═══════════════════════════════════════════════════════════════════════════

Transition table (predicate → allowed targets):
┌──────────────────┬─────────────────────────────────────────────────────────┐
│ From             │ To (on classifier output)                                │
├──────────────────┼─────────────────────────────────────────────────────────┤
│ IDLE             │ STANDING, LEVEL_WALKING, STAIR_*, RAMP_*, SIT_TO_STAND  │
│ STANDING         │ LEVEL_WALKING, STAIR_*, RAMP_*, SIT_TO_STAND, IDLE      │
│ LEVEL_WALKING    │ STAIR_ASCENT, STAIR_DESCENT, RAMP_ASCENT, RAMP_DESCENT, │
│                  │ STANDING, SIT_TO_STAND, IDLE                             │
│ STAIR_ASCENT     │ LEVEL_WALKING, STAIR_DESCENT, STANDING, IDLE            │
│ STAIR_DESCENT    │ LEVEL_WALKING, STAIR_ASCENT, STANDING, IDLE             │
│ RAMP_ASCENT      │ LEVEL_WALKING, RAMP_DESCENT, STANDING, IDLE             │
│ RAMP_DESCENT     │ LEVEL_WALKING, RAMP_ASCENT, STANDING, IDLE              │
│ SIT_TO_STAND     │ STANDING, LEVEL_WALKING, IDLE                           │
└──────────────────┴─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ── State definitions ────────────────────────────────────────────────────────

@dataclass
class FSMState:
    name: str
    torque_scale: float
    description: str
    color_code: str = ''     # for optional coloured terminal output


# Canonical 8 states
STATES: dict[str, FSMState] = {
    'IDLE':          FSMState('IDLE',          0.0, 'System off / initializing',         '\033[90m'),
    'STANDING':      FSMState('STANDING',      0.0, 'Static standing — no assist needed','\033[34m'),
    'LEVEL_WALKING': FSMState('LEVEL_WALKING', 1.0, 'Normal gait assistance',            '\033[32m'),
    'STAIR_ASCENT':  FSMState('STAIR_ASCENT',  1.5, 'Increased knee flexion assist',     '\033[33m'),
    'STAIR_DESCENT': FSMState('STAIR_DESCENT', 0.8, 'Eccentric brake mode',              '\033[35m'),
    'RAMP_ASCENT':   FSMState('RAMP_ASCENT',   1.2, 'Moderate uphill assist',            '\033[36m'),
    'RAMP_DESCENT':  FSMState('RAMP_DESCENT',  0.8, 'Downhill brake mode',               '\033[35m'),
    'SIT_TO_STAND':  FSMState('SIT_TO_STAND',  2.0, 'Hip/knee extension burst',          '\033[31m'),
}

_RESET = '\033[0m'

# Map from classifier output index → FSM state name
# (matches CLASS_NAMES in data_loader.py)
PREDICTION_TO_STATE: dict[int, str] = {
    0: 'LEVEL_WALKING',
    1: 'STAIR_ASCENT',
    2: 'STAIR_DESCENT',
    3: 'RAMP_ASCENT',
    4: 'RAMP_DESCENT',
    5: 'STANDING',
    6: 'SIT_TO_STAND',
}

# Allowed transitions (from_state → set of allowed to_states)
# All states can go to IDLE; we allow free transitions between locomotion modes
# (filtered by hysteresis).
ALLOWED_TRANSITIONS: dict[str, set] = {
    'IDLE':          {'STANDING', 'LEVEL_WALKING', 'STAIR_ASCENT', 'STAIR_DESCENT',
                      'RAMP_ASCENT', 'RAMP_DESCENT', 'SIT_TO_STAND'},
    'STANDING':      {'LEVEL_WALKING', 'STAIR_ASCENT', 'STAIR_DESCENT',
                      'RAMP_ASCENT', 'RAMP_DESCENT', 'SIT_TO_STAND', 'IDLE'},
    'LEVEL_WALKING': {'STAIR_ASCENT', 'STAIR_DESCENT', 'RAMP_ASCENT', 'RAMP_DESCENT',
                      'STANDING', 'SIT_TO_STAND', 'IDLE'},
    'STAIR_ASCENT':  {'LEVEL_WALKING', 'STAIR_DESCENT', 'STANDING', 'IDLE'},
    'STAIR_DESCENT': {'LEVEL_WALKING', 'STAIR_ASCENT', 'STANDING', 'IDLE'},
    'RAMP_ASCENT':   {'LEVEL_WALKING', 'RAMP_DESCENT', 'STANDING', 'IDLE'},
    'RAMP_DESCENT':  {'LEVEL_WALKING', 'RAMP_ASCENT', 'STANDING', 'IDLE'},
    'SIT_TO_STAND':  {'STANDING', 'LEVEL_WALKING', 'IDLE'},
}


# ── FSM class ────────────────────────────────────────────────────────────────

class ExoskeletonFSM:
    """
    8-state finite-state machine for exoskeleton assistive control.

    Hysteresis: the FSM only switches state when the prediction buffer
    contains `hysteresis` identical consecutive predictions.  This prevents
    rapid chattering on ambiguous transitions (e.g., walking ↔ standing).

    Torque scales:
      IDLE / STANDING  → 0.0  (no active assistance)
      LEVEL_WALKING    → 1.0  (baseline assistance)
      RAMP_ASCENT      → 1.2  (moderate uphill)
      STAIR_DESCENT /
      RAMP_DESCENT     → 0.8  (eccentric braking)
      STAIR_ASCENT     → 1.5  (high knee-flexion demand)
      SIT_TO_STAND     → 2.0  (burst mode, transient)
    """

    def __init__(self, hysteresis: int = 3):
        if hysteresis < 1:
            raise ValueError("hysteresis must be >= 1")
        self.hysteresis      = hysteresis
        self.current_state   = STATES['IDLE']
        self.pred_buffer: deque = deque(maxlen=hysteresis)
        self.step_count      = 0
        self.transition_log: List[Tuple[int, str, str]] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, prediction: int) -> FSMState:
        """
        Feed a new classifier prediction and optionally trigger a transition.

        Args:
            prediction: class index (0–6) from the classifier.

        Returns:
            Current FSMState after (potential) transition.
        """
        self.step_count += 1
        self.pred_buffer.append(prediction)

        # Only consider switching when buffer is full and unanimous
        if len(self.pred_buffer) == self.hysteresis and \
                len(set(self.pred_buffer)) == 1:
            desired_state_name = PREDICTION_TO_STATE.get(prediction, 'STANDING')
            self._try_transition(desired_state_name)

        return self.current_state

    def force_idle(self) -> None:
        """Immediately switch to IDLE (e.g., on emergency stop)."""
        if self.current_state.name != 'IDLE':
            self._log_transition(self.current_state.name, 'IDLE')
            self.current_state = STATES['IDLE']
        self.pred_buffer.clear()

    @property
    def torque(self) -> float:
        """Current torque scale for the actuators."""
        return self.current_state.torque_scale

    @property
    def state_name(self) -> str:
        """Name of the current FSM state."""
        return self.current_state.name

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _try_transition(self, desired: str) -> None:
        """Attempt a transition to `desired`; silently skip if not allowed."""
        current = self.current_state.name
        if desired == current:
            return
        allowed = ALLOWED_TRANSITIONS.get(current, set())
        if desired in allowed:
            self._log_transition(current, desired)
            self.current_state = STATES[desired]

    def _log_transition(self, from_name: str, to_name: str) -> None:
        self.transition_log.append((self.step_count, from_name, to_name))

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_transition_table(self) -> None:
        """Print a formatted table of all state transitions that occurred."""
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║             FSM Transition History                   ║")
        print("╠══════════════════════════════════════════════════════╣")
        print(f"║ {'Step':>5}  {'From':>16}  {'To':>16}  {'Torque':>6} ║")
        print("╠══════════════════════════════════════════════════════╣")
        if not self.transition_log:
            print("║  (no transitions recorded)                           ║")
        else:
            for step, frm, to in self.transition_log:
                t_scale = STATES[to].torque_scale
                print(f"║ {step:>5}  {frm:>16}  {to:>16}  {t_scale:>6.1f} ║")
        print("╚══════════════════════════════════════════════════════╝")

    def reset(self) -> None:
        """Reset FSM to initial IDLE state."""
        self.current_state = STATES['IDLE']
        self.pred_buffer.clear()
        self.step_count     = 0
        self.transition_log = []


# ── Demo / standalone runner ──────────────────────────────────────────────────

def demo(verbose: bool = True) -> None:
    """
    Run a 30-step synthetic prediction sequence through the FSM and
    print the step-by-step state transitions.

    Scenario:
      Steps 1–5   : STANDING (class 5)
      Steps 6–11  : LEVEL_WALKING (class 0)
      Steps 12–16 : STAIR_ASCENT  (class 1)
      Steps 17–21 : STAIR_DESCENT (class 2)
      Steps 22–26 : RAMP_ASCENT   (class 3)
      Steps 27–30 : STANDING      (class 5)
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Exoskeleton FSM Demo — 30-step synthetic sequence     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nHysteresis N=3: need 3 consecutive same predictions to switch.\n")

    fsm = ExoskeletonFSM(hysteresis=3)

    sequence = (
        [5] * 5   +   # STANDING
        [0] * 6   +   # LEVEL_WALKING
        [1] * 5   +   # STAIR_ASCENT
        [2] * 5   +   # STAIR_DESCENT
        [3] * 4   +   # RAMP_ASCENT
        [5] * 5       # STANDING
    )

    CLASS_NAMES = [
        'level_walking', 'stair_ascent', 'stair_descent',
        'ramp_ascent', 'ramp_descent', 'standing', 'sit_to_stand',
    ]

    prev_state = None
    header = f"{'Step':>4}  {'Prediction':>16}  {'FSM State':>16}  {'Torque':>7}  Note"
    print(header)
    print('─' * len(header))

    for i, pred in enumerate(sequence):
        state = fsm.update(pred)
        pred_name = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else '?'
        note = ''
        if prev_state is not None and state.name != prev_state:
            note = f'<-- TRANSITION from {prev_state}'
        if verbose:
            print(f"{i+1:>4}  {pred_name:>16}  "
                  f"{state.name:>16}  "
                  f"{state.torque_scale:>7.1f}  {note}")
        prev_state = state.name

    fsm.print_transition_table()

    # Torque scale reference table
    print("\n┌──────────────────────────────────────────────────────┐")
    print("│        Torque Scale Reference (all states)           │")
    print("├──────────────────┬───────────┬───────────────────────┤")
    print(f"│ {'State':16s} │ {'Scale':9s} │ {'Description':21s} │")
    print("├──────────────────┼───────────┼───────────────────────┤")
    for sname, st in STATES.items():
        print(f"│ {sname:16s} │ {st.torque_scale:9.1f} │ {st.description[:21]:21s} │")
    print("└──────────────────┴───────────┴───────────────────────┘")


if __name__ == '__main__':
    demo()
