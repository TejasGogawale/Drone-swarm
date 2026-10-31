#!/usr/bin/env python3
"""LLM client for drone command generation using Gemini (cloud), Groq/OpenAI, or Ollama.

Sends structured prompts (telemetry + YOLO detections + user command)
to an LLM and returns parsed JSON drone commands.

Maintains a rolling conversation history so the LLM remembers prior
missions, positions, and actions within a session.
"""

import json
import logging
import os
import re
import time
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


def _load_env_if_needed():
    env_path = os.path.expanduser('~/Desktop/swarm/.env')
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        k, v = k.strip(), v.strip()
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


SYSTEM_PROMPT = """You are the autonomous flight controller for a PX4 drone.
Given live telemetry, YOLO camera detections, and a user mission, reason step-by-step
then output a single JSON command.

You have memory of previous commands in this conversation — use that context when
the user gives follow-up instructions (e.g. "now go back", "continue", "stop that").

═══════════════════════════════════════════
COORDINATE SYSTEM — NED (North-East-Down)

  x = North   (metres from home,  positive → north)
  y = East    (metres from home,  positive → east)
  z = Down    (NEGATIVE = above home,  z=-20 means 20 m AGL)

  Altitude limits: z between -2 and -500 (e.g. 130 m height -> z = -130.0).

GPS → NED CONVERSION (when telemetry gives GPS coords):
  Δx = (lat − home_lat) × 111,139
  Δy = (lon − home_lon) × 111,139 × cos(home_lat_radians)
  z  = −(alt_msl − home_alt_msl)

  The telemetry includes both GPS and NED. Use NED x/y/z directly in commands.

═══════════════════════════════════════════
AVAILABLE PX4 COMMANDS

arm_offboard      Arm motors and enter PX4 OFFBOARD mode.
                  Required once before any movement. Check "Armed" in telemetry.
  {"action": "arm_offboard"}

position_ned      Fly to an absolute NED position (PX4 TrajectorySetpoint).
                  z must be negative for altitude above home.
                  yaw_rad: heading in radians, omit to keep current heading.
  {"action": "position_ned", "x": <m>, "y": <m>, "z": <m>, "yaw_rad": <float>}

orbit             Continuous circular loiter around a NED centre (runs at 10 Hz).
                  cx/cy = orbit centre in NED metres. alt_z must be negative.
                  target_class: YOLO class name to search for while orbiting (e.g.
                                "person", "car", "truck", "bicycle", "dog"). Include
                                ONLY when the user wants to find a specific object;
                                omit for pure loitering with no search goal.
  {"action": "orbit", "cx": <m>, "cy": <m>, "alt_z": <m>, "radius": <m>, "speed": <m/s>, "target_class": "<yolo_class>"}

square_survey     Fly a square survey pattern centred on current NED position.
  {"action": "square_survey", "side": <m>, "alt_z": <m>, "speed": <m/s>}

set_heading       Rotate to compass bearing without translating (0=North, 90=East).
  {"action": "set_heading", "heading_deg": <0-360>}

set_speed         Change cruise speed.
  {"action": "set_speed", "speed": <m/s>}

look_at_gps       Point camera/nose at a GPS coordinate.
  {"action": "look_at_gps", "lat": <deg>, "lon": <deg>, "alt": <m_msl>}

hold              Stop all motion immediately. Hover at current position.
  {"action": "hold"}

land              Land vertically at current position (MAV_CMD_NAV_LAND).
  {"action": "land"}

rtl               Return to launch and land (MAV_CMD_NAV_RETURN_TO_LAUNCH).
  {"action": "rtl"}

═══════════════════════════════════════════
DECISION RULES

- Arming and entering OFFBOARD mode are handled automatically by all movement commands.
- For takeoff (e.g. "take off to 15m"): output position_ned with current x, current y, and z = -altitude (e.g. z = -15.0).
- Relative movement: "go 100m north" → new_x = current_x + 100, keep y and z.
- Relative movement: "go 50m east"  → new_y = current_y + 50,  keep x and z.
- GPS target: apply GPS→NED formula, then output position_ned.
- "Search / scan / look around / survey / move in circles / fly in circles / circle / map the area / scan surrounding / map whole surrounding" (no specific object) → orbit, no target_class.
  If user asks to "map and scan the whole surrounding", use action="orbit" with radius=22.0, alt_z=-12.0 (or user's requested height), speed=5.0 so the 3D LiDAR sweeps all surrounding buildings, walls, and trees!
  If user specifies a height/altitude (e.g. "move in circles or go at 130 m height"), set alt_z = -<height> (e.g. -130.0) in the orbit or position_ned command!
  If user does not specify orbit centre, use current x and y (or 0, 0 if near home).
- "Go to <N> m height / fly at <N>m / climb to <N>m" → position_ned with current x, current y, and z = -<N> (or orbit with alt_z = -<N> if also asked to circle).
- "Find / look for / search for <object>" → orbit with target_class = YOLO class name.
  Common YOLO class names: "person", "car", "truck", "motorcycle", "bicycle",
                            "bus", "dog", "cat", "boat", "airplane".
- "Stop / cancel / abort / wait / freeze" → hold.
- Default orbit radius: 20 m. Default speed: 5 m/s.
- If intent is unclear or mission is done → hold.
- Use conversation history to handle follow-up commands like "go back", "repeat",
  "now land", "continue scanning", etc.

═══════════════════════════════════════════
OUTPUT FORMAT — always this exact shape, no markdown:

{
  "thought": "<step-by-step: what user wants, NED maths, which command and why>",
  "action": "<action name>",
  <parameters for that action>
}
"""


class LLMClient:
    """LLM client with rolling conversation memory supporting Gemini, Groq/OpenAI, and Ollama."""

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        ollama_url: str | None = None,
        max_history_turns: int = 10,
    ):
        _load_env_if_needed()

        self.ollama_url = (
            ollama_url
            or os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        ).rstrip('/')
        self.ollama_model = 'qwen2.5:3b'
        self.gemini_key = os.environ.get('GEMINI_API_KEY')
        self.groq_key = os.environ.get('GROQ_API_KEY')
        self.groq_url = os.environ.get('GROQ_BASE_URL', 'https://api.groq.com/openai/v1').rstrip('/')

        self.backend = 'ollama'
        self.model = self.ollama_model
        self.max_history_turns = max_history_turns
        self.history: list = []

        logger.info(
            f'LLM client initialized: primary=ollama ({self.ollama_model}) with instant Groq/Gemini fallback '
            f'(memory: {self.max_history_turns} turns)'
        )

    def reset_memory(self):
        """Clear conversation history."""
        self.history = []
        logger.info('LLM memory cleared.')

    def _call_ollama(self, user_content: str) -> str:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            *self.history,
            {'role': 'user', 'content': user_content},
        ]
        payload = json.dumps({
            'model': self.ollama_model,
            'messages': messages,
            'stream': False,
            'options': {'temperature': 0.2},
            'format': 'json',
        }).encode('utf-8')
        req = urllib.request.Request(
            f'{self.ollama_url}/api/chat',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        self.backend = 'ollama'
        self.model = self.ollama_model
        return result['message']['content'].strip()

    def _call_groq(self, user_content: str) -> str:
        if not self.groq_key:
            raise RuntimeError('No GROQ_API_KEY')
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            *self.history,
            {'role': 'user', 'content': user_content},
        ]
        for groq_model in ('qwen/qwen3.8-27b', 'openai/gpt-oss-20b', 'allam-2-7b'):
            payload = json.dumps({
                'model': groq_model,
                'messages': messages,
                'temperature': 0.2,
                'response_format': {'type': 'json_object'},
            }).encode('utf-8')
            req = urllib.request.Request(
                f'{self.groq_url}/chat/completions',
                data=payload,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.groq_key}',
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) drone_agent/0.1',
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read().decode('utf-8'))
                self.backend = 'qwen'
                self.model = groq_model
                return result['choices'][0]['message']['content'].strip()
            except Exception:
                continue
        raise RuntimeError('Groq models failed')

    def _call_gemini(self, user_content: str) -> str:
        if not self.gemini_key:
            raise RuntimeError('No GEMINI_API_KEY')
        gemini_contents = []
        for item in self.history:
            role = 'user' if item['role'] == 'user' else 'model'
            gemini_contents.append({'role': role, 'parts': [{'text': item['content']}]})
        gemini_contents.append({'role': 'user', 'parts': [{'text': user_content}]})

        payload = json.dumps({
            'systemInstruction': {'parts': [{'text': SYSTEM_PROMPT}]},
            'contents': gemini_contents,
            'generationConfig': {
                'responseMimeType': 'application/json',
                'temperature': 0.2,
            },
        }).encode('utf-8')

        for model_name in ('gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-flash', 'gemini-2.0-flash-lite'):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_key}"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={'Content-Type': 'application/json'},
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    result = json.loads(resp.read().decode('utf-8'))
                candidate = result['candidates'][0]
                self.backend = 'gemini'
                self.model = model_name
                return ''.join(
                    p['text'] for p in candidate.get('content', {}).get('parts', []) if 'text' in p
                ).strip()
            except Exception:
                continue
        raise RuntimeError('Gemini models failed')

    def ask(self, drone_state: str, user_command: str, yolo_detections: str) -> dict:
        """Send telemetry + command to LLM and return parsed JSON command."""
        user_content = (
            f"USER MISSION: {user_command}\n\n"
            f"CURRENT DRONE STATE:\n{drone_state}\n\n"
            f"YOLO DETECTIONS:\n{yolo_detections}\n\n"
            "Respond with ONLY a JSON command."
        )

        text = ''
        try:
            for caller in (self._call_groq, self._call_ollama, self._call_gemini):
                try:
                    text = caller(user_content)
                    if text:
                        break
                except Exception:
                    continue
            if not text:
                logger.warning('All LLM backends unreachable; using rule-based command parser fallback.')
                return self._rule_based_fallback(user_command, drone_state)

            # Clean markdown code fences if present
            text = text.replace('```json', '').replace('```', '').strip()
            command = json.loads(text)

            # Some models wrap the action under a nested key or use take_off
            if 'action' not in command:
                if 'command' in command and isinstance(command['command'], dict) and 'action' in command['command']:
                    command = command['command']
                else:
                    raise ValueError(f"LLM response missing 'action' key: {command}")

            # Store in rolling history
            self.history.append({'role': 'user', 'content': user_content})
            self.history.append({'role': 'assistant', 'content': text})

            max_msgs = self.max_history_turns * 2
            if len(self.history) > max_msgs:
                self.history = self.history[-max_msgs:]

            logger.info(
                f'LLM [{self.backend}/{self.model}] command: {command}  '
                f'[memory: {len(self.history) // 2}/{self.max_history_turns} turns]'
            )
            return command

        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse LLM response as JSON: {text!r}; using rule-based fallback.')
            return self._rule_based_fallback(user_command, drone_state)
        except Exception as e:
            logger.error(f'LLM API call failed ({self.backend}): {e}; using rule-based fallback.')
            return self._rule_based_fallback(user_command, drone_state)

    def _rule_based_fallback(self, user_command: str, drone_state: str) -> dict:
        """Deterministic rule-based command parser if cloud/local LLM is offline or throttled."""
        self.backend = 'rule_fallback'
        self.model = 'builtin_parser'
        cmd_lower = user_command.lower()

        # Extract current NED from drone_state if present
        cur_x, cur_y, cur_z = 0.0, 0.0, -10.0
        m_pos = re.search(r'Local position \(NED\):\s*x=([-\d.]+)m,\s*y=([-\d.]+)m,\s*z=([-\d.]+)m', drone_state)
        if m_pos:
            cur_x, cur_y, cur_z = float(m_pos.group(1)), float(m_pos.group(2)), float(m_pos.group(3))

        # Extract any height/altitude or generic metres number
        alt_val = None
        m_alt = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre|meters|metres)?\s*(?:height|hight|alt|altitude|high|up|agl)', cmd_lower)
        if not m_alt:
            m_alt = re.search(r'(?:at|to|height|hight|alt|altitude)\s*(\d+(?:\.\d+)?)\s*(?:m|meter|metre|meters|metres)?', cmd_lower)
        if m_alt:
            alt_val = float(m_alt.group(1))

        # Check for circle / orbit / map surrounding
        if any(w in cmd_lower for w in ('circle', 'orbit', 'loop', 'around', 'search', 'scan', 'map', 'surrounding', 'find', 'survivor', 'victim')):
            default_alt = -9.5 if ('map' in cmd_lower or 'surrounding' in cmd_lower or 'scan' in cmd_lower or 'search' in cmd_lower or 'find' in cmd_lower) else (cur_z if cur_z < -5.0 else -12.0)
            alt_z = -abs(alt_val) if alt_val is not None else default_alt
            radius = 15.0
            m_rad = re.search(r'radius\s*(?:of\s*)?(\d+(?:\.\d+)?)', cmd_lower)
            if m_rad:
                radius = float(m_rad.group(1))
            target_class = None
            if any(w in cmd_lower for w in ('person', 'survivor', 'victim', 'human', 'people')):
                target_class = 'person'
            elif any(w in cmd_lower for w in ('car', 'vehicle', 'prius', 'ambulance', 'suv')):
                target_class = 'car'
            else:
                for cls in ('truck', 'dog', 'bicycle'):
                    if cls in cmd_lower:
                        target_class = cls
                        break
            res = {
                'thought': f'Rule fallback: orbiting central disaster plaza (0.0, 0.0) at {abs(alt_z):.1f}m height, radius {radius}m.',
                'action': 'orbit',
                'cx': 0.0,
                'cy': 0.0,
                'alt_z': alt_z,
                'radius': radius,
                'speed': 3.2,
            }
            if target_class:
                res['target_class'] = target_class
            return res

        # Square pattern
        if 'square' in cmd_lower or 'box' in cmd_lower:
            alt_z = -abs(alt_val) if alt_val is not None else (cur_z if cur_z < -3.0 else -15.0)
            return {
                'thought': f'Rule fallback: flying square survey at {abs(alt_z):.1f}m.',
                'action': 'square_survey',
                'side': 25.0,
                'alt_z': alt_z,
                'speed': 4.0,
            }

        # Land / RTL / Hold
        if 'rtl' in cmd_lower or 'return' in cmd_lower or 'home' in cmd_lower:
            return {'thought': 'Rule fallback: returning to launch.', 'action': 'rtl'}
        if 'land' in cmd_lower:
            return {'thought': 'Rule fallback: landing.', 'action': 'land'}
        if any(w in cmd_lower for w in ('stop', 'hold', 'hover', 'wait', 'freeze')):
            return {'thought': 'Rule fallback: holding position.', 'action': 'hold'}

        # Directional movement (north/south/east/west) — supports combined directions e.g. "12m north and 14m west at 10m height"
        nx, ny, nz = cur_x, cur_y, (cur_z if cur_z < -2.0 else -12.0)
        if alt_val is not None:
            nz = -abs(alt_val)

        moved_dir = False
        for d_val, d_word in re.findall(r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre|meters|metres)?\s*(north|south|east|west)', cmd_lower):
            v = float(d_val)
            moved_dir = True
            if d_word == 'north':
                nx = v if 'fly to' in cmd_lower else (nx + v)
            elif d_word == 'south':
                nx = -v if 'fly to' in cmd_lower else (nx - v)
            elif d_word == 'east':
                ny = v if 'fly to' in cmd_lower else (ny + v)
            elif d_word == 'west':
                ny = -v if 'fly to' in cmd_lower else (ny - v)

        if not moved_dir:
            dist = 20.0
            m_dist = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre|meters|metres)', cmd_lower)
            if m_dist:
                dist = float(m_dist.group(1))
            if 'north' in cmd_lower:
                nx += dist
                moved_dir = True
            if 'south' in cmd_lower:
                nx -= dist
                moved_dir = True
            if 'east' in cmd_lower:
                ny += dist
                moved_dir = True
            if 'west' in cmd_lower:
                ny -= dist
                moved_dir = True

        if moved_dir or alt_val is not None or 'takeoff' in cmd_lower or 'take off' in cmd_lower or 'fly' in cmd_lower or 'go' in cmd_lower:
            if not moved_dir and alt_val is None:
                nz = -abs(dist)
            return {
                'thought': f'Rule fallback: flying to NED ({nx:.1f}, {ny:.1f}, {nz:.1f}).',
                'action': 'position_ned',
                'x': nx,
                'y': ny,
                'z': nz,
            }

        return {'thought': 'Rule fallback: holding.', 'action': 'hold'}
