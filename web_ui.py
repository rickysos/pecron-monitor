"""Built-in web dashboard for pecron-monitor.

Standard-library only (no extra dependencies). Run a monitor with --web to
start a live dashboard:

    pecron-monitor --web                 # http://<host>:8090/
    pecron-monitor --web --web-port 9000

Or enable it in config.yaml:

    web:
      enabled: true
      host: 0.0.0.0
      port: 8090

API:
    GET  /              dashboard
    GET  /api/status    live telemetry for all devices
    GET  /api/controls  writable controls + current values
    POST /api/control   {"device_key", "code", "value"} -> send_control()
    GET  /api/logs      recent log lines
    GET  /api/raw       raw device data
"""

import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from constants import (
    BATTERY_CAPACITY_WH,
    DEVICE_STATUS_LABELS,
    ENUM_LABELS,
    SENSOR_FIELDS,
)
from helpers import _get_kv
from monitor_data import coerce_switch, extract_power, extract_soc, extract_voltage

log = logging.getLogger("pecron")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8090
LOG_BUFFER_SIZE = 200
SAMPLE_INTERVAL = 5.0
HISTORY_SECONDS = 24 * 3600
SAMPLE_KEYS = ("battery_percent", "voltage", "temperature_c", "total_input_w", "total_output_w")


class HistoryBuffer:
    """Per-device ring buffer of sampled telemetry snapshots."""

    def __init__(self, max_seconds=HISTORY_SECONDS, interval=SAMPLE_INTERVAL):
        self.max_samples = max(10, int(max_seconds / interval))
        self.samples = {}  # device_key -> deque[(ts, {field: value})]

    def add(self, device_key: str, payload: dict):
        rec = (time.time(), {k: payload.get(k) for k in SAMPLE_KEYS})
        dq = self.samples.setdefault(device_key, deque(maxlen=self.max_samples))
        dq.append(rec)

    def since(self, device_key: str, since_ts: float):
        return [s for s in self.samples.get(device_key, []) if s[0] >= since_ts]


class LogBufferHandler(logging.Handler):
    """Keeps the last N formatted log lines in a ring buffer."""

    def __init__(self, maxlen=LOG_BUFFER_SIZE):
        super().__init__()
        self.lines = deque(maxlen=maxlen)
        self._fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    def emit(self, record):
        try:
            self.lines.append(self._fmt.format(record))
        except Exception:
            pass


def _fmt_hours(hours: float) -> str:
    if hours < 0 or hours != hours:
        return "N/A"
    total_min = int(hours * 60)
    days, rem = divmod(total_min, 24 * 60)
    h, m = divmod(rem, 60)
    if days > 0:
        return f"{days}d {h}h {m}m"
    return f"{h}h {m}m"


class PecronWebUI:
    """Serves a live dashboard + JSON API over a monitor instance."""

    def __init__(self, monitor, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.monitor = monitor
        self.host = host
        self.port = port
        self.log_buffer = LogBufferHandler()
        self.history = HistoryBuffer()
        self._started_at = time.time()
        self._last_data_at = {}  # device_key -> ts of last observed telemetry change
        self._last_sample_sig = {}
        self._server = None
        self._thread = None
        self._stop_evt = threading.Event()

    # --- lifecycle ---

    def start(self):
        logging.getLogger("pecron").addHandler(self.log_buffer)
        handler = _make_handler(self)
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), handler)
        except OSError as e:
            log.error("Web UI failed to bind %s:%d — %s", self.host, self.port, e)
            raise
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="pecron-web", daemon=True
        )
        self._thread.start()
        self._stop_evt.clear()
        sampler = threading.Thread(target=self._sample_loop, name="pecron-web-sampler", daemon=True)
        sampler.start()
        shown = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        log.info(
            "Web UI: http://%s:%d/ (LAN: http://<this-machine's IP>:%d/)",
            shown,
            self.port,
            self.port,
        )

    def _sample_loop(self):
        while not self._stop_evt.is_set():
            try:
                for d in self.monitor.devices:
                    dk = d["device_key"]
                    p = _device_payload(self, d)
                    self.history.add(dk, p)
                    sig = (
                        p.get("battery_percent"),
                        p.get("voltage"),
                        p.get("total_input_w"),
                        p.get("total_output_w"),
                        p.get("net_power_w"),
                        (p.get("switches") or {}).get("ac"),
                        (p.get("switches") or {}).get("dc"),
                    )
                    if sig != self._last_sample_sig.get(dk):
                        self._last_sample_sig[dk] = sig
                        self._last_data_at[dk] = time.time()
            except Exception:
                pass
            self._stop_evt.wait(SAMPLE_INTERVAL)

    def stop(self):
        self._stop_evt.set()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


# --- data builders ---


def _device_payload(ui: PecronWebUI, d: dict) -> dict:
    m = ui.monitor
    dk = d["device_key"]
    kv = m.latest_data.get(dk, {})
    source = m.data_sources.get(dk)

    soc = extract_soc(kv)
    voltage = extract_voltage(kv)
    temp = _get_kv(kv, SENSOR_FIELDS["temperature"], None)
    try:
        temp = int(temp) if temp is not None else None
    except (TypeError, ValueError):
        temp = None

    total_in = extract_power(kv, "total_input_power", "ac_input_power", "dc_input_power")
    total_out = extract_power(kv, "total_output_power", "ac_output_power", "dc_output_power")

    def _i(key):
        v = _get_kv(kv, key, 0)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    current = _get_kv(kv, SENSOR_FIELDS["current"], 0)
    try:
        net_power = round(float(voltage or 0) * float(current or 0), 1)
    except (TypeError, ValueError):
        net_power = None

    remain = _i("remain_time")
    remain_ch = _i("remain_charging_time")
    # Local transports report unreliable remain_time (see monitor._process_data)
    if source in ("LOCAL TCP", "BLE") and remain <= 5 and (soc or 0) > 50:
        remain = None
    if remain is not None and (remain <= 0 or remain >= 65535):
        remain = None
    if remain_ch <= 0:
        remain_ch = None

    status_val = _get_kv(kv, SENSOR_FIELDS["device_status_hm"], None)
    try:
        status_val = int(status_val)
    except (TypeError, ValueError):
        status_val = None
    status_label = DEVICE_STATUS_LABELS.get(status_val, None) if status_val is not None else None

    capacity_wh = None
    name = d.get("device_name") or d.get("product_name") or d.get("name")
    if name in BATTERY_CAPACITY_WH:
        try:
            capacity_wh = float(BATTERY_CAPACITY_WH[name])
        except (TypeError, ValueError):
            capacity_wh = None

    est_full = est_empty = None
    if capacity_wh and soc is not None and 0 <= soc <= 100 and net_power:
        charge_wh = capacity_wh * (soc / 100.0)
        if net_power < 0:
            est_empty = _fmt_hours(charge_wh / abs(net_power))
        elif net_power > 0:
            est_full = _fmt_hours((capacity_wh - charge_wh) / net_power)

    def _switch(field_name):
        return coerce_switch(_get_kv(kv, SENSOR_FIELDS[field_name], None))

    return {
        "device_key": dk,
        "name": name,
        "source": source,
        "status": {"value": status_val, "label": status_label},
        "battery_percent": soc,
        "voltage": round(voltage, 2) if voltage else None,
        "temperature_c": temp,
        "total_input_w": total_in,
        "total_output_w": total_out,
        "ac_input_w": _i(SENSOR_FIELDS["ac_input_power"]),
        "dc_input_w": _i(SENSOR_FIELDS["dc_input_power"]),
        "ac_output_w": _i(SENSOR_FIELDS["ac_output_power"]),
        "dc_output_w": _i(SENSOR_FIELDS["dc_output_power"]),
        "ac_output_voltage": _get_kv(kv, SENSOR_FIELDS["ac_output_voltage"], None),
        "net_power_w": net_power,
        "remain_discharge_min": remain,
        "remain_charge_min": remain_ch,
        "capacity_wh": capacity_wh,
        "est_time_to_empty": est_empty,
        "est_time_to_full": est_full,
        "switches": {
            "ac": _switch("ac_switch"),
            "dc": _switch("dc_switch"),
            "ups": _switch("ups_mode"),
        },
    }


def _controls_payload(ui: PecronWebUI) -> dict:
    m = ui.monitor
    out = {}
    for d in m.devices:
        dk = d["device_key"]
        kv = m.latest_data.get(dk, {})
        controls = d.get("controls") or d.get("tsl_cache") or {}
        writable = []
        for code, info in sorted(controls.items(), key=lambda x: x[1].get("id", 0)):
            access = str(info.get("access", "R")).upper()
            if "W" not in access:
                continue
            ctype = str(info.get("type", "BOOL")).upper()
            # STRUCT/ARRAY/TEXT entries are telemetry payloads, not settable controls
            if ctype in ("STRUCT", "ARRAY", "TEXT"):
                continue
            value = m._extract_value_by_key(kv, code)
            writable.append(
                {
                    "code": code,
                    "id": info.get("id"),
                    "type": ctype,
                    "desc": info.get("desc", ""),
                    "value": value,
                    "enum_labels": ENUM_LABELS.get(code),
                }
            )
        out[dk] = writable
    return out


def _history_from_prometheus(ui: PecronWebUI, minutes: int):
    """Fetch history from a local Prometheus instance. Returns None to fall back."""
    import urllib.parse
    import urllib.request

    m = ui.monitor
    if not m.devices:
        return None
    base_url = (m.config.get("web") or {}).get("prometheus_url", "http://127.0.0.1:9090")
    if not base_url:
        return None
    dk = m.devices[0]["device_key"]
    seconds = minutes * 60
    step = (
        15 if seconds <= 3600 else 60 if seconds <= 86400 else 300 if seconds <= 7 * 86400 else 900
    )
    end = int(time.time())
    start = end - seconds

    series = {}
    for key, metric in PROMETHEUS_HISTORY_METRICS.items():
        query = f'{metric}{{device_key="{dk}"}}'
        url = (
            base_url.rstrip("/")
            + "/api/v1/query_range?"
            + urllib.parse.urlencode({"query": query, "start": start, "end": end, "step": step})
        )
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                result = json.load(resp)["data"]["result"]
        except Exception:
            return None
        if not result:
            return None  # nothing stored yet — fall back to memory buffer
        series[key] = {round(float(ts)): float(val) for ts, val in result[0]["values"]}

    ts_axis = sorted(set().union(*[s.keys() for s in series.values()]))
    if not ts_axis:
        return None
    return {
        "minutes": minutes,
        "source": "prometheus",
        "devices": {
            dk: {
                "t": [float(t) for t in ts_axis],
                "v": {k: [series[k].get(t) for t in ts_axis] for k in PROMETHEUS_HISTORY_METRICS},
            }
        },
    }


def _history_payload(ui: PecronWebUI, full_path: str) -> dict:
    from urllib.parse import parse_qs

    query = parse_qs(full_path.split("?", 1)[1] if "?" in full_path else "")
    try:
        minutes = min(max(int(query.get("minutes", ["60"])[0] or 60), 5), 43200)
    except ValueError:
        minutes = 60

    prom = _history_from_prometheus(ui, minutes)
    if prom:
        return prom

    since = time.time() - minutes * 60
    devices = {}
    for dk in list(ui.history.samples):
        sel = ui.history.since(dk, since)
        if not sel:
            continue
        devices[dk] = {
            "t": [round(s[0], 1) for s in sel],
            "v": {k: [s[1].get(k) for s in sel] for k in SAMPLE_KEYS},
        }
    return {"minutes": minutes, "source": "memory", "devices": devices}


# --- Prometheus exporter ---

PROMETHEUS_SOURCES = ("BLE", "LOCAL TCP", "CLOUD MQTT", "REST API")

# Dashboard key -> Prometheus metric name (for reading history back)
PROMETHEUS_HISTORY_METRICS = {
    "battery_percent": "pecron_battery_percent",
    "voltage": "pecron_battery_voltage",
    "temperature_c": "pecron_battery_temperature_c",
    "total_input_w": "pecron_input_power_w",
    "total_output_w": "pecron_output_power_w",
}


def _esc_label(value) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def build_metrics(ui: PecronWebUI) -> str:
    """Render the Prometheus text exposition format (version 0.0.4)."""
    m = ui.monitor
    now = time.time()
    ts_ms = int(now * 1000)
    poll = float(m.config.get("poll_interval", 70) or 70)
    online_threshold = max(120.0, 3 * poll)

    families = {}  # name -> (help, type, [(labels, value)])

    def add(name, help_text, mtype, labels, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if v != v:  # NaN
            return
        fam = families.setdefault(name, (help_text, mtype, []))
        fam[2].append((labels, v))

    add(
        "pecron_monitor_running",
        "Monitor process is running (1 = yes).",
        "gauge",
        {},
        1.0 if m._running else 0.0,
    )
    add(
        "pecron_monitor_uptime_seconds",
        "Seconds since the web UI started.",
        "gauge",
        {},
        round(now - ui._started_at, 1),
    )

    for d in m.devices:
        dk = d["device_key"]
        name = d.get("device_name") or d.get("product_name") or d.get("name")
        base = {"device": name, "device_key": dk}
        p = _device_payload(ui, d)

        last_at = ui._last_data_at.get(dk)
        age = (now - last_at) if last_at else None
        add(
            "pecron_device_online",
            "Device reported telemetry within the expected window (1 = online).",
            "gauge",
            base,
            1.0 if (age is None or age < online_threshold) else 0.0,
        )
        if age is not None:
            add(
                "pecron_data_age_seconds",
                "Seconds since the device last reported changed telemetry.",
                "gauge",
                base,
                round(age, 1),
            )
        source = p.get("source")
        for src in PROMETHEUS_SOURCES:
            add(
                "pecron_data_source",
                "Current telemetry source (1 = active).",
                "gauge",
                {**base, "source": src},
                1.0 if source == src else 0.0,
            )

        add(
            "pecron_battery_percent",
            "Battery state of charge (percent).",
            "gauge",
            base,
            p.get("battery_percent"),
        )
        add("pecron_battery_voltage", "Battery pack voltage (V).", "gauge", base, p.get("voltage"))
        add(
            "pecron_battery_temperature_c",
            "Battery temperature (deg C).",
            "gauge",
            base,
            p.get("temperature_c"),
        )
        add("pecron_capacity_wh", "Battery capacity (Wh).", "gauge", base, p.get("capacity_wh"))
        add(
            "pecron_net_power_w",
            "Net battery power; positive = charging, negative = draining (W).",
            "gauge",
            base,
            p.get("net_power_w"),
        )
        add("pecron_input_power_w", "Total input power (W).", "gauge", base, p.get("total_input_w"))
        add(
            "pecron_output_power_w",
            "Total output power (W).",
            "gauge",
            base,
            p.get("total_output_w"),
        )
        add("pecron_ac_input_power_w", "AC input power (W).", "gauge", base, p.get("ac_input_w"))
        add("pecron_dc_input_power_w", "DC input power (W).", "gauge", base, p.get("dc_input_w"))
        add("pecron_ac_output_power_w", "AC output power (W).", "gauge", base, p.get("ac_output_w"))
        add("pecron_dc_output_power_w", "DC output power (W).", "gauge", base, p.get("dc_output_w"))
        add(
            "pecron_ac_output_voltage",
            "AC output voltage (V).",
            "gauge",
            base,
            p.get("ac_output_voltage"),
        )
        add(
            "pecron_remaining_discharge_minutes",
            "Estimated remaining discharge time (minutes).",
            "gauge",
            base,
            p.get("remain_discharge_min"),
        )
        add(
            "pecron_remaining_charge_minutes",
            "Estimated remaining charge time (minutes).",
            "gauge",
            base,
            p.get("remain_charge_min"),
        )

        cap, soc, net = p.get("capacity_wh"), p.get("battery_percent"), p.get("net_power_w")
        if cap and soc is not None and net:
            charge_wh = cap * (soc / 100.0)
            if net < 0:
                add(
                    "pecron_est_time_to_empty_seconds",
                    "Estimated time to empty at current draw (seconds).",
                    "gauge",
                    base,
                    round(charge_wh / abs(net) * 3600, 0),
                )
            elif net > 0:
                add(
                    "pecron_est_time_to_full_seconds",
                    "Estimated time to full at current charge (seconds).",
                    "gauge",
                    base,
                    round((cap - charge_wh) / net * 3600, 0),
                )

        sw = p.get("switches") or {}
        for key, fname in (
            ("ac", "pecron_ac_switch"),
            ("dc", "pecron_dc_switch"),
            ("ups", "pecron_ups_switch"),
        ):
            state = sw.get(key)
            if state is not None:
                add(
                    fname,
                    "Output switch state (1 = on)." if key != "ups" else "UPS mode state (1 = on).",
                    "gauge",
                    base,
                    1.0 if state else 0.0,
                )

    out = []
    for name, (help_text, mtype, samples) in families.items():
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} {mtype}")
        for labels, v in samples:
            lbl = (
                "{" + ",".join(f'{k}="{_esc_label(vv)}"' for k, vv in labels.items()) + "}"
                if labels
                else ""
            )
            out.append(f"{name}{lbl} {v} {ts_ms}")
    return "\n".join(out) + "\n"


def build_status(ui: PecronWebUI) -> dict:
    m = ui.monitor
    return {
        "ts": time.time(),
        "monitor": {
            "running": m._running,
            "offline_mode": m.offline_mode,
            "devices_configured": len(m.devices),
        },
        "devices": [_device_payload(ui, d) for d in m.devices],
    }


# --- HTTP ---


def _make_handler(ui: PecronWebUI):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep monitor logs clean
            log.debug("web: %s", fmt % args)

        def _send(self, code, body, ctype):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=str), "application/json")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, HTML_PAGE, "text/html; charset=utf-8")
            elif path == "/api/status":
                self._json(build_status(ui))
            elif path == "/api/controls":
                self._json(_controls_payload(ui))
            elif path == "/api/history":
                self._json(_history_payload(ui, self.path))
            elif path == "/metrics":
                self._send(200, build_metrics(ui), "text/plain; version=0.0.4; charset=utf-8")
            elif path == "/api/logs":
                self._json({"lines": list(ui.log_buffer.lines)})
            elif path == "/api/raw":
                self._json({dk: kv for dk, kv in ui.monitor.latest_data.items()})
            elif path == "/healthz":
                self._send(200, "ok", "text/plain")
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path != "/api/control":
                self._json({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, TypeError):
                self._json({"error": "invalid JSON body"}, 400)
                return
            device_key = body.get("device_key")
            code = body.get("code")
            value = body.get("value")
            if not device_key or not code or "value" not in body:
                self._json({"error": "need device_key, code, value"}, 400)
                return
            if not isinstance(value, (bool, int, float)):
                word = str(value).strip().lower()
                if word in ("true", "on", "1"):
                    value = True
                elif word in ("false", "off", "0"):
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        self._json({"error": f"invalid value: {value!r}"}, 400)
                        return
            sent = ui.monitor.send_control(device_key, code, value)
            self._json({"ok": bool(sent), "device_key": device_key, "code": code, "value": value})

    return Handler


# --- dashboard page ---

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pecron Monitor</title>
<style>
  :root {
    --bg: #0e1116; --card: #161b23; --card2: #1c2330; --line: #2a3444;
    --text: #e6edf3; --dim: #8b98a9; --ok: #3fb950; --warn: #d29922;
    --bad: #f85149; --accent: #58a6ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 16px 16px 40px; }
  header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; background: var(--card2);
           border: 1px solid var(--line); color: var(--dim); }
  .badge.src { color: var(--accent); }
  .dim { color: var(--dim); font-size: 12px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
  .card .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--dim); }
  .card .value { font-size: 24px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .card .sub { font-size: 12px; color: var(--dim); margin-top: 2px; }
  .battery-card { display: flex; align-items: center; gap: 18px; grid-column: span 2; }
  .ring-wrap { position: relative; width: 132px; height: 132px; flex: none; }
  .ring-wrap svg { transform: rotate(-90deg); }
  .ring-bg { fill: none; stroke: var(--card2); stroke-width: 10; }
  .ring-fg { fill: none; stroke: var(--ok); stroke-width: 10; stroke-linecap: round;
             transition: stroke-dashoffset .6s ease, stroke .6s ease; }
  .ring-label { position: absolute; inset: 0; display: flex; flex-direction: column;
                align-items: center; justify-content: center; }
  .ring-label .pct { font-size: 30px; font-weight: 700; }
  .ring-label .st { font-size: 11px; color: var(--dim); }
  .battery-info { flex: 1; min-width: 0; }
  .battery-info .row { display: flex; justify-content: space-between; padding: 3px 0;
                       border-bottom: 1px dashed var(--line); }
  .battery-info .row:last-child { border-bottom: 0; }
  .battery-info .k { color: var(--dim); }
  .chg { font-weight: 600; }
  .chg.in  { color: var(--ok); }
  .chg.out { color: var(--warn); }
  .switches { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
  .sw { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .sw .name { font-size: 13px; font-weight: 600; }
  .sw .state { font-size: 11px; color: var(--dim); }
  button.tog { cursor: pointer; border-radius: 18px; border: 1px solid var(--line);
               background: var(--card2); color: var(--dim); width: 46px; height: 24px;
               position: relative; transition: background .15s; flex: none; }
  button.tog .knob { position: absolute; top: 2px; left: 2px; width: 18px; height: 18px;
                     border-radius: 50%; background: var(--dim); transition: left .15s, background .15s; }
  button.tog.on { background: #1f4d2b; border-color: var(--ok); }
  button.tog.on .knob { left: 24px; background: var(--ok); }
  button.tog:disabled { opacity: .5; cursor: default; }
  .controls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }
  .ctrl { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  .ctrl .meta { min-width: 0; }
  .ctrl .code { font-size: 12px; font-weight: 600; word-break: break-all; }
  .ctrl .desc { font-size: 11px; color: var(--dim); }
  .ctrl .cur { font-size: 11px; color: var(--accent); }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--dim);
       margin: 22px 0 10px; font-weight: 600; }
  .graphs { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .g-card { padding: 10px 12px; }
  .g-title { font-size: 12px; color: var(--dim); margin-bottom: 6px; font-weight: 600; }
  .g-card canvas { width: 100%; height: 140px; display: block; }
  .range-btns { margin-left: 10px; }
  .range-btn { cursor: pointer; font-size: 11px; padding: 2px 10px; border-radius: 10px;
               border: 1px solid var(--line); background: var(--card2); color: var(--dim); margin-left: 4px; }
  .range-btn.active { color: var(--accent); border-color: var(--accent); }
  #logs { background: #0a0d12; border: 1px solid var(--line); border-radius: 10px; padding: 10px;
          font: 11px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; color: #9fb0c3;
          height: 170px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
  details.raw summary { cursor: pointer; color: var(--dim); font-size: 12px; }
  details.raw pre { background: #0a0d12; border: 1px solid var(--line); border-radius: 10px;
                    padding: 10px; font: 11px/1.4 ui-monospace, Menlo, monospace; overflow-x: auto;
                    max-height: 300px; overflow-y: auto; }
  .offline { background: #2b1d1d; border-color: var(--bad); color: #ffb4ab; padding: 10px 14px;
             border-radius: 10px; margin-bottom: 14px; font-size: 13px; }
  a { color: var(--accent); text-decoration: none; }
  @media (max-width: 700px) { .graphs { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>⚡ Pecron Monitor</h1>
    <span class="badge" id="devname"></span>
    <span class="badge src" id="source">connecting…</span>
    <span class="badge" id="updated"></span>
  </header>

  <div id="banner"></div>
  <div id="devices"></div>

  <h2>History
    <span class="range-btns">
      <button class="range-btn" onclick="setRange(60,this)">1h</button>
      <button class="range-btn active" onclick="setRange(360,this)">6h</button>
      <button class="range-btn" onclick="setRange(1440,this)">24h</button>
      <button class="range-btn" onclick="setRange(10080,this)">7d</button>
      <button class="range-btn" onclick="setRange(43200,this)">30d</button>
    </span>
    <span class="badge" id="hist-src">…</span>
  </h2>
  <div class="graphs">
    <div class="card g-card"><div class="g-title">Battery %</div><canvas id="g-battery"></canvas></div>
    <div class="card g-card"><div class="g-title">Power (W) — input / output</div><canvas id="g-power"></canvas></div>
    <div class="card g-card"><div class="g-title">Voltage (V)</div><canvas id="g-voltage"></canvas></div>
    <div class="card g-card"><div class="g-title">Temperature (°C)</div><canvas id="g-temp"></canvas></div>
  </div>

  <h2>Logs</h2>
  <div id="logs"></div>

  <h2>Raw data</h2>
  <details class="raw">
    <summary>Show raw JSON</summary>
    <pre id="raw"></pre>
  </details>
</div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
let lastSwState = {};

function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function fmtMin(m) {
  if (m == null) return "N/A";
  if (m >= 1440) return Math.floor(m / 1440) + "d " + Math.floor((m % 1440) / 60) + "h";
  return Math.floor(m / 60) + "h " + (m % 60) + "m";
}

function card(label, value, sub, cls) {
  return `<div class="card"><div class="label">${esc(label)}</div>
    <div class="value ${cls || ''}">${value == null ? 'N/A' : esc(value)}</div>
    ${sub ? `<div class="sub">${esc(sub)}</div>` : ''}</div>`;
}

function toggle(deviceKey, code, name, state) {
  const on = state ? 'on' : '';
  const checked = state ? 1 : 0;
  return `<div class="card sw"><div><div class="name">${esc(name)}</div>
    <div class="state">${state ? 'ON' : 'OFF'}</div></div>
    <button class="tog ${on}" data-device="${esc(deviceKey)}" data-code="${esc(code)}"
      onclick="sendControl(this)" ${state ? '' : ''}><span class="knob"></span></button></div>`;
}

async function sendControl(btn) {
  const device = btn.dataset.device, code = btn.dataset.code;
  const want = !btn.classList.contains('on');
  btn.disabled = true;
  btn.title = "sending…";
  try {
    const r = await fetch('/api/control', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({device_key: device, code, value: want})});
    const j = await r.json();
    if (!j.ok) btn.title = "failed — see logs";
  } catch (e) { btn.title = "error: " + e; }
  btn.disabled = false;
  refresh();
}

async function loadControls(devicesEl) {
  let ctl = {};
  try { ctl = await (await fetch('/api/controls')).json(); } catch (e) { return; }
  devicesEl.querySelectorAll('.controls-grid').forEach((g) => {
    const dk = g.dataset.device;
    const list = ctl[dk] || [];
    g.innerHTML = list.map(c => {
      const val = c.value == null ? '' : (c.value === true ? 'ON' : c.value === false ? 'OFF' : c.value);
      let input;
      if (c.type === 'BOOL') {
        input = `<button class="tog ${c.value ? 'on' : ''}" data-device="${esc(dk)}"
          data-code="${esc(c.code)}" onclick="sendControl(this)"><span class="knob"></span></button>`;
      } else if (c.type === 'ENUM' && c.enum_labels) {
        input = `<select data-device="${esc(dk)}" data-code="${esc(c.code)}" class="ctl-select">
          ${Object.entries(c.enum_labels).map(([k, v]) =>
            `<option value="${esc(k)}" ${Number(k) === Number(c.value) ? 'selected' : ''}>${esc(v)}</option>`).join('')}
        </select>`;
      } else {
        input = `<input type="number" data-device="${esc(dk)}" data-code="${esc(c.code)}"
          class="ctl-input" value="${val === '' ? '' : esc(val)}">`;
      }
      return `<div class="card ctrl"><div class="meta">
        <div class="code">${esc(c.code)}</div>
        <div class="desc">${esc(c.desc)}</div>
        ${val !== '' ? `<div class="cur">current: ${esc(val)}</div>` : ''}
      </div>${input}</div>`;
    }).join('');
    g.querySelectorAll('select.ctl-select').forEach(s => s.onchange = () => sendCtl(dk, s.dataset.code, s.value));
    g.querySelectorAll('input.ctl-input').forEach(i => {
      i.onchange = () => sendCtl(dk, i.dataset.code, i.value === '' ? null : parseInt(i.value, 10));
    });
  });
}

async function sendCtl(device, code, value) {
  if (value === null || value === '') return;
  const r = await fetch('/api/control', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({device_key: device, code, value})});
  const j = await r.json();
  if (!j.ok) { alert('Command failed — check logs'); }
  refresh();
}

function ringColor(pct) {
  if (pct == null) return 'var(--dim)';
  if (pct <= 15) return 'var(--bad)';
  if (pct <= 35) return 'var(--warn)';
  return 'var(--ok)';
}

function renderDevice(d, idx) {
  const R = 56, C = 2 * Math.PI * R;
  const pct = d.battery_percent;
  const off = C - (pct == null ? 0 : (pct / 100) * C);
  const netCls = d.net_power == null ? '' : (d.net_power > 0 ? 'in' : 'out');
  const netLbl = d.net_power == null ? '—' :
    (d.net_power > 0 ? '▲ charging ' : '▼ draining ') + Math.abs(d.net_power) + ' W';
  const sw = d.switches || {};
  const acCode = 'ac_switch_hm', dcCode = 'dc_switch_hm', upsCode = 'ups_status_hm';

  return `<div>
  <div class="grid">
    <div class="card battery-card">
      <div class="ring-wrap">
        <svg width="132" height="132" viewBox="0 0 132 132">
          <circle class="ring-bg" cx="66" cy="66" r="${R}"></circle>
          <circle class="ring-fg" cx="66" cy="66" r="${R}" stroke="${ringColor(pct)}"
            stroke-dasharray="${C}" stroke-dashoffset="${off}"></circle>
        </svg>
        <div class="ring-label">
          <div class="pct">${pct == null ? '—' : pct + '%'}</div>
          <div class="st">${esc(d.status && d.status.label || 'no data')}</div>
        </div>
      </div>
      <div class="battery-info">
        <div class="row"><span class="k">Voltage</span><span>${d.voltage == null ? 'N/A' : d.voltage + ' V'}</span></div>
        <div class="row"><span class="k">Temperature</span><span>${d.temperature_c == null ? 'N/A' : d.temperature_c + ' °C'}</span></div>
        <div class="row"><span class="k">Battery</span><span class="chg ${netCls}">${netLbl}</span></div>
        <div class="row"><span class="k">Capacity</span><span>${d.capacity_wh ? d.capacity_wh + ' Wh' : '—'}</span></div>
        <div class="row"><span class="k">${d.est_empty ? 'Est. empty' : 'Est. full'}</span>
          <span>${d.est_empty || d.est_full || '—'}</span></div>
      </div>
    </div>
    ${card('Total input', (d.total_input_w || 0) + ' W', (d.ac_input_w || 0) + ' AC · ' + (d.dc_input_w || 0) + ' DC')}
    ${card('Total output', (d.total_output_w || 0) + ' W',
      (d.ac_output_w || 0) + ' AC' + (d.ac_output_voltage ? ' @ ' + d.ac_output_voltage + 'V' : '') + ' · ' + (d.dc_output_w || 0) + ' DC')}
    ${card('Run time', fmtMin(d.remain_discharge_min), 'at current draw')}
    ${card('Charge time', fmtMin(d.remain_charge_min), 'at current charge')}
  </div>

  <h2>Outputs</h2>
  <div class="switches">
    ${toggle(d.device_key, acCode, 'AC Output', sw.ac)}
    ${toggle(d.device_key, dcCode, 'DC Output', sw.dc)}
    ${toggle(d.device_key, upsCode, 'UPS Mode', sw.ups)}
  </div>

  <h2>Other controls</h2>
  <div class="controls-grid" data-device="${esc(d.device_key)}"></div>
  </div>`;
}

async function refresh() {
  let s;
  try { s = await (await fetch('/api/status')).json(); } catch (e) {
    $('banner').innerHTML = '<div class="offline">Cannot reach monitor API: ' + esc(e) + '</div>';
    return;
  }
  $('banner').innerHTML = '';
  $('devname').textContent = s.devices.map(d => d.name).join(' · ');
  const src = s.devices.map(d => d.source || '…');
  $('source').textContent = src.join(' · ');
  $('updated').textContent = new Date(s.ts * 1000).toLocaleTimeString();

  const el = $('devices');
  el.innerHTML = s.devices.map(renderDevice).join('');
  loadControls(el);

  // raw + logs + history (throttled to keep network quiet)
  try {
    const raw = await (await fetch('/api/raw')).json();
    $('raw').textContent = JSON.stringify(raw, null, 2);
    const lg = await (await fetch('/api/logs')).json();
    $('logs').textContent = (lg.lines || []).slice(-60).join('\n');
    const box = $('logs'); box.scrollTop = box.scrollHeight;
  } catch (e) { /* ignore */ }
  refreshHistory();
}

let rangeMin = 360;
function setRange(min, btn) {
  rangeMin = min;
  document.querySelectorAll('.range-btn').forEach(b => b.classList.toggle('active', b === btn));
  refreshHistory();
}

function fmtTick(v) {
  const a = Math.abs(v);
  if (a >= 1000) return (v / 1000).toFixed(1) + 'k';
  if (a >= 10) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

function drawChart(id, series) {
  const cv = document.getElementById(id);
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 300, h = cv.clientHeight || 140;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 44, r: 8, t: 8, b: 20 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;

  let t0 = Infinity, t1 = -Infinity, vmin = Infinity, vmax = -Infinity;
  for (const s of series) for (const p of s.pts) {
    if (p.t < t0) t0 = p.t;
    if (p.t > t1) t1 = p.t;
    if (p.v < vmin) vmin = p.v;
    if (p.v > vmax) vmax = p.v;
  }
  if (t0 > t1) {
    ctx.fillStyle = 'rgba(139,152,169,.6)';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('collecting data…', w / 2, h / 2);
    return;
  }
  if (t1 - t0 < 60) t1 = t0 + 60;
  if (vmax - vmin < 1e-6) { vmax += 1; vmin -= 1; }
  const padV = (vmax - vmin) * 0.08;
  vmin -= padV; vmax += padV;

  const X = t => pad.l + ((t - t0) / (t1 - t0)) * iw;
  const Y = v => pad.t + ih - ((v - vmin) / (vmax - vmin)) * ih;

  ctx.strokeStyle = 'rgba(42,52,68,.9)';
  ctx.fillStyle = '#8b98a9';
  ctx.font = '10px ui-monospace, monospace';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = vmin + ((vmax - vmin) * i) / 4;
    const y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(fmtTick(v), pad.l - 5, y + 3);
  }
  ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const t = t0 + ((t1 - t0) * i) / 4;
    ctx.fillText(new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), X(t), h - 6);
  }

  for (const s of series) {
    if (!s.pts.length) continue;
    const step = Math.max(1, Math.floor(s.pts.length / iw));
    ctx.beginPath();
    for (let i = 0; i < s.pts.length; i += step) {
      const p = s.pts[i];
      const x = X(p.t), y = Y(p.v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    const last = s.pts[s.pts.length - 1];
    ctx.lineTo(X(last.t), Y(last.v));
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    if (s.fill) {
      ctx.lineTo(X(last.t), h - pad.b);
      ctx.lineTo(X(s.pts[0].t), h - pad.b);
      ctx.closePath();
      ctx.fillStyle = s.color + '1e';
      ctx.fill();
    }
  }
}

async function refreshHistory() {
  let j;
  try { j = await (await fetch('/api/history?minutes=' + rangeMin)).json(); } catch (e) { return; }
  const src = $('hist-src');
  if (src) src.textContent = j.source === 'prometheus' ? 'stored: prometheus' : 'buffer: memory';
  const dks = Object.keys(j.devices || {});
  const d = dks.length ? j.devices[dks[0]] : null;
  const S = key => d ? d.t.map((t, i) => ({ t, v: (d.v[key] || [])[i] })).filter(p => p.v != null) : [];
  drawChart('g-battery', [{ color: '#3fb950', fill: true, pts: S('battery_percent') }]);
  drawChart('g-power', [
    { color: '#58a6ff', pts: S('total_input_w') },
    { color: '#d29922', pts: S('total_output_w') },
  ]);
  drawChart('g-voltage', [{ color: '#bc8cff', pts: S('voltage') }]);
  drawChart('g-temp', [{ color: '#f85149', pts: S('temperature_c') }]);
}

setInterval(refresh, 3000);
refresh();
</script>
</body>
</html>
"""
