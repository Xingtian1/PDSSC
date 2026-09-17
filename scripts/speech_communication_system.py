import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as TAT
from flask import Flask, abort, redirect, render_template_string, request, send_from_directory, url_for
from scipy.io import wavfile
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FRAME_MS_BASE = 20
PLR_OPTIONS = [0.0, 0.05, 0.10, 0.20, 0.30]
CHUNK_MS_OPTIONS = [20, 40, 80, 160]
PROPOSED_OPTIONS = [(2, "1.0 kbps (N=2)"), (3, "1.5 kbps (N=3)"), (4, "2.0 kbps (N=4)"),
                    (5, "2.5 kbps (N=5)"), (6, "3.0 kbps (N=6)"), (7, "3.5 kbps (N=7)"),
                    (8, "4.0 kbps (N=8)")]
N_STEPS_OPTIONS = [4, 6, 8, 10]
OPUS_OPTIONS = [(6.0, "6 kbps"), (8.0, "8 kbps"), (12.0, "12 kbps")]
ENCODEC_OPTIONS = [(1.5, "1.5 kbps"), (3.0, "3.0 kbps"), (6.0, "6.0 kbps")]


SpeechTokenizer = None
FlowMatchingModel = None
PretrainedSpeakerEncoder = None
EncodecModel = None
HAS_ENCODEC = None


def ensure_speechtokenizer_imports() -> None:
    global SpeechTokenizer, FlowMatchingModel, PretrainedSpeakerEncoder
    if SpeechTokenizer is not None and FlowMatchingModel is not None and PretrainedSpeakerEncoder is not None:
        return
    from speechtokenizer import SpeechTokenizer as _SpeechTokenizer
    from speechtokenizer.flow import (
        FlowMatchingModel as _FlowMatchingModel,
        PretrainedSpeakerEncoder as _PretrainedSpeakerEncoder,
    )
    SpeechTokenizer = _SpeechTokenizer
    FlowMatchingModel = _FlowMatchingModel
    PretrainedSpeakerEncoder = _PretrainedSpeakerEncoder


def check_encodec_available() -> bool:
    global EncodecModel, HAS_ENCODEC
    if HAS_ENCODEC is not None:
        return HAS_ENCODEC
    try:
        from encodec import EncodecModel as _EncodecModel
        EncodecModel = _EncodecModel
        HAS_ENCODEC = True
    except Exception:
        EncodecModel = None
        HAS_ENCODEC = False
    return HAS_ENCODEC


def get_bind_urls(host: str, port: int) -> List[str]:
    urls = []
    if host in ("0.0.0.0", "::"):
        urls.append(f"http://127.0.0.1:{port}")
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            if local_ip and local_ip not in ("127.0.0.1", "0.0.0.0"):
                urls.append(f"http://{local_ip}:{port}")
        except Exception:
            pass
    else:
        urls.append(f"http://{host}:{port}")
    return urls


def write_json_atomic(path: str, payload: Dict[str, object]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def update_status_file(path: str, payload: Dict[str, object]) -> None:
    payload["updated_at"] = time.time()
    write_json_atomic(path, payload)


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Speech Communication System</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --paper: #fffdf8;
      --ink: #1d1d1b;
      --muted: #6f6a61;
      --line: #d8d1c3;
      --accent: #b22a1d;
      --accent2: #164fb3;
      --ok: #19765a;
      --soft: #fff7e7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, #f7e8cf 0, transparent 30%),
        linear-gradient(180deg, #f5f1e8 0%, #f0ece4 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }
    .page {
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 18px;
      margin-bottom: 18px;
    }
    .panel {
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px 18px 16px;
      box-shadow: 0 10px 30px rgba(40, 30, 10, 0.06);
    }
    .hero-main {
      min-height: 260px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      background:
        linear-gradient(135deg, rgba(255,253,248,0.96), rgba(255,246,226,0.92)),
        repeating-linear-gradient(90deg, transparent 0 18px, rgba(178,42,29,0.04) 18px 19px);
    }
    h1 {
      margin: 0 0 10px;
      font-size: 42px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    .lede {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 15px;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      margin-bottom: 14px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fffaf1;
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 20px;
    }
    .chip {
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: white;
      color: var(--muted);
      font-size: 12px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #fffaf1;
      min-height: 92px;
    }
    .stat b {
      display: block;
      font-size: 13px;
      margin-bottom: 6px;
    }
    .stat span {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    form {
      display: grid;
      gap: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .triple {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    label {
      display: block;
      font-size: 13px;
      margin-bottom: 6px;
      color: var(--muted);
    }
    input[type="text"], input[type="number"], select, input[type="file"] {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      color: var(--ink);
      font-size: 14px;
    }
    .checkrow {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 4px;
      font-size: 14px;
    }
    .system-block {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #fffaf1;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .system-block:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 20px rgba(40, 30, 10, 0.06);
    }
    .system-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-size: 15px;
    }
    .system-note {
      color: var(--muted);
      font-size: 12px;
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 14px 16px;
      background: linear-gradient(135deg, #b22a1d, #7e1f16);
      color: white;
      font-size: 16px;
      font-family: inherit;
      cursor: pointer;
      box-shadow: 0 10px 20px rgba(126, 31, 22, 0.16);
    }
    .footer-note {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    @media (max-width: 900px) {
      .hero, .grid, .triple {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <div class="panel hero-main">
        <div>
        <div class="eyebrow">Packetized Speech Communication</div>
        <h1>Speech Communication System</h1>
        <p class="lede">
          This demo simulates packet-level speech transmission under packet loss, then uses
          <b>buffered full-utterance reconstruction</b> to keep the final audio quality stable.
          It compares <b>Proposed</b>, <b>Opus</b>, and optionally <b>EnCodec</b>.
        </p>
        </div>
        <div class="chips">
          <span class="chip">{{ frame_ms }} ms packet</span>
          <span class="chip">{{ sample_rate }} Hz speech</span>
          <span class="chip">Global PLR control</span>
          <span class="chip">Full mel display</span>
        </div>
      </div>
      <div class="panel">
        <h2>Platform Notes</h2>
        <div class="stats">
          <div class="stat">
            <b>Mode</b>
            <span>Packetized channel simulation with buffered reconstruction.</span>
          </div>
          <div class="stat">
            <b>Runtime</b>
            <span>Processing time and RTF are shown for each reconstructed system.</span>
          </div>
          <div class="stat">
            <b>Output</b>
            <span>Audio playback, mel spectrogram, processing time, and RTF.</span>
          </div>
          <div class="stat">
            <b>EnCodec</b>
            <span>{{ "Available" if has_encodec else "Not available in current Python environment" }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="panel">
      <form action="{{ url_for('run_demo') }}" method="post" enctype="multipart/form-data">
        <div class="grid">
          <div>
            <label>Upload Speech File</label>
            <input type="file" name="audio_file" accept=".wav,.flac,.mp3,.ogg,.m4a,.aac">
          </div>
          <div>
            <label>Or Use Local Path</label>
            <input type="text" name="audio_path" value="samples/example_input.wav">
          </div>
        </div>

        <div class="triple">
          <div>
            <label>Processing Mode</label>
            <input type="text" value="Packetized channel + buffered reconstruction" readonly>
          </div>
          <div>
            <label>Global PLR</label>
            <select name="plr">
              {% for v in plr_options %}
              <option value="{{ v }}" {% if v == 0.05 %}selected{% endif %}>{{ (v * 100)|round(0)|int }}%</option>
              {% endfor %}
            </select>
          </div>
          <div>
            <label>Max Duration</label>
            <input type="number" name="max_sec" value="6" min="1" max="30" step="1">
          </div>
        </div>

        <div class="triple">
          <div class="system-block">
            <div class="system-head">
              <b>Proposed</b>
              <label class="checkrow"><input type="checkbox" name="enable_proposed" checked> enable</label>
            </div>
            <label>Bitrate</label>
            <select name="proposed_n_layers">
              {% for value, text in proposed_options %}
              <option value="{{ value }}" {% if value == 6 %}selected{% endif %}>{{ text }}</option>
              {% endfor %}
            </select>
          </div>

          <div class="system-block">
            <div class="system-head">
              <b>Opus</b>
              <label class="checkrow"><input type="checkbox" name="enable_opus" checked> enable</label>
            </div>
            <label>Bitrate</label>
            <select name="opus_bitrate">
              {% for value, text in opus_options %}
              <option value="{{ value }}" {% if value == 8.0 %}selected{% endif %}>{{ text }}</option>
              {% endfor %}
            </select>
          </div>

          <div class="system-block">
            <div class="system-head">
              <b>EnCodec</b>
              <label class="checkrow">
                <input type="checkbox" name="enable_encodec" {% if has_encodec %}checked{% else %}disabled{% endif %}> enable
              </label>
            </div>
            <label>Bitrate</label>
            <select name="encodec_bitrate" {% if not has_encodec %}disabled{% endif %}>
              {% for value, text in encodec_options %}
              <option value="{{ value }}" {% if value == 3.0 %}selected{% endif %}>{{ text }}</option>
              {% endfor %}
            </select>
            <div class="system-note">
              {% if has_encodec %}Uses token-level LFR-PLC.{% else %}Unavailable until `encodec` is installed in this environment.{% endif %}
            </div>
          </div>
        </div>

        <button type="submit">Run Communication Demo</button>
      </form>
      <div class="footer-note">
        This final demo is intentionally not strict neural chunk decoding. It keeps packet-loss communication settings,
        but reconstructs from the buffered utterance so the listening demo reflects the model's stable audio quality.
      </div>
    </div>
  </div>
</body>
</html>
"""


RESULT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Speech Communication System Result</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --paper: #fffdf8;
      --ink: #1d1d1b;
      --muted: #6f6a61;
      --line: #d8d1c3;
      --accent: #b22a1d;
      --accent2: #164fb3;
      --ok: #19765a;
      --bad: #b22a1d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, #f7e8cf 0, transparent 30%),
        linear-gradient(180deg, #f5f1e8 0%, #f0ece4 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }
    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }
    .topbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: start;
      margin-bottom: 18px;
    }
    .topbar a {
      text-decoration: none;
      color: white;
      background: #1d1d1b;
      padding: 10px 14px;
      border-radius: 12px;
      font-size: 14px;
    }
    .panel {
      background: rgba(255, 253, 248, 0.94);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(40, 30, 10, 0.06);
      margin-bottom: 18px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 40px;
      line-height: 1.06;
      letter-spacing: -0.03em;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      margin-bottom: 12px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fffaf1;
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .summary div {
      background: #fffaf1;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      font-size: 14px;
      line-height: 1.5;
      min-height: 86px;
    }
    .summary b {
      display: block;
      margin-bottom: 4px;
      font-size: 13px;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }
    .card {
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 10px 24px rgba(40, 30, 10, 0.045);
    }
    .card-head {
      padding: 14px 16px 12px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #fff9ef, #fffdf8);
    }
    .model-row {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .card-head h2 {
      margin: 0;
      font-size: 20px;
    }
    .badge {
      flex: 0 0 auto;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: white;
      color: var(--muted);
      font-size: 12px;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      background: #fffaf1;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 5px;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }
    .metric b {
      display: block;
      font-size: 18px;
      line-height: 1.1;
    }
    .card-body {
      padding: 14px 16px 18px;
    }
    audio {
      width: 100%;
      margin-bottom: 12px;
    }
    .mel-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 5;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: #111;
    }
    .mel-base, .mel-reveal {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .mel-base {
      filter: grayscale(1) brightness(0.35);
      opacity: 0.5;
    }
    .mel-reveal {
      clip-path: inset(0 100% 0 0);
    }
    .ticks {
      display: flex;
      justify-content: space-between;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .download {
      margin-top: 10px;
      display: inline-block;
      font-size: 13px;
      text-decoration: none;
      color: #164fb3;
    }
    .note {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .sec-badge {
      display: inline-block;
      margin-bottom: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f2eadb;
      border: 1px solid var(--line);
      font-size: 12px;
      color: var(--ink);
    }
    .mel-placeholder {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      aspect-ratio: 16 / 5;
      border: 1px dashed var(--line);
      border-radius: 12px;
      background: #f6f0e5;
      color: var(--muted);
      font-size: 14px;
    }
    @media (max-width: 960px) {
      .summary, .cards {
        grid-template-columns: 1fr;
      }
      .topbar {
        grid-template-columns: 1fr;
      }
      .metric-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="topbar">
      <div>
        <div class="eyebrow">Packetized Communication Result</div>
        <h1>Speech Communication System Result</h1>
        <div class="note">
          Packet loss is simulated at the communication layer. The final listening result uses buffered full-utterance reconstruction
          so the audio remains stable and directly comparable across systems.
        </div>
      </div>
      <a href="{{ url_for('index') }}">Run Another Demo</a>
    </div>

    <div class="panel">
      <div class="summary">
        <div><b>Input</b>{{ run.input_name }}</div>
        <div><b>Global PLR</b>{{ (run.plr * 100)|round(0)|int }}%</div>
        <div><b>Duration</b>{{ "%.2f"|format(run.duration_sec) }} s</div>
        <div><b>Run ID</b>{{ run.run_id }}</div>
      </div>
    </div>

    <div class="cards">
      {% for item in items %}
      <div class="card">
        <div class="card-head">
          <div class="model-row">
            <h2>{{ item.title }}</h2>
            <span class="badge">{{ "Reference" if item.name == "source" else "Reconstructed" }}</span>
          </div>
          <div class="meta" id="{{ item.dom_id }}_meta">
            {% if item.show_timing %}
            {{ item.subtitle }}
            <div class="metric-grid">
              <div class="metric">
                <span>Processing</span>
                <b>{{ "%.3f"|format(item.proc_sec) }} s</b>
              </div>
              <div class="metric">
                <span>RTF</span>
                <b>{{ "%.3f"|format(item.rtf) }}</b>
              </div>
            </div>
            {% else %}
            {{ item.subtitle }}
            <div class="metric-grid">
              <div class="metric">
                <span>Duration</span>
                <b>{{ "%.2f"|format(run.duration_sec) }} s</b>
              </div>
              <div class="metric">
                <span>Sample Rate</span>
                <b>{{ sample_rate }} Hz</b>
              </div>
            </div>
            {% endif %}
          </div>
        </div>
        <div class="card-body">
          <audio controls preload="metadata" id="{{ item.dom_id }}_audio">
            <source src="{{ item.audio_url }}" type="audio/wav">
          </audio>
          <div class="mel-wrap" id="{{ item.dom_id }}">
            <img class="mel-base" src="{{ item.mel_url }}" alt="mel full">
            <img class="mel-reveal" src="{{ item.mel_url }}" alt="mel full reveal" id="{{ item.dom_id }}_reveal" style="clip-path: inset(0 0 0 0)">
          </div>
          <div class="ticks">
            <span>0.0 s</span>
            <span>{{ "%.2f"|format(run.duration_sec / 2.0) }} s</span>
            <span>{{ "%.2f"|format(run.duration_sec) }} s</span>
          </div>
          <a class="download" href="{{ item.audio_url }}" download id="{{ item.dom_id }}_download">Download WAV</a>
        </div>
      </div>
      {% endfor %}
    </div>

    <div class="panel">
      <div class="note">
        Interpretation: <b>Processing</b> and <b>RTF</b> measure full buffered reconstruction time for the selected system.
      </div>
    </div>
  </div>
</body>
</html>
"""


WAIT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Speech Communication System Processing</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --paper: #fffdf8;
      --ink: #1d1d1b;
      --muted: #6f6a61;
      --line: #d8d1c3;
      --accent: #b22a1d;
      --accent2: #164fb3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at top left, #f7e8cf 0, transparent 30%),
        linear-gradient(180deg, #f5f1e8 0%, #f0ece4 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }
    .panel {
      width: min(760px, calc(100vw - 32px));
      background: rgba(255, 253, 248, 0.95);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 18px 50px rgba(40, 30, 10, 0.09);
    }
    .eyebrow {
      display: inline-flex;
      margin-bottom: 14px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fffaf1;
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 38px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }
    .note {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
      margin-bottom: 20px;
    }
    .progress {
      width: 100%;
      height: 13px;
      border-radius: 999px;
      background: #e8e1d4;
      overflow: hidden;
      border: 1px solid var(--line);
    }
    .bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      transition: width 0.25s ease;
    }
    .status {
      margin-top: 14px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 14px;
    }
    .systems {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }
    .system {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fffaf1;
      padding: 12px;
      font-size: 14px;
    }
    .system b { display: block; margin-bottom: 5px; }
    .system span { color: var(--muted); font-size: 12px; }
    .error {
      display: none;
      margin-top: 14px;
      padding: 12px;
      border-radius: 12px;
      background: #fff1ed;
      border: 1px solid #e3b0a7;
      color: #7e1f16;
      font-size: 13px;
      line-height: 1.5;
    }
    @media (max-width: 760px) {
      .systems { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="panel">
    <div class="eyebrow">Buffered Reconstruction Running</div>
    <h1>Generating Results</h1>
    <div class="note">
      The page has switched to a waiting view first. The server is reconstructing the selected systems in the background;
      when all outputs are ready, this page will automatically open the final result.
    </div>
    <div class="progress"><div class="bar" id="bar"></div></div>
    <div class="status">
      <span id="statusText">Queued</span>
      <span id="percent">0%</span>
    </div>
    <div class="systems" id="systems"></div>
    <div class="error" id="errorBox"></div>
  </div>
  <script>
    const statusUrl = "{{ url_for('run_status', run_id=run_id) }}";
    const resultUrl = "{{ url_for('show_run', run_id=run_id) }}";
    const bar = document.getElementById("bar");
    const statusText = document.getElementById("statusText");
    const percent = document.getElementById("percent");
    const systems = document.getElementById("systems");
    const errorBox = document.getElementById("errorBox");

    function renderItems(items) {
      systems.innerHTML = "";
      for (const item of items || []) {
        if (item.name === "source") continue;
        const div = document.createElement("div");
        div.className = "system";
        const state = item.finished ? "finished" : (item.started ? "running" : "waiting");
        const elapsed = Number(item.elapsed_sec || 0).toFixed(2);
        div.innerHTML = `<b>${item.title}</b><span>${state} · ${elapsed}s</span>`;
        systems.appendChild(div);
      }
    }

    async function poll() {
      try {
        const res = await fetch(statusUrl, {cache: "no-store"});
        if (!res.ok) throw new Error("status request failed: " + res.status);
        const data = await res.json();
        const progress = Math.max(0, Math.min(1, Number(data.progress || 0)));
        bar.style.width = (progress * 100).toFixed(0) + "%";
        percent.textContent = (progress * 100).toFixed(0) + "%";
        statusText.textContent = data.status_text || "Running";
        renderItems(data.items);
        if (data.failed) {
          errorBox.style.display = "block";
          errorBox.textContent = data.status_text || "Generation failed.";
          return;
        }
        if (data.done) {
          window.location.href = resultUrl;
          return;
        }
      } catch (err) {
        errorBox.style.display = "block";
        errorBox.textContent = String(err);
      }
      setTimeout(poll, 600);
    }
    poll();
  </script>
</body>
</html>
"""


@dataclass
class SystemResult:
    name: str
    title: str
    subtitle: str
    wav: np.ndarray
    proc_sec: float
    rtf: float


def _strip_compile_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def _to_float32_audio(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    return data.astype(np.float32)


def load_audio_ffmpeg(path: str, target_sr: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(target_sr), tmp_path],
            capture_output=True,
            timeout=60,
        )
        if ret.returncode != 0:
            raise RuntimeError(ret.stderr.decode(errors="ignore")[-500:])
        sr, audio = wavfile.read(tmp_path)
        if sr != target_sr:
            raise RuntimeError(f"ffmpeg decode sr mismatch: {sr} != {target_sr}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return _to_float32_audio(audio)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def save_wav(path: str, wav_np: np.ndarray, sr: int) -> None:
    wav = np.clip(wav_np.astype(np.float32), -1.0, 1.0)
    wav_i16 = (wav * 32767.0).astype(np.int16)
    wavfile.write(path, sr, wav_i16)


def ensure_len(wav_np: np.ndarray, target_len: int) -> np.ndarray:
    wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
    if len(wav_np) == target_len:
        return wav_np
    if len(wav_np) > target_len:
        return wav_np[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:len(wav_np)] = wav_np
    return out


def waveform_linear_interp(wav_np: np.ndarray, lost_mask: np.ndarray) -> np.ndarray:
    out = wav_np.copy().astype(np.float64)
    n = len(out)
    i = 0
    while i < n:
        if i < len(lost_mask) and lost_mask[i]:
            start = i
            while i < n and i < len(lost_mask) and lost_mask[i]:
                i += 1
            end = i
            left = float(out[start - 1]) if start > 0 else 0.0
            right = float(out[end]) if end < n else 0.0
            length = end - start
            if length > 0:
                out[start:end] = np.linspace(left, right, length + 2)[1:-1]
        else:
            i += 1
    return out.astype(np.float32)


def latent_linear_interp(lat: torch.Tensor, recv_mask: torch.Tensor) -> torch.Tensor:
    lat_out = lat.clone()
    t_total = lat_out.shape[-1]
    recv = recv_mask.tolist()
    i = 0
    while i < t_total:
        if not recv[i]:
            start = i
            while i < t_total and not recv[i]:
                i += 1
            end = i
            left = lat_out[:, :, start - 1] if start > 0 else torch.zeros_like(lat_out[:, :, 0])
            right = lat_out[:, :, end] if end < t_total else torch.zeros_like(lat_out[:, :, 0])
            length = end - start
            for k in range(length):
                alpha = (k + 1) / (length + 1)
                lat_out[:, :, start + k] = (1.0 - alpha) * left + alpha * right
        else:
            i += 1
    return lat_out


@torch.no_grad()
def channel_simulate(st_model, wav: torch.Tensor, n_layers: int, p_loss: float, device) -> Dict[str, torch.Tensor]:
    x = wav.unsqueeze(0).to(device)
    codes_all = st_model.encode(x)
    t_enc = codes_all.shape[2]
    d_dim = st_model.quantizer.dimension

    def safe_decode(idx: int) -> torch.Tensor:
        vq_l = st_model.quantizer.vq.layers[idx]
        decoded = vq_l.decode(codes_all[idx])
        if decoded.shape[-1] == d_dim:
            decoded = decoded.permute(0, 2, 1)
        return decoded.contiguous()

    latent_q1 = safe_decode(0)
    q1_recv = (torch.rand(t_enc, device=device) >= p_loss)
    if not bool(q1_recv.all()):
        latent_q1 = latent_linear_interp(latent_q1, q1_recv)

    latent_ch = latent_q1.clone()
    for l_idx in range(1, n_layers):
        decoded = safe_decode(l_idx)
        recv_mask = (torch.rand(t_enc, device=device) >= p_loss).float()
        latent_ch = latent_ch + decoded * recv_mask.unsqueeze(0).unsqueeze(0)

    latent_8 = st_model.quantizer.decode(codes_all)
    return {
        "latent_ch": latent_ch,
        "latent_8": latent_8,
        "wav_8": st_model.decoder(latent_8).squeeze(0).cpu(),
    }


@torch.no_grad()
def flow_sample(flow_model, latent: torch.Tensor, spk_emb: torch.Tensor, n_steps: int = 10, n_layers: Optional[int] = None) -> torch.Tensor:
    import inspect
    sig = inspect.signature(flow_model.sample)
    if "n_layers" in sig.parameters and n_layers is not None:
        n_t = torch.tensor([n_layers], device=latent.device)
        return flow_model.sample(latent, spk_emb, n_steps=n_steps, n_layers=n_t)
    return flow_model.sample(latent, spk_emb, n_steps=n_steps)


class EncodecHelper:
    def __init__(self):
        self.cache: Dict[Tuple[float, str], object] = {}

    @torch.no_grad()
    def decode_with_lfrplc(self, wav: torch.Tensor, sr: int, source_bw: float, p_loss: float, device) -> Optional[np.ndarray]:
        if not check_encodec_available():
            return None
        cache_key = (source_bw, str(device))
        if cache_key not in self.cache:
            model = EncodecModel.encodec_model_24khz()
            model.set_target_bandwidth(source_bw)
            self.cache[cache_key] = model.to(device).eval()
        enc_model = self.cache[cache_key]
        enc_model.set_target_bandwidth(source_bw)
        enc_sr = enc_model.sample_rate
        wav_enc = torchaudio.functional.resample(wav.squeeze(), sr, enc_sr).unsqueeze(0).unsqueeze(0).to(device)
        frames = enc_model.encode(wav_enc)
        new_frames = []
        for codes, scale in frames:
            t_frames = codes.shape[-1]
            codes_out = codes.clone()
            for t in range(t_frames):
                if np.random.rand() < p_loss and t > 0:
                    codes_out[:, :, t] = codes_out[:, :, t - 1]
            new_frames.append((codes_out, scale))
        wav_dec = enc_model.decode(new_frames).squeeze()
        out = torchaudio.functional.resample(wav_dec.unsqueeze(0), enc_sr, sr).squeeze()
        return out.cpu().numpy().astype(np.float32)


def opus_lbrr_with_plr_ffmpeg(wav: torch.Tensor, sr: int, bitrate_kbps: float, p_loss: float, frame_ms: float = 20.0) -> Optional[np.ndarray]:
    codec_sr = 48000
    wav_rs = torchaudio.functional.resample(wav.squeeze(), sr, codec_sr).cpu().numpy().astype(np.float32)
    fd_ref, ref_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd_ref)
    enc_path = ref_path.replace(".wav", ".ogg")
    dec_path = ref_path.replace(".wav", "_dec.wav")
    try:
        save_wav(ref_path, wav_rs, codec_sr)
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", ref_path,
             "-c:a", "libopus",
             "-b:a", str(int(bitrate_kbps * 1000)),
             "-vbr", "constrained",
             "-application", "voip",
             "-frame_duration", str(frame_ms),
             "-packet_loss", str(max(0, min(100, int(round(p_loss * 100))))),
             "-fec", "1",
             enc_path],
            capture_output=True,
            timeout=60,
        )
        if ret.returncode != 0:
            ret = subprocess.run(
                ["ffmpeg", "-y", "-i", ref_path,
                 "-c:a", "libopus",
                 "-b:a", str(int(bitrate_kbps * 1000)),
                 "-vbr", "constrained",
                 "-frame_duration", str(frame_ms),
                 "-application", "voip",
                 enc_path],
                capture_output=True,
                timeout=60,
            )
            if ret.returncode != 0:
                raise RuntimeError(ret.stderr.decode(errors="ignore")[-500:])
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", enc_path, dec_path],
            capture_output=True,
            timeout=60,
        )
        if ret.returncode != 0:
            raise RuntimeError(ret.stderr.decode(errors="ignore")[-500:])
        codec_sr_read, dec_audio = wavfile.read(dec_path)
        if dec_audio.ndim > 1:
            dec_audio = dec_audio.mean(axis=1)
        dec_audio = _to_float32_audio(dec_audio)
        effective_plr = p_loss ** 2
        frame_len = int(codec_sr_read * frame_ms / 1000.0)
        n_frames = len(dec_audio) // max(1, frame_len)
        lost_samples = np.zeros(len(dec_audio), dtype=bool)
        for fi in range(n_frames):
            if np.random.rand() < effective_plr:
                s = fi * frame_len
                e = min(s + frame_len, len(dec_audio))
                lost_samples[s:e] = True
        dec_audio = waveform_linear_interp(dec_audio, lost_samples)
        out = torchaudio.functional.resample(
            torch.from_numpy(dec_audio).unsqueeze(0), codec_sr_read, sr
        ).squeeze(0).cpu().numpy().astype(np.float32)
        return out
    except Exception:
        return None
    finally:
        for path in (ref_path, enc_path, dec_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def save_mel_image(wav_np: np.ndarray, sr: int, path: str) -> None:
    wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
    # Keep the same analysis configuration as the full-length reference mel so
    # the streaming view stays visually consistent with the source mel.
    min_len = 1024
    if wav_np.size == 0:
        wav_np = np.zeros(min_len, dtype=np.float32)
    elif wav_np.size < min_len:
        wav_pad = np.zeros(min_len, dtype=np.float32)
        wav_pad[:wav_np.size] = wav_np
        wav_np = wav_pad

    wav_t = torch.from_numpy(wav_np).unsqueeze(0)
    mel_fn = TAT.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        power=2.0,
        center=True,
        pad_mode="constant",
    )
    mel = mel_fn(wav_t).squeeze(0)
    mel_db = (10.0 * torch.log10(mel + 1e-9)).cpu().numpy()
    plt.figure(figsize=(10, 3))
    plt.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    plt.axis("off")
    plt.tight_layout(pad=0)
    tmp_path = path + ".tmp.png"
    plt.savefig(tmp_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()
    os.replace(tmp_path, path)


class SimulatorEngine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sr = 16000
        self.st_model = None
        self.flow_model = None
        self.spk_encoder = None
        self.encodec_helper = EncodecHelper()
        self.min_flow_frames = None
        self.io_lock = threading.Lock()
        self.warmup_done = False

    def ensure_models(self) -> None:
        if self.st_model is not None:
            return
        ensure_speechtokenizer_imports()
        self.st_model = SpeechTokenizer.load_from_checkpoint(self.args.config_path, self.args.ckpt_path)
        self.st_model.eval().to(self.device)
        for p in self.st_model.parameters():
            p.requires_grad_(False)
        ckpt = torch.load(self.args.flow_ckpt, map_location="cpu")
        saved_args = argparse.Namespace(**ckpt.get("args", {}))
        self.flow_model = FlowMatchingModel(
            latent_dim=self.st_model.quantizer.dimension,
            base_ch=getattr(saved_args, "base_ch", 512),
            ch_mults=tuple(getattr(saved_args, "ch_mults", [1, 1, 2])),
            cond_dim=getattr(saved_args, "cond_dim", 512),
            spk_dim=getattr(saved_args, "spk_dim", 256),
            time_dim=getattr(saved_args, "time_dim", 128),
            n_res=getattr(saved_args, "n_res", 2),
            n_mid_res=getattr(saved_args, "n_mid_res", 2),
        ).to(self.device)
        self.flow_model.load_state_dict(_strip_compile_prefix(ckpt["flow_model"]))
        self.flow_model.eval()
        self.spk_encoder = PretrainedSpeakerEncoder(
            emb_dim=getattr(saved_args, "spk_dim", 256),
            save_dir=os.path.join(os.path.dirname(self.args.flow_ckpt), "spkrec-ecapa"),
        ).to(self.device)
        self.spk_encoder.load_state_dict(_strip_compile_prefix(ckpt["spk_encoder"]), strict=False)
        self.spk_encoder.eval()
        n_down_levels = len(getattr(self.flow_model, "down_blocks", []))
        self.min_flow_frames = max(1, 2 ** n_down_levels)

    def _cuda_sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def warmup_models(self) -> None:
        if self.warmup_done:
            return
        self.ensure_models()
        warm_len = 3200
        wav_np = np.zeros(warm_len, dtype=np.float32)
        wav = torch.from_numpy(wav_np).unsqueeze(0)
        ref = wav[:, :min(wav.shape[-1], int(1 * self.sr))].unsqueeze(0).to(self.device)
        self._cuda_sync()
        spk_emb = self.spk_encoder(ref)
        sim = channel_simulate(self.st_model, wav, n_layers=3, p_loss=0.0, device=self.device)
        lat_in = sim["latent_ch"]
        lat_flow_in = self._ensure_flow_min_frames(lat_in)
        lat_8 = flow_sample(self.flow_model, lat_flow_in, spk_emb, n_steps=min(int(self.args.n_steps), 4), n_layers=3)
        lat_8 = lat_8[..., :lat_in.shape[-1]]
        _ = self.st_model.decoder(lat_8)
        self._cuda_sync()
        self.warmup_done = True

    def _ensure_flow_min_frames(self, latent: torch.Tensor) -> torch.Tensor:
        cur_t = int(latent.shape[-1])
        if cur_t >= self.min_flow_frames:
            return latent
        pad_right = self.min_flow_frames - cur_t
        if cur_t > 1:
            return F.pad(latent, (0, pad_right), mode="replicate")
        return F.pad(latent, (0, pad_right), mode="constant", value=0.0)

    @torch.no_grad()
    def run_proposed(self, wav_np: np.ndarray, n_layers: int, plr: float) -> np.ndarray:
        wav = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
        ref = wav[:, :min(wav.shape[-1], int(5 * self.sr))].unsqueeze(0).to(self.device)
        spk_emb = self.spk_encoder(ref)
        sim = channel_simulate(self.st_model, wav, n_layers, plr, self.device)
        lat_in = sim["latent_ch"]
        lat_flow_in = self._ensure_flow_min_frames(lat_in)
        lat_8 = flow_sample(self.flow_model, lat_flow_in, spk_emb, n_steps=self.args.n_steps, n_layers=n_layers)
        lat_8 = lat_8[..., :lat_in.shape[-1]]
        wav_out = self.st_model.decoder(lat_8).squeeze().cpu().numpy().astype(np.float32)
        return ensure_len(wav_out, len(wav_np))

    def run_opus(self, wav_np: np.ndarray, bitrate_kbps: float, plr: float) -> Optional[np.ndarray]:
        wav = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
        out = opus_lbrr_with_plr_ffmpeg(wav, self.sr, bitrate_kbps, plr, frame_ms=FRAME_MS_BASE)
        if out is None:
            return None
        return ensure_len(out, len(wav_np))

    def run_encodec(self, wav_np: np.ndarray, bitrate_kbps: float, plr: float) -> Optional[np.ndarray]:
        if not check_encodec_available():
            return None
        wav = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)
        out = self.encodec_helper.decode_with_lfrplc(wav, self.sr, bitrate_kbps, plr, self.device)
        if out is None:
            return None
        return ensure_len(out, len(wav_np))

    def _update_item_status(self, status_path: str, item_name: str, **kwargs) -> None:
        with self.io_lock:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
            for item in status.get("items", []):
                if item.get("name") == item_name:
                    item.update(kwargs)
                    break
            update_status_file(status_path, status)

    def _update_global_status(self, status_path: str, **kwargs) -> None:
        with self.io_lock:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
            status.update(kwargs)
            update_status_file(status_path, status)

    def _update_meta_item(self, meta_path: str, item_name: str, **kwargs) -> None:
        with self.io_lock:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for item in meta.get("items", []):
                if item.get("name") == item_name:
                    item.update(kwargs)
                    break
            write_json_atomic(meta_path, meta)

    def simulate(self, input_path: str, out_root: str, chunk_ms: int, plr: float, max_sec: float,
                 proposed_n_layers: int, opus_bitrate: float, encodec_bitrate: float,
                 enable_proposed: bool, enable_opus: bool, enable_encodec: bool) -> Dict[str, object]:
        self.ensure_models()
        wav_np = load_audio_ffmpeg(input_path, self.sr)
        wav_np = wav_np[:int(max_sec * self.sr)]
        duration_sec = len(wav_np) / float(self.sr)
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = os.path.join(out_root, run_id)
        os.makedirs(run_dir, exist_ok=True)

        source_wav_path = os.path.join(run_dir, "source.wav")
        source_mel_path = os.path.join(run_dir, "source_mel.png")
        save_wav(source_wav_path, wav_np, self.sr)
        save_mel_image(wav_np, self.sr, source_mel_path)

        results: List[SystemResult] = []

        if enable_proposed:
            t0 = time.perf_counter()
            wav_out = self.run_proposed(wav_np, proposed_n_layers, plr)
            proc_sec = time.perf_counter() - t0
            results.append(SystemResult(
                name="proposed",
                title="Proposed",
                subtitle=f"{proposed_n_layers * 0.5:.1f} kbps, FLOW, PLR={plr * 100:.0f}%",
                wav=wav_out,
                proc_sec=proc_sec,
                rtf=proc_sec / max(duration_sec, 1e-6),
            ))

        if enable_opus:
            t0 = time.perf_counter()
            wav_out = self.run_opus(wav_np, opus_bitrate, plr)
            proc_sec = time.perf_counter() - t0
            if wav_out is not None:
                results.append(SystemResult(
                    name="opus",
                    title="Opus",
                    subtitle=f"{opus_bitrate:.1f} kbps, LBRR, PLR={plr * 100:.0f}%",
                    wav=wav_out,
                    proc_sec=proc_sec,
                    rtf=proc_sec / max(duration_sec, 1e-6),
                ))

        if enable_encodec and check_encodec_available():
            t0 = time.perf_counter()
            wav_out = self.run_encodec(wav_np, encodec_bitrate, plr)
            proc_sec = time.perf_counter() - t0
            if wav_out is not None:
                results.append(SystemResult(
                    name="encodec",
                    title="EnCodec",
                    subtitle=f"{encodec_bitrate:.1f} kbps, LFR-PLC, PLR={plr * 100:.0f}%",
                    wav=wav_out,
                    proc_sec=proc_sec,
                    rtf=proc_sec / max(duration_sec, 1e-6),
                ))

        items = []
        source_item = {
            "name": "source",
            "title": "Source",
            "subtitle": f"Reference speech, reveal step = {chunk_ms} ms",
            "audio_rel": "source.wav",
            "mel_rel": "source_mel.png",
            "proc_sec": 0.0,
            "rtf": 0.0,
        }
        items.append(source_item)

        for res in results:
            wav_path = f"{res.name}.wav"
            mel_path = f"{res.name}_mel.png"
            save_wav(os.path.join(run_dir, wav_path), res.wav, self.sr)
            save_mel_image(res.wav, self.sr, os.path.join(run_dir, mel_path))
            items.append({
                "name": res.name,
                "title": res.title,
                "subtitle": res.subtitle,
                "audio_rel": wav_path,
                "mel_rel": mel_path,
                "proc_sec": res.proc_sec,
                "rtf": res.rtf,
            })

        meta = {
            "run_id": run_id,
            "input_name": os.path.basename(input_path),
            "input_path": input_path,
            "duration_sec": duration_sec,
            "plr": plr,
            "chunk_ms": chunk_ms,
            "frame_ms": FRAME_MS_BASE,
            "items": items,
            "settings": {
                "proposed_n_layers": proposed_n_layers,
                "opus_bitrate": opus_bitrate,
                "encodec_bitrate": encodec_bitrate,
                "enable_proposed": enable_proposed,
                "enable_opus": enable_opus,
                "enable_encodec": enable_encodec and check_encodec_available(),
            },
        }
        with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta

    def launch_async_job(self, input_path: str, out_root: str, chunk_ms: int, plr: float, max_sec: float,
                         proposed_n_layers: int, opus_bitrate: float, encodec_bitrate: float,
                         enable_proposed: bool, enable_opus: bool, enable_encodec: bool) -> Dict[str, object]:
        wav_np = load_audio_ffmpeg(input_path, self.sr)
        wav_np = wav_np[:int(max_sec * self.sr)]
        duration_sec = len(wav_np) / float(self.sr)
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = os.path.join(out_root, run_id)
        os.makedirs(run_dir, exist_ok=True)

        source_wav_path = os.path.join(run_dir, "source.wav")
        source_mel_path = os.path.join(run_dir, "source_mel.png")
        save_wav(source_wav_path, wav_np, self.sr)
        save_mel_image(wav_np, self.sr, source_mel_path)

        items = [{
            "name": "source",
            "title": "Source",
            "subtitle": "Reference speech",
            "audio_rel": "source.wav",
            "mel_rel": "source_mel.png",
            "base_mel_rel": "source_mel.png",
            "proc_sec": 0.0,
            "rtf": 0.0,
            "audio_ready": True,
            "mel_ready": True,
            "generated_sec": duration_sec,
            "mel_generated_sec": duration_sec,
        }]

        tasks = []
        if enable_proposed:
            items.append({
                "name": "proposed",
                "title": "Proposed",
                "subtitle": f"{proposed_n_layers * 0.5:.1f} kbps, FLOW, PLR={plr * 100:.0f}%",
                "audio_rel": "proposed.wav",
                "mel_rel": "proposed_mel.png",
                "base_mel_rel": "source_mel.png",
                "proc_sec": 0.0,
                "rtf": 0.0,
                "audio_ready": False,
                "mel_ready": False,
                "generated_sec": 0.0,
                "mel_generated_sec": 0.0,
            })
            tasks.append(("proposed", lambda: self.run_proposed(wav_np, proposed_n_layers, plr)))
        if enable_opus:
            items.append({
                "name": "opus",
                "title": "Opus",
                "subtitle": f"{opus_bitrate:.1f} kbps, LBRR, PLR={plr * 100:.0f}%",
                "audio_rel": "opus.wav",
                "mel_rel": "opus_mel.png",
                "base_mel_rel": "source_mel.png",
                "proc_sec": 0.0,
                "rtf": 0.0,
                "audio_ready": False,
                "mel_ready": False,
                "generated_sec": 0.0,
                "mel_generated_sec": 0.0,
            })
            tasks.append(("opus", lambda: self.run_opus(wav_np, opus_bitrate, plr)))
        if enable_encodec and check_encodec_available():
            items.append({
                "name": "encodec",
                "title": "EnCodec",
                "subtitle": f"{encodec_bitrate:.1f} kbps, LFR-PLC, PLR={plr * 100:.0f}%",
                "audio_rel": "encodec.wav",
                "mel_rel": "encodec_mel.png",
                "base_mel_rel": "source_mel.png",
                "proc_sec": 0.0,
                "rtf": 0.0,
                "audio_ready": False,
                "mel_ready": False,
                "generated_sec": 0.0,
                "mel_generated_sec": 0.0,
            })
            tasks.append(("encodec", lambda: self.run_encodec(wav_np, encodec_bitrate, plr)))

        run_meta = {
            "run_id": run_id,
            "input_name": os.path.basename(input_path),
            "input_path": input_path,
            "duration_sec": duration_sec,
            "plr": plr,
            "chunk_ms": chunk_ms,
            "frame_ms": FRAME_MS_BASE,
            "items": items,
            "settings": {
                "proposed_n_layers": proposed_n_layers,
                "opus_bitrate": opus_bitrate,
                "encodec_bitrate": encodec_bitrate,
                "enable_proposed": enable_proposed,
                "enable_opus": enable_opus,
                "enable_encodec": enable_encodec and check_encodec_available(),
            },
        }
        write_json_atomic(os.path.join(run_dir, "run_meta.json"), run_meta)

        status_path = os.path.join(run_dir, "status.json")
        initial_status = {
            "run_id": run_id,
            "done": False,
            "failed": False,
            "progress": 0.0,
            "status_text": "Queued",
            "duration_sec": duration_sec,
            "items": [],
        }
        for idx, item in enumerate(items):
            initial_status["items"].append({
                "name": item["name"],
                "title": item["title"],
                "subtitle": item["subtitle"],
                "dom_id": f"mel_{idx}",
                "audio_url": f"/runs/{run_id}/files/{item['audio_rel']}",
                "mel_url": f"/runs/{run_id}/files/{item['mel_rel']}",
                "audio_ready": bool(item["audio_ready"]),
                "mel_ready": bool(item["mel_ready"]),
                "generated_sec": float(item["generated_sec"]),
                "elapsed_sec": 0.0,
                "started_at": None,
                "started": item["name"] == "source",
                "finished": item["name"] == "source",
                "show_timing": item["name"] in ("proposed", "opus", "encodec"),
            })
        update_status_file(status_path, initial_status)

        def worker() -> None:
            try:
                active_names = [name for name, _ in tasks]
                total_tasks = max(1, len(active_names))
                self._update_global_status(status_path, status_text="Loading models", progress=0.01)
                self.ensure_models()
                self.warmup_models()
                self._update_global_status(status_path, status_text="Running", progress=0.03)

                def update_overall_progress() -> None:
                    with self.io_lock:
                        with open(status_path, "r", encoding="utf-8") as f:
                            status_now = json.load(f)
                    finished_count = 0
                    for item in status_now.get("items", []):
                        if item.get("name") in active_names:
                            if item.get("finished"):
                                finished_count += 1
                    prog = min(0.99, finished_count / total_tasks)
                    running_names = [item.get("name") for item in status_now.get("items", []) if item.get("name") in active_names and not item.get("finished")]
                    text = "Running: " + (", ".join(running_names) if running_names else "finalizing")
                    self._update_global_status(status_path, status_text=text, progress=prog)

                def run_one(name: str, fn) -> None:
                    started_at = time.perf_counter()
                    self._update_item_status(status_path, name, started=True, finished=False, elapsed_sec=0.0, started_at=started_at)
                    stop_event = threading.Event()

                    def tick_elapsed() -> None:
                        while not stop_event.is_set():
                            self._update_item_status(status_path, name, elapsed_sec=time.perf_counter() - started_at)
                            time.sleep(0.2)

                    ticker = threading.Thread(target=tick_elapsed, daemon=True)
                    ticker.start()
                    self._cuda_sync()
                    wav_out = fn()
                    self._cuda_sync()
                    proc_sec = time.perf_counter() - started_at
                    stop_event.set()
                    ticker.join(timeout=0.5)
                    if wav_out is None:
                        self._update_item_status(
                            status_path,
                            name,
                            generated_sec=0.0,
                            mel_ready=False,
                            audio_ready=False,
                            elapsed_sec=proc_sec,
                            started_at=started_at,
                            finished=True,
                        )
                        update_overall_progress()
                        return

                    self._update_meta_item(
                        os.path.join(run_dir, "run_meta.json"),
                        name,
                        audio_ready=True,
                        mel_ready=True,
                        generated_sec=duration_sec,
                        mel_generated_sec=duration_sec,
                        proc_sec=proc_sec,
                        rtf=proc_sec / max(duration_sec, 1e-6),
                    )
                    save_wav(os.path.join(run_dir, f"{name}.wav"), wav_out, self.sr)
                    save_mel_image(wav_out, self.sr, os.path.join(run_dir, f"{name}_mel.png"))
                    self._update_item_status(
                        status_path,
                        name,
                        elapsed_sec=proc_sec,
                        audio_ready=True,
                        mel_ready=True,
                        generated_sec=duration_sec,
                        mel_generated_sec=duration_sec,
                        finished=True,
                        started_at=started_at,
                    )
                    update_overall_progress()

                for name, fn in tasks:
                    with self.io_lock:
                        with open(status_path, "r", encoding="utf-8") as f:
                            status_now = json.load(f)
                    done_before = sum(1 for item in status_now.get("items", []) if item.get("name") in active_names and item.get("finished"))
                    self._update_global_status(status_path, status_text=f"Running: {name}", progress=min(0.99, done_before / max(total_tasks, 1)))
                    run_one(name, fn)
                self._update_global_status(status_path, done=True, status_text="Completed", progress=1.0)
            except Exception as e:
                self._update_global_status(status_path, done=True, failed=True, status_text=f"Failed: {type(e).__name__}: {e}", progress=1.0)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return run_meta


def create_app(args):
    app = Flask(__name__)
    engine = SimulatorEngine(args)
    demo_root = os.path.abspath(args.demo_dir)
    upload_root = os.path.join(demo_root, "uploads")
    os.makedirs(upload_root, exist_ok=True)
    os.makedirs(demo_root, exist_ok=True)

    @app.route("/")
    def index():
        return render_template_string(
            INDEX_HTML,
            frame_ms=FRAME_MS_BASE,
            samples_per_frame=int(args.sample_rate * FRAME_MS_BASE / 1000),
            sample_rate=args.sample_rate,
            has_encodec=check_encodec_available(),
            chunk_ms_options=CHUNK_MS_OPTIONS,
            plr_options=PLR_OPTIONS,
            proposed_options=PROPOSED_OPTIONS,
            opus_options=OPUS_OPTIONS,
            encodec_options=ENCODEC_OPTIONS,
        )

    @app.route("/run", methods=["POST"])
    def run_demo():
        local_path = (request.form.get("audio_path") or "").strip()
        upload = request.files.get("audio_file")
        if upload and upload.filename:
            filename = secure_filename(upload.filename) or f"upload_{uuid.uuid4().hex[:8]}.wav"
            local_path = os.path.join(upload_root, filename)
            upload.save(local_path)
        if not local_path:
            return "No input file provided.", 400
        if not os.path.isabs(local_path):
            local_path = os.path.abspath(os.path.join(os.getcwd(), local_path))

        plr = float(request.form.get("plr", 0.05))
        max_sec = float(request.form.get("max_sec", 6))
        proposed_n_layers = int(request.form.get("proposed_n_layers", 6))
        opus_bitrate = float(request.form.get("opus_bitrate", 8.0))
        encodec_bitrate = float(request.form.get("encodec_bitrate", 3.0))
        enable_proposed = bool(request.form.get("enable_proposed"))
        enable_opus = bool(request.form.get("enable_opus"))
        enable_encodec = bool(request.form.get("enable_encodec"))

        try:
            meta = engine.launch_async_job(
                input_path=local_path,
                out_root=demo_root,
                chunk_ms=FRAME_MS_BASE,
                plr=plr,
                max_sec=max_sec,
                proposed_n_layers=proposed_n_layers,
                opus_bitrate=opus_bitrate,
                encodec_bitrate=encodec_bitrate,
                enable_proposed=enable_proposed,
                enable_opus=enable_opus,
                enable_encodec=enable_encodec,
            )
        except Exception as e:
            return f"Simulation failed: {type(e).__name__}: {e}", 500
        return redirect(url_for("wait_run", run_id=meta["run_id"]))

    @app.route("/runs/<run_id>/wait")
    def wait_run(run_id: str):
        run_dir = os.path.join(demo_root, run_id)
        if not os.path.exists(os.path.join(run_dir, "status.json")):
            abort(404)
        return render_template_string(WAIT_HTML, run_id=run_id)

    @app.route("/runs/<run_id>")
    def show_run(run_id: str):
        run_dir = os.path.join(demo_root, run_id)
        meta_path = os.path.join(run_dir, "run_meta.json")
        if not os.path.exists(meta_path):
            abort(404)
        with open(meta_path, "r", encoding="utf-8") as f:
            run = json.load(f)
        items = []
        for idx, item in enumerate(run["items"]):
            items.append({
                "name": item["name"],
                "title": item["title"],
                "subtitle": item["subtitle"],
                "audio_url": url_for("serve_run_file", run_id=run_id, filename=item["audio_rel"]),
                "mel_url": url_for("serve_run_file", run_id=run_id, filename=item["mel_rel"]),
                "base_mel_url": url_for("serve_run_file", run_id=run_id, filename=item.get("base_mel_rel", item["mel_rel"])),
                "proc_sec": float(item["proc_sec"]),
                "rtf": float(item["rtf"]),
                "dom_id": f"mel_{idx}",
                "audio_ready": bool(item.get("audio_ready", False)),
                "mel_ready": bool(item.get("mel_ready", False)),
                "generated_sec": float(item.get("generated_sec", 0.0)),
                "show_timing": item["name"] in ("proposed", "opus", "encodec"),
            })
        return render_template_string(
            RESULT_HTML,
            frame_ms=FRAME_MS_BASE,
            sample_rate=args.sample_rate,
            run=run,
            items=items,
        )

    @app.route("/runs/<run_id>/status")
    def run_status(run_id: str):
        run_dir = os.path.join(demo_root, run_id)
        status_path = os.path.join(run_dir, "status.json")
        if not os.path.exists(status_path):
            abort(404)
        with open(status_path, "r", encoding="utf-8") as f:
            return app.response_class(f.read(), mimetype="application/json")

    @app.route("/runs/<run_id>/files/<path:filename>")
    def serve_run_file(run_id: str, filename: str):
        run_dir = os.path.join(demo_root, run_id)
        if not os.path.isdir(run_dir):
            abort(404)
        return send_from_directory(run_dir, filename)

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Web speech communication system for headless servers")
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt", default="output/flow_checkpoints_stage3/best.pt")
    p.add_argument("--demo_dir", default="output/speech_communication_system")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    app = create_app(args)
    print("Speech communication system ready.")
    print(f"Output dir: {os.path.abspath(args.demo_dir)}")
    for url in get_bind_urls(args.host, args.port):
        print(f"Open: {url}")
    print(f"Base frame: {FRAME_MS_BASE} ms ({int(args.sample_rate * FRAME_MS_BASE / 1000)} samples @ {args.sample_rate} Hz)")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
