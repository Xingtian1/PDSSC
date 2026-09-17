import base64
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "scene_assets"
OUT = ROOT / "scene_assets" / "scene_redraw.svg"


def png_data_uri(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def image_tag(name: str, x: int, y: int, w: int, h: int) -> str:
    href = png_data_uri(ASSET_DIR / f"{name}.png")
    return f'<image href="{href}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" />'


def text_tag(x: int, y: int, text: str, size: int = 20, weight: str = "400", anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111111">{text}</text>'
    )


def arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#0d2f6b", width: int = 4, dashed: bool = False, opacity: float = 1.0) -> str:
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
        f'opacity="{opacity}" marker-end="url(#arrowhead)"{dash} />'
    )


def packet_loss(x: int, y: int, text: str = "Packet loss") -> str:
    return f"""
    <g transform="translate({x},{y})">
      <rect x="-58" y="-18" rx="14" ry="14" width="116" height="36" fill="#f16522"/>
      <text x="0" y="7" font-family="Times New Roman, Times, serif" font-size="20" text-anchor="middle" fill="white">{text}</text>
    </g>
    """


def packet_icon(x: int, y: int) -> str:
    return f"""
    <g transform="translate({x},{y})" opacity="0.9">
      <rect x="-8" y="-7" width="16" height="12" rx="2" ry="2" fill="#eef6ff" stroke="#355e8e" stroke-width="1.4"/>
      <path d="M -3 5 L -6 9 L 1 5" fill="#eef6ff" stroke="#355e8e" stroke-width="1.2"/>
    </g>
    """


SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#eef5fb"/>
    </linearGradient>
    <marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0,0 10,4 0,8" fill="#0d2f6b"/>
    </marker>
    <marker id="arrowhead_cyan" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
      <polygon points="0,0 10,4 0,8" fill="#17c5d8"/>
    </marker>
  </defs>

  <rect width="1200" height="720" fill="url(#bg)"/>
  <rect x="0" y="185" width="1200" height="150" fill="#f4f8fc" opacity="0.9"/>
  <rect x="0" y="335" width="1200" height="165" fill="#ffffff" opacity="0.95"/>

  {image_tag("satellite_left", 185, 8, 150, 120)}
  {image_tag("satellite_mid", 487, 5, 146, 78)}
  {image_tag("satellite_right", 805, 10, 128, 82)}
  {image_tag("uav_1", 375, 230, 86, 48)}
  {image_tag("uav_2", 646, 226, 90, 54)}
  {image_tag("uav_3", 852, 226, 92, 50)}
  {image_tag("gateway", 548, 345, 98, 74)}
  {image_tag("tower_left", 130, 360, 84, 112)}
  {image_tag("tower_mid", 360, 408, 78, 100)}
  {image_tag("tower_right", 963, 362, 74, 104)}
  {image_tag("remote_mountains", 0, 270, 255, 206)}
  {image_tag("offshore_platform", 488, 438, 180, 182)}
  {image_tag("moving_vehicle", 813, 454, 155, 118)}
  {image_tag("ambulance_scene", 1010, 419, 146, 145)}
  {image_tag("speech_service_left", 0, 520, 185, 132)}
  {image_tag("speech_service_right", 945, 520, 230, 132)}

  {arrow(255, 98, 559, 70)}
  {arrow(633, 70, 807, 86)}
  {arrow(424, 248, 560, 88)}
  {arrow(684, 245, 561, 88)}
  {arrow(895, 240, 868, 90)}
  {arrow(560, 92, 696, 232)}
  {arrow(560, 92, 876, 229)}
  {arrow(255, 108, 422, 240)}
  {arrow(422, 246, 607, 358)}
  {arrow(690, 246, 607, 358)}
  {arrow(893, 242, 608, 358)}
  {arrow(171, 417, 558, 360)}
  {arrow(400, 455, 559, 365)}
  {arrow(1001, 409, 651, 362)}

  <line x1="427" y1="255" x2="684" y2="251" stroke="#17c5d8" stroke-width="4" stroke-linecap="round" marker-end="url(#arrowhead_cyan)" />
  <line x1="685" y1="251" x2="895" y2="250" stroke="#17c5d8" stroke-width="4" stroke-linecap="round" marker-end="url(#arrowhead_cyan)" />
  <line x1="173" y1="415" x2="425" y2="255" stroke="#17c5d8" stroke-width="3.5" stroke-dasharray="10 7" marker-end="url(#arrowhead_cyan)" opacity="0.9" />
  <line x1="645" y1="503" x2="690" y2="269" stroke="#17c5d8" stroke-width="3.5" stroke-dasharray="10 7" marker-end="url(#arrowhead_cyan)" opacity="0.9" />
  <line x1="905" y1="490" x2="894" y2="267" stroke="#17c5d8" stroke-width="3.5" stroke-dasharray="10 7" marker-end="url(#arrowhead_cyan)" opacity="0.9" />

  {arrow(120, 545, 174, 470, color="#0d2f6b", width=3)}
  {arrow(183, 550, 364, 463, color="#17c5d8", width=3)}
  {arrow(1040, 546, 999, 466, color="#0d2f6b", width=3)}
  {arrow(1002, 544, 905, 520, color="#17c5d8", width=3)}

  {packet_loss(268, 222)}
  {packet_loss(748, 342)}
  {packet_loss(912, 362)}

  {packet_icon(486, 82)}
  {packet_icon(527, 95)}
  {packet_icon(594, 112)}
  {packet_icon(731, 109)}
  {packet_icon(863, 131)}
  {packet_icon(445, 289)}
  {packet_icon(716, 265)}
  {packet_icon(913, 268)}
  {packet_icon(337, 421)}
  {packet_icon(541, 392)}
  {packet_icon(950, 414)}

  {text_tag(156, 77, "Sat 1", 20)}
  {text_tag(604, 48, "Sat 2", 20)}
  {text_tag(934, 78, "Sat 3", 20)}
  {text_tag(427, 297, "UAV 1", 21)}
  {text_tag(690, 297, "UAV 2", 21)}
  {text_tag(899, 294, "UAV 3", 21)}
  {text_tag(599, 433, "Gateway", 24)}

  {text_tag(93, 420, "Remote", 20)}
  {text_tag(93, 446, "Mountain Area", 20)}
  {text_tag(165, 503, "Remote Edge", 18)}
  {text_tag(165, 526, "Base Station", 18)}
  {text_tag(398, 507, "Base", 18)}
  {text_tag(398, 529, "Station 2", 18)}
  {text_tag(997, 502, "Ground", 18)}
  {text_tag(997, 525, "Station 3", 18)}
  {text_tag(576, 619, "Offshore platform", 22)}
  {text_tag(874, 599, "Moving vehicle", 22)}
  {text_tag(1089, 514, "Emergency rescue", 22)}
  {text_tag(88, 670, "Speech service", 22)}
  {text_tag(1082, 670, "Speech service", 22)}

  {text_tag(1140, 46, "Space Region", 28, "700")}
  {text_tag(1140, 78, "Satellites provide realistic", 16)}
  {text_tag(1140, 99, "coordinated backhaul support", 16)}
  {text_tag(1140, 260, "Air Region", 28, "700")}
  {text_tag(1140, 291, "UAVs provide flexible relay", 16)}
  {text_tag(1140, 312, "and access support", 16)}
  {text_tag(1140, 406, "Ground Region", 28, "700")}
  {text_tag(1140, 436, "Gateway-centered access", 16)}
  {text_tag(1140, 457, "and service scenarios", 16)}

  {text_tag(581, 236, "Backhaul", 18)}
  {text_tag(813, 205, "ISL", 18)}
  {text_tag(275, 330, "Weak-coverage", 16)}
  {text_tag(275, 350, "UAV", 16)}
  {text_tag(756, 281, "Inter-UAV links", 16)}
  {text_tag(756, 301, "where spatially reasonable", 16)}
</svg>
"""


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(SVG, encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
