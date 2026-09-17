from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp_scene_ntn.png"
OUT = ROOT / "scene_assets"


# Manually selected regions from tmp_scene_ntn.png.
# These crops are intended as reusable visual materials for PPT/Figma redraw,
# not as pixel-perfect semantic segmentation.
REGIONS = {
    "satellite_left": (188, 5, 366, 146),
    "satellite_mid": (438, 0, 625, 99),
    "satellite_right": (706, 8, 854, 109),
    "uav_1": (306, 211, 404, 267),
    "uav_2": (553, 205, 652, 270),
    "uav_3": (710, 210, 821, 270),
    "gateway": (474, 318, 586, 403),
    "tower_left": (117, 340, 210, 462),
    "tower_mid": (325, 384, 419, 501),
    "tower_right": (842, 344, 924, 472),
    "remote_mountains": (0, 248, 286, 480),
    "offshore_platform": (438, 390, 616, 570),
    "moving_vehicle": (703, 392, 867, 556),
    "ambulance_scene": (872, 367, 1024, 555),
    "speech_service_left": (0, 428, 199, 572),
    "speech_service_right": (785, 432, 1024, 572),
    "packet_loss_marker": (587, 302, 657, 385),
    "speech_packet_icon": (406, 142, 442, 172),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    im = Image.open(SRC).convert("RGBA")

    manifest = []
    for name, box in REGIONS.items():
        crop = im.crop(box)
        out_path = OUT / f"{name}.png"
        crop.save(out_path)
        manifest.append((name, out_path.name, crop.size))

    manifest_path = OUT / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        for name, file_name, size in manifest:
            f.write(f"{name}: {file_name} size={size[0]}x{size[1]}\n")

    print(f"saved assets to {OUT}")
    print(f"saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
