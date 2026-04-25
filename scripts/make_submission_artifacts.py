import os
import struct
import zlib
from pathlib import Path


FONT = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11110", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def make_canvas(width, height, color=(255, 255, 255)):
    return [[color for _ in range(width)] for _ in range(height)]


def set_px(img, x, y, color):
    if 0 <= y < len(img) and 0 <= x < len(img[0]):
        img[y][x] = color


def draw_rect(img, x0, y0, x1, y1, color, fill=False):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if fill or x in (x0, x1) or y in (y0, y1):
                set_px(img, x, y, color)


def draw_line(img, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        set_px(img, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_text(img, x, y, text, color=(30, 30, 30), scale=2):
    cursor = x
    for char in text.upper():
        glyph = FONT.get(char, FONT[" "])
        for row_i, row in enumerate(glyph):
            for col_i, bit in enumerate(row):
                if bit == "1":
                    draw_rect(
                        img,
                        cursor + col_i * scale,
                        y + row_i * scale,
                        cursor + (col_i + 1) * scale - 1,
                        y + (row_i + 1) * scale - 1,
                        color,
                        fill=True,
                    )
        cursor += 6 * scale


def save_png(path, img):
    height = len(img)
    width = len(img[0])
    raw = b"".join(b"\x00" + b"".join(bytes(px) for px in row) for row in img)

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    Path(path).write_bytes(png)


def plot_line(path, title, x_label, y_label, points, y_min=None, y_max=None):
    img = make_canvas(900, 560)
    black = (30, 30, 30)
    grid = (215, 220, 225)
    blue = (31, 119, 180)
    left, top, right, bottom = 90, 70, 850, 470
    y_values = [p[1] for p in points]
    y_min = min(y_values) if y_min is None else y_min
    y_max = max(y_values) if y_max is None else y_max
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)

    draw_text(img, 250, 25, title, black, scale=3)
    draw_text(img, 380, 510, x_label, black, scale=2)
    draw_text(img, 15, 250, y_label, black, scale=2)
    draw_rect(img, left, top, right, bottom, black)

    for i in range(1, 5):
        x = left + (right - left) * i // 5
        y = top + (bottom - top) * i // 5
        draw_line(img, x, top, x, bottom, grid)
        draw_line(img, left, y, right, y, grid)

    last = None
    for x_val, y_val in points:
        x = int(left + (x_val - x_min) / max(1e-9, x_max - x_min) * (right - left))
        y = int(bottom - (y_val - y_min) / max(1e-9, y_max - y_min) * (bottom - top))
        if last is not None:
            draw_line(img, last[0], last[1], x, y, blue)
            draw_line(img, last[0], last[1] + 1, x, y + 1, blue)
        draw_rect(img, x - 3, y - 3, x + 3, y + 3, blue, fill=True)
        last = (x, y)

    draw_text(img, left, bottom + 10, str(x_min), black, scale=1)
    draw_text(img, right - 35, bottom + 10, str(x_max), black, scale=1)
    draw_text(img, 35, top - 5, f"{y_max:.1f}", black, scale=1)
    draw_text(img, 35, bottom - 5, f"{y_min:.1f}", black, scale=1)
    save_png(path, img)


def plot_bars(path):
    img = make_canvas(900, 560)
    black = (30, 30, 30)
    green = (44, 160, 44)
    orange = (255, 127, 14)
    red = (214, 39, 40)
    left, top, right, bottom = 95, 80, 850, 455
    metrics = [
        ("REWARD", -13.52, -12.212),
        ("USERS", 116.9, 141.75),
        ("SURVIVAL", 0.95, 0.95),
        ("EFFICIENCY", 0.16, 0.207),
    ]

    draw_text(img, 120, 25, "BASELINE VS TRAINED CEO", black, scale=3)
    draw_rect(img, left, top, right, bottom, black)
    group_w = (right - left) // len(metrics)
    for i, (name, base, trained) in enumerate(metrics):
        x0 = left + i * group_w + 35
        max_val = max(abs(base), abs(trained), 1)
        base_h = int(abs(base) / max_val * 230)
        trained_h = int(abs(trained) / max_val * 230)
        draw_rect(img, x0, bottom - base_h, x0 + 35, bottom, orange, fill=True)
        draw_rect(img, x0 + 50, bottom - trained_h, x0 + 85, bottom, green, fill=True)
        draw_text(img, x0 - 10, bottom + 18, name, black, scale=1)
        draw_text(img, x0 - 8, bottom - base_h - 22, f"{base:.1f}", black, scale=1)
        draw_text(img, x0 + 43, bottom - trained_h - 22, f"{trained:.1f}", black, scale=1)

    draw_rect(img, 210, 500, 230, 520, orange, fill=True)
    draw_text(img, 240, 503, "BASELINE", black, scale=1)
    draw_rect(img, 430, 500, 450, 520, green, fill=True)
    draw_text(img, 460, 503, "TRAINED CEO", black, scale=1)
    save_png(path, img)


def plot_reward_curve(path):
    img = make_canvas(900, 560)
    black = (30, 30, 30)
    orange = (255, 127, 14)
    green = (44, 160, 44)
    grid = (215, 220, 225)
    left, top, right, bottom = 120, 80, 820, 460
    values = [("BASELINE", -13.52, orange), ("TRAINED CEO", -12.212, green)]
    y_min, y_max = -16.0, 0.0

    draw_text(img, 165, 25, "AVERAGE EPISODE REWARD", black, scale=3)
    draw_rect(img, left, top, right, bottom, black)
    for i in range(1, 5):
        y = top + (bottom - top) * i // 5
        draw_line(img, left, y, right, y, grid)

    zero_y = int(bottom - (0 - y_min) / (y_max - y_min) * (bottom - top))
    for i, (name, value, color) in enumerate(values):
        x0 = left + 180 + i * 230
        y = int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))
        draw_rect(img, x0, zero_y, x0 + 90, y, color, fill=True)
        draw_text(img, x0 - 12, bottom + 20, name, black, scale=1)
        draw_text(img, x0 + 5, y - 24, f"{value:.2f}", black, scale=1)

    draw_text(img, 35, top - 5, "0", black, scale=1)
    draw_text(img, 25, bottom - 5, "-16", black, scale=1)
    draw_text(img, 300, 510, "TRAINED IS LESS NEGATIVE", black, scale=2)
    save_png(path, img)


def main():
    output_dir = Path("docs/assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_points = [
        (10, 3.02),
        (20, 2.32),
        (30, 1.59),
        (40, 1.03),
        (50, 0.69),
        (60, 0.49),
        (70, 0.37),
        (90, 0.28),
        (120, 0.23),
        (180, 0.21),
        (260, 0.20),
        (380, 0.19),
        (520, 0.18),
        (746, 0.18),
    ]
    plot_line(
        output_dir / "loss_curve.png",
        "CEO TRAINING LOSS",
        "TRAINING STEP",
        "LOSS",
        loss_points,
        y_min=0.0,
        y_max=3.2,
    )
    plot_bars(output_dir / "reward_comparison.png")
    plot_reward_curve(output_dir / "reward_curve.png")
    print(f"Wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
