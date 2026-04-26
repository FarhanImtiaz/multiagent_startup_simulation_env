import json
import struct
import zlib
from pathlib import Path

POLICY_METRICS = {
    "baseline": {
        "label": "Baseline CEO",
        "episodes": 20,
        "average_total_reward": -13.520,
        "average_final_money": 19244.960,
        "average_final_users": 116.900,
        "survival_rate": 0.950,
        "decision_efficiency": 0.160,
        "main_failure_mode": "no_users in 1/20",
    },
    "raw_grpo": {
        "label": "Raw GRPO CEO ablation",
        "episodes": 20,
        "average_total_reward": -5.243,
        "average_final_money": -2538.432,
        "average_final_users": 103.550,
        "survival_rate": 0.000,
        "decision_efficiency": 0.097,
        "main_failure_mode": "bankruptcy-heavy",
    },
    "governed_grpo": {
        "label": "GRPO + Governed CEO",
        "episodes": 20,
        "average_total_reward": -13.520,
        "average_final_money": 19244.960,
        "average_final_users": 116.900,
        "survival_rate": 0.950,
        "decision_efficiency": 0.160,
        "main_failure_mode": "no_users in 1/20",
    },
}


def metric(agent, name):
    return POLICY_METRICS[agent][name]


def metric_label(value):
    return f"{value:.3f}"


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
    red = (214, 39, 40)
    green = (44, 160, 44)
    orange = (255, 127, 14)
    metrics = [
        ("AVG REWARD", "average_total_reward"),
        ("FINAL MONEY", "average_final_money"),
        ("FINAL USERS", "average_final_users"),
        ("SURVIVAL", "survival_rate"),
        ("EFFICIENCY", "decision_efficiency"),
    ]

    draw_text(img, 180, 25, "EXACT FINAL EVAL METRICS", black, scale=3)
    draw_text(img, 70, 95, "METRIC", black, scale=2)
    draw_text(img, 275, 95, "BASELINE", orange, scale=2)
    draw_text(img, 480, 95, "RAW GRPO", red, scale=2)
    draw_text(img, 685, 95, "GOVERNED", green, scale=2)
    draw_line(img, 60, 128, 850, 128, black)

    y = 165
    for name, key in metrics:
        draw_text(img, 70, y, name, black, scale=1)
        draw_text(img, 275, y, metric_label(metric("baseline", key)), black, scale=1)
        draw_text(img, 480, y, metric_label(metric("raw_grpo", key)), black, scale=1)
        draw_text(img, 685, y, metric_label(metric("governed_grpo", key)), black, scale=1)
        draw_line(img, 60, y + 28, 850, y + 28, (230, 230, 230))
        y += 58

    draw_text(img, 70, 475, "RAW GRPO IS AN ABLATION", black, scale=1)
    draw_text(img, 70, 505, "GOVERNED IS THE FINAL POLICY", black, scale=1)
    save_png(path, img)


def plot_reward_curve(path):
    img = make_canvas(900, 560)
    black = (30, 30, 30)
    orange = (255, 127, 14)
    red = (214, 39, 40)
    green = (44, 160, 44)
    grid = (215, 220, 225)
    left, top, right, bottom = 120, 80, 820, 460
    values = [("BASELINE", metric("baseline", "average_total_reward"), orange), ("RAW GRPO", metric("raw_grpo", "average_total_reward"), red), ("GOVERNED", metric("governed_grpo", "average_total_reward"), green)]
    y_min, y_max = -16.0, 0.0

    draw_text(img, 165, 25, "AVERAGE EPISODE REWARD", black, scale=3)
    draw_rect(img, left, top, right, bottom, black)
    for i in range(1, 5):
        y = top + (bottom - top) * i // 5
        draw_line(img, left, y, right, y, grid)

    zero_y = int(bottom - (0 - y_min) / (y_max - y_min) * (bottom - top))
    for i, (name, value, color) in enumerate(values):
        x0 = left + 110 + i * 190
        y = int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))
        draw_rect(img, x0, zero_y, x0 + 90, y, color, fill=True)
        draw_text(img, x0 - 12, bottom + 20, name, black, scale=1)
        draw_text(img, x0 + 5, y - 24, metric_label(value), black, scale=1)

    draw_text(img, 35, top - 5, "0", black, scale=1)
    draw_text(img, 25, bottom - 5, "-16", black, scale=1)
    draw_text(img, 240, 510, "RAW REWARD HIDES SURVIVAL RISK", black, scale=2)
    save_png(path, img)


def plot_policy_summary(path):
    img = make_canvas(900, 560)
    black = (30, 30, 30)
    red = (214, 39, 40)
    green = (44, 160, 44)
    orange = (255, 127, 14)

    draw_text(img, 125, 25, "CEO POLICY SAFETY SUMMARY", black, scale=3)
    draw_text(img, 70, 110, "BASELINE CEO", orange, scale=2)
    draw_text(img, 70, 150, "SURVIVAL 0.950", black, scale=2)
    draw_text(img, 70, 185, "19 OF 20 MAX DAYS", black, scale=2)

    draw_text(img, 350, 110, "RAW GRPO", red, scale=2)
    draw_text(img, 350, 150, "SURVIVAL 0.000", black, scale=2)
    draw_text(img, 350, 185, "BANKRUPTCY LOOP", black, scale=2)

    draw_text(img, 620, 110, "GOVERNED", green, scale=2)
    draw_text(img, 620, 150, "SURVIVAL 0.950", black, scale=2)
    draw_text(img, 620, 185, "19 OF 20 MAX DAYS", black, scale=2)

    draw_text(img, 105, 300, "LESSON", black, scale=2)
    draw_text(img, 105, 340, "GRPO LEARNED VALID ACTIONS", black, scale=2)
    draw_text(img, 105, 375, "RAW POLICY WAS NOT SAFE", black, scale=2)
    draw_text(img, 105, 410, "GOVERNOR RECOVERED SURVIVAL", black, scale=2)
    save_png(path, img)


def write_comparison_files(output_dir):
    summary = {
        "baseline": POLICY_METRICS["baseline"],
        "raw_grpo_ablation": POLICY_METRICS["raw_grpo"],
        "governed_grpo": POLICY_METRICS["governed_grpo"],
        "deltas": {
            "governed_vs_raw_survival_rate": round(
                metric("governed_grpo", "survival_rate") - metric("raw_grpo", "survival_rate"), 3
            ),
            "governed_vs_raw_average_final_money": round(
                metric("governed_grpo", "average_final_money") - metric("raw_grpo", "average_final_money"), 3
            ),
            "governed_vs_raw_decision_efficiency": round(
                metric("governed_grpo", "decision_efficiency") - metric("raw_grpo", "decision_efficiency"), 3
            ),
            "governed_vs_baseline_survival_rate": round(
                metric("governed_grpo", "survival_rate") - metric("baseline", "survival_rate"), 3
            ),
        },
        "interpretation": [
            "Raw GRPO is reported as an ablation, not as the final deployed system.",
            "The governed GRPO CEO recovers survival from 0.000 to 0.950 relative to raw GRPO.",
            "The governed GRPO CEO matches the baseline CEO survival rate while retaining a trained-policy path in safe states.",
        ],
    }
    (output_dir.parent / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rows = [
        ("Average total reward", "average_total_reward"),
        ("Average final money", "average_final_money"),
        ("Average final users", "average_final_users"),
        ("Survival rate", "survival_rate"),
        ("Decision efficiency", "decision_efficiency"),
    ]
    lines = [
        "# MASS Policy Comparison",
        "",
        "| Metric | Baseline CEO | Raw GRPO CEO ablation | GRPO + Governed CEO |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in rows:
        lines.append(
            f"| {label} | {metric_label(metric('baseline', key))} | "
            f"{metric_label(metric('raw_grpo', key))} | {metric_label(metric('governed_grpo', key))} |"
        )
    lines.extend(
        [
            f"| Main failure mode | {metric('baseline', 'main_failure_mode')} | "
            f"{metric('raw_grpo', 'main_failure_mode')} | {metric('governed_grpo', 'main_failure_mode')} |",
            "",
            "## Interpretation",
            "",
            "Raw GRPO is included as an ablation. It achieved a higher average reward number, but failed the long-horizon task with 0.000 survival and bankruptcy-heavy endings. The governed GRPO CEO recovered survival to 0.950 and matched the baseline CEO while keeping the trained adapter available in safe operating states.",
            "",
            "## Artifacts",
            "",
            "- `comparison_summary.json`",
            "- `loss_curve.png`",
            "- `reward_curve.png`",
            "- `reward_comparison.png`",
            "- `policy_comparison.png`",
            "- `policy_summary.png`",
            "",
        ]
    )
    (output_dir.parent / "comparison_report.md").write_text("\n".join(lines))


def main():
    output_dir = Path("docs/assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_points = list(
        enumerate(
            [
                -0.01865,
                -0.05499,
                -0.007192,
                -0.01457,
                -0.0537,
                -0.0519,
                -0.04173,
                -0.004797,
                -0.03055,
                -0.02548,
                -0.07764,
                -0.03239,
                -0.03825,
                -0.06323,
                -0.01778,
                -0.06385,
                -0.03933,
                -0.04885,
                -0.005103,
                -0.05227,
                -0.02987,
            ],
            start=1,
        )
    )
    plot_line(
        output_dir / "loss_curve.png",
        "GRPO LOGGED LOSS",
        "LOG INDEX",
        "LOSS",
        loss_points,
        y_min=-0.09,
        y_max=0.01,
    )
    reward_points = list(
        enumerate(
            [
                -0.05464,
                0.05913,
                0.1205,
                0.259,
                0.1678,
                0.2651,
                0.1069,
                0.2306,
                0.3428,
                0.1926,
                0.2253,
                0.3479,
                0.219,
                0.3115,
                0.2347,
                0.3444,
                0.4937,
                0.3684,
                0.4169,
                0.4157,
                0.5277,
            ],
            start=1,
        )
    )
    plot_line(
        output_dir / "reward_curve.png",
        "GRPO LOGGED REWARD",
        "LOG INDEX",
        "REWARD",
        reward_points,
        y_min=-0.1,
        y_max=0.6,
    )
    plot_bars(output_dir / "reward_comparison.png")
    plot_reward_curve(output_dir / "policy_comparison.png")
    plot_policy_summary(output_dir / "policy_summary.png")
    write_comparison_files(output_dir)
    print(f"Wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
