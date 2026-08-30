from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_csv(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        data = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                data[name].append(float(row[name]))
    if not data["time_s"]:
        raise RuntimeError(f"CSV contains no samples: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Q250 body-rate step CSV")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it in the Isaac Lab Python environment.") from exc

    log_path = args.log.resolve()
    data = load_csv(log_path)
    output_dir = (args.output_dir or log_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = log_path.stem

    t = data["time_s"]

    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(t, data["cmd_p_deg_s"], label="p cmd")
    ax1.plot(t, data["p_deg_s"], label="p measured")
    ax1.plot(t, data["cmd_q_deg_s"], label="q cmd")
    ax1.plot(t, data["q_deg_s"], label="q measured")
    ax1.plot(t, data["cmd_r_deg_s"], label="r cmd")
    ax1.plot(t, data["r_deg_s"], label="r measured")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Body rate [deg/s]")
    ax1.set_title("Q250 body-rate response")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    rate_png = output_dir / f"{stem}_rates.png"
    fig1.savefig(rate_png, dpi=160)

    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    for motor in range(1, 5):
        ax2.plot(t, data[f"m{motor}_rpm"], label=f"M{motor} actual")
        ax2.plot(t, data[f"m{motor}_cmd_rpm"], linestyle="--", label=f"M{motor} cmd")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Motor speed [rpm]")
    ax2.set_title("Q250 motor command and first-order response")
    ax2.grid(True)
    ax2.legend(ncol=2)
    fig2.tight_layout()
    motor_png = output_dir / f"{stem}_motors.png"
    fig2.savefig(motor_png, dpi=160)

    print(f"rate plot  : {rate_png}")
    print(f"motor plot : {motor_png}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
