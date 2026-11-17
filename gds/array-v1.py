"""Square grid of uECoG test arrays inside the square inscribed in a round Si wafer. Lengths in um.
Layer 1 = metal (fabricated), layer 10 = annotations (not fabricated).
Writes two GDS files: <name>_annotated.gds (metal + annotations) and <name>_clean.gds (metal only).
Usage: python array-v1.py [--save NAME]
  no --save overwrites the GDS files next to this script
  --save NAME writes yamla and GDS files into folder NAME
"""

import argparse
import os
import numpy as np
import gdspy

METAL, ANNOTATIONS = 1, 10  # layer levels
IN = 25400.0

# ---------- parameters ----------
WAFER_DIAM_IN = 4  # wafer diameter in inches
WAFER = WAFER_DIAM_IN * IN  # wafer diameter
SQUARE = WAFER / np.sqrt(2)  # edge of the inscribed square = usable area for the array grid
GRID_ROWS, GRID_COLS = 4, 4  # array grid arrangement (rows x columns)
GRID_GAP = 200.0  # gap between neighbouring arrays and to the inscribed square's edge
_N = max(GRID_ROWS, GRID_COLS)
ARRAY = (SQUARE - (_N + 1) * GRID_GAP) / _N  # array edge (derived): square space divided into the grid
N_ARRAYS = GRID_ROWS * GRID_COLS
N_ROWS = 2  # electrode rows per array
N_ELEC = 12  # per array; half small / half large in every row
D_SMALL = 35.0  # small electrode diameter
D_LARGE = 500.0  # large electrode diameter
ELEC_PITCH = 2000.0  # same-row spacing; rows are staggered by ELEC_PITCH / N_ROWS
ROW_SPACE = 1000.0  # row-to-row spacing
ELEC_Y0 = 3000.0  # bottom row height above array bottom edge
TRACE_W_SMALL = 10.0  # trace width to small electrodes (keep thin: low parasitic C)
TRACE_W_LARGE = 50.0  # trace width to large electrodes (keep wide: low series R)
FAN_Y0 = 6000.0  # height of the lowest fan-out jog
FAN_STEP = 200.0  # height difference between neighbouring jogs
MARGIN = 500.0  # pads/array edge clearance
PAD_GAP = 600.0  # gap between neighbouring pads
PAD_H = 4000.0  # pad height
MIN_SPACE = 10.0  # minimum metal-metal gap
OUTLINE_W = 100.0  # annotation outline line width


def make_array(lib, annotate):
    cell = lib.new_cell("ARRAY")
    if annotate:
        cell.add(gdspy.Rectangle((0, 0), (ARRAY, ARRAY), layer=ANNOTATIONS))

    per_row = N_ELEC // N_ROWS
    span = (per_row - 1) * ELEC_PITCH + (N_ROWS - 1) * ELEC_PITCH / N_ROWS
    x0 = (ARRAY - span) / 2
    # (x, y, diameter)
    elecs = sorted(
        (
            x0 + k * ELEC_PITCH + r * ELEC_PITCH / N_ROWS,
            ELEC_Y0 + r * ROW_SPACE,
            D_SMALL if k % 2 == 0 else D_LARGE,
        )
        for r in range(N_ROWS)
        for k in range(per_row)
    )

    pitch = (ARRAY - 2 * MARGIN) / N_ELEC
    pad_w = pitch - PAD_GAP
    pad_y1 = ARRAY - MARGIN
    pad_y0 = pad_y1 - PAD_H
    assert (
        ELEC_PITCH / N_ROWS - D_LARGE / 2 - TRACE_W_LARGE / 2 >= MIN_SPACE
    ), "trace too close to electrode"
    assert FAN_STEP - TRACE_W_LARGE >= MIN_SPACE, "fan-out jogs too close"

    lengths = []
    for i, (x, y, d) in enumerate(
        elecs
    ):  # i-th electrode from the left -> i-th pad from the left
        px = MARGIN + (i + 0.5) * pitch
        h = FAN_Y0 + FAN_STEP * min(
            i, N_ELEC - 1 - i
        )  # outer traces jog lowest -> no crossings
        w = TRACE_W_SMALL if d == D_SMALL else TRACE_W_LARGE
        pts = [(x, y), (x, h), (px, h), (px, pad_y0 + 1)]
        cell.add(gdspy.Round((x, y), d / 2, layer=METAL))
        cell.add(gdspy.FlexPath(pts, w, layer=METAL))
        cell.add(
            gdspy.Rectangle(
                (px - pad_w / 2, pad_y0), (px + pad_w / 2, pad_y1), layer=METAL
            )
        )
        lengths.append(
            (d, w, sum(np.hypot(*np.subtract(a, b)) for a, b in zip(pts, pts[1:])))
        )
    if annotate:
        print(f"pad {pad_w:.0f} x {PAD_H:.0f} um")
        for d, w, L in sorted(
            set(
                (d, w, round(max(l for dd, _, l in lengths if dd == d)))
                for d, w, _ in lengths
            )
        ):
            print(
                f"D={d:.0f} um: w={w:.0f} um, longest trace {L / 1e3:.1f} mm = {L / w:.0f} squares"
            )
    return cell


def build(annotate, path):
    gdspy.current_library = (
        gdspy.GdsLibrary()
    )  # fresh global library so cell names don't collide
    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
    arr = make_array(lib, annotate)
    top = lib.new_cell("WAFER")
    sq0 = (WAFER - SQUARE) / 2  # inscribed square corner (wafer centred at WAFER/2)
    if annotate:
        c = WAFER / 2
        top.add(  # wafer outline
            gdspy.Round(
                (c, c), WAFER / 2, inner_radius=WAFER / 2 - OUTLINE_W,
                number_of_points=720, layer=ANNOTATIONS,
            )
        )
        top.add(  # inscribed square outline
            gdspy.boolean(
                gdspy.Rectangle((sq0, sq0), (sq0 + SQUARE, sq0 + SQUARE)),
                gdspy.Rectangle(
                    (sq0 + OUTLINE_W, sq0 + OUTLINE_W),
                    (sq0 + SQUARE - OUTLINE_W, sq0 + SQUARE - OUTLINE_W),
                ),
                "not",
                layer=ANNOTATIONS,
            )
        )

    assert ARRAY > 0, "arrays do not fit in inscribed square"
    # grid is centred in the square if it is not square itself
    gx = sq0 + GRID_GAP + (_N - GRID_COLS) * (ARRAY + GRID_GAP) / 2
    gy = sq0 + GRID_GAP + (_N - GRID_ROWS) * (ARRAY + GRID_GAP) / 2
    for j in range(N_ARRAYS):
        r, c = divmod(j, GRID_COLS)
        o = (gx + c * (ARRAY + GRID_GAP), gy + r * (ARRAY + GRID_GAP))
        top.add(gdspy.CellReference(arr, o))
        if annotate:
            top.add(
                gdspy.Text(
                    f"A{j + 1}", 500, (o[0] + MARGIN, o[1] + 500), layer=ANNOTATIONS
                )
            )

    lib.write_gds(path)
    print("wrote", path)


def params():
    """Design parameters (the UPPERCASE constants in the parameters section)."""
    names = ["WAFER_DIAM_IN", "SQUARE", "ARRAY", "GRID_ROWS", "GRID_COLS", "GRID_GAP", "N_ROWS", "N_ELEC", "D_SMALL", "D_LARGE",
             "ELEC_PITCH", "ROW_SPACE", "ELEC_Y0", "TRACE_W_SMALL", "TRACE_W_LARGE",
             "FAN_Y0", "FAN_STEP", "MARGIN", "PAD_GAP", "PAD_H", "MIN_SPACE"]
    return {k: globals()[k] for k in names}


def write_yaml(path):
    with open(path, "w") as f:
        f.write("# lengths in um\n")
        for k, v in params().items():
            f.write(f"{k.lower()}: {v}\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="NAME",
                    help="save params.yaml and both GDS files into folder NAME")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    stem = os.path.splitext(os.path.basename(__file__))[0]
    if args.save:
        out = os.path.join(here, args.save)
        os.makedirs(out, exist_ok=True)
        write_yaml(os.path.join(out, "params.yaml"))
    else:
        out = here
    build(True, os.path.join(out, stem + "_annotated.gds"))
    build(False, os.path.join(out, stem + "_clean.gds"))


if __name__ == "__main__":
    main()
