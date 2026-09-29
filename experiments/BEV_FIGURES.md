# BEV Bonsai Figure Exports

Use `bev_map_context_v9_schematic_figure.py` for the paper-style bonsai BEV
figures. It follows the preferred v3-style circular train/test camera schematic
with fixed, uniformly spaced representative camera glyphs. The glyph positions
are a clean schematic designed for large paper-readable cameras: test glyphs are
inside the marked test sector and train glyphs are outside it, while the
displayed train/test counts still come from the real
split files. The context background is a top-down rasterization of the COLMAP
SfM point cloud, replacing the earlier Ours RGB/Gaussian render. The test sector
is marked with a light orange outer boundary/band rather than a central arrow.
Camera glyphs use the same visual language as `pic1.pdf`: a solid camera body,
a trapezoid lens, and a translucent triangular frustum. Their colors also follow
`pic1.pdf`: train cameras are red and test cameras are blue.
Earlier
`bev_*context*_figure.py` scripts are prototypes kept for provenance and
debugging; they are visually cluttered or use less explicit ordering.

## Recommended Command

Use the clean version with no embedded context legend. The camera legend is
exported as a standalone panel.

```bash
conda run -n gsprior python experiments/bev_map_context_v9_schematic_figure.py
```

## Main Outputs

- `global_all/BEV_overlay_bonsai_mapcontext_v9_schematic.png`
- `global_all/BEV_overlay_bonsai_mapcontext_v9_schematic.pdf`

## Separate Ordered Exports

The separate panels are written to:

```text
/data1/zhangao/outputs/viz_multiangle/SEPARATE/BEV/bonsai_mapcontext_v9_schematic/
```

The exported context panels use ordered names:

```text
00_camera_legend.png
00_camera_legend.pdf
00_context_row.png
01_angle020_train_test_context.png
02_angle040_train_test_context.png
03_angle060_train_test_context.png
04_angle080_train_test_context.png
05_angle100_train_test_context.png
```

Body panels are under `panels_all/` with `rowXX` and `colXX` prefixes so file
browsers preserve the figure order. `README.txt` in the export directory records
the angle order, method order, split sources, camera source, crop extent, and
all body-panel source files.
