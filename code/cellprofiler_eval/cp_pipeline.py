"""Build the BBBC021 crop pipeline with CellProfiler's own API, save a .cppipe.

Run in the env where CellProfiler is installed:

    python cellprofiler_eval/cp_pipeline.py --thresholds $D/classic_thresholds.json \\
                          --out $D/classic.cppipe
    python cellprofiler_eval/cp_pipeline.py --validate $D/classic.cppipe

Building through the API takes the installed version's defaults and overrides
only the settings this analysis depends on. Every override is read back, and
all failures are reported together with the module's real setting names
(``--list_settings`` prints those names). Modules: LoadData, Identify{Primary,
Secondary,Tertiary}Objects (or Cellpose masks, or none = whole-crop features),
size/shape, intensity, texture, granularity, colocalization, image quality,
ExportToSpreadsheet. Border objects are kept: the crop's cell often touches
the edge.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Optional

# --------------------------------------------------------------------------- #
# module lookup: CP4 split modules between cellprofiler and cellprofiler_core
# --------------------------------------------------------------------------- #
_PKGS = ("cellprofiler.modules.", "cellprofiler_core.modules.")


def _mod_class(name: str, cls: str):
    last = None
    for pkg in _PKGS:
        try:
            return getattr(importlib.import_module(pkg + name), cls)
        except Exception as e:                                    # noqa: BLE001
            last = e
    raise SystemExit(
        f"cannot import {cls} from {' or '.join(p + name for p in _PKGS)}\n"
        f"  last error: {type(last).__name__}: {last}\n"
        f"Run this inside the conda env where CellProfiler is installed "
        f"(`python -c 'import cellprofiler; print(cellprofiler.__version__)'`).")


def granularity_is_broken() -> Optional[str]:
    """Reason MeasureGranularity would raise at run time, or None.

    CellProfiler 4.2 passes ``selem=`` to ``skimage.morphology.erosion``, which
    scikit-image removed in 0.20.
    """
    try:
        import inspect
        import skimage
        import skimage.morphology as m
        if "selem" in inspect.signature(m.erosion).parameters:
            return None
        return (f"scikit-image {skimage.__version__} removed the `selem` "
                f"argument that CellProfiler's MeasureGranularity still "
                f"passes")
    except Exception as e:                                        # noqa: BLE001
        return f"could not probe scikit-image ({type(e).__name__}: {e})"


class _Failures:
    """Collect every setting that could not be applied, report once, then exit."""

    def __init__(self):
        self.items = []

    def add(self, module, key, value, names):
        self.items.append((type(module).__name__, key, value, names))

    def raise_if_any(self):
        if not self.items:
            return
        msg = ["Some settings could not be applied. CellProfiler renamed them, "
               "or this build differs from the one this script was written "
               "against. Fix the key on the left; the module's real names are "
               "on the right.\n"]
        for mod, key, val, names in self.items:
            msg.append(f"  {mod}.{key} = {val!r}")
            msg.append(f"     attributes: "
                       f"{', '.join(n for n in names['attrs'])[:400]}")
            msg.append(f"     setting texts: "
                       f"{' | '.join(names['texts'])[:600]}")
            if names.get("choices"):
                msg.append(f"     choices: {names['choices']}")
            msg.append("")
        raise SystemExit("\n".join(msg))


FAIL = _Failures()


def _names(module):
    attrs = sorted(a for a in dir(module)
                   if not a.startswith("_") and not callable(getattr(module, a, None)))
    try:
        texts = [str(getattr(s, "text", "")) for s in module.settings()]
    except Exception:                                             # noqa: BLE001
        texts = []
    ch = {}
    try:
        for st in module.settings():
            c = list(getattr(st, "choices", []) or [])
            if c:
                ch[str(getattr(st, "text", ""))[:40]] = c
    except Exception:                                             # noqa: BLE001
        pass
    return {"attrs": attrs, "texts": [t for t in texts if t],
            "choices": str(ch)[:800]}


def _assign(setting, value) -> bool:
    try:
        setting.value = value
    except Exception:                                             # noqa: BLE001
        return False
    return True


def _same(got, want) -> bool:
    # a list setting is written as "a, b, c" and read back as ["a","b","c"]
    if isinstance(want, str) and isinstance(got, (list, tuple)):
        want = [x.strip() for x in want.split(",") if x.strip()]
    if isinstance(want, (list, tuple)) or isinstance(got, (list, tuple)):
        return set(map(str, got or [])) == set(map(str, want or []))
    return str(got) == str(want) or got == want


def _resolve_choice(setting, value):
    """Map a shorthand onto the setting's exact choice string, or None.

    CellProfiler ``Choice`` settings accept any string and only fail at run
    time, so an unresolvable value is a build-time failure here.
    """
    choices = list(getattr(setting, "choices", []) or [])
    if not choices or not isinstance(value, str):
        return value if not choices else (value if value in choices else None)
    if value in choices:
        return value
    v = value.lower()
    hit = [c for c in choices if c.lower() == v]
    hit = hit or [c for c in choices if c.lower().startswith(v)]
    hit = hit or [c for c in choices if v in c.lower()]
    return hit[0] if len(set(hit)) == 1 or len(hit) == 1 else None


def setv(module, key, value, text=None):
    """Set ``module.<key>`` (or the setting whose text contains ``text``).

    List settings take a comma-joined string while the getter returns a list,
    so a list is offered both ways. Choices go through ``_resolve_choice``.
    Every assignment is read back and compared; a mismatch is recorded in FAIL.
    """
    cands = [value]
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        cands.append(", ".join(value))

    targets = []
    s = getattr(module, key, None)
    if s is not None and hasattr(s, "value"):
        targets.append(s)
    if text:
        targets += [st for st in module.settings()
                    if text.lower() in str(getattr(st, "text", "")).lower()]
    for st in targets:
        for v in cands:
            r = _resolve_choice(st, v)
            if r is None:
                continue
            if _assign(st, r) and _same(st.value, r):
                return
    FAIL.add(module, key, value, _names(module))


# --------------------------------------------------------------------------- #
def build(a):
    from cellprofiler_core.pipeline import Pipeline               # noqa: E402
    import cellprofiler_core.preferences as prefs                 # noqa: E402
    prefs.set_headless()

    dna, cyto = a.dna_channel, a.cyto_channel
    chans = [c.strip() for c in a.channels.split(",")]
    if dna not in chans or cyto not in chans:
        raise SystemExit(f"--dna_channel/--cyto_channel must be in --channels "
                         f"{chans}")

    pipe = Pipeline()
    n = 0

    def add(m):
        nonlocal n
        n += 1
        m.module_num = n
        pipe.add_module(m)
        return m

    # 1. LoadData ---------------------------------------------------------- #
    ld = add(_mod_class("loaddata", "LoadData")())
    setv(ld, "csv_directory", f"Elsewhere...|{a.csv_dir}", text="input data file location")
    setv(ld, "csv_file_name", a.csv_name, text="name of the file")
    setv(ld, "wants_images", True, text="load images based on this data")
    setv(ld, "image_directory", "Elsewhere...|/", text="base image location")
    setv(ld, "rescale", True, text="rescale intensities")

    if a.segmentation == "none":
        print("  segmentation: none -- whole-crop features only")
        measure_and_export(a, add, chans, [])
        FAIL.raise_if_any()
        return pipe
    if a.segmentation == "cellpose":
        # Nuclei and Cells arrive from LoadData as objects (cp_cellpose.py).
        print("  segmentation: Cellpose masks via LoadData "
              "(Image_ObjectsFileName_Nuclei / _Cells)")
    else:
        add_classic_segmentation(a, add, dna, cyto)

    # 4. Cytoplasm --------------------------------------------------------- #
    it = add(_mod_class("identifytertiaryobjects", "IdentifyTertiaryObjects")())
    setv(it, "secondary_objects_name", "Cells", text="select the larger")
    setv(it, "primary_objects_name", "Nuclei", text="select the smaller")
    setv(it, "subregion_objects_name", "Cytoplasm", text="name the tertiary")

    objs = ["Nuclei", "Cells", "Cytoplasm"]
    measure_and_export(a, add, chans, objs)
    FAIL.raise_if_any()
    return pipe


def add_classic_segmentation(a, add, dna, cyto):
    """``--segmentation classic``: thresholded nuclei, propagated cells.

    ``a.frozen`` (from ``--thresholds classic_thresholds.json``): one global
    threshold per channel, entered as each module's Manual threshold, with
    ``use_advanced`` on so the saved settings are the ones that run.

    ``--thresholds per_image``: ``use_advanced`` stays off, so CP 4.2 runs
    basic mode (per-image Minimum Cross-Entropy, intensity declumping) and
    ignores the Otsu / Shape settings set below.
    """
    # 2. Nuclei ------------------------------------------------------------ #
    ip = add(_mod_class("identifyprimaryobjects", "IdentifyPrimaryObjects")())
    setv(ip, "x_name", dna, text="select the input image")
    setv(ip, "y_name", "Nuclei", text="name the primary objects")
    setv(ip, "size_range", (a.nucleus_min, a.nucleus_max), text="typical diameter")
    setv(ip, "exclude_size", True, text="discard objects outside the diameter")
    # keep border objects: the crop's cell often touches the edge
    setv(ip, "exclude_border_objects", False, text="discard objects touching")
    if a.frozen:
        setv(ip, "use_advanced", True, text="use advanced settings")
        _manual(ip.threshold, a.frozen["Nuclei"])
    else:
        setv(ip, "unclump_method", "Shape", text="method to distinguish clumped")
        setv(ip, "watershed_method", "Shape", text="method to draw dividing lines")
        if hasattr(ip, "threshold"):
            setv(ip.threshold, "threshold_scope", "Global", text="threshold strategy")
            setv(ip.threshold, "global_operation", "Otsu", text="thresholding method")
            setv(ip.threshold, "two_class_otsu", "Three classes",
                 text="two-class or three")
            setv(ip.threshold, "assign_middle_to_foreground", "Foreground",
                 text="assign pixels in the middle")
            setv(ip.threshold, "threshold_correction_factor",
                 a.threshold_correction, text="threshold correction factor")

    # 3. Cells ------------------------------------------------------------- #
    isec = add(_mod_class("identifysecondaryobjects", "IdentifySecondaryObjects")())
    setv(isec, "x_name", "Nuclei", text="select the input objects")
    setv(isec, "y_name", "Cells", text="name the objects to be identified")
    setv(isec, "image_name", cyto, text="select the input image")
    setv(isec, "method", "Propagation", text="select the method to identify")
    setv(isec, "wants_discard_edge", False, text="discard secondary objects")
    if a.frozen:
        _manual(isec.threshold, a.frozen["Cells"])
    elif hasattr(isec, "threshold"):
        setv(isec.threshold, "threshold_scope", "Global", text="threshold strategy")
        setv(isec.threshold, "global_operation", "Otsu", text="thresholding method")


def _manual(th, value: float) -> None:
    setv(th, "threshold_scope", "Global", text="threshold strategy")
    setv(th, "global_operation", "Manual", text="thresholding method")
    setv(th, "manual_threshold", value, text="manual threshold")


def load_frozen(a) -> None:
    """``--thresholds`` -> ``a.frozen`` ({"Nuclei": t, "Cells": t}) or None.

    The channels the thresholds were computed on must be the ones the modules
    read."""
    a.frozen = None
    if a.segmentation != "classic":
        return
    if a.thresholds is None:
        raise SystemExit("--segmentation classic needs --thresholds: the "
                         "classic_thresholds.json from `cp_classic.py "
                         "thresholds`, or `per_image` for the pre-2026-09-18 "
                         "build")
    if a.thresholds == "per_image":
        print("  ! --thresholds per_image: CellProfiler's basic mode, per-image "
              "Minimum Cross-Entropy (see add_classic_segmentation)")
        return
    j = json.load(open(a.thresholds))
    want = {"Nuclei": a.dna_channel, "Cells": a.cyto_channel}
    if j["channels"] != want:
        raise SystemExit(f"{a.thresholds} was computed on {j['channels']}, but "
                         f"this build reads {want} (--dna_channel / "
                         f"--cyto_channel)")
    a.frozen = {k: float(v) for k, v in j["thresholds"].items()}
    print(f"  segmentation: classic, frozen thresholds {a.frozen} "
          f"from {a.thresholds}")


def measure_and_export(a, add, chans, objs):
    # 5-9. measurements ---------------------------------------------------- #
    if objs:
        ss = add(_mod_class("measureobjectsizeshape", "MeasureObjectSizeShape")())
        setv(ss, "objects_list", objs, text="select object sets")
        setv(ss, "calculate_zernikes", True, text="zernike")

        mi = add(_mod_class("measureobjectintensity", "MeasureObjectIntensity")())
        setv(mi, "images_list", chans, text="select images")
        setv(mi, "objects_list", objs, text="select objects")
    else:
        # whole crop: intensity of the image itself, plus five percentiles
        # so a shift in the tail is not averaged away
        mi = add(_mod_class("measureimageintensity", "MeasureImageIntensity")())
        setv(mi, "images_list", chans, text="select images to measure")
        setv(mi, "wants_objects", False, text="enclosed by objects")
        setv(mi, "wants_percentiles", True, text="custom percentiles")
        setv(mi, "percentiles", "5,25,50,75,95", text="percentiles to measure")

    tx = add(_mod_class("measuretexture", "MeasureTexture")())
    setv(tx, "images_list", chans, text="select images")
    if objs:
        setv(tx, "objects_list", objs, text="select objects")
    # scale_groups is a plain list of SettingsGroup grown by add_scale(), not a
    # settable value; scale_count is a HiddenCount that maintains itself.
    while len(tx.scale_groups) < len(a.texture_scales):
        tx.add_scale()
    for grp, sc in zip(tx.scale_groups, a.texture_scales):
        grp.scale.value = int(sc)
    if [g.scale.value for g in tx.scale_groups] != list(a.texture_scales):
        FAIL.add(tx, "texture_scales", a.texture_scales, _names(tx))
    setv(tx, "images_or_objects", "Both" if objs else "Images",
         text="measure images or objects")

    broken = granularity_is_broken()
    if broken and not a.force_granularity:
        print(f"  ! MeasureGranularity SKIPPED: {broken}.\n"
              f"    It would raise mid-run, after MeasureTexture has already "
              f"spent its time, and take the whole chunk with it.\n"
              f"    To get it back: `conda install 'scikit-image<0.20'` in the "
              f"CellProfiler env, then rebuild. Or --force_granularity to "
              f"build it anyway.\n"
              f"    Cost: the two Image_Granularity_* rows of "
              f"gate_features.yaml drop out. Everything else is unaffected.")
    elif "granularity" not in a.skip:
        gr = add(_mod_class("measuregranularity", "MeasureGranularity")())
        setv(gr, "images_list", chans, text="select images")
        setv(gr, "subsample_size", 1.0, text="subsampling factor for granularity")
        setv(gr, "image_sample_size", 1.0, text="subsampling factor for background")
        setv(gr, "element_size", 5, text="radius of structuring element")
        setv(gr, "granular_spectrum_length", a.granularity_length,
             text="range of the granular spectrum")

    co = add(_mod_class("measurecolocalization", "MeasureColocalization")())
    setv(co, "images_list", chans, text="select images")
    if objs:
        setv(co, "objects_list", objs, text="select objects")
    setv(co, "images_or_objects", "Both" if objs else "Across entire image",
         text="select where to measure")
    setv(co, "thr", 15.0, text="threshold as a percentage")

    iq = add(_mod_class("measureimagequality", "MeasureImageQuality")())
    setv(iq, "images_choice", "All loaded images", text="calculate metrics for which")

    # 10. export ----------------------------------------------------------- #
    ex = add(_mod_class("exporttospreadsheet", "ExportToSpreadsheet")())
    setv(ex, "delimiter", "Comma", text="select the column delimiter")
    setv(ex, "directory", "Default Output Folder|", text="output file location")
    setv(ex, "wants_everything", True, text="export all measurement types")
    setv(ex, "add_metadata", True, text="add image metadata columns")
    setv(ex, "nan_representation", "NaN", text="representation of nan")
    # no MyExpt_ prefix: cp_features.py looks for Image.csv / Nuclei.csv /
    # Cells.csv / Cytoplasm.csv by those exact names (--prefix overrides).
    setv(ex, "wants_prefix", False, text="add a prefix to file names")


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="bbbc021_crops.cppipe")
    p.add_argument("--csv_dir", default="/PATH/TO/DUMP",
                   help="dir holding load_data.csv (the --dump_images dir). "
                        "Overridable at run time with --data-file, so the "
                        "value baked in here only matters for the GUI.")
    p.add_argument("--csv_name", default=None,
                   help="default load_data_cellpose.csv (cellpose) or "
                        "load_data.csv (classic)")
    p.add_argument("--segmentation", default="classic",
                   choices=("classic", "cellpose", "none"),
                   help="classic: IdentifyPrimaryObjects + "
                        "IdentifySecondaryObjects with --thresholds. cellpose: "
                        "Nuclei/Cells are cp_cellpose.py's masks, loaded by "
                        "LoadData. none: no objects, whole-crop features only")
    p.add_argument("--thresholds", default=None,
                   help="classic only: the classic_thresholds.json from "
                        "cp_classic.py thresholds (global thresholds frozen "
                        "from the real control pool), or per_image for "
                        "CellProfiler's basic mode, the build every run before "
                        "2026-09-18 used")
    p.add_argument("--channels", default="DNA,Actin,Tubulin",
                   help="must match cp_io's channel_names exactly")
    p.add_argument("--dna_channel", default="DNA")
    p.add_argument("--cyto_channel", default="Actin",
                   help="channel for IdentifySecondaryObjects. Actin gives the "
                        "cell body; on the other channel-order convention this "
                        "is the tubulin image and the Cells masks change.")
    p.add_argument("--nucleus_min", type=int, default=8)
    p.add_argument("--nucleus_max", type=int, default=40)
    p.add_argument("--threshold_correction", type=float, default=1.0,
                   help="--thresholds per_image only; frozen thresholds take "
                        "cp_classic.py thresholds --correction")
    p.add_argument("--texture_scales", type=int, nargs="+", default=[3, 5])
    p.add_argument("--granularity_length", type=int, default=8)
    p.add_argument("--force_granularity", action="store_true",
                   help="build MeasureGranularity even when the scikit-image "
                        "probe says it will raise at run time")
    p.add_argument("--skip", default="", help="comma-separated module keywords "
                                              "to leave out, e.g. granularity")
    p.add_argument("--list_settings", action="store_true",
                   help="print every module's attribute and setting names and "
                        "exit. Use this first when a build mismatches.")
    p.add_argument("--validate", default=None,
                   help="load an existing .cppipe and print its modules")
    a = p.parse_args()

    if a.validate:
        from cellprofiler_core.pipeline import Pipeline
        import cellprofiler_core.preferences as prefs
        prefs.set_headless()
        pipe = Pipeline()
        pipe.load(a.validate)
        print(f"{a.validate}: {len(pipe.modules())} modules")
        for m in pipe.modules():
            print(f"  {m.module_num:2d}. {m.module_name}")
        return 0

    if a.list_settings:
        for nm, cls in (("loaddata", "LoadData"),
                        ("identifyprimaryobjects", "IdentifyPrimaryObjects"),
                        ("identifysecondaryobjects", "IdentifySecondaryObjects"),
                        ("identifytertiaryobjects", "IdentifyTertiaryObjects"),
                        ("measureobjectsizeshape", "MeasureObjectSizeShape"),
                        ("measureobjectintensity", "MeasureObjectIntensity"),
                        ("measuretexture", "MeasureTexture"),
                        ("measuregranularity", "MeasureGranularity"),
                        ("measurecolocalization", "MeasureColocalization"),
                        ("measureimagequality", "MeasureImageQuality"),
                        ("exporttospreadsheet", "ExportToSpreadsheet")):
            m = _mod_class(nm, cls)()
            n = _names(m)
            print(f"\n== {cls}\n  attrs : {', '.join(n['attrs'])}")
            print(f"  texts : {' | '.join(n['texts'])}")
        return 0

    if a.csv_name is None:
        a.csv_name = ("load_data_cellpose.csv" if a.segmentation == "cellpose"
                      else "load_data.csv")
    load_frozen(a)
    pipe = build(a)
    with open(a.out, "w") as f:
        # CP4 Pipeline.dump; save_image_plane_details=False keeps the file a
        # pipeline rather than a pipeline plus a frozen image list.
        pipe.dump(f, save_image_plane_details=False)
    cfg = {k: v for k, v in vars(a).items() if k not in ("list_settings", "validate")}
    with open(a.out + ".json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"-> {a.out}  ({len(pipe.modules())} modules)")
    for m in pipe.modules():
        print(f"  {m.module_num:2d}. {m.module_name}")
    print(f"-> {a.out}.json  (the settings this was built with)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
