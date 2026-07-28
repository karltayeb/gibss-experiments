"""Pull a supercollection's results back from the cluster (Midway), selecting exactly the
relevant batches so you never drag unrelated experiments or the 78 GB of raw fits along.

Selectors (union; at least one required):
  -e/--experiments      experiment NUMBERS (5, 05, 011) -> all supercollections of that experiment
  -s/--supercollections  SC names, exact/prefix or fnmatch glob ('005-*loc')
  -b/--batches          batch hashes, exact or prefix (grab one cell/batch directly)

Data kinds (-d, repeatable; default reductions):
  reductions  by_batch/<bh>/fits/<mh>/reductions/*.parquet  (+ manifest_cache.json)  [~0.16 GB/exp]
  plots       supercollections/<sc>/**                        (analysis PDFs + .done)
  sims        by_batch/<bh>/{simulations,sample_metadata}.parquet                     [~6 GB]
  fits        by_batch/<bh>/fits/<mh>/fits.parquet             (the raw fits)          [~78 GB]
  all         reductions + plots + sims + fits

Batches are resolved from the LOCAL config + content-addressed hashes (identical to the
cluster), so `-e/-s` -> the SC's cells -> (batch_hash, method_hash) set; `fits` is filtered
to the selected SCs' method hashes. A remote `find` over just those batch dirs gives the exact
file list + sizes; `--dry-run` prints that and the rsync command without transferring.

Run:  uv run --project .. python scripts/sync_simulations.py -e 011 -d reductions [-n]
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from experiments import loader as L  # noqa: E402

DEFAULT_HOST = "midway3"
DEFAULT_REMOTE_ROOT = "/project/mstephens/ktayeb/gibss-experiments/twogroup_experiments"
DATA_KINDS = ["reductions", "plots", "sims", "fits", "all"]
_GLOB_CHARS = set("*?[")


# ---------------------------------------------------------------------------
# Selector resolution -> matched supercollections + {batch_hash: methods|None}
# (methods=None means "all methods on that batch", used for bare -b selection).
# ---------------------------------------------------------------------------
def _leading_number(name: str) -> int | None:
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else None


def resolve(experiments, supercollections, batches):
    config = L.load_config()
    lib = config["library"]
    scs = config["supercollections"]
    default_nb = int(lib["defaults"]["n_batches"])

    matched_sc: set[str] = set()
    for sel in experiments or []:
        if not sel.isdigit():
            raise SystemExit(f"-e/--experiments takes numbers (got {sel!r}); use -s for names/globs")
        n = int(sel)
        hits = {name for name in scs if _leading_number(name) == n}
        if not hits:
            raise SystemExit(f"no supercollection for experiment {sel!r}")
        matched_sc |= hits
    for sel in supercollections or []:
        if _GLOB_CHARS & set(sel):
            hits = {name for name in scs if fnmatch.fnmatch(name, sel)}
        else:
            hits = {name for name in scs if name == sel or name.startswith(sel)}
        if not hits:
            raise SystemExit(f"no supercollection matches {sel!r}")
        matched_sc |= hits

    batch_methods: dict[str, set[str] | None] = {}
    for name in sorted(matched_sc):
        sc = scs[name]
        nb = int(sc["n_batches"]) if sc.get("n_batches") is not None else default_nb
        mhs = {L.method_hash(c) for c in L.resolve_methods_for_sc(lib, sc).values()}
        for coll in L.supercollection_collections(lib, name, sc):
            for member in coll["coordinates"]:
                sh = L.sim_hash(member["coordinate"])
                for i in range(nb):
                    bh = L._batch_hash(sh, i)
                    if batch_methods.get(bh) is None and bh in batch_methods:
                        continue  # already all-methods (from -b); keep it broad
                    batch_methods.setdefault(bh, set())
                    if batch_methods[bh] is not None:
                        batch_methods[bh] |= mhs

    if batches:
        from manifest_cache import load_manifest_cached
        all_bh = list(load_manifest_cached()["batches"])
        for sel in batches:
            hits = [h for h in all_bh if h.startswith(sel)]
            if not hits:
                raise SystemExit(f"no batch hash starts with {sel!r}")
            for h in hits:
                batch_methods[h] = None  # bare batch: all methods

    return sorted(matched_sc), batch_methods


# ---------------------------------------------------------------------------
# Remote listing (one ssh `find`) -> exact relative paths + sizes for the kinds.
# ---------------------------------------------------------------------------
def _remote_find(host, remote_root, rel_dirs, extra_globs):
    """List files under the given dirs (relative to remote_root) with sizes.
    Returns list of (relpath, size_bytes). rel_dirs that don't exist are skipped."""
    if not rel_dirs:
        return []
    # -print0-ish via tab; skip missing dirs with 2>/dev/null per-dir is awkward, so filter args
    args = " ".join(f"'{d}'" for d in rel_dirs)
    cmd = (f"cd {remote_root} && for d in {args}; do [ -e \"$d\" ] && "
           f"find \"$d\" -type f -printf '%p\\t%s\\n'; done")
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                         capture_output=True, text=True)
    rows = []
    for line in out.stdout.splitlines():
        if "\t" not in line:
            continue
        p, s = line.rsplit("\t", 1)
        rows.append((p, int(s)))
    return rows


def select_paths(host, remote_root, matched_sc, batch_methods, kinds):
    """Return (paths, total_bytes). paths are relative to remote_root."""
    want_red = "reductions" in kinds
    want_fits = "fits" in kinds
    want_sims = "sims" in kinds
    want_plots = "plots" in kinds

    chosen: dict[str, int] = {}

    # batch-scoped kinds: one find over the selected batch dirs, filter in python
    if (want_red or want_fits or want_sims) and batch_methods:
        bdirs = [f"results/by_batch/{bh}" for bh in batch_methods]
        for p, size in _remote_find(host, remote_root, bdirs, None):
            parts = p.split("/")  # results by_batch <bh> ...
            if len(parts) < 3:
                continue
            bh = parts[2]
            methods = batch_methods.get(bh)
            leaf = parts[-1]
            is_reduction = "reductions" in parts and leaf.endswith(".parquet")
            is_fit = leaf == "fits.parquet"
            is_sim = leaf in ("simulations.parquet", "sample_metadata.parquet")
            # method filter (fits + reductions live under fits/<mh>/...)
            mh = parts[4] if len(parts) > 4 and parts[3] == "fits" else None
            method_ok = methods is None or mh is None or mh in methods
            if want_red and is_reduction and method_ok:
                chosen[p] = size
            elif want_fits and is_fit and method_ok:
                chosen[p] = size
            elif want_sims and is_sim:
                chosen[p] = size

    if want_red:  # manifest always needed to resolve hash->name locally
        for p, size in _remote_find(host, remote_root, ["results/manifest_cache.json"], None):
            chosen[p] = size

    if want_plots:
        if not matched_sc:
            print("  (note) -d plots needs -e/-s (batches have no supercollection); skipping plots",
                  file=sys.stderr)
        pdirs = [f"results/supercollections/{sc}" for sc in matched_sc]
        for p, size in _remote_find(host, remote_root, pdirs, None):
            chosen[p] = size

    return sorted(chosen), sum(chosen.values())


def _rsync_progress_flags():
    """`--info=progress2` (clean overall bar) needs rsync >= 3.1.0; older rsync (e.g. macOS's
    bundled 2.6.9) only has per-file `--progress`."""
    try:
        out = subprocess.run(["rsync", "--version"], capture_output=True, text=True).stdout
        m = re.search(r"version (\d+)\.(\d+)(?:\.(\d+))?", out)
        if m and (int(m.group(1)), int(m.group(2))) >= (3, 1):
            return ["--info=progress2"]
    except Exception:
        pass
    return ["--progress"]


def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="sync-simulations",
        description="Pull a supercollection's results back from Midway (reductions/plots/sims/fits).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("-e", "--experiments", nargs="+", metavar="N",
                    help="experiment numbers (5, 05, 011) -> all SCs of that experiment")
    ap.add_argument("-s", "--supercollections", nargs="+", metavar="NAME",
                    help="SC names, exact/prefix or glob ('005-*loc')")
    ap.add_argument("-b", "--batches", nargs="+", metavar="HASH",
                    help="batch hashes, exact or prefix")
    ap.add_argument("-d", "--data", nargs="+", choices=DATA_KINDS, default=["reductions"],
                    help="what to pull (default: reductions)")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    ap.add_argument("--local-root", default=str(_HERE),
                    help="local twogroup_experiments dir (default: this repo)")
    ap.add_argument("--delete", action="store_true", help="pass rsync --delete")
    ap.add_argument("-f", "--force", action="store_true",
                    help="skip the confirmation prompt (non-interactive transfers need this)")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="resolve + list files + size, transfer nothing")
    ap.add_argument("--list", action="store_true", help="print supercollection names and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for name in sorted(L.load_config()["supercollections"]):
            print(name)
        return 0
    if not (args.experiments or args.supercollections or args.batches):
        ap.error("give at least one of -e / -s / -b")

    kinds = set(["reductions", "plots", "sims", "fits"] if "all" in args.data else args.data)
    matched_sc, batch_methods = resolve(args.experiments, args.supercollections, args.batches)

    print(f"matched supercollections ({len(matched_sc)}): {', '.join(matched_sc) or '(none)'}")
    print(f"batches: {len(batch_methods)}  |  data: {', '.join(sorted(kinds))}")
    print("resolving remote file list ...", flush=True)
    paths, total = select_paths(args.host, args.remote_root, matched_sc, batch_methods, kinds)
    print(f"-> {len(paths)} files, {_human(total)}")
    if not paths:
        print("nothing to sync.")
        return 0

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(paths) + "\n")
        files_from = fh.name

    rsync = ["rsync", "-a", *_rsync_progress_flags(), "--files-from", files_from,
             *(["--delete"] if args.delete else []),
             f"{args.host}:{args.remote_root}/", f"{args.local_root}/"]
    if args.dry_run:
        print("\nDRY RUN — would run:\n  " + " ".join(rsync))
        if args.verbose:
            print("\nfiles:\n" + "\n".join("  " + p for p in paths[:50])
                  + (f"\n  ... (+{len(paths)-50} more)" if len(paths) > 50 else ""))
        return 0

    # Confirm before every transfer (bypass with -f/--force).
    if not args.force:
        action = "pull + DELETE local extras" if args.delete else "pull"
        if not sys.stdin.isatty():
            print(f"refusing to {action} {_human(total)} without -f/--force (non-interactive).",
                  file=sys.stderr)
            return 1
        if input(f"About to {action} {_human(total)} ({len(paths)} files). Proceed? [y/N] ") \
                .strip().lower() not in ("y", "yes"):
            print("aborted.")
            return 1

    print("\n" + " ".join(rsync), flush=True)
    return subprocess.call(rsync)


if __name__ == "__main__":
    raise SystemExit(main())
