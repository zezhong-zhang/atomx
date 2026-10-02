"""FAC postprocessing threshold controls with unchanged archived RR strengths."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import bench_systematic_xray as b
from fac_runner import parse_levels

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "systematic_xray_2026_10_02"
OLD = HERE.parent / "atomic_consistency_2026_10_02"
refs = json.loads((REF / "references.json").read_text())


def binary_energies(path):
    """Read native FAC 1.1.5 little-endian EN records without text rounding."""
    content = path.read_bytes()
    header = struct.unpack_from("<qiiiif4si", content)
    assert header[1:5] == (1, 1, 5, 1)
    offset, energies = 36, {}
    for _ in range(header[-1]):
        _position, length, _nele, count = struct.unpack_from("<qqii", content, offset)
        offset += 24
        assert length == count * 228
        for _ in range(count):
            _p, _j, ilev, ibase, energy = struct.unpack_from("<hhiid", content, offset)
            assert ibase == -1
            energies[ilev] = energy * 27.211386018  # FAC consts.h
            offset += 228
    assert offset == len(content)
    return energies


def run(el):
    archive = json.loads((OLD / f"{el}-gu.json").read_text())
    photons = refs[el]["photons_eV"]
    shells = b.CASES[el][1]
    ni = b.csf_count(shells)
    out = {
        "element": el,
        "control_only": True,
        "unchanged_rr_strengths": True,
        "shells": {},
    }
    # An empty valid AI database lets AdjustEnergy complete its EN/AI API.
    # Copy just the serialized 36-byte header, setting NBlocks=0.
    fmt = "<qiiiif4si"
    source = REF / "raw" / el / "fac-1s-hole/ai.b"
    header = list(struct.unpack(fmt, source.read_bytes()[: struct.calcsize(fmt)]))
    assert header[4] == 5 and header[-1] > 0
    header[-1] = 0
    empty = HERE / f"{el}-empty-ai.bin"
    empty.write_bytes(struct.pack(fmt, *header))
    for shell in ("1s", "2s", "2p"):
        raw = REF / "raw" / el / f"fac-pi-{shell}"
        levels = parse_levels(raw / "levels.txt")
        exact = binary_energies(raw / "levels.b")
        groups = [levels[:ni], levels[ni:]]
        weights = [sum(r["TWOJ"] + 1 for r in rows) for rows in groups]
        centroid = [
            sum(exact[r["ILEV"]] * (r["TWOJ"] + 1) for r in rows) / w
            for rows, w in zip(groups, weights, strict=True)
        ]
        directory = HERE / "raw" / el / f"threshold-{shell}"
        lines = b.fac_header(el)
        for variant in ("resolved", "centroid", "native"):
            en = raw / "levels.b"
            if variant != "resolved":
                target = [centroid[0], centroid[1]]
                if variant == "native":
                    # Archive assigns the same relaxed threshold to both jj holes.
                    thresholds = [
                        v
                        for k, v in archive["thresholds_eV"]["relaxed"].items()
                        if k.startswith(shell)
                    ]
                    assert thresholds and np.ptp(thresholds) < 1e-12
                    target[1] = target[0] + thresholds[0]
                shifts = [
                    (r["ILEV"], target[g] - exact[r["ILEV"]])
                    for g, rows in enumerate(groups)
                    for r in rows
                ]
                # AdjustEnergy treats any exact zero as an absolute-energy flag.
                assert all(shift != 0.0 for _, shift in shifts)
                table = HERE / f"{el}-{shell}-{variant}-shifts.txt"
                table.write_text("".join(f"{i} {e:.16e}\n" for i, e in shifts))
                lines.append(
                    f"# shift-input-sha256 {hashlib.sha256(table.read_bytes()).hexdigest()}"
                )
                en = directory / f"levels-{variant}.b"
                lines.append(
                    f"AdjustEnergy('{table}','{raw / 'levels.b'}','{en}','{empty}','empty-{variant}.b')"
                )
                lines.append(f"PrintTable('{en}','levels-{variant}.txt',1)")
            lines.extend(["FreeMemENTable()", f"MemENTable('{en}')"])
            for i in range(ni):
                lines.append(
                    f"TotalPICross('{raw / 'pi.b'}','{variant}-{i}.txt',{i},{photons!r})"
                )
        b.fac_job(directory, lines)
        variants = {}
        for variant in ("resolved", "centroid", "native"):
            total = np.zeros(len(photons))
            for i in range(ni):
                values = np.loadtxt(directory / f"{variant}-{i}.txt", ndmin=2)
                assert np.allclose(values[:, 0], photons) and np.all(
                    np.isfinite(values)
                )
                total += values[:, 1] * (levels[i]["TWOJ"] + 1) / weights[0] * 0.01
            assert np.all(total > 0)
            variants[variant] = total.tolist()
            if variant != "resolved":
                shifted = parse_levels(directory / f"levels-{variant}.txt")
                assert len(shifted) == len(levels)
                assert np.ptp([r["ENERGY_eV"] for r in shifted[:ni]]) < 1e-7
                assert np.ptp([r["ENERGY_eV"] for r in shifted[ni:]]) < 1e-5
        np.testing.assert_allclose(
            variants["resolved"], refs[el]["fac_pi"][shell], rtol=1e-10
        )
        out["shells"][shell] = {
            "variants_Mb": variants,
            "fac_centroid_eV": centroid[1] - centroid[0],
            "initial_level_count": ni,
            "final_level_count": len(groups[1]),
        }
    (HERE / f"{el}-fac-threshold.json").write_text(json.dumps(out, indent=2) + "\n")
    print(el, "threshold control complete", flush=True)


if __name__ == "__main__":
    for element in sys.argv[1:] or ("Fe",):
        run(element)
