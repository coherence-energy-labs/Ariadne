"""External labelled-corpus adapters for discovery inference benchmarks.

The offline benchmark suite is intentionally small and deterministic. This
module is the bridge to actual survey/orbit corpora: MPCORB known-object
catalog rows, ZTF-style alert exports, and Rubin/LSST alert packets or JSON
exports. Each adapter returns `LabelledCase` rows that can be passed directly
to `run_inference_benchmark`.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

from .brokers.base import Alert
from .benchmarking import LabelledCase
from .inference import Evidence
from .taxonomy import classify_orbit


MPCORB_URL = "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
ZTF_IRSA_DOCS_URL = "https://irsa.ipac.caltech.edu/docs/program_interface/ztf_api.html"
ZTF_ALERT_ARCHIVE_URL = "https://ztf.uw.edu/"
RUBIN_ALERT_DOCS_URL = "https://prompt-products.lsst.io/products/alerts/index.html"
RUBIN_ALERT_SCHEMA_URL = "https://sdm-schemas.lsst.io/"


@dataclass(frozen=True)
class ExternalCorpusManifest:
    """Sources used to construct an external benchmark."""
    mpcorb_url: str = MPCORB_URL
    ztf_docs_url: str = ZTF_IRSA_DOCS_URL
    ztf_alert_archive_url: str = ZTF_ALERT_ARCHIVE_URL
    rubin_alert_docs_url: str = RUBIN_ALERT_DOCS_URL
    rubin_schema_url: str = RUBIN_ALERT_SCHEMA_URL


@dataclass(frozen=True)
class CorpusBuildRecord:
    """One reproducible import step used to assemble a real-data corpus."""
    source: str
    path_or_url: str
    n_records: int = 0
    n_cases: int = 0
    n_alerts: int = 0
    status: str = "ok"
    error: str = ""


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_labelled_cases_jsonl(cases: Iterable[LabelledCase],
                               path: str | Path) -> int:
    """Write benchmark cases as portable JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(_jsonable(case), sort_keys=True) + "\n")
            n += 1
    return n


def read_labelled_cases_jsonl(path: str | Path) -> list[LabelledCase]:
    """Read JSONL produced by `write_labelled_cases_jsonl`."""
    out: list[LabelledCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        evidence = Evidence(**rec["evidence"])
        out.append(LabelledCase(
            case_id=rec["case_id"],
            evidence=evidence,
            truth_label=rec["truth_label"],
            source=rec["source"],
            split=rec.get("split", "validation"),
            adversarial=bool(rec.get("adversarial", False)),
            metadata=rec.get("metadata", {}),
        ))
    return out


def write_alerts_jsonl(alerts: Iterable[Alert], path: str | Path) -> int:
    """Write operational alerts as portable JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for alert in alerts:
            f.write(json.dumps(_jsonable(alert), sort_keys=True) + "\n")
            n += 1
    return n


def read_alerts_jsonl(path: str | Path) -> list[Alert]:
    """Read JSONL produced by `write_alerts_jsonl`."""
    alerts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            alerts.append(Alert(**json.loads(line)))
    return alerts


def records_to_alerts(records: Iterable[dict], *, survey: str,
                      id_fields: tuple[str, ...] = ("candid", "diaSourceId", "alertId", "id"),
                      obj_fields: tuple[str, ...] = ("objectId", "objectid", "oid", "ssObjectId")) -> list[Alert]:
    """Convert generic alert exports into Ariadne operational `Alert` rows."""
    alerts = []
    for idx, rec in enumerate(records):
        src = rec.get("diaSource") or rec.get("dia_source") or rec
        def pick(fields, default=""):
            for key in fields:
                value = src.get(key, rec.get(key))
                if value not in (None, ""):
                    return str(value)
            return default
        mjd = _float(src.get("mjd"), _mjd_from_jd(src.get("jd")))
        ra = _float(src.get("ra"))
        dec = _float(src.get("dec"))
        if mjd is None or ra is None or dec is None:
            continue
        band = src.get("band") or src.get("filterName") or src.get("filter") or _band_from_ztf_fid(src.get("fid"))
        alerts.append(Alert(
            survey=survey,
            alert_id=pick(id_fields, f"{survey}_{idx}"),
            obj_id=pick(obj_fields, pick(id_fields, f"{survey}_{idx}")),
            mjd=float(mjd),
            ra=float(ra),
            dec=float(dec),
            mag=_float(src.get("magpsf"), _float(src.get("psFluxMag"), _float(src.get("mag"), -99.0))),
            band=str(band or ""),
            meta={k: _jsonable(v) for k, v in rec.items()
                  if k not in {"diaSource", "dia_source"}},
        ))
    return alerts


def alerts_from_ztf_file(path: str | Path) -> list[Alert]:
    return records_to_alerts(_load_records(path), survey="ZTF")


def alerts_from_rubin_file(path: str | Path) -> list[Alert]:
    return records_to_alerts(_load_records(path), survey="LSST")


def _float(text, default=None):
    try:
        if text is None:
            return default
        s = str(text).strip()
        if not s or s.lower() in {"nan", "null", "none"}:
            return default
        return float(s)
    except Exception:
        return default


def _int(text, default=0):
    try:
        if text is None:
            return default
        s = str(text).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def classify_orbit_label(*, a_au: float | None, e: float | None,
                         i_deg: float | None = None) -> str:
    """Map orbital elements to Ariadne's inference labels."""
    if a_au is None or e is None or not math.isfinite(a_au) or not math.isfinite(e):
        return "moving_object"
    tax = classify_orbit(a_au, e, i_deg or 0.0)
    return tax.label if tax.label != "UNCLASSIFIED" else "moving_object"


def evidence_from_orbit(*, a_au: float, e: float, i_deg: float = 0.0,
                        H: float | None = None, rms_arcsec: float | None = None,
                        n_obs: int = 6) -> Evidence:
    """Build inference evidence from a known orbit row.

    This is not an ephemeris. It is a benchmark abstraction that presents the
    inference engine with the observable hints implied by a labelled orbit.
    """
    distance = max(a_au, 0.2)
    if H is None:
        apparent_mag = None
    else:
        delta = max(distance - 1.0, 0.2)
        apparent_mag = H + 5.0 * math.log10(distance * delta)
    rate = 30.0 / math.sqrt(max(distance, 0.2))
    if distance >= 30.0:
        rate = min(rate, 3.0)
    return Evidence(
        rate_arcsec_hr=rate,
        apparent_mag=apparent_mag,
        band="r",
        morphology_label="POINT",
        morphology_confidence=0.9,
        n_detections=max(4, min(n_obs, 12)),
        arc_days=10.0 if n_obs >= 6 else 2.0,
        rms_arcsec=rms_arcsec if rms_arcsec is not None else 1.0,
        skybot_match_names=[],
        sky_context={"source": "MPCORB", "a_au": a_au, "e": e, "i_deg": i_deg},
    )


def parse_mpcorb_line(line: str) -> dict | None:
    """Parse one fixed-width MPCORB row into orbital fields."""
    if len(line) < 103:
        return None
    packed = line[0:7].strip()
    H = _float(line[8:13])
    e = _float(line[70:79])
    a_au = _float(line[92:103])
    if not packed or e is None or a_au is None:
        return None
    i_deg = _float(line[59:68], 0.0)
    rms = _float(line[137:141])
    n_obs = _int(line[117:122], 6)
    readable = line[166:194].strip() if len(line) >= 194 else ""
    return {
        "packed_designation": packed,
        "designation": readable or packed,
        "H": H,
        "a_au": a_au,
        "e": e,
        "i_deg": i_deg,
        "rms_arcsec": rms,
        "n_obs": n_obs,
    }


def labelled_cases_from_mpcorb_lines(lines: Iterable[str],
                                     *,
                                     limit: int | None = None,
                                     source: str = "mpc_mpcorb_live") -> list[LabelledCase]:
    """Convert MPCORB fixed-width rows into labelled inference cases."""
    cases: list[LabelledCase] = []
    for line in lines:
        row = parse_mpcorb_line(line.rstrip("\n"))
        if row is None:
            continue
        truth = classify_orbit_label(
            a_au=row["a_au"], e=row["e"], i_deg=row.get("i_deg"))
        ev = evidence_from_orbit(
            a_au=row["a_au"], e=row["e"], i_deg=row.get("i_deg") or 0.0,
            H=row.get("H"), rms_arcsec=row.get("rms_arcsec"),
            n_obs=row.get("n_obs") or 6,
        )
        cases.append(LabelledCase(
            case_id=f"mpcorb_{row['packed_designation']}",
            evidence=ev,
            truth_label=truth,
            source=source,
            metadata=row,
        ))
        if limit is not None and len(cases) >= limit:
            break
    return cases


def fetch_mpcorb_cases(*, limit: int = 200, url: str = MPCORB_URL,
                       timeout: float = 30.0) -> list[LabelledCase]:
    """Fetch a streaming sample from the live MPCORB catalog."""
    req = Request(url, headers={"User-Agent": "Ariadne external benchmark importer"})
    with urlopen(req, timeout=timeout) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            lines = (b.decode("latin-1", errors="replace") for b in gz)
            return labelled_cases_from_mpcorb_lines(lines, limit=limit)


def _load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".json", ".avro"}:
        if suffix == ".avro":
            try:
                import fastavro  # type: ignore
            except Exception as e:
                raise RuntimeError("fastavro is required to read Avro alert packets") from e
            with path.open("rb") as f:
                return list(fastavro.reader(f))
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("alerts"), list):
            return data["alerts"]
        return [data]
    if suffix in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"unsupported corpus file type: {path}")


def _band_from_ztf_fid(fid) -> str:
    return {1: "g", 2: "r", 3: "i"}.get(_int(fid, -1), "")


def _mjd_from_jd(jd) -> float | None:
    v = _float(jd)
    return None if v is None else v - 2400000.5


def _motion_rate_from_records(records: list[dict]) -> float | None:
    if len(records) < 2:
        return None
    first, last = records[0], records[-1]
    mjd0 = _float(first.get("mjd")) or _mjd_from_jd(first.get("jd"))
    mjd1 = _float(last.get("mjd")) or _mjd_from_jd(last.get("jd"))
    ra0 = _float(first.get("ra"))
    ra1 = _float(last.get("ra"))
    dec0 = _float(first.get("dec"))
    dec1 = _float(last.get("dec"))
    if None in {mjd0, mjd1, ra0, ra1, dec0, dec1} or mjd0 == mjd1:
        return None
    cosd = math.cos(math.radians((dec0 + dec1) / 2.0))
    sep_deg = math.hypot((ra1 - ra0) * cosd, dec1 - dec0)
    return abs(sep_deg * 3600.0 / ((mjd1 - mjd0) * 24.0))


def _group_by(records: Iterable[dict], key_options: tuple[str, ...]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for idx, rec in enumerate(records):
        key = None
        for opt in key_options:
            if rec.get(opt) not in (None, ""):
                key = str(rec[opt])
                break
        if key is None:
            key = f"row_{idx}"
        groups.setdefault(key, []).append(rec)
    return groups


def ztf_records_to_labelled_cases(records: Iterable[dict],
                                  *,
                                  mpc_labels: dict[str, str] | None = None,
                                  require_truth: bool = True) -> list[LabelledCase]:
    """Convert ZTF alert/export records into labelled benchmark cases.

    Records with an explicit `truth_label` column are accepted directly. Known
    solar-system matches (`ssnamenr`) can be labelled through `mpc_labels`.
    Low-realbogus records can be used if they explicitly carry an artifact label.
    """
    mpc_labels = mpc_labels or {}
    cases = []
    for obj_id, group in _group_by(records, ("objectId", "objectid", "oid", "candid")).items():
        group = sorted(group, key=lambda r: _float(r.get("mjd")) or _mjd_from_jd(r.get("jd")) or 0.0)
        last = group[-1]
        truth = last.get("truth_label") or last.get("label")
        ss_name = str(last.get("ssnamenr") or last.get("ssName") or "").strip()
        if not truth and ss_name and ss_name not in {"null", "None", "-999"}:
            truth = mpc_labels.get(ss_name) or mpc_labels.get(ss_name.replace(" ", ""))
        if not truth and not require_truth:
            truth = "moving_object" if ss_name else "artefact"
        if not truth:
            continue
        mjds = [
            _float(r.get("mjd")) or _mjd_from_jd(r.get("jd"))
            for r in group
            if (_float(r.get("mjd")) or _mjd_from_jd(r.get("jd"))) is not None
        ]
        rb = _float(last.get("drb"), _float(last.get("rb"), 1.0))
        elong = _float(last.get("elong"), _float(last.get("aimage")))
        morph = "STREAK" if elong is not None and elong > 2.5 else "POINT"
        if rb is not None and rb < 0.2 and truth in {"artefact", "subtraction_residual"}:
            morph = "BLEND"
        ev = Evidence(
            mjd=mjds[-1] if mjds else None,
            ra_deg=_float(last.get("ra")),
            dec_deg=_float(last.get("dec")),
            rate_arcsec_hr=_motion_rate_from_records(group),
            apparent_mag=_float(last.get("magpsf"), _float(last.get("mag"))),
            band=str(last.get("filter") or _band_from_ztf_fid(last.get("fid"))),
            morphology_label=morph,
            morphology_confidence=max(0.05, min(0.99, rb or 0.7)),
            n_detections=len(group),
            arc_days=(max(mjds) - min(mjds)) if len(mjds) >= 2 else 0.0,
            skybot_match_names=[ss_name] if ss_name else [],
            sky_context={"survey": "ZTF", "objectId": obj_id, "realbogus": rb},
        )
        cases.append(LabelledCase(
            case_id=f"ztf_{obj_id}",
            evidence=ev,
            truth_label=str(truth),
            source="ztf_external_alerts",
            adversarial=str(truth) in {"artefact", "subtraction_residual", "satellite_trail"},
            metadata={"objectId": obj_id, "ssnamenr": ss_name, "n_records": len(group)},
        ))
    return cases


def labelled_cases_from_ztf_file(path: str | Path, **kwargs) -> list[LabelledCase]:
    return ztf_records_to_labelled_cases(_load_records(path), **kwargs)


def rubin_records_to_labelled_cases(records: Iterable[dict],
                                   *,
                                   require_truth: bool = True) -> list[LabelledCase]:
    """Convert Rubin/LSST alert JSON or Avro records into LabelledCase rows."""
    cases = []
    for idx, rec in enumerate(records):
        src = rec.get("diaSource") or rec.get("dia_source") or rec
        obj = rec.get("ssObject") or rec.get("ss_object") or {}
        orbit = rec.get("mpcOrbit") or rec.get("mpc_orbit") or rec.get("orbit") or {}
        truth = rec.get("truth_label") or src.get("truth_label")
        if not truth and orbit:
            truth = classify_orbit_label(
                a_au=_float(orbit.get("a") or orbit.get("a_au") or orbit.get("ssObject_a")),
                e=_float(orbit.get("e") or orbit.get("ssObject_e")),
                i_deg=_float(orbit.get("i") or orbit.get("incl") or orbit.get("i_deg")),
            )
        if not truth and not require_truth:
            truth = "moving_object" if obj or orbit else "artefact"
        if not truth:
            continue
        mid = src.get("diaSourceId") or rec.get("alertId") or idx
        band = src.get("band") or src.get("filterName") or src.get("filter")
        trail = _float(src.get("trailLength"), 0.0)
        ev = Evidence(
            mjd=_float(src.get("midPointTai"), _float(src.get("mjd"))),
            ra_deg=_float(src.get("ra")),
            dec_deg=_float(src.get("dec")),
            rate_arcsec_hr=_float(src.get("skyVelocity"), _float(src.get("rate_arcsec_hr"))),
            apparent_mag=_float(src.get("psFluxMag"), _float(src.get("mag"))),
            band=str(band or ""),
            morphology_label="STREAK" if trail and trail > 1.0 else "POINT",
            morphology_confidence=0.85,
            n_detections=_int(src.get("nDiaSources"), _int(rec.get("n_detections"), 1)),
            arc_days=_float(src.get("arc_days"), 0.0) or 0.0,
            skybot_match_names=[str(obj.get("ssObjectId"))] if obj.get("ssObjectId") else [],
            sky_context={"survey": "LSST", "diaSourceId": mid},
        )
        cases.append(LabelledCase(
            case_id=f"rubin_{mid}",
            evidence=ev,
            truth_label=str(truth),
            source="rubin_lsst_external_alerts",
            metadata={"diaSourceId": mid},
        ))
    return cases


def labelled_cases_from_rubin_file(path: str | Path, **kwargs) -> list[LabelledCase]:
    return rubin_records_to_labelled_cases(_load_records(path), **kwargs)
