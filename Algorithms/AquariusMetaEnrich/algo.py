#!/usr/bin/env python3
"""
Pontus-X / Ocean Compute-to-Data algorithm (AgrospAI)
Enrich Aquarius DCAT JSON-LD with dcat:theme using AgroPortal Annotator.

Key Pontus-X behaviors:
- Reads dataset DID from env var DIDS (JSON array of DIDs) when available.
- Fallback: inspects /data/inputs/* directory names for did:... folders.
- Reads optional parameters from algoCustomData.json (if present).
- Writes outputs to /data/outputs/.

Outputs:
- /data/outputs/enriched_dcat.json
- /data/outputs/debug_env.json (helpful for first pilot runs)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

import requests

INPUTS_DIR = Path("/data/inputs")
OUTPUTS_DIR = Path("/data/outputs")

DEFAULT_AQUARIUS_BASE = "https://dcat.agrospai.udl.cat"
DEFAULT_ANNOTATOR_BASE = "https://data.agroportal.lirmm.fr"
DEFAULT_LABEL_BASE = "https://data.agroportal.lirmm.fr"
DEFAULT_ONTOLOGIES = "AGROVOC"
DEFAULT_LANG = "en"

# Two-slice policy for long descriptions (single annotator request)
DEFAULT_HEAD_CHARS = 15000
DEFAULT_TAIL_CHARS = 5000

HTTP_TIMEOUT_GET = 45
HTTP_TIMEOUT_POST = 180
LABEL_FETCH_TIMEOUT = 45
LABEL_FETCH_MAX = 500  # hard cap on distinct class label lookups (per run)


@dataclass
class Params:
    aquarius_base: str = DEFAULT_AQUARIUS_BASE
    annotator_base: str = DEFAULT_ANNOTATOR_BASE
    label_base: str = DEFAULT_LABEL_BASE
    api_key: str = ""
    ontologies: str = DEFAULT_ONTOLOGIES
    lang: str = DEFAULT_LANG
    head_chars: int = DEFAULT_HEAD_CHARS
    tail_chars: int = DEFAULT_TAIL_CHARS
    debug: bool = True


# ------------------------
# Pontus-X / CtD helpers
# ------------------------

def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def get_dataset_did() -> str:
    """
    Primary: env var DIDS (JSON array of DID strings), as per Ocean CtD convention.
    Fallback: infer from folder names in /data/inputs (often /data/inputs/<did>/...).
    """
    dids_raw = os.getenv("DIDS", "").strip()
    if dids_raw:
        parsed = _safe_json_loads(dids_raw)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], str) and parsed[0].startswith("did:"):
            return parsed[0]
        # Sometimes DIDS can be a comma-separated string (defensive)
        if "," in dids_raw:
            parts = [p.strip() for p in dids_raw.split(",") if p.strip()]
            for p in parts:
                if p.startswith("did:"):
                    return p
        if dids_raw.startswith("did:"):
            return dids_raw

    # Fallback: scan /data/inputs subfolders
    if INPUTS_DIR.exists():
        for child in INPUTS_DIR.iterdir():
            if child.is_dir() and child.name.startswith("did:"):
                return child.name

    raise RuntimeError("Could not determine dataset DID (no DIDS env var and no did:* folder under /data/inputs).")


def find_algo_custom_data() -> Optional[Path]:
    """
    Pontus-X commonly provides algoCustomData.json somewhere under /data/inputs.
    We'll search a few typical places.
    """
    candidates = [
        INPUTS_DIR / "algoCustomData.json",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    if INPUTS_DIR.exists():
        # Search shallowly (avoid huge walks)
        for p in INPUTS_DIR.rglob("algoCustomData.json"):
            if p.is_file():
                return p
    return None


def load_params_from_custom_data(p: Params) -> Params:
    """
    Merge algorithm parameters from algoCustomData.json if present.
    Expected shape (flexible):
      {
        "aquarius_base": "...",
        "annotator_base": "...",
        "label_base": "...",
        "agroportal_api_key": "...",
        "ontologies": "FOODON",
        "lang": "en",
        "head_chars": 15000,
        "tail_chars": 5000,
        "debug": true
      }
    """
    custom_path = find_algo_custom_data()
    if not custom_path:
        return p

    try:
        data = json.loads(custom_path.read_text(encoding="utf-8"))
    except Exception:
        return p

    if not isinstance(data, dict):
        return p

    def pick(*keys: str) -> Any:
        for k in keys:
            if k in data:
                return data[k]
        return None

    aquarius_base = pick("aquarius_base", "AQUARIUS_BASE")
    annotator_base = pick("annotator_base", "AGROPORTAL_ANNOTATOR_BASE")
    label_base = pick("label_base", "AGROPORTAL_LABEL_BASE")
    api_key = pick("agroportal_api_key", "api_key", "AGROPORTAL_API_KEY")
    ontologies = pick("ontologies", "ontology")
    lang = pick("lang", "language")
    head_chars = pick("head_chars", "head")
    tail_chars = pick("tail_chars", "tail")
    debug = pick("debug")

    if isinstance(aquarius_base, str) and aquarius_base.strip():
        p.aquarius_base = aquarius_base.strip()
    if isinstance(annotator_base, str) and annotator_base.strip():
        p.annotator_base = annotator_base.strip()
    if isinstance(label_base, str) and label_base.strip():
        p.label_base = label_base.strip()
    if isinstance(api_key, str) and api_key.strip():
        p.api_key = api_key.strip()
    if isinstance(ontologies, str) and ontologies.strip():
        p.ontologies = ontologies.strip()
    if isinstance(lang, str) and lang.strip():
        p.lang = lang.strip()
    if isinstance(head_chars, int) and head_chars > 0:
        p.head_chars = head_chars
    if isinstance(tail_chars, int) and tail_chars >= 0:
        p.tail_chars = tail_chars
    if isinstance(debug, bool):
        p.debug = debug

    return p


# ------------------------
# DCAT / text helpers
# ------------------------

def first_value(v: Any) -> Any:
    return v[0] if isinstance(v, list) and v else v


def normalize_label(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.isupper():
        return s.capitalize()
    return s


def build_text(title: str, keywords: List[str], desc: str, head: int, tail: int) -> str:
    title = (title or "").strip()
    desc = desc or ""
    kw = ", ".join([k for k in (keywords or []) if isinstance(k, str) and k.strip()])

    parts: List[str] = []
    if title:
        parts.append(title)
    if kw:
        parts.append(f"Keywords: {kw}")

    prefix_txt = "\n\n".join(parts).strip()
    if not desc.strip():
        return prefix_txt

    if tail <= 0 or len(desc) <= head + tail:
        desc_part = desc[: max(head, 0)]
    else:
        desc_part = desc[:head].rstrip() + "\n\n[...] \n\n" + desc[-tail:].lstrip()

    return (prefix_txt + "\n\n" + desc_part) if prefix_txt else desc_part


def ensure_context_has_skos(dcat: Dict[str, Any]) -> None:
    ctx = dcat.get("@context")
    if isinstance(ctx, dict):
        ctx.setdefault("skos", "http://www.w3.org/2004/02/skos/core#")
    elif ctx is None:
        dcat["@context"] = {"skos": "http://www.w3.org/2004/02/skos/core#"}


def merge_themes(dcat: Dict[str, Any], new_themes: List[Dict[str, Any]]) -> Tuple[int, int]:
    existing = dcat.get("dcat:theme")
    if not isinstance(existing, list):
        existing = []

    seen: Set[str] = set()
    merged: List[Any] = []

    for t in existing:
        if isinstance(t, dict) and isinstance(t.get("@id"), str):
            seen.add(t["@id"])
        merged.append(t)

    added = 0
    for t in new_themes:
        tid = t.get("@id") if isinstance(t, dict) else None
        if isinstance(tid, str) and tid not in seen:
            merged.append(t)
            seen.add(tid)
            added += 1

    dcat["dcat:theme"] = merged
    return len(existing), added


# ------------------------
# HTTP calls
# ------------------------

def fetch_aquarius_ddo(aquarius_base: str, did: str) -> Dict[str, Any]:
    url = f"{aquarius_base.rstrip('/')}/api/aquarius/assets/ddo/{did}"
    r = requests.get(url, timeout=HTTP_TIMEOUT_GET)
    r.raise_for_status()
    obj = r.json()
    if not isinstance(obj, dict):
        raise RuntimeError("Aquarius did not return a JSON object.")
    return obj


def call_agroportal_annotator(api_key: str, text: str, annotator_base: str, ontologies: str) -> List[Dict[str, Any]]:
    url = f"{annotator_base.rstrip('/')}/annotator?ontologies={ontologies}"
    headers = {"Authorization": f"apikey token={api_key}"}
    r = requests.post(url, headers=headers, data={"text": text}, timeout=HTTP_TIMEOUT_POST)
    r.raise_for_status()
    out = r.json()
    return out if isinstance(out, list) else []


def rewrite_base(url: str, new_base: str) -> str:
    try:
        u = urlparse(url)
        b = urlparse(new_base)
        return urlunparse((b.scheme or u.scheme, b.netloc, u.path, u.params, u.query, u.fragment))
    except Exception:
        return url


def fetch_class_pref_label_localized(self_url: str, api_key: str, label_base: str, lang: str) -> Optional[str]:
    """
    Fetch class prefLabel using lang=<lang> (confirmed necessary).
    Keeps payload small with include=prefLabel.
    """
    url = rewrite_base(self_url, label_base)

    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}include=prefLabel&display_context=false&display_links=false&lang={lang}"

    headers = {"Authorization": f"apikey token={api_key}", "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=LABEL_FETCH_TIMEOUT)
    r.raise_for_status()
    obj = r.json()
    if not isinstance(obj, dict):
        return None

    pl = obj.get("prefLabel") or obj.get("skos:prefLabel") or obj.get("rdfs:label") or obj.get("label")
    if isinstance(pl, str) and pl.strip():
        return pl.strip()

    # Defensive fallback if deployment returns other shapes
    if isinstance(pl, dict) and isinstance(pl.get("@value"), str):
        return pl["@value"].strip() or None
    if isinstance(pl, list):
        for it in pl:
            if isinstance(it, dict) and it.get("value"):
                return str(it["value"]).strip() or None
    return None


def extract_surface_label(item: Dict[str, Any]) -> Optional[str]:
    anns = item.get("annotations")
    if isinstance(anns, list) and anns:
        a0 = anns[0]
        if isinstance(a0, dict):
            t = a0.get("text")
            if isinstance(t, str) and t.strip():
                return normalize_label(t)
    return None


def annotations_to_themes(
    ann: List[Dict[str, Any]],
    api_key: str,
    label_base: str,
    lang: str,
) -> List[Dict[str, Any]]:
    themes: Dict[str, Dict[str, Any]] = {}
    label_cache: Dict[str, Optional[str]] = {}
    fetched = 0

    for item in ann or []:
        if not isinstance(item, dict):
            continue

        cls = item.get("annotatedClass") or {}
        if not isinstance(cls, dict):
            continue

        uri = cls.get("@id")
        if not isinstance(uri, str) or not uri.startswith("http"):
            continue

        links = cls.get("links")
        self_url = links.get("self") if isinstance(links, dict) else None

        label: Optional[str] = None

        # 1) Localized canonical label via classes endpoint (+lang=...)
        if isinstance(self_url, str) and self_url.startswith("http"):
            if uri in label_cache:
                label = label_cache[uri]
            elif fetched < LABEL_FETCH_MAX:
                try:
                    label = fetch_class_pref_label_localized(self_url, api_key, label_base=label_base, lang=lang)
                except Exception:
                    label = None
                label_cache[uri] = label
                fetched += 1

        # 2) Fallback to surface match text
        if not label:
            label = extract_surface_label(item)

        theme_obj: Dict[str, Any] = {"@id": uri, "@type": "skos:Concept"}
        if label:
            theme_obj["skos:prefLabel"] = {"@language": lang, "@value": label}

        if uri not in themes:
            themes[uri] = theme_obj
        else:
            if "skos:prefLabel" not in themes[uri] and "skos:prefLabel" in theme_obj:
                themes[uri]["skos:prefLabel"] = theme_obj["skos:prefLabel"]

    return list(themes.values())


# ------------------------
# Debug / main
# ------------------------

def write_debug(did: str, params: Params) -> None:
    if not params.debug:
        return
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    debug = {
        "dataset_did": did,
        "env": {
            "DIDS": os.getenv("DIDS"),
            "TRANSFORMATION_DID": os.getenv("TRANSFORMATION_DID"),
            "AQUARIUS_BASE": os.getenv("AQUARIUS_BASE"),
            "AGROPORTAL_API_KEY": "***set***" if os.getenv("AGROPORTAL_API_KEY") else None,
            "AGROPORTAL_ANNOTATOR_BASE": os.getenv("AGROPORTAL_ANNOTATOR_BASE"),
            "AGROPORTAL_LABEL_BASE": os.getenv("AGROPORTAL_LABEL_BASE"),
        },
        "params": {
            "aquarius_base": params.aquarius_base,
            "annotator_base": params.annotator_base,
            "label_base": params.label_base,
            "ontologies": params.ontologies,
            "lang": params.lang,
            "head_chars": params.head_chars,
            "tail_chars": params.tail_chars,
            "debug": params.debug,
        },
        "inputs_dir_listing": [p.name for p in INPUTS_DIR.iterdir()] if INPUTS_DIR.exists() else [],
        "custom_data_path": str(find_algo_custom_data()) if find_algo_custom_data() else None,
    }
    (OUTPUTS_DIR / "debug_env.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")


def main() -> int:
    # Start with defaults + env overrides
    params = Params(
        aquarius_base=os.getenv("AQUARIUS_BASE", DEFAULT_AQUARIUS_BASE),
        annotator_base=os.getenv("AGROPORTAL_ANNOTATOR_BASE", DEFAULT_ANNOTATOR_BASE),
        label_base=os.getenv("AGROPORTAL_LABEL_BASE", DEFAULT_LABEL_BASE),
        api_key=os.getenv("AGROPORTAL_API_KEY", "").strip(),
        ontologies=os.getenv("AGROPORTAL_ONTOLOGIES", DEFAULT_ONTOLOGIES),
        lang=os.getenv("AGROPORTAL_LANG", DEFAULT_LANG),
    )

    # Merge custom data file if present
    params = load_params_from_custom_data(params)

    # Final: require API key
    if not params.api_key:
        print("ERROR: missing AgroPortal API key (set AGROPORTAL_API_KEY env var or in algoCustomData.json).", file=sys.stderr)
        return 2

    # Determine dataset DID
    try:
        did = get_dataset_did()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    write_debug(did, params)

    # Fetch DCAT JSON-LD from Aquarius
    try:
        dcat = fetch_aquarius_ddo(params.aquarius_base, did)
    except Exception as e:
        print(f"ERROR: Aquarius fetch failed: {e}", file=sys.stderr)
        return 3

    title = first_value(dcat.get("dct:title")) or ""
    desc = first_value(dcat.get("dct:description")) or ""
    keywords = dcat.get("dcat:keyword") or []
    if not isinstance(keywords, list):
        keywords = []

    text = build_text(str(title), keywords, str(desc), params.head_chars, params.tail_chars)

    # Annotate
    try:
        ann = call_agroportal_annotator(
            api_key=params.api_key,
            text=text,
            annotator_base=params.annotator_base,
            ontologies=params.ontologies,
        )
    except Exception as e:
        print(f"ERROR: Annotator call failed: {e}", file=sys.stderr)
        return 4

    # Convert to themes + merge
    try:
        new_themes = annotations_to_themes(ann, api_key=params.api_key, label_base=params.label_base, lang=params.lang)
        ensure_context_has_skos(dcat)
        before, added = merge_themes(dcat, new_themes)
    except Exception as e:
        print(f"ERROR: Theme generation failed: {e}", file=sys.stderr)
        return 5

    # Write output
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "enriched_dcat.json"
    out_path.write_text(json.dumps(dcat, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: wrote {out_path}")
    print(f"Themes: {before} existing, {added} added, {len(dcat.get('dcat:theme', []))} total")
    print(f"DID: {did}")
    print(f"Annotator base: {params.annotator_base}")
    print(f"Label base:     {params.label_base}")
    print(f"Ontologies:     {params.ontologies}")
    print(f"Lang:           {params.lang}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
