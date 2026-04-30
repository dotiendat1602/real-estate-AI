from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a genuinely harder single-turn dataset from the medium tuned set.")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to evaluation folder",
    )
    parser.add_argument(
        "--input-dataset",
        default="single_turn_goldens_medium_tuned.json",
        help="Source dataset filename under evaluation/datasets",
    )
    parser.add_argument(
        "--output-dataset",
        default="single_turn_goldens_medium_genuinely_hard.json",
        help="Output dataset filename under evaluation/datasets",
    )
    parser.add_argument(
        "--smoke-output-dataset",
        default="single_turn_goldens_medium_genuinely_hard_smoke.json",
        help="Smoke output dataset filename under evaluation/datasets",
    )
    parser.add_argument(
        "--smoke-cases",
        type=int,
        default=8,
        help="Number of generated smoke cases",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_context_blob(blob: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in (blob or "").split(","):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _strip_title_noise(text: str) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if not compact:
        return ""
    compact = compact.replace('"', "")
    return compact


def _sparse_title(text: str, max_words: int = 10) -> str:
    words = _strip_title_noise(text).split()
    if not words:
        return ""
    return " ".join(words[:max_words])


def _pick_real_estate_hint(meta: dict[str, str]) -> str:
    title = _strip_title_noise(meta.get("title", ""))
    highlights = _strip_title_noise(meta.get("highlights", "")).replace("|", ", ")
    address = _strip_title_noise(meta.get("address", ""))
    blob = " ".join(part for part in (title, highlights, address) if part).lower()

    if "oto" in blob:
        return "nha trong ngo co loi vao o to"
    if "view song" in blob:
        return "can ho 2PN view song khu trung tam"
    if "lotte mall" in blob:
        return "lo dat doi dien Lotte Mall gan Ho Tay"
    if "mini" in blob or "bep rieng" in blob:
        return "nha thiet ke van hanh can ho mini, moi phong co bep rieng"
    if "dai hoc" in blob or "bach mai" in blob:
        return "nha phu hop nguoi hoc tap lam viec gan truong va benh vien lon"
    if "aeon mall" in blob:
        return "nha gan truong hoc va Aeon Mall, hop sinh hoat gia dinh"
    if "xanhpon" in blob or "nguyen thai hoc" in blob:
        return "nha nguyen can gan Xanhpon va Nguyen Thai Hoc"
    if "vinhomes" in blob:
        return f"can ho trong khu { _sparse_title(title, max_words=6) }".strip()
    if title:
        return _sparse_title(title, max_words=10).lower()
    if address:
        return _sparse_title(address, max_words=10).lower()
    return "bat dong san co mo ta gan giong"


def _pick_real_estate_persona(meta: dict[str, str]) -> str:
    blob = " ".join(meta.values()).lower()
    bedrooms = _safe_int(meta.get("bedrooms"))
    if "oto" in blob:
        return "gia dinh co o to"
    if "mini" in blob or (bedrooms is not None and bedrooms >= 6):
        return "nguoi muon van hanh cho thue nhieu phong"
    if "view song" in blob or "ho boi" in blob or "gym" in blob:
        return "nguoi uu tien can ho trung tam co tien ich"
    if "lotte mall" in blob or "ho tay" in blob:
        return "nguoi mua can tai san giu gia tri vi tri"
    if "2 years" in blob:
        return "nhom o lau dai"
    if "dai hoc" in blob or "benh vien" in blob:
        return "nguoi hoc tap hoac lam viec khu trung tam"
    return "nguoi dang can chon dung tai san theo nhu cau"


def _real_estate_target_facts(meta: dict[str, str]) -> list[str]:
    facts: list[str] = []
    price = meta.get("price")
    area = meta.get("area")
    bedrooms = meta.get("bedrooms")
    bathrooms = meta.get("bathrooms")
    furnishing = meta.get("furnishing")
    title = meta.get("title")
    highlights = meta.get("highlights", "").replace("|", ", ")

    if title:
        facts.append(f"Tai san dung la: {_sparse_title(title, max_words=14)}")
    if price or area:
        parts = []
        if price:
            parts.append(f"gia {price}")
        if area:
            parts.append(f"dien tich {area}")
        facts.append(", ".join(parts))
    if bedrooms or bathrooms:
        parts = []
        if bedrooms and bedrooms != "N/A":
            parts.append(f"{bedrooms} phong ngu")
        if bathrooms and bathrooms != "N/A":
            parts.append(f"{bathrooms} phong ve sinh")
        if parts:
            facts.append(", ".join(parts))
    if furnishing and furnishing != "N/A":
        facts.append(f"Noi that: {furnishing}")
    if highlights:
        facts.append(f"Dau hieu nhan dien dung: {highlights}")
    return facts[:4]


def _real_estate_distractor_fact(meta: dict[str, str]) -> str:
    title = _sparse_title(meta.get("title", ""), max_words=12)
    highlights = meta.get("highlights", "").replace("|", ", ")
    if highlights:
        return f"Chi tiet khong duoc gan nham tu phuong an khac: {title} co dac diem {highlights}"
    price = meta.get("price")
    area = meta.get("area")
    return f"Chi tiet khong duoc gan nham tu phuong an khac: {title} co gia {price} va dien tich {area}".strip()


def _build_real_estate_case(index: int, item: dict[str, Any], distractor: dict[str, Any]) -> dict[str, Any]:
    target_meta = _parse_context_blob((item.get("context") or [""])[0])
    distractor_meta = _parse_context_blob((distractor.get("context") or [""])[0])
    pattern = index % 3

    target_hint = _pick_real_estate_hint(target_meta)
    distractor_hint = _pick_real_estate_hint(distractor_meta)
    persona = _pick_real_estate_persona(target_meta)

    if pattern == 0:
        input_text = (
            f"Khach dang nho mo ho ve hai lua chon gan giong nhau: mot phuong an la {target_hint}, "
            f"phuong an de bi nham sang la {distractor_hint}. "
            f"Hay chon dung bat dong san phu hop nhat cho {persona}, neu gia, dien tich, so phong ngu/ve sinh neu co, "
            f"va chi ro 1 dau hieu cho thay phuong an con lai khong phai dap an."
        )
        hard_pattern = "candidate_disambiguation"
    elif pattern == 1:
        input_text = (
            f"Khach dang can mot tai san cho {persona} va phan van giua {target_hint} voi {distractor_hint}. "
            f"Hay so sanh ngan gon hai phuong an nhung phai ket luan ro nen chon phuong an nao, "
            f"kem gia, dien tich, so phong ngu/ve sinh neu co, va 1 ly do phuong an kia kem phu hop hon."
        )
        hard_pattern = "compare_and_recommend"
    else:
        input_text = (
            f"Khach chi nho rang can tim {target_hint}, nhung lai de lan voi {distractor_hint}. "
            f"Hay nhan dien dung tai san can tim, tom tat thong tin chinh de ra quyet dinh, "
            f"va canh bao 1 chi tiet thuoc lua chon khac de tranh tra loi lech nhu cau."
        )
        hard_pattern = "underspecified_selection"

    expected_output_outline = _real_estate_target_facts(target_meta)
    expected_output_outline.append(_real_estate_distractor_fact(distractor_meta))

    target_post_id = _safe_int(target_meta.get("postId"))
    target_property_id = _safe_int(target_meta.get("propertyId"))
    distractor_post_id = _safe_int(distractor_meta.get("postId"))
    distractor_property_id = _safe_int(distractor_meta.get("propertyId"))

    return {
        "difficulty": "medium_genuinely_hard",
        "question_type": "hard_selection",
        "domain": "real_estate",
        "dataset_variant": "genuine_hard_selection",
        "hard_pattern": hard_pattern,
        "input": input_text,
        "expected_output_outline": expected_output_outline,
        "context": [item["context"][0], distractor["context"][0]],
        "target_metadata": {
            "evalOnly": True,
            "retrievalIntent": "broad_query_candidate_disambiguation",
            "targetPostIds": [value for value in [target_post_id, distractor_post_id] if value is not None],
            "targetPropertyIds": [value for value in [target_property_id, distractor_property_id] if value is not None],
            "preferredPostId": target_post_id,
            "preferredPropertyId": target_property_id,
            "sourceBaseId": item.get("id"),
            "distractorBaseId": distractor.get("id"),
        },
        "source_base_id": item.get("id"),
        "distractor_source_base_id": distractor.get("id"),
    }


def _pick_planning_hint(meta: dict[str, str]) -> str:
    blob = " ".join(meta.values()).lower()
    if "districtduties" in {k.lower() for k in meta.keys()}:
        return "phan nhiem vu trien khai sau khi ke hoach duoc phe duyet"
    if "agricultural2025" in {k.lower() for k in meta.keys()}:
        return "chi tieu bien dong dat nong nghiep giua hien trang va nam 2025"
    if "nonagricultural2025" in {k.lower() for k in meta.keys()}:
        return "so lieu dat phi nong nghiep nam 2025 va muc bien dong"
    if "gpmb2024" in {k.lower() for k in meta.keys()} or "notices" in {k.lower() for k in meta.keys()}:
        return "chi so giai phong mat bang va boi thuong nam 2024"
    if "article67project1" in {k.lower() for k in meta.keys()} or "projectname" in {k.lower() for k in meta.keys()}:
        return "thong tin chi tiet cua mot du an cu the trong ke hoach"
    if "transferredprojects2024to2025" in {k.lower() for k in meta.keys()} or "transferredprojectscount" in {k.lower() for k in meta.keys()}:
        return "nhom du an chuyen tiep sang ke hoach nam 2025"
    source = meta.get("source", "")
    return f"chi tiet trong tai lieu {source}".strip()


def _planning_target_facts(item: dict[str, Any]) -> list[str]:
    facts = [line.strip() for line in item.get("expected_output_outline") or [] if str(line).strip()]
    return facts[:4]


def _planning_distractor_fact(distractor: dict[str, Any]) -> str:
    lines = [line.strip() for line in distractor.get("expected_output_outline") or [] if str(line).strip()]
    if lines:
        return f"Chi tiet de nham tu tai lieu khac, khong duoc gan vao dap an: {lines[0]}"
    return f"Chi tiet de nham tu tai lieu khac, khong duoc gan vao dap an: {distractor.get('input')}"


def _build_planning_case(index: int, item: dict[str, Any], distractor: dict[str, Any]) -> dict[str, Any]:
    target_meta = _parse_context_blob((item.get("context") or [""])[0])
    distractor_meta = _parse_context_blob((distractor.get("context") or [""])[0])
    pattern = index % 3

    target_hint = _pick_planning_hint(target_meta)
    distractor_hint = _pick_planning_hint(distractor_meta)

    if pattern == 0:
        input_text = (
            f"Trong cac tai lieu ke hoach su dung dat, nguoi hoi dang can dung tai lieu noi ve {target_hint}, "
            f"nhung rat de lan voi tai lieu noi ve {distractor_hint}. "
            f"Hay chon dung truong hop, neu 2-3 y chinh can thiet, va chi ro 1 chi tiet thuoc tai lieu kia de tranh gan nham."
        )
        hard_pattern = "planning_document_disambiguation"
    elif pattern == 1:
        input_text = (
            f"Nguoi dung can mot cau tra loi ngan gon nhung dung trong boi canh quy hoach: "
            f"tap trung vao {target_hint}, khong bi troi sang {distractor_hint}. "
            f"Hay neu ro so lieu hoac nhiem vu cot loi, dong thoi nhac 1 dau hieu cho thay tai lieu con lai khong phai dap an."
        )
        hard_pattern = "planning_focus_control"
    else:
        input_text = (
            f"Hay nhan dien dung tai lieu quy hoach phu hop voi yeu cau {target_hint}. "
            f"Nguoi hoi de nham voi tai lieu khac noi ve {distractor_hint}, "
            f"vi vay can tra loi dung y trong tam va canh bao ngan 1 chi tiet khong nen suy dien sang."
        )
        hard_pattern = "planning_underspecified_selection"

    target_metadata = dict(item.get("target_metadata") or {})
    planning_doc_id = target_metadata.get("planningDocumentId")
    if planning_doc_id is not None:
        target_metadata["planningDocumentId"] = planning_doc_id
    target_metadata["evalOnly"] = True
    target_metadata["retrievalIntent"] = "planning_document_disambiguation"
    target_metadata["sourceBaseId"] = item.get("id")
    target_metadata["distractorBaseId"] = distractor.get("id")
    distractor_doc_id = (distractor.get("target_metadata") or {}).get("planningDocumentId")
    if distractor_doc_id is not None:
        target_metadata["distractorPlanningDocumentId"] = distractor_doc_id

    expected_output_outline = _planning_target_facts(item)
    expected_output_outline.append(_planning_distractor_fact(distractor))

    return {
        "difficulty": "medium_genuinely_hard",
        "question_type": "hard_selection",
        "domain": "land_use_planning",
        "dataset_variant": "genuine_hard_selection",
        "hard_pattern": hard_pattern,
        "input": input_text,
        "expected_output_outline": expected_output_outline,
        "context": [item["context"][0], distractor["context"][0]],
        "target_metadata": target_metadata,
        "source_base_id": item.get("id"),
        "distractor_source_base_id": distractor.get("id"),
    }


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return int(str(value).strip())
    except Exception:
        return None


def _assign_ids(items: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        cloned = dict(item)
        cloned["id"] = f"{prefix}_{index:03d}"
        assigned.append(cloned)
    return assigned


def _build_smoke_cases(cases: list[dict[str, Any]], smoke_cases: int) -> list[dict[str, Any]]:
    real_patterns = [
        "candidate_disambiguation",
        "compare_and_recommend",
        "underspecified_selection",
    ]
    planning_patterns = [
        "planning_document_disambiguation",
        "planning_focus_control",
        "planning_underspecified_selection",
    ]
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    target_real = max(1, smoke_cases // 2)
    target_planning = max(1, smoke_cases - target_real)

    def pick_by_patterns(domain: str, patterns: list[str], limit: int) -> None:
        if limit <= 0:
            return
        for pattern in patterns:
            for item in cases:
                if item.get("domain") != domain or item.get("hard_pattern") != pattern:
                    continue
                source_id = str(item.get("source_base_id") or item.get("id"))
                if source_id in seen_ids:
                    continue
                selected.append(item)
                seen_ids.add(source_id)
                break
            if len([entry for entry in selected if entry.get("domain") == domain]) >= limit:
                return

        for item in cases:
            if item.get("domain") != domain:
                continue
            source_id = str(item.get("source_base_id") or item.get("id"))
            if source_id in seen_ids:
                continue
            selected.append(item)
            seen_ids.add(source_id)
            if len([entry for entry in selected if entry.get("domain") == domain]) >= limit:
                return

    pick_by_patterns("real_estate", real_patterns, target_real)
    pick_by_patterns("land_use_planning", planning_patterns, target_planning)

    if len(selected) < smoke_cases:
        for item in cases:
            source_id = str(item.get("source_base_id") or item.get("id"))
            if source_id in seen_ids:
                continue
            selected.append(item)
            seen_ids.add(source_id)
            if len(selected) >= smoke_cases:
                break
    return selected[:smoke_cases]


def main() -> None:
    args = parse_args()
    eval_root = Path(args.root)
    datasets_dir = eval_root / "datasets"

    base_cases = _load_json(datasets_dir / args.input_dataset)
    real_estate = [item for item in base_cases if item.get("domain") == "real_estate"]
    planning = [item for item in base_cases if item.get("domain") == "land_use_planning"]

    generated: list[dict[str, Any]] = []

    for index, item in enumerate(real_estate):
        distractor = real_estate[(index + 1) % len(real_estate)]
        generated.append(_build_real_estate_case(index, item, distractor))

    for index, item in enumerate(planning):
        distractor = planning[(index + 1) % len(planning)]
        generated.append(_build_planning_case(index, item, distractor))

    generated = _assign_ids(generated, "ST_MEDIUM_GENUINE_HARD")
    smoke = _assign_ids(_build_smoke_cases(generated, max(1, int(args.smoke_cases))), "ST_MEDIUM_GENUINE_HARD_SMOKE")

    _write_json(datasets_dir / args.output_dataset, generated)
    _write_json(datasets_dir / args.smoke_output_dataset, smoke)

    print(
        f"Generated {len(generated)} cases -> {datasets_dir / args.output_dataset}\n"
        f"Generated {len(smoke)} smoke cases -> {datasets_dir / args.smoke_output_dataset}"
    )


if __name__ == "__main__":
    main()
