from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

PLANNING_KEYWORDS = (
    "quy hoach",
    "quy hoach su dung dat",
    "ke hoach su dung dat",
    "ke hoach",
    "khsd",
    "khsdd",
    "thong tin quy hoach",
    "loai dat",
    "thua dat",
    "to ban do",
    "muc dich su dung dat",
    "quy hoach do thi",
)

PLANNING_STRUCTURAL_TERMS = (
    "thua dat",
    "to ban do",
    "quy hoach",
    "ke hoach su dung dat",
    "muc dich su dung dat",
    "loai dat",
    "dat o",
    "hanh lang",
    "chi gioi",
    "thu hoi dat",
    "dat nong nghiep",
    "dat phi nong nghiep",
    "dat chua su dung",
    "dua dat chua su dung vao su dung",
    "dien tich tu nhien",
    "don vi hanh chinh",
    "luat dat dai",
)

PLANNING_LAND_ADMIN_TERMS = (
    "thu hoi",
    "dat nong nghiep",
    "dat phi nong nghiep",
    "dat chua su dung",
    "dien tich tu nhien",
    "don vi hanh chinh",
    "cong trinh",
    "du an",
    "luat dat dai",
)

PLANNING_REASON_CONTEXT_TERMS = (
    "quan ly dat dai",
    "giai phap",
    "ha tang",
    "thoat nuoc",
    "giao thong",
    "dich vu",
    "nha o",
    "do thi hoa",
    "dan so co hoc",
    "bai giua",
    "song hong",
    "giai phong mat bang",
)

PLANNING_FACT_MARKERS = (
    "ma ho so",
    "dossier",
    "dossier code",
    "dossiercode",
    "ap dung cho nam nao",
    "nam nao",
    "thuoc khu vuc nao",
    "khu vuc nao",
    "thanh pho nao",
    "thuoc thanh pho nao",
    "quan nao",
    "huyen nao",
    "ten day du",
    "ten van ban",
    "tieu de",
    "title",
    "noi gi",
    "neu gi",
    "dang noi gi",
    "dang neu gi",
    "la gi",
    "bao nhieu",
    "co bao nhieu",
    "tong dien tich",
    "dien tich tu nhien",
    "thu hoi bao nhieu",
    "dat nong nghiep",
    "dat phi nong nghiep",
    "dat chua su dung",
    "vi sao",
    "tai sao",
    "nhu the nao",
    "ra sao",
    "phan tich",
    "danh gia",
    "tac dong",
    "anh huong",
    "rui ro",
)

PLANNING_EXPLANATORY_MARKERS = (
    "vi sao",
    "tai sao",
    "ly do",
    "nguyen nhan",
    "do dau",
    "nhu the nao",
    "ra sao",
    "phan tich",
    "danh gia",
    "giai thich",
    "co so",
    "can cu",
    "tac dong",
    "anh huong",
    "rui ro",
    "thach thuc",
    "giai phap",
    "khuyen nghi",
    "how",
    "why",
    "analysis",
)

PLANNING_ANALYTICAL_FACT_MARKERS = (
    "phan loai",
    "phan nhom",
    "cau thanh",
    "hinh thanh",
    "chi tieu",
    "bien dong",
    "so voi",
    "hien trang",
    "uoc hien trang",
    "chuyen tiep",
    "dang ky moi",
    "da thuc hien",
    "chua thuc hien",
    "chua to chuc",
    "huy bo",
    "ket qua thuc hien",
    "dua vao ke hoach",
    "thuoc doi tuong",
    "khong thuoc doi tuong",
    "bao cao thuyet minh",
    "tong so",
    "tong cong",
    "ty le",
    "tien do",
    "dau gia quyen su dung dat",
    "phuong an boi thuong",
    "tai dinh cu",
)

PLANNING_IMPLEMENTATION_CARRY_FORWARD_MARKERS = (
    "ubnd thanh pho phe duyet",
    "ket qua thuc hien",
    "da thuc hien",
    "du kien thuc hien den",
    "31/12/2024",
    "chua to chuc",
    "chuyen tiep",
    "chuyen ky sau",
)

PLANNING_REGISTERED_PLAN_EVIDENCE_MARKERS = (
    "dang ky lap danh muc",
    "tong so cong trinh du an dang ky thuc hien",
    "tong so cong trinh du an",
    "ke hoach su dung dat nam 2025 cap huyen",
    "danh muc cac cong trinh du an thu hoi dat nam 2025",
    "danh muc cac du an chuyen muc dich",
    "nghi quyet",
    "dua vao ke hoach su dung dat",
)

def normalize_planning_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = unicodedata.normalize("NFD", lowered)
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    lowered = lowered.replace("đ", "d")
    lowered = re.sub(r"[^a-z0-9\s/.\-]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def strip_accents(text: str) -> str:
    lowered = unicodedata.normalize("NFD", (text or "").lower())
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    return lowered.replace("đ", "d")


def planning_focus_phrases(message: str) -> tuple[str, ...]:
    normalized = normalize_planning_text(message)
    if not normalized:
        return ()

    phrases: list[str] = []
    seen: set[str] = set()
    ascii_text = strip_accents(message or "")
    for token in re.findall(r"\b[a-z0-9]+(?:/[a-z0-9]+)+\b", ascii_text):
        key = token.strip()
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        phrases.append(key)

    for pattern in (
        r"(?:phuong|xa)\s+[a-z0-9]{2,}(?:\s+[a-z0-9]{2,}){0,3}",
        r"(?:quan|huyen|thi\s+xa)\s+[a-z0-9]{2,}(?:\s+[a-z0-9]{2,}){0,3}",
    ):
        for match in re.findall(pattern, normalized):
            key = re.sub(r"\s+", " ", match).strip()
            if len(key) < 5 or key in seen:
                continue
            seen.add(key)
            phrases.append(key)

    return tuple(phrases)


@dataclass(frozen=True)
class PlanningQueryProfile:
    fact_query: bool
    explanatory_query: bool
    project_structure: bool
    focus_area_reason: bool
    sector_land_demand: bool
    drainage_transport: bool
    implementation_carry_forward: bool
    project_delay_reason: bool


def build_planning_query_profile(message: str, *, planning_intent: bool = True) -> PlanningQueryProfile:
    normalized = normalize_planning_text(message)
    focus_phrases = planning_focus_phrases(message)

    if not normalized or not planning_intent:
        return PlanningQueryProfile(
            fact_query=False,
            explanatory_query=False,
            project_structure=False,
            focus_area_reason=False,
            sector_land_demand=False,
            drainage_transport=False,
            implementation_carry_forward=False,
            project_delay_reason=False,
        )

    project_structure = any(marker in normalized for marker in ("du an", "cong trinh", "danh muc")) and any(
        marker in normalized
        for marker in (
            "bieu 1a",
            "nam trong bieu",
            "cau thanh",
            "chuyen tiep",
            "da thuc hien",
            "chua thuc hien",
            "chua to chuc",
            "ket qua thuc hien",
            "dua vao ke hoach",
            "bao cao thuyet minh",
            "thong qua",
        )
    )

    focus_area_reason = any(marker in normalized for marker in ("vi sao", "tai sao", "trong diem", "nhan manh", "giai phap")) and any(
        phrase.startswith(("phuong ", "xa ")) for phrase in focus_phrases
    )

    sector_land_demand = any(
        marker in normalized for marker in ("vi sao", "tai sao", "can danh", "can tang", "can them", "nhu cau", "uu tien")
    ) and sum(
        1
        for marker in (
            "dat thuong mai",
            "thuong mai dich vu",
            "dat giao thong",
            "giao thong",
            "dat o do thi",
            "dat o tai do thi",
            "nha o do thi",
            "nha o",
        )
        if marker in normalized
    ) >= 2

    drainage_transport = any(marker in normalized for marker in ("thoat nuoc", "giao thong", "ung ngap", "ha tang")) and (
        any(marker in normalized for marker in ("vung trung", "dia hinh", "song", "ho", "tieu thoat"))
        or any(marker in normalized for marker in ("vi sao", "tai sao", "can", "chu trong", "uu tien"))
    )

    has_how_signal = any(
        marker in normalized
        for marker in ("nhu the nao", "ra sao", "phan tich", "danh gia", "how", "analysis", "vi sao", "tai sao")
    )
    has_fact_structure_signal = any(marker in normalized for marker in PLANNING_ANALYTICAL_FACT_MARKERS)
    has_numeric_cue = any(marker in normalized for marker in ("bao nhieu", "tong", "tong cong", "ty le", "dien tich")) or bool(
        re.search(r"\b\d+(?:[\.,]\d+)?\s*(?:ha|m2|km2)\b", normalized)
    )
    has_reasoning_only_signal = any(
        marker in normalized
        for marker in ("vi sao", "tai sao", "ly do", "nguyen nhan", "tac dong", "anh huong", "rui ro", "thach thuc", "giai phap", "khuyen nghi")
    )
    analytical_fact_query = any(
        (
            project_structure,
            focus_area_reason,
            sector_land_demand,
            drainage_transport,
        )
    )
    if not analytical_fact_query:
        analytical_fact_query = has_how_signal and (has_fact_structure_signal or has_numeric_cue) and not (
            has_reasoning_only_signal and not has_fact_structure_signal
        )

    explanatory_query = not analytical_fact_query and any(marker in normalized for marker in PLANNING_EXPLANATORY_MARKERS)
    fact_query = any(marker in normalized for marker in PLANNING_FACT_MARKERS) or explanatory_query

    implementation_carry_forward = False
    project_delay_reason = False

    if project_structure:
        asks_reason = any(marker in normalized for marker in ("vi sao", "tai sao", "ly do", "nguyen nhan", "do dau"))
        carry_forward_hits = sum(
            1 for marker in PLANNING_IMPLEMENTATION_CARRY_FORWARD_MARKERS if marker in normalized
        )
        result_grouping = any(
            marker in normalized
            for marker in (
                "ket qua thuc hien",
                "da thuc hien",
                "chua thuc hien",
                "chua to chuc",
                "chuyen tiep",
                "chuyen ky sau",
                "du kien thuc hien",
                "31 12 2024",
                "31/12/2024",
            )
        )

        implementation_carry_forward = not asks_reason and (carry_forward_hits >= 2 or result_grouping)
        project_delay_reason = asks_reason and any(
            marker in normalized
            for marker in ("chuyen tiep", "chuyen ky sau", "chua to chuc", "chua thuc hien", "ket qua thuc hien", "sang nam 2025")
        )

    return PlanningQueryProfile(
        fact_query=fact_query,
        explanatory_query=explanatory_query,
        project_structure=project_structure,
        focus_area_reason=focus_area_reason,
        sector_land_demand=sector_land_demand,
        drainage_transport=drainage_transport,
        implementation_carry_forward=implementation_carry_forward,
        project_delay_reason=project_delay_reason,
    )
