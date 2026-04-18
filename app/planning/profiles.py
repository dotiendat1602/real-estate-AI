from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

PLANNING_KEYWORDS = (
    "quy hoach",
    "quy hoạch",
    "quy hoach su dung dat",
    "quy hoạch sử dụng đất",
    "ke hoach su dung dat",
    "kế hoạch sử dụng đất",
    "ke hoach",
    "kế hoạch",
    "khsd",
    "khsdd",
    "thong tin quy hoach",
    "thông tin quy hoạch",
    "loai dat",
    "loại đất",
    "thua dat",
    "thửa đất",
    "to ban do",
    "tờ bản đồ",
    "muc dich su dung dat",
    "mục đích sử dụng đất",
    "quy hoach do thi",
    "quy hoạch đô thị",
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

PLANNING_REGISTERED_COMPOSITION_MARKERS = (
    "dang ky thuc hien",
    "dang ky",
    "hdnd thanh pho",
    "hoi dong nhan dan",
    "thu hoi dat",
    "chuyen muc dich",
    "dat trong lua",
    "dua vao ke hoach",
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

PLANNING_PROJECT_DELAY_REASON_MARKERS = (
    "nguyen nhan chu yeu",
    "lam cac thu tuc",
    "thu tuc phe duyet",
    "bao cao kinh te ky thuat",
    "do dac hien trang",
    "xac dinh nguon goc dat",
    "phuong an den bu",
    "cong bo cong khai quy hoach",
    "chua to chuc thuc hien",
    "chuyen tiep sang nam",
)

PLANNING_FOCUS_MANAGEMENT_MARKERS = (
    "ngoai de",
    "ngoai bai",
    "hanh lang de",
    "hanh lang thoat lu",
    "thoat lu",
    "hanh lang cau",
    "lan chiem",
    "bo bai",
    "bai giua",
    "song hong",
    "su dung sai muc dich",
    "kiem tra",
    "thu hoi dat",
    "du lich dich vu",
    "dich vu du lich",
)

PLANNING_GROWTH_PRESSURE_MARKERS = (
    "cua ngo phia tay",
    "quan noi thanh",
    "he thong giao thong",
    "ha tang",
    "thu hut",
    "von tai chinh",
    "nhan luc",
    "khoa hoc cong nghe",
    "thuong mai va dich vu",
    "trung tam thuong mai",
    "khach san",
    "dich vu tai chinh",
    "du lich",
    "do thi hoa",
    "dan so co hoc",
    "nha o",
    "ha tang ky thuat",
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


def planning_marker_hits(haystack: str, markers: tuple[str, ...]) -> int:
    return sum(1 for marker in markers if marker in haystack)


def planning_asks_reason(normalized: str) -> bool:
    return any(marker in normalized for marker in ("vi sao", "tai sao", "ly do", "nguyen nhan", "do dau"))


def planning_has_structure_composition_request(normalized: str) -> bool:
    return any(marker in normalized for marker in ("cau thanh", "bao gom", "phan loai", "phan nhom", "nhu the nao"))


def planning_has_result_grouping_request(normalized: str) -> bool:
    return any(
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


def planning_has_registered_plan_request(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in (
            "bao cao thuyet minh",
            "danh muc",
            "dua vao ke hoach",
            "ke hoach su dung dat",
            "nam ke hoach",
            "dang ky",
        )
    )


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
    normalized: str
    planning_intent: bool
    focus_phrases: tuple[str, ...]
    fact_query: bool
    analytical_fact_query: bool
    explanatory_query: bool
    project_listing: bool
    new_registration_unique: bool
    city_level_listing: bool
    article67: bool
    decision_total: bool
    land_recovery: bool
    admin_overview: bool
    land_change: bool
    public_purpose_composition: bool
    project_structure: bool
    gpmb_stats: bool
    environment_constraint: bool
    plan_necessity: bool
    focus_area_reason: bool
    sector_land_demand: bool
    drainage_transport: bool
    registered_plan_composition: bool
    implementation_carry_forward: bool
    project_delay_reason: bool
    focus_management: bool


def build_planning_query_profile(message: str, *, planning_intent: bool = True) -> PlanningQueryProfile:
    normalized = normalize_planning_text(message)
    focus_phrases = planning_focus_phrases(message)

    if not normalized or not planning_intent:
        return PlanningQueryProfile(
            normalized=normalized,
            planning_intent=bool(planning_intent and normalized),
            focus_phrases=focus_phrases,
            fact_query=False,
            analytical_fact_query=False,
            explanatory_query=False,
            project_listing=False,
            new_registration_unique=False,
            city_level_listing=False,
            article67=False,
            decision_total=False,
            land_recovery=False,
            admin_overview=False,
            land_change=False,
            public_purpose_composition=False,
            project_structure=False,
            gpmb_stats=False,
            environment_constraint=False,
            plan_necessity=False,
            focus_area_reason=False,
            sector_land_demand=False,
            drainage_transport=False,
            registered_plan_composition=False,
            implementation_carry_forward=False,
            project_delay_reason=False,
            focus_management=False,
        )

    project_listing = any(
        marker in normalized
        for marker in (
            "du an dang ky moi",
            "duy nhat",
            "cong trinh cap thanh pho",
            "bao nhieu cong trinh",
            "bao nhieu du an",
            "du an nao",
            "theo quyet dinh",
            "quyet dinh phe duyet",
            "khoan 4 dieu 67",
            "thuoc truong hop quy dinh",
            "phu luc",
            "danh muc du an",
        )
    )
    new_registration_unique = project_listing and any(marker in normalized for marker in ("dang ky moi", "duy nhat"))
    city_level_listing = project_listing and any(
        marker in normalized for marker in ("cap thanh pho", "thanh pho giao", "cong trinh cap thanh pho")
    )
    article67 = project_listing and any(marker in normalized for marker in ("khoan 4 dieu 67", "dieu 67", "thuoc truong hop quy dinh"))
    decision_total = project_listing and any(marker in normalized for marker in ("theo quyet dinh", "quyet dinh phe duyet")) and any(
        marker in normalized for marker in ("bao nhieu cong trinh", "bao nhieu du an")
    )

    land_recovery = any(
        marker in normalized
        for marker in (
            "thu hoi",
            "ke hoach thu hoi dat",
            "tong cong bao nhieu dat",
            "dat nong nghiep",
            "dat phi nong nghiep",
        )
    ) and (not any(marker in normalized for marker in ("chuyen muc dich", "chuyen doi")) or "thu hoi" in normalized)

    admin_overview = "dien tich tu nhien" in normalized and any(
        marker in normalized
        for marker in ("don vi hanh chinh", "cap phuong", "cap xa", "bao nhieu don vi")
    )

    land_change = any(
        marker in normalized
        for marker in ("dat nong nghiep", "dat phi nong nghiep", "dat chua su dung", "chi tieu su dung dat")
    ) and (
        any(marker in normalized for marker in ("chi tieu", "bien dong", "so voi", "hien trang", "uoc hien trang", "tang", "giam"))
        or ("2024" in normalized and "2025" in normalized)
    )

    public_purpose_composition = any(marker in normalized for marker in ("muc dich cong cong", "dat cong cong")) and any(
        marker in normalized for marker in ("cau thanh", "bao gom", "nhu the nao", "phan loai")
    )

    project_structure = any(marker in normalized for marker in ("du an", "cong trinh", "danh muc")) and any(
        marker in normalized
        for marker in (
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

    gpmb_stats = any(marker in normalized for marker in ("giai phong mat bang", "gpmb", "boi thuong", "tai dinh cu", "thong bao thu hoi")) and any(
        marker in normalized for marker in ("nhu the nao", "trien khai", "tien do", "phuong an", "ho gia dinh", "ty dong", "kho khan")
    )

    environment_constraint = any(
        marker in normalized
        for marker in ("moi truong", "o nhiem", "nuoc mat", "nuoc duoi dat", "khong khi", "bod5", "cod", "tss", "amoni", "h2s")
    )

    plan_necessity = "ke hoach su dung dat" in normalized and any(
        marker in normalized
        for marker in ("can lap", "phai lap", "su can thiet", "can co ke hoach", "vi sao can lap", "tai sao can lap")
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
            land_change,
            public_purpose_composition,
            project_structure,
            gpmb_stats,
            environment_constraint,
            plan_necessity,
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

    registered_plan_composition = False
    implementation_carry_forward = False
    project_delay_reason = False
    focus_management = False

    if project_structure:
        asks_reason = planning_asks_reason(normalized)
        has_composition = planning_has_structure_composition_request(normalized) or "bao cao thuyet minh" in normalized
        registered_hits = planning_marker_hits(normalized, PLANNING_REGISTERED_COMPOSITION_MARKERS)
        carry_forward_hits = planning_marker_hits(normalized, PLANNING_IMPLEMENTATION_CARRY_FORWARD_MARKERS)
        registered_context = planning_has_registered_plan_request(normalized)
        result_grouping = planning_has_result_grouping_request(normalized)

        registered_plan_composition = (
            not asks_reason
            and has_composition
            and (registered_hits >= 2 or registered_context)
            and not (carry_forward_hits >= 2 or result_grouping)
        )
        implementation_carry_forward = not asks_reason and (carry_forward_hits >= 2 or result_grouping)
        project_delay_reason = asks_reason and any(
            marker in normalized
            for marker in ("chuyen tiep", "chuyen ky sau", "chua to chuc", "chua thuc hien", "ket qua thuc hien", "sang nam 2025")
        )

    if focus_area_reason:
        focus_management = "quan ly dat dai" in normalized or "giai phap quan ly" in normalized

    return PlanningQueryProfile(
        normalized=normalized,
        planning_intent=True,
        focus_phrases=focus_phrases,
        fact_query=fact_query,
        analytical_fact_query=analytical_fact_query,
        explanatory_query=explanatory_query,
        project_listing=project_listing,
        new_registration_unique=new_registration_unique,
        city_level_listing=city_level_listing,
        article67=article67,
        decision_total=decision_total,
        land_recovery=land_recovery,
        admin_overview=admin_overview,
        land_change=land_change,
        public_purpose_composition=public_purpose_composition,
        project_structure=project_structure,
        gpmb_stats=gpmb_stats,
        environment_constraint=environment_constraint,
        plan_necessity=plan_necessity,
        focus_area_reason=focus_area_reason,
        sector_land_demand=sector_land_demand,
        drainage_transport=drainage_transport,
        registered_plan_composition=registered_plan_composition,
        implementation_carry_forward=implementation_carry_forward,
        project_delay_reason=project_delay_reason,
        focus_management=focus_management,
    )
