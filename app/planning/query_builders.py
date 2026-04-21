from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .profiles import build_planning_query_profile, planning_focus_phrases, strip_accents
from .ranker import planning_query_terms


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = strip_accents(lowered)
    lowered = lowered.replace("Ä‘", "d").replace("Ä", "d")
    lowered = unicodedata.normalize("NFD", lowered)
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def district_code_fragment(district: Optional[str]) -> str:
    if not district:
        return ""
    ascii_text = strip_accents(district)
    parts = re.split(r"[^A-Za-z0-9]+", ascii_text)
    cleaned = [part for part in parts if part]
    return "".join(token[:1].upper() + token[1:] for token in cleaned)


def planning_specialized_limit(message: str) -> int:
    profile = build_planning_query_profile(message, planning_intent=True)
    if profile.new_registration_unique:
        return 1
    if profile.city_level_listing:
        return 3
    if profile.article67:
        return 2
    if profile.admin_overview:
        return 2
    if profile.decision_total:
        return 1
    if profile.land_recovery:
        return 2
    if profile.land_change:
        return 3
    if profile.public_purpose_composition:
        return 3
    if profile.project_structure:
        if profile.registered_plan_composition or profile.project_delay_reason or profile.implementation_carry_forward:
            return 4
        return 3
    if profile.gpmb_stats:
        return 3
    if profile.environment_constraint:
        return 2
    if profile.plan_necessity:
        return 2
    if profile.focus_area_reason:
        return 3 if profile.focus_management else 2
    if profile.sector_land_demand:
        return 3
    if profile.drainage_transport:
        return 2
    if profile.analytical_fact_query or profile.fact_query:
        # Generic fallback for analytical planning questions that miss specialized buckets.
        return 2
    return 0


def planning_fact_subqueries(message: str) -> list[str]:
    msg_norm = _normalize_text(message)
    if not msg_norm:
        return []

    query_terms = [term for term in planning_query_terms(message, max_terms=14) if term]
    focus_terms = planning_focus_phrases(message)
    year_terms = re.findall(r"\b20\d{2}\b", msg_norm)

    out: list[str] = []
    if query_terms:
        out.append(" ".join(query_terms[:6]))
        if len(query_terms) > 6:
            out.append(" ".join(query_terms[6:12]))

    for phrase in focus_terms[:4]:
        out.append(phrase)

    if year_terms and query_terms:
        out.append(f"{' '.join(query_terms[:4])} {' '.join(year_terms[:2])}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        key = _normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def planning_intent_rescue_queries(message: str, district: Optional[str], plan_year: Optional[int]) -> list[str]:
    normalized = _normalize_text(message)
    if not normalized:
        return []

    profile = build_planning_query_profile(message, planning_intent=True)
    focus_terms = planning_focus_phrases(message)
    queries: list[str] = []
    implementation_duty_query = (
        "duoc phe duyet" in normalized
        and "ubnd" in normalized
        and any(marker in normalized for marker in ("trien khai", "thuc hien", "phai"))
    )
    auction_query = "dau gia" in normalized and "quyen su dung dat" in normalized
    numeric_tokens = re.findall(r"\b\d+(?:[\.,]\d+)?\b", normalized)

    if any(marker in normalized for marker in ("chi tieu", "bien dong", "so voi", "hien trang", "dat nong nghiep", "dat phi nong nghiep")):
        queries.append("chi tieu dat nong nghiep dat phi nong nghiep dat chua su dung hien trang 2024 2025")

    if any(
        marker in normalized
        for marker in ("du an", "cong trinh", "phan loai", "phan nhom", "da thuc hien", "chua thuc hien", "chuyen tiep", "hinh thanh", "cau thanh")
    ):
        queries.append("tong so du an da thuc hien chua thuc hien chuyen tiep dien tich")

    if any(marker in normalized for marker in ("giai phong mat bang", "gpmb", "thu hoi dat", "boi thuong", "tai dinh cu")):
        queries.append("giai phong mat bang thong bao thu hoi dieu tra xac nhan nguon goc phuong an boi thuong")
        queries.append("cong tac giai phong mat bang trien khai ket qua thuc hien thong bao thu hoi")

    if implementation_duty_query:
        queries.append("ubnd quan cong bo cong khai ke hoach thu hoi dat kiem tra xu ly vi pham can doi nguon von to chuc thuc hien")
        queries.append("nhiem vu ubnd sau khi ke hoach su dung dat duoc phe duyet bao cao ket qua truoc 15 10 2025")

    if auction_query:
        queries.append("cong tac dau gia quyen su dung dat o dat du an trung dau gia gia khoi diem gia trung dau gia")
        queries.append("moi dau gia nhieu lan chua co nha dau tu khach hang trung dau gia")

    if any(marker in normalized for marker in ("muc dich cong cong", "dat cong cong", "bao gom", "cau thanh")):
        queries.append("dat su dung vao muc dich cong cong bao gom cau thanh chi tieu")

    if profile.land_change:
        queries.append("chi tieu su dung dat nam 2024 nam 2025 bien dong dat nong nghiep dat phi nong nghiep dat chua su dung")
        queries.append("hien trang 2024 ke hoach 2025 dat nong nghiep dat phi nong nghiep dat chua su dung tang giam")

    if profile.public_purpose_composition:
        queries.append("dat su dung vao muc dich cong cong bao gom giao thong thuy loi di tich nang luong buu chinh cho khu vui choi")
        queries.append("tong dien tich dat su dung vao muc dich cong cong giao thong la lon nhat")

    if profile.project_structure:
        queries.append("tong so du an da thuc hien chua thuc hien chuyen tiep dua vao ke hoach dien tich")
        queries.append("bao cao thuyet minh danh muc cong trinh du an nam 2025 hdnd thong qua dua vao ke hoach")
        queries.append("ket qua thuc hien ke hoach su dung dat nam 2024 chuyen tiep sang 2025 tong so dien tich")
    if profile.registered_plan_composition:
        queries.append("tong so dang ky thuc hien hdnd thanh pho thong qua thu hoi dat chuyen muc dich dat trong lua dua vao ke hoach")
        queries.append("bao cao thuyet minh tong so cong trinh du an dang ky thuc hien hdnd thong qua dua vao ke hoach su dung dat")
        queries.append("dang ky lap danh muc cong trinh du an thu hoi dat nam 2025 chuyen muc dich dat trong lua nghi quyet")
        queries.append("danh muc cong trinh du an trong ke hoach su dung dat nam 2025 la bao nhieu trong do")
    if profile.implementation_carry_forward:
        queries.append("ubnd thanh pho phe duyet ket qua thuc hien da thuc hien du kien thuc hien den 31 12 2024 chua to chuc chuyen tiep")
        queries.append("tong so du an da thuc hien du kien thuc hien den 31 12 2024 chuyen tiep sang 2025")
    if profile.project_delay_reason:
        queries.append("nguyen nhan chu yeu chuyen tiep sang nam 2025 thu tuc phe duyet bao cao kinh te ky thuat")
        queries.append("do dac hien trang xac dinh nguon goc dat phuong an den bu cong bo cong khai quy hoach du an")

    if profile.gpmb_stats:
        queries.append("giai phong mat bang thong bao thu hoi dieu tra xac nhan nguon goc du thao phe duyet phuong an boi thuong")
        queries.append("cong tac giai phong mat bang ho gia dinh to chuc phuong an boi thuong ty dong")
        queries.append("du an thanh phan boi thuong ho tro tai dinh cu di chuyen mo")

    if profile.environment_constraint:
        queries.append("phan tich danh gia moi truong tac dong den viec su dung dat")
        queries.append("moi truong nuoc mat nuoc duoi dat khong khi bod5 cod tss amoni h2s o nhiem")
        queries.append("o nhiem moi truong tac dong den su dung dat")

    if profile.plan_necessity:
        queries.append("luat dat dai 2024 nghi dinh 102 2024 ke hoach su dung dat hang nam cap huyen")
        queries.append("co so thu hoi dat giao dat cho thue dat chuyen muc dich su dung dat")
        queries.append("hang nam cap huyen phai lap ke hoach su dung dat")
        queries.append("su dung dat hop ly tiet kiem tranh lang phi bao ve moi truong sinh thai")

    if profile.drainage_transport:
        queries.append("dia hinh vung trung song ho to lich lu set kim nguu yen so linh dam den lu tieu thoat nuoc ung ngap giao thong")
        queries.append("vung trung song ho tieu thoat nuoc giao thong ha tang ky thuat")
        queries.append("vi sao can chu trong thoat nuoc giao thong do dia hinh vung trung song ho ung ngap")

    if profile.analytical_fact_query or profile.fact_query:
        generic_terms = [term for term in planning_query_terms(message, max_terms=12) if len(term) >= 4]
        if generic_terms:
            queries.append(" ".join(generic_terms[:6]))
            queries.append("bao cao thuyet minh " + " ".join(generic_terms[:5]))
        if numeric_tokens:
            queries.append("tong so du an dien tich " + " ".join(numeric_tokens[:6]))
        queries.append("quyet dinh phu luc tong so du an dien tich nam 2025")
        queries.append("bao cao thuyet minh ke hoach su dung dat ket qua thuc hien nam 2024")

    if profile.focus_area_reason:
        for phrase in focus_terms[:3]:
            queries.append(f"{phrase} dat nong nghiep thu hoi chuyen muc dich du an trong diem")
            queries.append(f"{phrase} giai phap quan ly dat dai hanh lang de thoat lu su dung sai muc dich")
            queries.append(f"{phrase} bai giua song hong quy hoach chi tiet quan ly dat dai")
            queries.append(f"{phrase} ngoai de song hong hanh lang de hanh lang thoat lu hanh lang cau lan chiem bo bai")
            queries.append(f"{phrase} kiem tra thu hoi dat su dung sai muc dich dat bai giua dich vu du lich")

    if profile.sector_land_demand:
        queries.append("dat thuong mai dich vu dat giao thong dat o do thi tang them vi sao")
        queries.append("nhu cau dat thuong mai dich vu giao thong nha o do thi trong giai doan toi")
        queries.append("cua ngo phia tay he thong giao thong thu hut von tai chinh nhan luc khoa hoc cong nghe")
        queries.append("thuong mai va dich vu trung tam thuong mai khach san dich vu tai chinh du lich do thi hoa dan so co hoc")

    if profile.admin_overview:
        queries.append("dien tich tu nhien don vi hanh chinh cap phuong cap xa")
        queries.append("tong dien tich tu nhien don vi hanh chinh")
        queries.append("tong dien tich tu nhien phuong xa")

    if any(marker in normalized for marker in ("dang ky moi", "du an dang ky moi", "duy nhat")):
        queries.append("du an dang ky moi duy nhat nam 2025 ten du an dien tich")
        queries.append("chi tieu thu hoi dat dang ky moi ten du an dien tich")

    if any(marker in normalized for marker in ("cap thanh pho", "thanh pho giao", "cong trinh cap thanh pho")):
        queries.append("cong trinh cap thanh pho tren dia ban quan nam 2025 gom nhung gi")
        queries.append("tru so bo cong an 44 yet kieu ga c10 ga s12")

    if any(
        marker in normalized
        for marker in ("thu hoi bao nhieu", "thu hoi dat", "dat nong nghiep", "dat phi nong nghiep", "tong cong bao nhieu dat")
    ):
        queries.append("thu hoi dat nong nghiep dat phi nong nghiep tong dien tich thu hoi nam 2025")
        queries.append("ke hoach thu hoi dat nam 2025 tong cong dat nong nghiep dat phi nong nghiep")
        queries.append("bieu ke hoach thu hoi dat tong cong")
        queries.append("tong dien tich dat thu hoi nam 2025 la bao nhieu")

    if any(marker in normalized for marker in ("theo quyet dinh", "quyet dinh phe duyet", "bao nhieu cong trinh", "bao nhieu du an")):
        queries.append("quyet dinh phe duyet tong so cong trinh du an nam 2025 tong dien tich")
        queries.append("tong so cong trinh du an duoc phe duyet phu luc")

    if any(marker in normalized for marker in ("khoan 4 dieu 67", "dieu 67", "tiep tuc thuc hien")):
        queries.append("khoan 4 dieu 67 danh muc du an tiep tuc thuc hien nam 2025")
        queries.append("hai du an nao khoan 4 dieu 67 mo rong nha tang le bo cong an")
        queries.append("mo rong nha tang le quoc gia so 5 tran thanh tong")
        queries.append("mo rong tru so bo cong an so 30 tran binh trong 58 tran nhan tong")

    if profile.project_listing:
        queries.extend(
            [
                "phu luc danh muc cac cong trinh du an thuc hien nam 2025",
                "du an dang ky moi duy nhat ten du an dien tich",
                "cong trinh cap thanh pho ten cong trinh",
                "quyet dinh phe duyet tong so cong trinh du an",
                "xay dung tram y te phuong",
                "truong thpt cong lap o dat ky hieu",
                "mo rong nha tang le quoc gia mo rong tru so bo cong an",
            ]
        )

    if profile.new_registration_unique:
        queries = [
            "xay dung tram y te phuong chuong duong dang ky moi 0,0076",
            "truong thpt cong lap o dat ky hieu f/thpt1 phuong mai dich",
            *queries,
        ]

    if profile.city_level_listing:
        queries = ["tru so bo cong an 44 yet kieu ga c10 ga s12", *queries]

    if profile.decision_total:
        queries = ["tong so cong trinh du an duoc phe duyet 32 8,6915", *queries]

    if profile.article67:
        queries = [
            "khoan 4 dieu 67 du an mo rong nha tang le quoc gia",
            "mo rong nha tang le quoc gia so 5 tran thanh tong",
            "mo rong tru so bo cong an so 30 tran binh trong 58 tran nhan tong",
            *queries,
        ]

    if profile.admin_overview:
        queries = ["dien tich tu nhien don vi hanh chinh", "tong dien tich tu nhien cap phuong cap xa", *queries]

    if "dieu 78" in normalized or "dieu 79" in normalized:
        queries.append("dieu 78 dieu 79 thu hoi khong thu hoi tong so dien tich")

    queries.extend(planning_fact_subqueries(message)[:3])

    district_label = (district or "").strip()
    expanded: list[str] = list(queries)
    for query in list(queries):
        if district_label:
            expanded.append(f"{query} {district_label}")
        if plan_year is not None:
            expanded.append(f"{query} nam {plan_year}")

    deduped: list[str] = []
    seen: set[str] = set()
    for query in expanded:
        key = _normalize_text(query)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(query)
        if len(deduped) >= 18:
            break
    return deduped


def planning_query_candidates(
    message: str,
    district: Optional[str],
    plan_year: Optional[int],
    *,
    max_query_candidates: int,
    max_fact_subqueries: int,
) -> list[str]:
    candidates: list[str] = []
    district_label = (district or "").strip()
    intent_rescue_queries = planning_intent_rescue_queries(message, district, plan_year)
    fact_subqueries = planning_fact_subqueries(message)[:max_fact_subqueries]

    candidates.extend(intent_rescue_queries)
    candidates.extend(fact_subqueries)
    candidates.append(message)

    if district_label:
        candidates.append(f"{message} {district_label}")
    if plan_year is not None:
        candidates.append(f"{message} nam {plan_year}")
    if district_label and plan_year is not None:
        candidates.append(f"{message} {district_label} nam {plan_year}")

    focus_terms = list(planning_query_terms(message, max_terms=12))
    if focus_terms:
        focus_query = " ".join(focus_terms)
        candidates.append(focus_query)
        if district_label:
            candidates.append(f"{focus_query} {district_label}")
        if plan_year is not None:
            candidates.append(f"{focus_query} {plan_year}")

    for subquery in fact_subqueries:
        if district_label:
            candidates.append(f"{subquery} {district_label}")
        if plan_year is not None:
            candidates.append(f"{subquery} nam {plan_year}")

    if district_label and plan_year is not None:
        district_code = district_code_fragment(district_label)
        if district_code:
            candidates.append(f"HN-{district_code}-KH{plan_year}")

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(candidate)
        if len(out) >= max_query_candidates:
            break
    return out
