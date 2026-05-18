from __future__ import annotations

import re
import unicodedata

PLANNING_DISTRICT_ALIASES: dict[str, tuple[str, ...]] = {
    "Ba Đình": ("ba dinh", "quan ba dinh"),
    "Bắc Từ Liêm": ("bac tu liem", "quan bac tu liem"),
    "Cầu Giấy": ("cau giay", "quan cau giay"),
    "Đan Phượng": ("dan phuong", "huyen dan phuong"),
    "Đống Đa": ("dong da", "quan dong da"),
    "Hai Bà Trưng": ("hai ba trung", "quan hai ba trung"),
    "Hà Đông": ("ha dong", "quan ha dong"),
    "Hoài Đức": ("hoai duc", "huyen hoai duc"),
    "Hoàn Kiếm": ("hoan kiem", "quan hoan kiem"),
    "Hoàng Mai": ("hoang mai", "quan hoang mai"),
    "Long Biên": ("long bien", "quan long bien"),
    "Mê Linh": ("me linh", "huyen me linh"),
    "Nam Từ Liêm": ("nam tu liem", "quan nam tu liem"),
    "Phú Xuyên": ("phu xuyen", "huyen phu xuyen"),
    "Quốc Oai": ("quoc oai", "huyen quoc oai"),
    "Sơn Tây": ("son tay", "thi xa son tay", "son tay town"),
    "Tây Hồ": ("tay ho", "quan tay ho"),
    "Thạch Thất": ("thach that", "huyen thach that"),
    "Thanh Oai": ("thanh oai", "huyen thanh oai"),
    "Thanh Trì": ("thanh tri", "huyen thanh tri"),
    "Thanh Xuân": ("thanh xuan", "quan thanh xuan"),
}

_DOSSIER_HINTS: dict[str, str] = {
    "badinh": "Ba Đình",
    "bactuliem": "Bắc Từ Liêm",
    "caugiay": "Cầu Giấy",
    "danphuong": "Đan Phượng",
    "dongda": "Đống Đa",
    "haibatrung": "Hai Bà Trưng",
    "hadong": "Hà Đông",
    "hoaiduc": "Hoài Đức",
    "hoankiem": "Hoàn Kiếm",
    "hoangmai": "Hoàng Mai",
    "longbien": "Long Biên",
    "melinh": "Mê Linh",
    "namtuliem": "Nam Từ Liêm",
    "phuxuyen": "Phú Xuyên",
    "quocoai": "Quốc Oai",
    "sontay": "Sơn Tây",
    "tayho": "Tây Hồ",
    "thachthat": "Thạch Thất",
    "thanhoai": "Thanh Oai",
    "thanhtri": "Thanh Trì",
    "thanhxuan": "Thanh Xuân",
}


def _strip_accents(text: str) -> str:
    normalized = (text or "").replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", normalized)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _normalize_text(text: str) -> str:
    value = _strip_accents(text or "").lower()
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _compact_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _normalize_text(text))


def _match_from_text(text: str) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None

    padded = f" {normalized} "
    for canonical, aliases in PLANNING_DISTRICT_ALIASES.items():
        alias_candidates = {canonical, *aliases}
        for alias in alias_candidates:
            alias_norm = _normalize_text(alias)
            if not alias_norm:
                continue
            if f" {alias_norm} " in padded:
                return canonical

    compact = _compact_token(text)
    if compact:
        for canonical, aliases in PLANNING_DISTRICT_ALIASES.items():
            alias_compacts = {_compact_token(canonical), *(_compact_token(alias) for alias in aliases)}
            if compact in alias_compacts:
                return canonical
            if any(alias_compact and alias_compact in compact for alias_compact in alias_compacts):
                return canonical

    return None


def canonicalize_planning_district(
    value: str | None,
    *,
    title: str | None = None,
    dossier_code: str | None = None,
) -> str | None:
    for candidate in (value, title):
        matched = _match_from_text(candidate or "")
        if matched:
            return matched

    dossier = _compact_token(dossier_code or "")
    if dossier:
        for hint, canonical in _DOSSIER_HINTS.items():
            if hint in dossier:
                return canonical

    return None
