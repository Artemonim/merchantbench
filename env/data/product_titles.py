"""Deterministic marketplace-style product title generator.

Titles are built only from the caller-supplied ``rng``. Category noun
pools are keyed by the synthetic / mapped catalog categories so repeated
brands and nouns stay frequent enough for hot-search support.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# * Hard cap stays well under the catalog's 200-character name limit.
MAX_TITLE_LEN = 120
PROMO_CHANCE = 0.22
ATTRIBUTE_CHANCE = 0.62
MATERIAL_CHANCE = 0.48
SPEC_CHANCE = 0.58

# * Fictional 3–8 letter brands reused across listings for hot-search support.
BRANDS = [
    "Velora",
    "Noxen",
    "Brivio",
    "Aurel",
    "Kinta",
    "Solvi",
    "Mardex",
    "Quenlo",
    "Pixora",
    "Halden",
    "Orbex",
    "Tulio",
    "Fenra",
    "Calven",
    "Zorik",
    "Lumina",
    "Pedra",
    "Winton",
    "Sable",
    "Korvin",
    "Nexel",
    "Vorin",
    "Altek",
    "Miraq",
    "Boden",
    "Trevia",
    "Kasen",
    "Olvar",
    "Jenlo",
    "Rivex",
]

ATTRIBUTES = [
    "Portable",
    "Compact",
    "Premium",
    "Foldable",
    "Soft",
    "Durable",
    "Sturdy",
    "Classic",
    "Modern",
    "Slim",
    "Dual",
    "Mini",
    "Ultra",
]

PROMO_TOKENS = ["New", "Hot", "2026"]

SPECS = [
    "Set of 2",
    "Set of 4",
    "3-Pack",
    "2-Pack",
    "Pack of 6",
    "4pc",
]

_GENERIC_MATERIALS = [
    "Cotton",
    "Steel",
    "Plastic",
    "Rubber",
    "Glass",
    "Nylon",
    "Metal",
    "Ceramic",
]

_FALLBACK_NOUNS = ["item", "kit", "pack", "set"]

# * Category nouns reused from the synthetic catalog pools, plus a few extras.
CATEGORY_NOUNS = {
    "office": ["pen", "marker", "folder", "tape", "pad", "clip", "sticker", "binder"],
    "womenswear": ["dress", "tshirt", "skirt", "scarf", "belt", "jacket", "shoes", "blouse"],
    "pet_garden": ["leash", "collar", "planter", "treat", "toy", "bed", "brush", "carrier"],
    "appliances": ["kettle", "fan", "lamp", "scale", "heater", "blender", "humidifier", "iron"],
    "home_decor": ["towel", "blanket", "pillow", "rug", "curtain", "shelf", "hook", "basket"],
    "home_goods": ["mat", "hanger", "box", "cup", "rack", "hook", "basket", "tray"],
    "cleaning": ["mop", "brush", "bin", "sprayer", "cloth", "bucket", "organizer", "squeegee"],
    "toys": ["puzzle", "blocks", "robot", "plush", "kite", "doll", "set", "tracks"],
    "bags": ["tote", "wallet", "backpack", "satchel", "pouch", "suitcase", "duffel", "case"],
    "sports": ["ball", "rope", "bottle", "mat", "band", "gloves", "racket", "shorts"],
    "electronics": ["earbuds", "charger", "cable", "speaker", "mouse", "keyboard", "router", "lamp"],
    "home": ["towel", "blanket", "pillow", "rug", "curtain", "shelf", "hook", "basket"],
    "kitchen": ["pan", "knife", "spatula", "kettle", "blender", "scale", "mat", "jar"],
    "beauty": ["serum", "cream", "mask", "lipstick", "brush", "mirror", "balm", "tonic"],
    "apparel": ["tshirt", "socks", "scarf", "hat", "belt", "gloves", "jacket", "shoes"],
    "books": ["notebook", "novel", "guide", "diary", "planner", "comic", "manual", "almanac"],
    "pet": ["leash", "collar", "bowl", "treat", "toy", "bed", "brush", "carrier"],
}

# * Dummy-draw bounds for the product RNG: 8 nouns per category, 1 fallback,
# * 10 adjectives. Lengths keep catalog numeric fields bitwise-compatible
# * with the pre-generator catalog.
LEGACY_ADJECTIVES = [
    "pro",
    "lite",
    "max",
    "mini",
    "plus",
    "ultra",
    "classic",
    "smart",
    "eco",
    "prime",
]
LEGACY_FALLBACK_NOUNS = ["item"]
LEGACY_NOUN_POOLS = {
    "office": ["pen", "marker", "folder", "tape", "pad", "clip", "sticker", "binder"],
    "womenswear": ["dress", "tshirt", "skirt", "scarf", "belt", "jacket", "shoes", "blouse"],
    "pet_garden": ["leash", "collar", "planter", "treat", "toy", "bed", "brush", "carrier"],
    "appliances": ["kettle", "fan", "lamp", "scale", "heater", "blender", "humidifier", "iron"],
    "home_decor": ["towel", "blanket", "pillow", "rug", "curtain", "shelf", "hook", "basket"],
    "home_goods": ["mat", "hanger", "box", "cup", "rack", "hook", "basket", "tray"],
    "cleaning": ["mop", "brush", "bin", "sprayer", "cloth", "bucket", "organizer", "squeegee"],
    "toys": ["puzzle", "blocks", "robot", "plush", "kite", "doll", "set", "tracks"],
    "bags": ["tote", "wallet", "backpack", "satchel", "pouch", "suitcase", "duffel", "case"],
    "sports": ["ball", "rope", "bottle", "mat", "band", "gloves", "racket", "shorts"],
    "electronics": ["earbuds", "charger", "cable", "speaker", "mouse", "keyboard", "router", "lamp"],
    "home": ["towel", "blanket", "pillow", "rug", "curtain", "shelf", "hook", "basket"],
    "kitchen": ["pan", "knife", "spatula", "kettle", "blender", "scale", "mat", "jar"],
    "beauty": ["serum", "cream", "mask", "lipstick", "brush", "mirror", "balm", "tonic"],
    "apparel": ["tshirt", "socks", "scarf", "hat", "belt", "gloves", "jacket", "shoes"],
    "books": ["notebook", "novel", "guide", "diary", "planner", "comic", "manual", "almanac"],
    "pet": ["leash", "collar", "bowl", "treat", "toy", "bed", "brush", "carrier"],
}

ATTRIBUTES_BY_CATEGORY = {
    "office": ["Compact", "Durable", "Classic", "Slim", "Premium"],
    "womenswear": ["Soft", "Classic", "Premium", "Slim", "Modern"],
    "pet_garden": ["Durable", "Portable", "Sturdy", "Folding", "Soft"],
    "appliances": ["Compact", "Portable", "Dual", "Mini", "Premium"],
    "home_decor": ["Soft", "Modern", "Classic", "Premium", "Folding"],
    "home_goods": ["Compact", "Sturdy", "Durable", "Slim", "Premium"],
    "cleaning": ["Durable", "Compact", "Heavy", "Sturdy", "Portable"],
    "toys": ["Mini", "Classic", "Soft", "Durable", "Foldable"],
    "bags": ["Compact", "Durable", "Slim", "Premium", "Sturdy"],
    "sports": ["Durable", "Portable", "Compact", "Ultra", "Sturdy"],
    "electronics": ["Wireless", "Compact", "Portable", "Slim", "Dual"],
    "home": ["Soft", "Modern", "Classic", "Premium", "Compact"],
    "kitchen": ["Compact", "Durable", "Premium", "Mini", "Classic"],
    "beauty": ["Soft", "Mini", "Premium", "Compact", "Classic"],
    "apparel": ["Soft", "Classic", "Slim", "Premium", "Ultra"],
    "books": ["Classic", "Compact", "Premium", "Slim", "Modern"],
    "pet": ["Durable", "Soft", "Portable", "Sturdy", "Compact"],
}

SPECS_BY_CATEGORY = {
    "office": ["Set of 2", "12-Pack", "3-Pack", "4pc"],
    "womenswear": ["Size M", "Size L", "Size S"],
    "pet_garden": ["Set of 2", "3-Pack", "Size M"],
    "appliances": ["1.5L", "1.7L", "2-Pack", "10 inch"],
    "home_decor": ["Set of 2", "Set of 4", "12 inch", "3-Pack"],
    "home_goods": ["Set of 2", "4pc", "3-Pack", "12 inch"],
    "cleaning": ["Set of 2", "3-Pack", "500ml", "1L"],
    "toys": ["Set of 2", "3-Pack", "4pc"],
    "bags": ["15 inch", "12 inch", "Size M"],
    "sports": ["2-Pack", "Size M", "Size L", "500ml"],
    "electronics": ["2-Pack", "1.5m", "10 inch", "3-Pack"],
    "home": ["Set of 2", "Set of 4", "12 inch"],
    "kitchen": ["1.5L", "2L", "500ml", "350ml", "Set of 2", "4pc"],
    "beauty": ["50ml", "30ml", "Set of 2", "3-Pack"],
    "apparel": ["Size M", "Size L", "Size S", "2-Pack"],
    "books": ["Set of 2", "3-Pack", "Hardcover"],
    "pet": ["Set of 2", "3-Pack", "Size M"],
}

MATERIALS_BY_CATEGORY = {
    "office": ["Plastic", "Metal", "Paper", "Leather"],
    "womenswear": ["Cotton", "Linen", "Silk", "Knit"],
    "pet_garden": ["Nylon", "Canvas", "Rubber", "Cotton"],
    "appliances": ["Steel", "Plastic", "Glass", "Silicone"],
    "home_decor": ["Cotton", "Velvet", "Ceramic", "Wooden"],
    "home_goods": ["Plastic", "Steel", "Bamboo", "Cotton"],
    "cleaning": ["Plastic", "Microfiber", "Steel", "Rubber"],
    "toys": ["Plastic", "Wood", "Plush", "Foam"],
    "bags": ["Leather", "Canvas", "Nylon", "Suede"],
    "sports": ["Nylon", "Rubber", "Mesh", "Foam"],
    "electronics": ["Silicone", "Aluminum", "ABS", "Nylon"],
    "home": ["Cotton", "Linen", "Ceramic", "Wooden"],
    "kitchen": ["Steel", "Ceramic", "Glass", "Silicone"],
    "beauty": ["Glass", "Plastic", "Silk", "Mineral"],
    "apparel": ["Cotton", "Wool", "Linen", "Fleece"],
    "books": ["Paper", "Leather", "Cloth", "Card"],
    "pet": ["Nylon", "Cotton", "Rubber", "Canvas"],
}

_TYPO_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def parse_title_typo_rate(value: Any, *, field: str = "typo_rate") -> float:
    """Return a typo probability in ``[0, 1]``.

    Args:
        value: Raw numeric value from a scenario key or CLI flag.
        field: Name inserted into the error message.

    Returns:
        Probability in ``[0, 1]``.

    Raises:
        ValueError: If ``value`` is missing, non-numeric, or outside ``[0, 1]``.
    """
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a float in [0, 1], got {value!r}") from exc
    if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ValueError(f"{field} must be a float in [0, 1], got {value!r}")
    return rate


def generate_title(
    category: str,
    rng: np.random.Generator,
    *,
    typo_rate: float = 0.0,
) -> str:
    """Build a deterministic marketplace-style product title.

    Layout is brand + optional attribute/material + category noun +
    optional spec, with a moderate chance of a promo token. ``typo_rate=0``
    skips the typo branch so the title stays clean.

    Args:
        category: Catalog category key or display name.
        rng: Seeded generator from ``derive_rng``.
        typo_rate: Probability of one character-level typo. ``0`` disables
            typos.

    Returns:
        A title of at most ``MAX_TITLE_LEN`` characters (always ≤ 200).

    Raises:
        ValueError: If ``typo_rate`` is outside ``[0, 1]``.
    """
    rate = parse_title_typo_rate(typo_rate, field="typo_rate")
    key = _normalize_category(category)
    nouns = CATEGORY_NOUNS.get(key, _FALLBACK_NOUNS)
    materials = MATERIALS_BY_CATEGORY.get(key, _GENERIC_MATERIALS)
    attributes = ATTRIBUTES_BY_CATEGORY.get(key, ATTRIBUTES)
    specs = SPECS_BY_CATEGORY.get(key, SPECS)

    parts: list[str] = []
    if float(rng.random()) < PROMO_CHANCE:
        parts.append(_pick(rng, PROMO_TOKENS))
    parts.append(_pick(rng, BRANDS))
    if float(rng.random()) < ATTRIBUTE_CHANCE:
        parts.append(_pick(rng, attributes))
    if float(rng.random()) < MATERIAL_CHANCE:
        parts.append(_pick(rng, materials))
    parts.append(_pick(rng, nouns).title())
    if float(rng.random()) < SPEC_CHANCE:
        parts.append(_pick(rng, specs))

    title = _clip_title(" ".join(parts), MAX_TITLE_LEN)
    # * Typos are off by default so the catalog stream stays readable.
    if rate > 0.0 and float(rng.random()) < rate:
        title = _apply_typo(title, rng)
    return title


def _normalize_category(category: str) -> str:
    text = str(category or "").strip().lower().replace("-", "_")
    return "_".join(text.split())


def _pick(rng: np.random.Generator, pool: list[str]) -> str:
    if not pool:
        raise ValueError("title word pool is empty")
    return pool[int(rng.integers(0, len(pool)))]


def _clip_title(title: str, limit: int) -> str:
    compact = " ".join(str(title).split())
    if len(compact) <= limit:
        return compact
    clipped = compact[:limit].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped


def _apply_typo(title: str, rng: np.random.Generator) -> str:
    """Apply one swap, drop, or substitute on an alphabetic character."""
    chars = list(title)
    letter_idx = [i for i, ch in enumerate(chars) if ch.isalpha()]
    if not letter_idx:
        return title
    kind = int(rng.integers(0, 3))
    if kind == 0:
        swap_idx = [i for i in letter_idx if i + 1 < len(chars) and chars[i + 1].isalpha()]
        if swap_idx:
            index = swap_idx[int(rng.integers(0, len(swap_idx)))]
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
            return "".join(chars)
        kind = 1
    if kind == 1 and len(chars) > 1:
        index = letter_idx[int(rng.integers(0, len(letter_idx)))]
        del chars[index]
        return "".join(chars)
    index = letter_idx[int(rng.integers(0, len(letter_idx)))]
    current = chars[index].lower()
    choices = [ch for ch in _TYPO_ALPHABET if ch != current]
    repl = choices[int(rng.integers(0, len(choices)))]
    chars[index] = repl.upper() if chars[index].isupper() else repl
    return "".join(chars)
