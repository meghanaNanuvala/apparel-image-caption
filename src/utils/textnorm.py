"""Text helpers shared by data prep, the caption rewrite, evaluation and the notebooks."""

import re

# Raw H&M captions look like "<pattern label> <colour> <description>", for example
# "all over pattern dark green calf-length skirt in ...".

PATTERN_LABELS = [
    "all over pattern", "application/3d", "argyle", "chambray", "check", "colour blocking",
    "contrast", "denim", "dot", "embroidery", "front print", "glittering/metallic", "hologram",
    "jacquard", "lace", "melange", "mesh", "metallic", "mixed solid/pattern", "neps",
    "other pattern", "other structure", "placement print", "sequin", "slub", "solid", "stripe",
    "transparent", "treatment", "unknown",
]
COLOUR_MODIFIERS = ["dark", "light", "other", "off", "greenish", "greyish", "yellowish"]
COLOUR_NAMES = [
    "beige", "black", "blue", "bronze/copper", "brown", "gold", "green", "grey", "khaki",
    "lilac purple", "lilac", "mole", "orange", "pink", "purple", "red", "silver", "transparent",
    "turquoise", "unknown", "white", "yellow", "other",
]

_PATTERN_RE = "|".join(re.escape(p) for p in sorted(PATTERN_LABELS, key=len, reverse=True))
_COLOUR_RE = (
    "(?:(?:" + "|".join(COLOUR_MODIFIERS) + r")\s+)?"
    "(?:" + "|".join(re.escape(c) for c in sorted(COLOUR_NAMES, key=len, reverse=True)) + ")"
)
_PREFIX_RE = re.compile(rf"^(?P<pattern>{_PATTERN_RE})\s+(?P<colour>{_COLOUR_RE})\s+(?P<body>.+)$")


def split_caption(raw: str) -> dict:
    """Split a raw H&M caption into pattern label, colour and description body.
    If it does not follow the usual layout, pattern and colour are None."""
    text = " ".join(raw.lower().split())
    match = _PREFIX_RE.match(text)
    if not match:
        return {"pattern": None, "colour": None, "body": text}
    return match.groupdict()


# Words for things a camera cannot see. Fabric types you can recognise by eye (denim, lace,
# knit, jersey) are fine; fibre content, feel, hidden parts and marketing claims are not.

NON_VISIBLE_TERMS = [
    # fibre content
    "cotton", "polyester", "viscose", "wool", "cashmere", "linen", "silk", "lyocell", "modal",
    "elastane", "polyamide", "nylon", "acrylic", "mohair", "alpaca", "merino", "tencel",
    "spandex", "rayon",
    # sustainability
    "organic", "recycled", "conscious", "sustainable",
    # hidden construction
    "lined", "unlined", "lining", "inside", "inner", "concealed", "hidden", "built-in",
    "filling", "feathers",
    # feel and function
    "soft", "softly", "stretch", "stretchy", "brushed", "breathable", "fast-drying", "quick-drying",
    "comfort", "comfortable", "denier", "functional", "sturdy", "lightweight", "airy",
    # fit labels and marketing
    "regular fit", "slim fit", "relaxed fit", "loose fit", "skinny fit", "muscle fit",
    "premium", "optimum", "flattering",
]
_NON_VISIBLE_RE = re.compile(
    r"(?<![a-z-])(" + "|".join(re.escape(t) for t in NON_VISIBLE_TERMS) + r")(?![a-z])",
    re.IGNORECASE,
)
_META_RE = re.compile(r"\b(image|photo|photograph|picture)\b", re.IGNORECASE)
_PERSON_RE = re.compile(
    r"\b(woman|women|man|men|girl|boy|lady|person|people|model|she|he|her|his)\b", re.IGNORECASE
)


def non_visible_terms(text: str) -> list[str]:
    return sorted({m.group(1).lower() for m in _NON_VISIBLE_RE.finditer(text)})


def mentions_person(text: str) -> bool:
    return bool(_PERSON_RE.search(text))


def mentions_image(text: str) -> bool:
    return bool(_META_RE.search(text))


HARD_MAX_CHARS = 140  # the 125 target is a guideline; beyond this a rewrite is rejected


def alt_text_issues(text: str) -> list[str]:
    """Reasons a text is not acceptable as alt-text. An empty list means it is fine."""
    issues = []
    stripped = text.strip()
    if not stripped:
        return ["empty"]
    if len(stripped) > HARD_MAX_CHARS:
        issues.append(f"too long ({len(stripped)} characters)")
    if "\n" in stripped:
        issues.append("more than one line")
    if not stripped.endswith("."):
        issues.append("does not end with a period")
    if not stripped[0].isupper():
        issues.append("does not start with a capital letter")
    hidden = non_visible_terms(stripped)
    if hidden:
        issues.append("mentions things that cannot be seen: " + ", ".join(hidden))
    if mentions_image(stripped):
        issues.append("mentions the image itself")
    return issues


# Pattern labels that must show up in the rewrite, with the words that count as saying so.
_PATTERN_WORDS = {
    "stripe": r"strip",
    "check": r"check|plaid|tartan|gingham",
    "dot": r"dot|spot",
    "all over pattern": r"pattern|print|floral|flower|leopard|animal|paisley|camouflage|zebra|snake",
}
# Product photos show the front, so a detail that exists only at the back is not visible.
_BACK_RE = re.compile(r"\b(?:at|on|down) the back\b|\bback pockets?\b|\bback of the\b", re.IGNORECASE)


def rewrite_issues(text: str, pattern: str | None = None) -> list[str]:
    """The alt-text checks, plus two for LLM rewrites: the pattern must be mentioned and
    details at the back of the garment must be gone."""
    issues = alt_text_issues(text)
    words = _PATTERN_WORDS.get(pattern or "")
    if words and not re.search(words, text, re.IGNORECASE):
        issues.append(f"does not say the garment is patterned (pattern label: {pattern})")
    if _BACK_RE.search(text):
        issues.append("describes the back of the garment, which the photo does not show")
    return issues


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps hyphenated words such as 'v-neck' together."""
    return _TOKEN_RE.findall(text.lower())


def tidy_output(text: str) -> str:
    """Clean up model output: collapse whitespace, strip stray quotes."""
    text = " ".join(text.split())
    return text.strip("\"'` ")
