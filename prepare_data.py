import csv
import html
import re
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from datasets import load_dataset

OUT_PATH = "data/newsroom_all_pairs.tsv"

def _insert_case_boundaries(text: str) -> str:
    """Insert space at camelCase and letter/digit boundaries."""
    # Insert space between lowercase/number followed by uppercase (e.g., "outsidersintheSixties" -> "outsiders in the Sixties")
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    # Insert space between letters and digits (e.g., "June12" -> "June 12")
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    # Insert space between digit and letter, EXCEPT for:
    # - ordinal suffixes: st, nd, rd, th
    # - decade suffix: s (as in 80s, 90s)
    text = re.sub(r'(\d)(?!(?:st|nd|rd|th|s)\b)([A-Za-z])', r'\1 \2', text)
    return text

def _conservative_segment(token: str):
    # Optional fallback segmentation for very long alphabetic tokens.
    # Tries to split on common English short words and common prefixes/suffixes.
    # This is conservative and only applied to tokens >= 12 chars.
    if len(token) < 12 or not token.isalpha():
        return token
    # common small words to try splitting on
    small_words = ["the", "and", "for", "with", "that", "this", "they", "their", "from", "when", "which", "what", "were", "have", "been", "about", "into", "over", "under", "after", "before", "while", "where", "there", "these", "those"]
    token_lower = token.lower()
    for w in small_words:
        idx = token_lower.find(w)
        if idx > 3 and idx < len(token) - 3:
            # split around the small word, keep original casing roughly
            left = token[:idx]
            mid = token[idx:idx+len(w)]
            right = token[idx+len(w):]
            parts = []
            if left: parts.append(left)
            parts.append(mid)
            if right: parts.append(right)
            return " ".join(parts)
    # fallback: split into two roughly equal parts (very conservative)
    mid = len(token) // 2
    return token[:mid] + " " + token[mid:]

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")
_CTRL_BYTES_RE = re.compile(r"[\x80-\x9f]")
_WS_RE = re.compile(r"\s+")


def _normalize_unicode_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("‟", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("‚", "'")
        .replace("‛", "'")
        .replace("``", '"')
    )


def _normalize_newsroom_quote_markers(text: str) -> str:
        """Normalize newsroom-style doubled-quote markers into standard quotes.

        Many sources use doubled quotes as open/close markers with inconsistent spacing:
            - : ""I ...
            - , ""she said
            - ""hold his nose ""and

        This function processes "" markers left-to-right with alternating open/close
        logic to properly handle pairs and maintain correct spacing.
        """

        # First collapse triple+ quotes to doubled
        text = re.sub(r'"{3,}', '""', text)
        
        # Process "" occurrences left-to-right with alternating open/close
        parts = text.split('""')
        if len(parts) == 1:
            return text
        
        result = parts[0]
        is_opening = True  # First "" after text is opening
        for part in parts[1:]:
            if is_opening:
                # Opening quote: strip trailing space from prev, strip leading space from part
                result = result.rstrip() + ' "'
                part = part.lstrip()
                result += part
            else:
                # Closing quote: strip trailing space from prev, add quote + space
                result = result.rstrip() + '" '
                part = part.lstrip()
                result += part
            is_opening = not is_opening
        
        return result


def _fix_split_domain_tokens(text: str) -> str:
    # Join domains that were split by earlier case-boundary or punctuation rules:
    # "www. Instagram. Com" -> "www.instagram.com"
    def _join_domain(m: re.Match) -> str:
        a, b, c = m.group(1), m.group(2), m.group(3)
        return f"{a}.{b}.{c}".lower()

    text = re.sub(
        r"\b(www)\.\s*([A-Za-z0-9-]+)\.\s*(com|org|net|edu|gov|io)\b",
        _join_domain, text, flags=re.IGNORECASE
    )
    return text


def _normalize_whitespace_and_artifacts(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CTRL_BYTES_RE.sub("", text)
    # Common mojibake artifacts from CP1252/UTF-8 confusion
    text = text.replace("Â", "")
    return text


def _normalize_quotes_and_spacing(text: str) -> str:
    text = _normalize_unicode_quotes(text)

    # Normalize newsroom doubled-quote markers into standard quotes.
    text = _normalize_newsroom_quote_markers(text)

    # Collapse repeated same-quote runs: """" -> ", ''' -> '
    text = re.sub(r'("{2,})', '"', text)
    text = re.sub(r"('{2,})", "'", text)

    # Collapse mixed adjacent quotes into a single double quote
    text = re.sub(r'(["\'])(["\'])+', '"', text)

    # Collapse whitespace
    text = _WS_RE.sub(" ", text)

    return text


def _normalize_punctuation_spacing(text: str) -> str:
    # No space before punctuation; single space after when followed by a word/quote
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # Add space after punctuation before word, but NOT for:
    # - decimals (digit.digit)
    # - times (digit:digit)
    text = re.sub(r'([,!?;])(?=["\']?\w)', r"\1 ", text)
    # For colon: only add space if NOT followed by a digit (preserve times like 2:22)
    text = re.sub(r':(?=["\']?[A-Za-z])', r": ", text)
    # For period: add space if followed by uppercase letter (sentence boundary)
    # but NOT for lowercase (likely domain or abbreviation)
    text = re.sub(r'\.(?=["\']?[A-Z])', r". ", text)
    return text


def _fix_abbrev_dots(text: str) -> str:
    # "I. D." -> "ID"
    return re.sub(
        r"\b([A-Za-z])\.\s*([A-Za-z])\.\b",
        lambda m: m.group(1) + m.group(2),
        text,
    )


def _fix_common_abbrev_sequences(text: str) -> str:
    """Fix common multi-letter abbreviations with dots and spaces."""
    # Use lookahead to ensure we don't capture into the next word
    text = re.sub(r"\bU\.\s*S\.(?=\s|$)", "US", text)
    text = re.sub(r"\bU\.\s*K\.(?=\s|$)", "UK", text)
    text = re.sub(r"\bE\.\s*U\.(?=\s|$)", "EU", text)
    text = re.sub(r"\bA\.\s*M\.(?=\s|$)", "AM", text)
    text = re.sub(r"\bP\.\s*M\.(?=\s|$)", "PM", text)
    return text


def _optional_segment(text: str) -> str:
    try:
        from wordsegment import load, segment

        load()
        out = []
        for tok in text.split(" "):
            if len(tok) >= 12 and tok.isalpha():
                seg = segment(tok)
                out.extend(seg if len(seg) > 1 else [tok])
            else:
                out.append(tok)
        return " ".join(out)
    except Exception:
        return " ".join(
            _conservative_segment(tok) if (len(tok) >= 12 and tok.isalpha()) else tok
            for tok in text.split(" ")
        )


def cleanup_text(
    text: Optional[str],
    *,
    segment_long_tokens: bool = False,
    capitalize_sentences: bool = True,
) -> Optional[str]:
    """Single, comprehensive cleanup without duplicated logic."""
    if not text:
        return text

    text = _normalize_whitespace_and_artifacts(text)
    text = _normalize_quotes_and_spacing(text)

    # & spacing
    text = re.sub(r"\s*&\s*", " & ", text)

    # hyphens: keep single hyphen, remove surrounding spaces
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"-{2,}", "-", text)

    # Insert case boundaries, but protect apostrophe patterns like '80s
    # Temporarily replace '(digit) with a sentinel
    text = re.sub(r"'(\d)", r"⟪APO\1⟫", text)
    text = _insert_case_boundaries(text)
    text = re.sub(r"⟪APO(\d)⟫", r"'\1", text)
    
    # Fix spacing in ordinal suffixes AFTER case boundaries: "18 th" -> "18th"
    text = re.sub(r"\b(\d+)\s+(st|nd|rd|th)\b", r"\1\2", text, flags=re.IGNORECASE)

    # Fix time spacing: "1: 74" -> "1:74" ; "2: 22" -> "2:22"
    text = re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\b", r"\1:\2", text)

    # Fix decimal number spacing: "91. 2" -> "91.2"
    text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)

    text = _fix_abbrev_dots(text)
    text = _fix_common_abbrev_sequences(text)
    text = _normalize_punctuation_spacing(text)

    # Remove stray quote runs that sometimes remain after other normalizations.
    # Examples: Blames""bias -> Blames "bias ; GOP.""" -> GOP."
    text = re.sub(r'"{2,}', '"', text)
    text = re.sub(r"'{2,}", "'", text)

    # Remove spaces after an opening quote when it starts a sentence/segment.
    # Example: : ""I feel... -> : "I feel...
    text = re.sub(r'(^|[\s:(\[{])"\s+', r'\1"', text)

    # Ensure spacing when a word is immediately followed by a quote: and""thereby -> and "thereby
    text = re.sub(r'(\w)"(?=\w)', r'\1 "', text)

    # Collapse triple+ quotes that appear before punctuation/whitespace:
    # """When -> "When
    text = re.sub(r'"{3,}(?=\w)', '"', text)

    # Fix split domains like "www. Instagram. Com"
    text = _fix_split_domain_tokens(text)

    # Final spacing pass after aggressive quote normalization
    text = _normalize_punctuation_spacing(text)
    text = _WS_RE.sub(" ", text).strip()

    if segment_long_tokens and text:
        text = _optional_segment(text)
        text = _WS_RE.sub(" ", text).strip()

    if capitalize_sentences and text:
        text = text[0].upper() + text[1:]
        text = re.sub(
            r"([.!?]\s+)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )

    # ABSOLUTE FINAL GUARD: collapse any stray doubled quotes that somehow survived.
    if text:
        # Collapse any remaining doubled quotes
        text = re.sub(r'"{2,}', '"', text)
        text = re.sub(r"'{2,}", "'", text)
        
        # Handle word"word cases = ambiguous, treat as opening quote
        text = re.sub(r'(\w)"(\w)', r'\1 "\2', text)
        
        # Final whitespace/punctuation cleanup
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = _WS_RE.sub(" ", text).strip()

    return text

def load_all_pairs():
    ds = load_dataset("LogeshChandran/newsroom", split="train", streaming=True)
    for item in ds:
        summary = cleanup_text(item["summary"])
        article = cleanup_text(item["text"].split("\n")[0])
        if len(summary.split()) > 5 and len(article.split()) > 5:
            yield (summary, article)

def save_pairs(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for s, a in tqdm(load_all_pairs(), desc="Writing pairs", unit="pair"):
            writer.writerow([s, a])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NEWSROOM contrastive pairs")
    parser.add_argument("--out", type=str, default=OUT_PATH, help="Output TSV path")
    args = parser.parse_args()

    print("Writing all NEWSROOM pairs to TSV...")
    save_pairs(args.out)
    print(f"Done. Output saved to {args.out}")

