import os
import re
import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# ==================== CONFIG ====================
# Folder containing chat files in normalized format:
#   "[dd.mm.yy, HH:MM] Name: Message"
# with optional "[CONVERSATION_BREAK]" marker lines.
RAW_TXT_FOLDER = "data/unformatted_text"

# Folder for final JSON files in ShareGPT-style format
OUTPUT_JSON_FOLDER = "data/preprocessed"

# --- Burst merging ---
# Consecutive messages from the same sender within this many minutes
# are merged into a single turn (joined with newlines).
BURST_GAP_MINUTES = 10

# --- Session splitting ---
# A silence longer than this starts a new conversation.
SESSION_GAP_HOURS = 3

# --- Conversation size (in merged turns) ---
MIN_TURNS_PER_CONV = 4          # drop conversations shorter than this
MAX_TURNS_PER_CONV = 30         # split longer conversations at human boundaries

# --- Low-information filtering ---
# A conversation whose *average* gpt-turn length (chars) is below this
# is considered "pure ack" ("ok", "cool", emoji...) and only a fraction is kept.
ACK_AVG_CHAR_THRESHOLD = 25
ACK_KEEP_FRACTION = 0.25
RANDOM_SEED = 42

# --- Per-contact balancing ---
# Max conversations kept per contact (random sample). Prevents one chat
# from dominating the dataset. None = no cap.
CAP_CONVS_PER_CONTACT = {
    "Margheritadidouxe": 700,
}

# --- Persona system prompt ---
# If this file exists, its content is prepended to every conversation
# as a "system" turn. Use the SAME prompt at inference time.
SYSTEM_PROMPT_FILE = "data/persona/system_prompt.txt"
# =================================================


# "[12.03.24, 17:42] Name: Message"  (seconds optional)
LINE_RE = re.compile(
    r'^\[(\d{2})\.(\d{2})\.(\d{2}), (\d{2}):(\d{2})(?::\d{2})?\] (.+?): (.*)$'
)

BREAK_MARKER = "[CONVERSATION_BREAK]"

URL_RE = re.compile(r'https?://\S+|www\.\S+')

# Messages that are pure WhatsApp artifacts -> dropped entirely
ARTIFACT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'<media omitted>',
        r'\b(image|video|audio|sticker|gif|document) omitted\b',
        r'\(file attached\)',
        r'this message was deleted',
        r'you deleted this message',
        r'missed (voice|video) call',
        r'live location',
        r'^location:\s*https?',
        r'\.vcf\b',
        r'\[REDACTED',
    ]
]

EDITED_TAG_RE = re.compile(r'<this message was edited>', re.IGNORECASE)

# Letters/digits in any script -> used to detect emoji/punctuation-only messages
WORDCHAR_RE = re.compile(r'\w', re.UNICODE)


def clean_text(text):
    """
    Clean a single raw message. Returns "" if the message should be dropped.
    """
    # Strip invisible direction marks WhatsApp inserts around attachments
    text = text.replace('‎', '').replace('‏', '').strip()

    if not text:
        return ""

    for pat in ARTIFACT_PATTERNS:
        if pat.search(text):
            return ""

    # Keep edited messages but drop the "<This message was edited>" tag
    text = EDITED_TAG_RE.sub('', text).strip()

    # Drop messages that are nothing but a link
    if not URL_RE.sub('', text).strip():
        return ""

    return text


def parse_chat_file(path):
    """
    Parse a normalized chat file into a list of "segments".
    Each segment (separated by [CONVERSATION_BREAK]) is a list of messages:
        [{"dt": datetime, "name": str, "text": str}, ...]
    Continuation lines (no timestamp) are attached to the previous message.
    """
    segments = [[]]

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if BREAK_MARKER in line:
                if segments[-1]:
                    segments.append([])
                continue

            m = LINE_RE.match(line)
            if m:
                d, mo, y, hh, mm, name, text = m.groups()
                dt = datetime(2000 + int(y), int(mo), int(d), int(hh), int(mm))
                text = clean_text(text)
                if text:
                    segments[-1].append(
                        {"dt": dt, "name": name.strip(), "text": text}
                    )
            elif line.strip() and segments[-1]:
                # Continuation of a multiline message
                extra = clean_text(line)
                if extra:
                    msg = segments[-1][-1]
                    msg["text"] = f'{msg["text"]}\n{extra}'

    return [s for s in segments if s]


def merge_bursts(messages):
    """
    Merge consecutive messages from the same sender sent within
    BURST_GAP_MINUTES into single turns. Returns:
        [{"dt": datetime, "name": str, "text": str}, ...]
    """
    merged = []
    gap = timedelta(minutes=BURST_GAP_MINUTES)

    for msg in messages:
        if (
            merged
            and msg["name"] == merged[-1]["name"]
            and msg["dt"] - merged[-1]["dt"] <= gap
        ):
            merged[-1]["text"] += "\n" + msg["text"]
            merged[-1]["dt"] = msg["dt"]
        else:
            merged.append(dict(msg))

    return merged


def split_sessions(messages):
    """
    Split a list of merged turns into sessions wherever the time gap
    exceeds SESSION_GAP_HOURS.
    """
    if not messages:
        return []

    sessions = [[messages[0]]]
    gap = timedelta(hours=SESSION_GAP_HOURS)

    for prev, cur in zip(messages, messages[1:]):
        if cur["dt"] - prev["dt"] > gap:
            sessions.append([])
        sessions[-1].append(cur)

    return sessions


def to_sharegpt(session, whatsapp_name):
    """
    Convert a session (list of merged turns) into ShareGPT turns:
        [{"from": "human"/"gpt", "value": ...}, ...]
    Enforces: starts with "human", ends with "gpt".
    Consecutive same-role turns (possible after burst merging across a
    session boundary of >10min but <3h) are merged.
    """
    turns = []
    for msg in session:
        role = "gpt" if whatsapp_name in msg["name"] else "human"
        if turns and turns[-1]["from"] == role:
            turns[-1]["value"] += "\n" + msg["text"]
        else:
            turns.append({"from": role, "value": msg["text"]})

    # Must start with the other person...
    while turns and turns[0]["from"] == "gpt":
        turns.pop(0)
    # ...and end with you (a trailing human turn carries no training signal)
    while turns and turns[-1]["from"] == "human":
        turns.pop()

    return turns


def chunk_turns(turns):
    """
    Split long conversations into chunks of at most MAX_TURNS_PER_CONV,
    always starting on a "human" turn and ending on a "gpt" turn.
    """
    chunks = []
    for i in range(0, len(turns), MAX_TURNS_PER_CONV):
        chunk = turns[i:i + MAX_TURNS_PER_CONV]
        while chunk and chunk[0]["from"] == "gpt":
            chunk.pop(0)
        while chunk and chunk[-1]["from"] == "human":
            chunk.pop()
        if len(chunk) >= MIN_TURNS_PER_CONV:
            chunks.append(chunk)
    return chunks


def avg_gpt_length(turns):
    lengths = [len(t["value"]) for t in turns if t["from"] == "gpt"]
    return sum(lengths) / len(lengths) if lengths else 0.0


def is_ack_only(text):
    """True if a message contains no letters/digits at all (pure emoji etc.)."""
    return not WORDCHAR_RE.search(text)


def process_file(path, whatsapp_name):
    """
    Full pipeline for one chat file. Returns a list of conversations
    (each a list of ShareGPT turns).
    """
    conversations = []

    for segment in parse_chat_file(path):
        merged = merge_bursts(segment)
        for session in split_sessions(merged):
            turns = to_sharegpt(session, whatsapp_name)
            # Drop a conversation whose final gpt turn is emoji-only:
            # we don't want "👌🏼👌🏼" as the last training target.
            while (
                len(turns) >= 2
                and turns[-1]["from"] == "gpt"
                and is_ack_only(turns[-1]["value"])
            ):
                turns = turns[:-2] if turns[-2]["from"] == "human" else turns[:-1]
            if len(turns) >= MIN_TURNS_PER_CONV:
                conversations.extend(chunk_turns(turns))

    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="WhatsApp chats -> cleaned, burst-merged ShareGPT JSON."
    )
    parser.add_argument(
        "whatsapp_name",
        help="Your WhatsApp display name (your messages become 'gpt')."
    )
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)
    os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)

    # Clear previous output so stale files from earlier runs can't linger
    # (but keep synthetic conversations generated by generate_synthetic.py)
    for old in Path(OUTPUT_JSON_FOLDER).glob("*.json"):
        if not old.name.startswith("Synthetic_"):
            old.unlink()

    system_prompt = None
    if os.path.exists(SYSTEM_PROMPT_FILE):
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        print(f"[OK] Prepending system prompt from {SYSTEM_PROMPT_FILE}")

    txt_files = sorted(Path(RAW_TXT_FOLDER).glob("*.txt"))
    if not txt_files:
        print(f"[WARN] No .txt files found in {RAW_TXT_FOLDER}")
        return

    total_kept, total_dropped_ack = 0, 0
    stats_rows = []

    for path in txt_files:
        conversations = process_file(path, args.whatsapp_name)

        kept = []
        dropped_ack = 0
        for turns in conversations:
            if avg_gpt_length(turns) < ACK_AVG_CHAR_THRESHOLD:
                if rng.random() < ACK_KEEP_FRACTION:
                    kept.append(turns)
                else:
                    dropped_ack += 1
            else:
                kept.append(turns)

        contact = path.stem.replace("WhatsApp Chat with ", "")

        cap = CAP_CONVS_PER_CONTACT.get(contact)
        if cap is not None and len(kept) > cap:
            kept = rng.sample(kept, cap)

        for i, turns in enumerate(kept):
            if system_prompt:
                turns = [{"from": "system", "value": system_prompt}] + turns
            out_path = Path(OUTPUT_JSON_FOLDER) / f"{contact}_{i:04d}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"conversations": turns}, f,
                          ensure_ascii=False, indent=4)

        gpt_avg = (
            sum(avg_gpt_length(t) for t in kept) / len(kept) if kept else 0.0
        )
        stats_rows.append((contact, len(kept), dropped_ack, gpt_avg))
        total_kept += len(kept)
        total_dropped_ack += dropped_ack

    print(f"\n{'contact':<25}{'convs':>7}{'ack-dropped':>13}{'avg gpt chars':>15}")
    for contact, n, d, avg in stats_rows:
        print(f"{contact:<25}{n:>7}{d:>13}{avg:>15.1f}")
    print(
        f"\nKept {total_kept} conversations "
        f"({total_dropped_ack} ack-only conversations dropped). "
        f"Output: {OUTPUT_JSON_FOLDER}"
    )


if __name__ == "__main__":
    main()