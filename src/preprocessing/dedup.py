import re

GREETING = re.compile(
    r"^\s*(hello team|hi there|good morning|good afternoon|dear support|"
    r"please help|hello|hey|hi)\b[\s,\.!]*",
    re.IGNORECASE,
)

TAIL = re.compile(
    r"[\s,\.!]*\b(thanks in advance|thank you so much|thank you|thanks|"
    r"please advise|appreciate any help|any help appreciated|"
    r"this is time sensitive|kind regards|best regards|regards|best)\b"
    r"[\s,\.!]*$",
    re.IGNORECASE,
)

NUMBER = re.compile(r"[\$€£]?\s?\d+(?:[\.,]\d+)?")

ASSET = re.compile(
    r"\b(btc|bitcoin|eth|ethereum|usdt|usdc|sol|solana|polygon|matic|"
    r"ada|cardano|xrp|doge)\b",
    re.IGNORECASE,
)

WHITESPACE = re.compile(r"\s+")


def template_key(text):
    key = str(text).lower().strip()
    while True:
        stripped = TAIL.sub("", GREETING.sub("", key)).strip()
        if stripped == key:
            break
        key = stripped
    key = NUMBER.sub("NUM", key)
    key = ASSET.sub("ASSET", key)
    key = WHITESPACE.sub(" ", key)
    return key.strip(" .,!?")