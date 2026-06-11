import re

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_semicolon_list(value):
    return [item.strip() for item in str(value).split(";") if item.strip()]

def split_pipe_list(value):
    return [item.strip() for item in str(value).split("|") if item.strip()]

def matches_any_pattern(text, patterns):
    text = normalize_text(text)
    return any(re.search(pattern, text) for pattern in patterns)

def normalize_name(name):
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name