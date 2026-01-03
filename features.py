import re
import numpy as np

KEYWORDS = [
    "dp", "graph", "tree", "greedy", "recursion",
    "binary search", "dfs", "bfs", "mod", "segment tree"
]

def text_length(text):
    return len(text)

def math_symbol_count(text):
    return len(re.findall(r"[=<>+\-*/^]", text))

def keyword_features(text):
    text = text.lower()
    return [text.count(k) for k in KEYWORDS]

def handcrafted_features(texts):
    features = []
    for t in texts:
        row = [
            text_length(t),
            math_symbol_count(t),
            *keyword_features(t)
        ]
        features.append(row)
    return np.array(features)
