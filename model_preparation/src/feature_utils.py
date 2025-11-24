"""
Shared helpers for keeping feature names consistent across the training pipeline.
"""

def normalize_feature_name(name: str) -> str:
    """
    Normalize a feature/column name so it is compatible with the model:
    - Trim surrounding whitespace
    - Replace spaces, dots, slashes, and hyphens with underscores
    - Remove any remaining non-alphanumeric/underscore characters
    - Convert to lowercase
    """
    if name is None:
        return ""

    normalized = (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

    cleaned = []
    for ch in normalized:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)

    return "".join(cleaned).lower()


