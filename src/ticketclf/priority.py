import yaml

from .config import resolve

VALID_LEVELS = ("High", "Medium", "Low")


def _norm(label: str) -> str:
    
    return label.strip().lower().rstrip("?")


class PriorityMapper:
    

    def __init__(self, groups: dict, default: str = "Medium"):
        if default not in VALID_LEVELS:
            raise ValueError(f"Invalid default priority: {default}")
        self.default = default
        self.mapping = {}
        for level, intents in groups.items():
            if level not in VALID_LEVELS:
                raise ValueError(f"Invalid priority level: {level}")
            for intent in intents:
                key = _norm(intent)
                if key in self.mapping:
                    raise ValueError(f"Duplicate intent in priority map: {intent}")
                self.mapping[key] = level

    @classmethod
    def from_yaml(cls, path="configs/priority_map.yaml"):
        with open(resolve(path), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data["priorities"], data.get("default", "Medium"))

    def get(self, intent: str) -> str:
        return self.mapping.get(_norm(intent), self.default)

    def validate(self, labels) -> dict:
        label_keys = {_norm(l) for l in labels}
        return {
            "missing": sorted(label_keys - set(self.mapping)),
            "unknown": sorted(set(self.mapping) - label_keys),
        }