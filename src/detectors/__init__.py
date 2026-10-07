"""Three detectors with one interface: fit(orders), score(orders), explain(orders).

``score`` returns one number per order, higher = more suspicious.
``explain`` returns one short human-readable reason per order.
"""
from src.detectors.iforest import IsolationForestDetector
from src.detectors.lgbm import LightGBMDetector
from src.detectors.rules import RuleDetector

__all__ = ["RuleDetector", "IsolationForestDetector", "LightGBMDetector"]
