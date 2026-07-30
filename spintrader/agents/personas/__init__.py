"""Trading personas: the strategies that produce trade intents."""

from spintrader.agents.personas.baseline_trend import (
    BaselineTrendAgent, TrendReading,
)
from spintrader.agents.personas.mean_reversion import (
    MeanReversionAgent, ReversionReading,
)

__all__ = [
    "BaselineTrendAgent", "MeanReversionAgent", "ReversionReading", "TrendReading",
]
