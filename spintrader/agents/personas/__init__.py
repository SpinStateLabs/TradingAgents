"""Trading personas: the strategies that produce trade intents."""

from spintrader.agents.personas.baseline_trend import (
    BaselineTrendAgent, TrendReading,
)

__all__ = ["BaselineTrendAgent", "TrendReading"]
