"""Trading personas: the strategies that produce trade intents."""

from spintrader.agents.personas.baseline_trend import (
    BaselineTrendAgent, TrendReading,
)
from spintrader.agents.personas.hedge import HedgeEnsembleAgent, HedgeReading
from spintrader.agents.personas.markov_chain import (
    HighOrderMarkovAgent, MarkovReading,
)
from spintrader.agents.personas.mean_reversion import (
    MeanReversionAgent, ReversionReading,
)
from spintrader.agents.personas.regime_switch import (
    RegimeSwitchingAgent, RegimeSwitchReading,
)
from spintrader.agents.personas.sentiment import SentimentAgent, SentimentReading

__all__ = [
    "BaselineTrendAgent", "HedgeEnsembleAgent", "HedgeReading",
    "HighOrderMarkovAgent", "MarkovReading", "MeanReversionAgent",
    "RegimeSwitchReading", "RegimeSwitchingAgent", "ReversionReading",
    "SentimentAgent", "SentimentReading", "TrendReading",
]
