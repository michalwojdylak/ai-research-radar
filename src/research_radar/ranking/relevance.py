import json
import re

from research_radar.analysis.llm import LLMProvider, cached_generate
from research_radar.config import Settings
from research_radar.models.paper import Paper
from research_radar.models.ranking import QualityDimensions, RelevanceResult
from research_radar.prompts import load_prompt

AI_CATEGORIES = frozenset({"cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "cs.RO", "stat.ML"})
AI_TERMS = re.compile(
    r"\b(?:artificial intelligence|machine learning|deep learning|neural networks?|"
    r"large language models?|language models?|LLMs?|transformers?|reinforcement learning|"
    r"computer vision|natural language processing|generative models?|diffusion models?|"
    r"foundation models?|representation learning|AI|multimodal|fine[- ]tuning|"
    r"self[- ]supervised learning|supervised learning)\b",
    re.IGNORECASE,
)


def deterministic_prefilter(paper: Paper) -> bool:
    return bool(
        AI_CATEGORIES.intersection(paper.categories)
        or AI_TERMS.search(f"{paper.title}\n{paper.abstract}")
    )


class RelevanceFilter:
    def __init__(self, settings: Settings, provider: LLMProvider | None = None) -> None:
        self.settings = settings
        self.provider = provider

    async def classify(self, paper: Paper) -> RelevanceResult | None:
        if not deterministic_prefilter(paper):
            return None
        if self.provider is None:
            result = RelevanceResult(
                relevance_score=70,
                confidence=0.35,
                classifier="deterministic",
                reason="Category or keyword prefilter only; relevance is not AI-confirmed.",
            )
        else:
            result = await cached_generate(
                self.settings,
                self.provider,
                system_prompt=load_prompt("relevance", self.settings.relevance_prompt_version),
                user_prompt=json.dumps(
                    {"paper_id": paper.id, "title": paper.title, "abstract": paper.abstract},
                    ensure_ascii=False,
                ),
                schema=RelevanceResult,
                model=self.settings.relevance_model,
                paper_id=paper.id,
                stage="relevance",
                version=self.settings.relevance_prompt_version,
            )
            # Title and abstract cannot substantiate venue or author/lab prestige.
            quality = QualityDimensions.model_validate(
                {**result.quality.model_dump(), "venue": None, "author_lab": None}
            )
            result = result.model_copy(update={"classifier": "llm", "quality": quality})
        return result if result.relevance_score >= self.settings.relevance_threshold else None
