from importlib.resources import files
from pathlib import Path
from typing import Literal

PromptName = Literal["relevance", "analysis", "critic"]


def load_prompt(name: PromptName, version: str) -> str:
    """Use editable working-directory prompts, otherwise bundled wheel resources."""
    override = Path("prompts") / f"{name}.md"
    body = (
        override.read_text(encoding="utf-8")
        if override.is_file()
        else files("research_radar.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")
    )
    return f"Prompt: {name}; version: {version}\n\n{body}"
