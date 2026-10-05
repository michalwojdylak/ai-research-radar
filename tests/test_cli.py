import pytest

from research_radar.cli import parser


def test_default_command_is_full_run() -> None:
    assert parser().parse_args([]).command == "run"


@pytest.mark.parametrize("stage", ["discover", "enrich", "rank", "analyze", "generate", "validate"])
def test_independent_commands(stage: str) -> None:
    assert parser().parse_args([stage]).command == stage


def test_run_overrides() -> None:
    options = parser().parse_args(["run", "--days", "7", "--top", "10", "--dry-run"])
    assert options.days == 7
    assert options.top == 10
    assert options.dry_run


def test_conflicting_write_flags_are_rejected() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(["--dry-run", "--no-dry-run"])


def test_legacy_root_options_before_command_remain_supported() -> None:
    options = parser().parse_args(["--days", "7", "--top", "10", "--dry-run", "discover"])
    assert options.command == "discover"
    assert options.days == 7
    assert options.top == 10
    assert options.dry_run
