import pytest
from click.testing import CliRunner
from likert_graph_cli import main, blend_colors


def test_main_should_require_input():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 2
    assert result.output == '''Usage: main [OPTIONS] INPUT OUTPUT
Try 'main -h' for help.

Error: Missing argument 'INPUT'.
'''


@pytest.mark.parametrize("steps,expected", [
    (3, ['#3e6386', '#c7cdd0', '#e26d34']),
    (4, ['#3e6386', '#7f9bae', '#94ce88', '#e26d34']),
    (5, ['#3e6386', '#7f9bae', '#c7cdd0', '#94ce88', '#e26d34']),
])
def test_blend_colors_should_return_request_number_of_colors(steps, expected):
    colors = blend_colors(steps)
    assert colors == expected
