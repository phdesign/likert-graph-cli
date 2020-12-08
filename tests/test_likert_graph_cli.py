from click.testing import CliRunner
from likert_graph_cli import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert result.output == 'Hello Peter!\n'