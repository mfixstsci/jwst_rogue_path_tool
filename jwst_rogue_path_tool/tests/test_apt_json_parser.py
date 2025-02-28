"""Unit tests for apt_json_parser"""

import pytest

from pathlib import Path

from jwst_rogue_path_tool.program_data_parser import aptJsonFile
from jwst_rogue_path_tool.constants import PROJECT_DIRNAME

JSON_PATH = Path(PROJECT_DIRNAME) / "data" / "APT_test_4RPtool.json"


def test_json_exists(json_path=JSON_PATH):
    """Assert json test file exists"""
    assert json_path.exists()


def test_load_json(json_path=JSON_PATH):
    program = aptJsonFile(json_path)
    assert program is not None


@pytest.mark.parametrize(
    "tablename",
    [
        "exposures",
        "visit",
        "observation",
        "program",
        "fixed_target",
    ],
)
def test_tables_exist(tablename, json_path=JSON_PATH):
    program = aptJsonFile(json_path)
    assert tablename in program.tablenames
