"""Test `jwst_rogue_path_tool.utils` module.

Authors
-------
    - Mees Fix
"""

import pytest

from jwst_rogue_path_tool.utils import absolute_magnitude, get_pivot_wavelength


@pytest.mark.parametrize(
    "magnitude, expected",
    [
        (7.9, -2.2440677282261037),
        (15.6, -2.9828114958861542),
    ],
)
def test_absolute_magnitude(magnitude, expected):
    pivot_wavelength = absolute_magnitude(magnitude)
    pivot_wavelength == expected


@pytest.mark.parametrize(
    "pupil, filter, expected",
    [
        ("CLEAR", "F070W", 0.704),
        ("F070W", "F140M", 0.704),
        ("CLEAR", "F162M", 1.626),
        ("F200W", "F210M", 1.990),
    ],
)
def test_get_pivot_wavelength(pupil, filter, expected):
    pivot_wavelength = get_pivot_wavelength(pupil, filter)
    pivot_wavelength == expected

