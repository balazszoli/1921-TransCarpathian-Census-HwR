import pytest

import numpy as np

from src.table_recognizer.table_utils import continue_lines


def test_continue_lines__given_one_vertical_line__return_one_vertical_which_start_from_zero_to_img_height():
    horizontal, vertical = continue_lines(np.array([[[50, 60, 50, 30]]]), 100, 80)
    assert horizontal == []
    assert vertical == [[50, 0, 50, 100]]


def test_continue_lines__given_one_horizontal_line__return_one_horizontal_which_start_from_zero_to_img_width():
    horizontal, vertical = continue_lines(np.array([[[20, 60, 70, 60]]]), 100, 80)
    assert vertical == []
    assert horizontal == [[0, 60, 80, 60]]