from __future__ import absolute_import

import unittest

from laia.hooks import action


@action
def action_with_arg(v):
    return v


@action
def action_with_kwargs(v="Y"):
    return v


@action
def empty_action():
    return True


class ActionTest(unittest.TestCase):
    def test_normal_usage(self):
        self.assertEqual("X", action_with_arg("X"))
        self.assertEqual("X", action_with_kwargs(v="X"))
        self.assertTrue(empty_action())

    def test_with_extra_args(self):
        self.assertEqual("X", action_with_arg("X", "Y"))
        self.assertTrue(empty_action("X", "Y"))

    def test_with_extra_kwargs(self):
        self.assertTrue(empty_action("X", v="Y"))
        self.assertEqual("X", action_with_arg("X", v2="Y"))
        self.assertEqual("X", action_with_kwargs(v="X", v2="Y"))

    def test_multiple_values_for_same_argument(self):
        with self.assertRaises(TypeError):
            action_with_arg("X", v="Y")

    def test_with_no_args(self):
        with self.assertRaises(TypeError):
            action_with_arg()
        self.assertEqual("Y", action_with_kwargs())


if __name__ == "__main__":
    unittest.main()
