

import unittest

from laia.hooks import Hook, action


@action
def action_with_arg(v):
    return v


@action
def action_with_args(v1, v2):
    return v1 + v2


@action
def action_with_kwargs(v="Y"):
    return v


@action
def empty_action():
    return True


# Mock conditions
def T():
    return True


def F():
    return False


class HookTest(unittest.TestCase):
    def test_with_args(self):
        self.assertEqual("X", Hook(T, action_with_arg, "X")())
        self.assertEqual("X", Hook(T, action_with_arg)("X"))

        self.assertEqual("XY", Hook(T, action_with_args, "X", "Y")())
        self.assertEqual("XY", Hook(T, action_with_args)("X", "Y"))

    def test_with_kwargs(self):
        self.assertEqual("X", Hook(T, action_with_kwargs, v="X")())
        self.assertEqual("X", Hook(T, action_with_kwargs)(v="X"))

    def test_with_extra_args(self):
        self.assertTrue(Hook(T, empty_action, "X")())
        self.assertTrue(Hook(T, empty_action)("X"))

        self.assertEqual("X", Hook(T, action_with_arg, "X", "Y")())
        self.assertEqual("X", Hook(T, action_with_arg)("X", "Y"))

    def test_with_extra_kwargs(self):
        self.assertTrue(Hook(T, empty_action, v="X")())
        self.assertTrue(Hook(T, empty_action)(v="X"))

        self.assertEqual("X", Hook(T, action_with_arg, "X", v2="Y")())
        self.assertEqual("X", Hook(T, action_with_arg)("X", v2="Y"))

    def test_multiple_values_for_same_argument(self):
        with self.assertRaises(TypeError):
            self.assertEqual("X", Hook(T, action_with_arg, "X", v="Y")())
        with self.assertRaises(TypeError):
            self.assertEqual("X", Hook(T, action_with_arg)("X", v="Y"))

    def test_hook_and_call_args_joined(self):
        self.assertEqual("XY", Hook(T, action_with_args, "X")("Y"))

    def test_call_kwarg_overwritten(self):
        self.assertEqual("X", Hook(T, action_with_kwargs, v="X")(v="X2"))

    def test_with_no_args(self):
        with self.assertRaises(TypeError):
            self.assertEqual("X", Hook(T, action_with_args)())
        with self.assertRaises(TypeError):
            self.assertEqual("X", Hook(T, action_with_args)())

    def test_false_condition(self):
        self.assertFalse(Hook(F, action_with_arg, "X")())
        self.assertFalse(Hook(F, action_with_arg)("X"))
        self.assertFalse(Hook(F, action_with_kwargs, v="X")())
        self.assertFalse(Hook(F, action_with_kwargs)(v="X"))


if __name__ == "__main__":
    unittest.main()
