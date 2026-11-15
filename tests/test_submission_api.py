"""Keep submission eligibility separate from development bots and controls."""

import importlib
import inspect
import unittest
from pathlib import Path

import custom_bots
from bots import Bot
from simulation import bot_catalog


class SubmissionTests(unittest.TestCase):
    def test_only_the_two_intended_bots_are_submitted(self):
        self.assertEqual(set(custom_bots.__all__), {"SamplingBot", "ValuationBot"})
        self.assertEqual(len(custom_bots.__all__), 2)
        self.assertTrue(
            set(custom_bots.__all__).isdisjoint(custom_bots.EXPERIMENTAL_BOTS)
        )
        for name in custom_bots.__all__:
            self.assertTrue(issubclass(getattr(custom_bots, name), Bot))

    def test_submission_definitions_and_strategy_contributor_documentation(self):
        directory = Path(custom_bots.__file__).resolve().parent / "bot_definitions"
        for name in custom_bots.__all__:
            with self.subTest(bot=name):
                cls = getattr(custom_bots, name)
                self.assertEqual(Path(inspect.getfile(cls)).resolve().parent, directory)
                module = importlib.import_module(cls.__module__)
                documentation = inspect.getdoc(module) or ""
                self.assertGreater(len(documentation.splitlines()), 5)
                self.assertIn("Submission role:", documentation)
                self.assertIn("Contributors:", documentation)
                self.assertIn("Grace Guo", documentation)
                self.assertIn("OpenAI Codex", documentation)
                self.assertTrue(inspect.getdoc(cls))

    def test_development_bots_remain_usable_without_becoming_submissions(self):
        catalog = bot_catalog()
        for name, cls in custom_bots.EXPERIMENTAL_BOTS.items():
            with self.subTest(bot=name):
                self.assertIs(catalog[name], cls)
                self.assertIs(getattr(custom_bots, name), cls)
                self.assertNotIn(name, custom_bots.__all__)
                module = importlib.import_module(cls.__module__)
                self.assertIn("Contributors:", inspect.getdoc(module) or "")
        self.assertNotIn("PassiveBidder", custom_bots.__all__)
        self.assertIs(catalog["ValuationBot"], custom_bots.ValuationBot)


if __name__ == "__main__":
    unittest.main()
