
# test_main.py
import unittest
from main import main

class TestMain(unittest.TestCase):
    def test_run(self):
        try:
            main()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"main() raised {e} unexpectedly!")

if __name__ == "__main__":
    unittest.main()
