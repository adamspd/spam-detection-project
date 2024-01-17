# spam_detector_ai/tests/launcher.py

import unittest
import cProfile

if __name__ == "__main__":
    # Create a test suite combining all test cases
    loader = unittest.TestLoader()
    start_dir = 'spam_detector_ai/tests'  # Adjust the path to where your tests are located
    suite = loader.discover(start_dir)

    # Run the test suite with profiling
    profiler = cProfile.Profile()
    profiler.enable()

    runner = unittest.TextTestRunner()
    runner.run(suite)

    profiler.disable()
    profiler.print_stats(sort='time')
