#!/usr/bin/env python
"""Test runner script for mmrag package."""

import unittest
import argparse
import sys
import os
from pathlib import Path
import time
from typing import Optional # Added for type hinting

def discover_tests(start_dir: Path, pattern: str = "test_*.py", top_level_dir: Optional[Path] = None):
    """Discover tests in the given directory."""
    if not start_dir.exists():
        print(f"Warning: Test directory {start_dir} does not exist. Skipping discovery.")
        return None
    
    loader = unittest.defaultTestLoader
    return unittest.defaultTestLoader.discover(
        start_dir=str(start_dir),
        pattern=pattern,
        # Use top_level_dir if provided, otherwise default behavior
        top_level_dir=str(top_level_dir) if top_level_dir else None
    )

def run_tests(test_type=None, verbose=False, failfast=False, pattern=None):
    """Run tests of the specified type."""
    verbosity = 2 if verbose else 1
    test_pattern = pattern or "test_*.py"
    
    # Get project root directory (two levels up from this file)
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "tests"  # Now points to the tests directory
    
    # Add src directory to Python path
    sys.path.insert(0, str(project_root / "src"))
    
    # Determine which tests to run
    if test_type == "unit":
        print("Running unit tests...")
        test_suite = discover_tests(test_dir / "unit", test_pattern, project_root)
    elif test_type == "integration":
        print("Running integration tests...")
        test_suite = discover_tests(test_dir / "integration", test_pattern, project_root)
    elif test_type == "stress":
        print("Running stress tests...")
        test_suite = discover_tests(test_dir / "stress", test_pattern, project_root)
    else:
        print("Running all tests...")
        # Explicitly discover from known subdirectories and combine
        loader = unittest.defaultTestLoader
        test_suite = unittest.TestSuite()
        subdirs_to_discover = ["unit", "integration", "stress"]
        found_tests = False
        
        for subdir in subdirs_to_discover:
            subdir_path = test_dir / subdir
            if subdir_path.exists():
                print(f"Discovering tests in {subdir_path}...")
                discovered = loader.discover(
                    start_dir=str(subdir_path),
                    pattern=test_pattern,
                    top_level_dir=str(project_root)
                )
                if discovered and discovered.countTestCases() > 0:
                    print(f"Found {discovered.countTestCases()} tests in {subdir}.")
                    test_suite.addTest(discovered)
                    found_tests = True
    
    if not test_suite or test_suite.countTestCases() == 0:
        print("No tests discovered.")
        return 1
    
    # Run tests
    start_time = time.time()
    
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = runner.run(test_suite)
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"  Ran {result.testsRun} test{'s' if result.testsRun != 1 else ''} in {elapsed_time:.2f} seconds")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the mmrag package")
    parser.add_argument(
        "--suite", # Renamed argument
        choices=["unit", "integration", "stress", "all"], # Keep choices consistent
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--failfast", "-f",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        help="Pattern for test file names"
    )
    
    args = parser.parse_args()
    # Keep test_type as 'all' if specified, otherwise use the specific type
    test_type_arg = args.suite if args.suite != "all" else None # Use args.suite
    
    sys.exit(run_tests(test_type_arg, args.verbose, args.failfast, args.pattern))
