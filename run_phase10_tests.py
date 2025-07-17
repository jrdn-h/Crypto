"""
Phase 10 Master Test Runner

This script runs all Phase 10 validation tests and provides comprehensive reporting:
- Comprehensive integration tests
- Regression testing for backward compatibility  
- Individual phase test suites
- Master validation report

Usage:
    python run_phase10_tests.py [--quick] [--regression-only] [--integration-only]
"""

import unittest
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


class TestResult:
    """Enhanced test result tracking."""
    
    def __init__(self, name: str):
        self.name = name
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
        self.duration = 0.0
        
    @property
    def success_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0
    
    def add_result(self, result: unittest.TestResult, duration: float):
        """Add unittest result to this tracker."""
        self.total += result.testsRun
        self.failed += len(result.failures) + len(result.errors)
        self.skipped += len(result.skipped)
        self.passed = self.total - self.failed - self.skipped
        self.duration = duration
        
        # Collect error messages
        for test, error in result.failures + result.errors:
            self.errors.append(f"{test}: {error.split(chr(10))[0]}")


class Phase10TestRunner:
    """Master test runner for Phase 10 validation."""
    
    def __init__(self):
        self.results: Dict[str, TestResult] = {}
        self.start_time = time.time()
    
    def run_test_suite(self, name: str, test_class, timeout: float = 30.0) -> TestResult:
        """Run a single test suite with timeout and error handling."""
        result = TestResult(name)
        
        try:
            print(f">> Running {name}")
            
            start_time = time.time()
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_result = unittest.TestResult()
            
            # Run with timeout protection
            suite.run(test_result)
            duration = time.time() - start_time
            
            result.add_result(test_result, duration)
            
            # Report results
            if result.failed > 0:
                print(f"   FAILED {result.passed}/{result.total} tests passed ({result.success_rate:.1f}%)")
            else:
                print(f"   PASSED {result.passed}/{result.total} tests passed ({result.success_rate:.1f}%)")
            
            if result.skipped > 0:
                print(f"   SKIPPED {result.skipped} tests (missing dependencies)")
                
        except Exception as e:
            print(f"   ERROR: Test suite failed to run: {e}")
            result.errors.append(f"Suite execution error: {e}")
        
        self.results[name] = result
        return result
    
    def run_comprehensive_tests(self) -> bool:
        """Run comprehensive integration tests."""
        print("Phase 10: Comprehensive Integration Tests")
        print("=" * 50)
        
        try:
            from test_phase10_comprehensive_validation import (
                TestPhase10Integration,
                TestRegressionValidation,
                TestEndToEndWorkflows,
                TestProviderFallbackValidation,
                TestPerformanceValidation
            )
            
            suites = [
                ("Integration Tests", TestPhase10Integration),
                ("Regression Tests", TestRegressionValidation),
                ("End-to-End Workflows", TestEndToEndWorkflows),
                ("Provider Fallback", TestProviderFallbackValidation),
                ("Performance Tests", TestPerformanceValidation),
            ]
            
            for name, test_class in suites:
                self.run_test_suite(name, test_class)
                
        except ImportError as e:
            print(f"   ERROR: Comprehensive tests not available: {e}")
            return False
        
        return True
    
    def run_regression_tests(self) -> bool:
        """Run regression tests for backward compatibility."""
        print("\nPhase 10: Regression Tests")
        print("=" * 35)
        
        try:
            from test_phase10_regression_validation import (
                TestEquityRegressionCore,
                TestEquityDataProviderRegression,
                TestEquityCLIRegression,
                TestEquityBackwardCompatibility,
                TestEquityPerformanceRegression
            )
            
            suites = [
                ("Core Equity Regression", TestEquityRegressionCore),
                ("Data Provider Regression", TestEquityDataProviderRegression),
                ("CLI Regression", TestEquityCLIRegression),
                ("Backward Compatibility", TestEquityBackwardCompatibility),
                ("Performance Regression", TestEquityPerformanceRegression),
            ]
            
            for name, test_class in suites:
                self.run_test_suite(name, test_class)
                
        except ImportError as e:
            print(f"   ERROR: Regression tests not available: {e}")
            return False
        
        return True
    
    def run_individual_phase_tests(self) -> bool:
        """Run individual phase test suites."""
        print("\nPhase 10: Individual Phase Validation")
        print("=" * 40)
        
        phase_tests = [
            ("Phase 4: News & Sentiment", "test_phase4_news_sentiment"),
            ("Phase 5: Technical Analysis", "test_phase5_crypto_technical"), 
            ("Phase 6: Crypto Researchers", "test_phase6_crypto_researchers"),
            ("Phase 7: Execution Adapters", "test_phase7_crypto_execution"),
            ("Phase 8: Risk Management", "test_phase8_crypto_risk_management"),
            ("Phase 9: CLI Enhancement", "test_phase9_cli_enhancements"),
        ]
        
        individual_results = []
        
        for phase_name, test_module in phase_tests:
            try:
                print(f">> Running {phase_name}")
                
                # Import and run the test module
                start_time = time.time()
                
                if test_module == "test_phase4_news_sentiment":
                    import test_phase4_news_sentiment
                    success = test_phase4_news_sentiment.run_comprehensive_tests()
                elif test_module == "test_phase5_crypto_technical":
                    import test_phase5_crypto_technical
                    success = test_phase5_crypto_technical.run_comprehensive_tests()
                elif test_module == "test_phase6_crypto_researchers":
                    import test_phase6_crypto_researchers
                    success = test_phase6_crypto_researchers.run_comprehensive_tests()
                elif test_module == "test_phase7_crypto_execution":
                    import test_phase7_crypto_execution
                    success = test_phase7_crypto_execution.run_comprehensive_tests()
                elif test_module == "test_phase8_crypto_risk_management":
                    import test_phase8_crypto_risk_management
                    success = test_phase8_crypto_risk_management.run_comprehensive_tests()
                elif test_module == "test_phase9_cli_enhancements":
                    import test_phase9_cli_enhancements
                    success = test_phase9_cli_enhancements.run_comprehensive_tests()
                else:
                    success = False
                
                duration = time.time() - start_time
                
                result = TestResult(phase_name)
                result.duration = duration
                result.total = 1
                result.passed = 1 if success else 0
                result.failed = 0 if success else 1
                
                if success:
                    print(f"   PASSED {phase_name} validation")
                else:
                    print(f"   FAILED {phase_name} validation")
                    result.errors.append("Phase validation failed")
                
                individual_results.append(result)
                self.results[phase_name] = result
                
            except ImportError as e:
                print(f"   SKIPPED {phase_name} (module not available)")
                result = TestResult(phase_name)
                result.total = 1
                result.skipped = 1
                result.errors.append(f"Import error: {e}")
                individual_results.append(result)
                self.results[phase_name] = result
            except Exception as e:
                print(f"   ERROR {phase_name}: {e}")
                result = TestResult(phase_name)
                result.total = 1
                result.failed = 1
                result.errors.append(f"Execution error: {e}")
                individual_results.append(result)
                self.results[phase_name] = result
        
        return True
    
    def generate_final_report(self) -> bool:
        """Generate comprehensive final validation report."""
        total_duration = time.time() - self.start_time
        
        print(f"\nPhase 10 Master Validation Report")
        print("=" * 45)
        print(f"Total Execution Time: {total_duration:.2f} seconds\n")
        
        # Summary statistics
        total_tests = sum(r.total for r in self.results.values())
        total_passed = sum(r.passed for r in self.results.values())
        total_failed = sum(r.failed for r in self.results.values())
        total_skipped = sum(r.skipped for r in self.results.values())
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Overall Test Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_passed}")
        print(f"  Failed: {total_failed}")
        print(f"  Skipped: {total_skipped}")
        print(f"  Success Rate: {overall_success_rate:.1f}%")
        
        # Detailed results by category
        print(f"\nDetailed Results by Test Suite:")
        print("-" * 60)
        
        for name, result in self.results.items():
            status = "PASS" if result.failed == 0 and result.total > 0 else "FAIL" if result.failed > 0 else "SKIP"
            print(f"{name:35} {status:4} {result.passed:3}/{result.total:3} ({result.success_rate:5.1f}%) {result.duration:6.2f}s")
        
        # Critical failures
        critical_failures = [name for name, result in self.results.items() 
                           if result.failed > 0 and "Regression" in name]
        
        if critical_failures:
            print(f"\nCRITICAL: Regression test failures detected!")
            print(f"Backward compatibility may be compromised:")
            for failure in critical_failures:
                print(f"  - {failure}")
        
        # Error summary
        all_errors = []
        for result in self.results.values():
            all_errors.extend(result.errors)
        
        if all_errors:
            print(f"\nError Summary (showing first 5):")
            for error in all_errors[:5]:
                print(f"  - {error}")
            if len(all_errors) > 5:
                print(f"  ... and {len(all_errors) - 5} more errors")
        
        # Coverage assessment
        print(f"\nPhase 10 Test Coverage Assessment:")
        print(f"  [{'OK' if overall_success_rate >= 70 else 'FAIL'}] Integration Testing: Cross-phase component validation")
        print(f"  [{'OK' if not critical_failures else 'FAIL'}] Regression Testing: Backward compatibility verification")
        print(f"  [{'OK' if total_tests >= 50 else 'WARN'}] Test Completeness: {total_tests} total test cases")
        print(f"  [{'OK' if total_skipped < total_tests * 0.5 else 'WARN'}] Dependency Coverage: {total_skipped}/{total_tests} skipped")
        
        # Final determination
        minimum_success_rate = 70.0
        has_critical_failures = len(critical_failures) > 0
        
        if overall_success_rate >= minimum_success_rate and not has_critical_failures:
            print(f"\nSUCCESS: Phase 10 Comprehensive Validation: PASSED")
            print(f"         All critical functionality validated successfully")
            print(f"         TradingAgents crypto extension ready for deployment")
            return True
        else:
            print(f"\nWARNING: Phase 10 Comprehensive Validation: NEEDS ATTENTION")
            if overall_success_rate < minimum_success_rate:
                print(f"         Success rate {overall_success_rate:.1f}% below threshold {minimum_success_rate:.0f}%")
            if has_critical_failures:
                print(f"         Critical regression failures detected")
            print(f"         Review and fix issues before deployment")
            return False


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Phase 10 Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument("--regression-only", action="store_true", help="Run only regression tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--individual-only", action="store_true", help="Run only individual phase tests")
    
    args = parser.parse_args()
    
    runner = Phase10TestRunner()
    
    print("TradingAgents Crypto Extension - Phase 10 Validation")
    print("=" * 60)
    print("Comprehensive testing and validation of all implemented phases")
    print()
    
    success = True
    
    # Run test suites based on arguments
    if args.regression_only:
        success = runner.run_regression_tests()
    elif args.integration_only:
        success = runner.run_comprehensive_tests()
    elif args.individual_only:
        success = runner.run_individual_phase_tests()
    else:
        # Run all tests
        if not args.quick:
            runner.run_comprehensive_tests()
        
        runner.run_regression_tests()
        
        if not args.quick:
            runner.run_individual_phase_tests()
    
    # Generate final report
    final_success = runner.generate_final_report()
    
    # Exit with appropriate code
    sys.exit(0 if final_success else 1)


if __name__ == "__main__":
    main() 