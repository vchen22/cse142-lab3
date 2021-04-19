# Students should not edit this file, since changes here will _only_
# affect how your code runs locally.  It will not change how your code
# executes in the cloud.
from ArchLab.CSE141Lab import CSE141Lab,test_configs # The labs for the class have lots in common, so we have base class.
import unittest
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
import io
import unittest
from gradescope_utils.autograder_utils.decorators import weight, leaderboard, partial_credit
import parameterized
import logging as log

class ThisLab(CSE141Lab):
    def __init__(self, **kwargs):
        super(ThisLab, self).__init__(
            lab_name = "Caches I",
            short_name = "caches-I",
            output_files = ['code.csv', 'code.gprof', 'code.s', 'regressions.out', 'benchmark.csv', 'benchmark.gprof'],
            input_files = ['code.cpp', 'opt_cnn.hpp', 'config.env'],
            default_cmd = ['make'],
            repo = kwargs.get("repo") or "https://github.com/NVSL/CSE141pp-Lab-Caches.git",
            reference_tag = kwargs.get("repo") or "master",
            timeout = 360
        )
    def filter_command(self, command):
        return self.make_target_filter(command)
    
    class MetaRegressions(CSE141Lab.MetaRegressions):
        def setUp(self):
            self.lab_spec = ThisLab.load(".")

        @parameterized.parameterized.expand(test_configs("solution", "."))
        def test_solution(self, solution, flags):
            if flags.devel and "solution" == solution:
                self.skipTest("Skipping since this solution doesn't work in devel mode")
            
            result, tag = self.run_solution(solution, flags)

            js = result.results
            
            c = self.read_text_file("code.csv",root=".")
            b = self.read_text_file("benchmark.csv",root=".")
            self.assertEqual(len(c.strip().split("\n")), 2, "code.csv should have 2 lines.  It has a different number.")
            self.assertEqual(len(b.strip().split("\n")), 2, "benchmark.csv should have 2 lines.  It has a different number.")

            runtime = self.lab_spec.csv_extract_by_line(b, 'runtime')
            if flags.grades_valid():
                self.assertEqual(len(js["gradescope_test_output"]['leaderboard']), 1, "too many or too few leader board entries")

                


