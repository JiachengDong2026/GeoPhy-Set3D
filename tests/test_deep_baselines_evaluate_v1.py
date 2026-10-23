"""Synthetic case/draw statistical checks; never open measurement or GT arrays."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/deep_baselines_v1/evaluate.py'
SPEC = importlib.util.spec_from_file_location('deep_baseline_evaluation', PATH)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def rows(mini=False):
    result = []
    for i in range(2 if mini else 36):
        for group_index, group in enumerate(evaluation.GROUPS):
            for draw in range(1 if mini else 5):
                record = dict(case_id=f'case{i:02d}', case_index=i, group=group, draw=draw,
                              geometry_id=f'V4-{group}-d{draw}', station_ids=[0, 1, 2, 3], H_station_ids=[4, 5, 6, 7])
                value = float((i + 1) * (group_index + 1) + draw)
                result.append(dict(record=record, complete_input_sha256='synthetic-input', complete_H_sha256='synthetic-H',
                                   **{metric: value for metric in evaluation.METRICS}))
    return result


class EvaluationStatisticsTests(unittest.TestCase):
    def test_actual_frozen_records_are_full_val_geometry_membership(self):
        records = evaluation.read(evaluation.RECORDS)
        evaluation.validate_records(records)
        self.assertEqual(len(records), 900)

    def test_draw_mean_precedes_case_bootstrap(self):
        data = rows()
        cases = evaluation.aggregate_cases(data)
        self.assertEqual(len(cases), 180)
        first = next(r for r in cases if r['case_id'] == 'case00' and r['group'] == 'G1')
        self.assertEqual(first['relative_l1'], 3.)
        summary = evaluation.summarize(cases)
        actual = next(r for r in summary['summary'] if r['group'] == 'G1' and r['metric'] == 'relative_l1')
        values = np.arange(1, 37, dtype=np.float64) + 2
        self.assertEqual(actual['mean_per_case'], float(values.mean()))
        # Independent weighted-count implementation of the same case bootstrap.
        rng = np.random.default_rng(2026101801)
        draws = rng.integers(0, 36, (10000, 36))
        weights = np.stack([np.bincount(sample, minlength=36) for sample in draws])
        low, high = np.quantile(weights @ values / 36, [.025, .975])
        np.testing.assert_allclose([actual['ci_low'], actual['ci_high']], [low, high], atol=1e-14, rtol=0)
        self.assertEqual(actual['worst_case_id'], 'case35')
        self.assertEqual(summary['bootstrap']['unit'], 'case after draw mean')

    def test_geometry_ratio_is_ratio_of_grand_means(self):
        cases = evaluation.aggregate_cases(rows(mini=True), mini=True)
        for item in cases:
            for metric in evaluation.METRICS:
                item[metric] = (1. if item['case_id'] == 'case00' else 9.)
                if item['group'] == 'G5':
                    item[metric] = (2. if item['case_id'] == 'case00' else 9.)
        summary = evaluation.summarize(cases, mini=True)
        contrast = summary['contrasts'][0]
        self.assertAlmostEqual(contrast['G5_div_G1_grand_mean_ratio'], 1.1)
        self.assertAlmostEqual(contrast['G5_minus_G1'], .5)
        self.assertFalse(summary['formal'])
        self.assertEqual(summary['scope'], 'NONFORMAL_MINI_INTEGRATION')

    def test_missing_duplicate_overlap_and_nonfinite_fail_closed(self):
        for malformed in (rows()[:-1], rows() + rows()[:1]):
            with self.assertRaises(ValueError):
                evaluation.aggregate_cases(malformed)
        invalid = rows()
        invalid[0]['record']['H_station_ids'] = [0, 4, 5, 6]
        with self.assertRaises(ValueError):
            evaluation.aggregate_cases(invalid)
        invalid = rows()
        invalid[0]['relative_l1'] = float('nan')
        with self.assertRaises(ValueError):
            evaluation.aggregate_cases(invalid)
        incomplete = rows(mini=True)
        with self.assertRaises(ValueError):
            evaluation.aggregate_cases(incomplete)

    def test_zero_ratio_denominator_is_undefined_not_infinity(self):
        cases = evaluation.aggregate_cases(rows(mini=True), mini=True)
        for item in cases:
            if item['group'] == 'G1':
                for metric in evaluation.METRICS:
                    item[metric] = 0.
        result = evaluation.summarize(cases, mini=True)
        self.assertIsNone(result['contrasts'][0]['G5_div_G1_grand_mean_ratio'])
        self.assertIsNone(result['contrasts'][0]['ratio_ci_low'])

    def test_case_paired_statistics_and_input_identity(self):
        reference = rows(mini=True)
        baseline = copy.deepcopy(reference)
        for row in baseline:
            for metric in evaluation.METRICS:
                row[metric] += 3.
        result = evaluation.paired_statistics(reference, baseline, mini=True)
        for row in result:
            self.assertEqual(row['mean_delta'], 3.)
            self.assertEqual(row['ci_low'], 3.)
            self.assertEqual(row['ci_high'], 3.)
            self.assertEqual(row['baseline_worse'], 2)
        baseline[0]['complete_H_sha256'] = 'different'
        with self.assertRaises(ValueError):
            evaluation.paired_statistics(reference, baseline, mini=True)


if __name__ == '__main__':
    unittest.main()
