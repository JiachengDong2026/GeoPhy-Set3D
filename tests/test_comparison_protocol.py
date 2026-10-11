import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from geophy_set3d.training.utils import load_config
from geophy_set3d.training.comparison import assert_fair_configs, assert_fair_b1_b2


class ComparisonProtocolTests(unittest.TestCase):
    def setUp(self):
        repo = Path(__file__).resolve().parents[1]
        self.b0 = load_config(repo/'configs/train_supervised_b0_300.yaml')
        self.b1 = load_config(repo/'configs/train_supervised_b1_300.yaml')
        self.b2 = load_config(repo/'configs/train_supervised_b2_300.yaml')

    @patch('geophy_set3d.training.comparison.yaml.safe_load', return_value={
        'geometry': {'radius_m': [5.5, 6.5], 'azimuth_deg': [0., 180.]}})
    @patch('pathlib.Path.read_text', return_value='')
    def test_fixed_protocol_and_reject_unfair_settings(self, *unused):
        assert_fair_configs(self.b0, self.b1)
        for section, key, value in [('training', 'max_epochs', 150), ('training', 'early_stopping_patience', 20),
                                    ('train', 'geometry_pool', 'ood'), ('validation', 'split', 'test'),
                                    ('scheduler', 'T_max', 150), ('optimizer', 'lr', 1e-3)]:
            with self.subTest(section=section, key=key):
                other = copy.deepcopy(self.b1)
                other[section][key] = value
                with self.assertRaises(ValueError):
                    assert_fair_configs(self.b0, other)

    @patch('geophy_set3d.training.comparison.yaml.safe_load', return_value={
        'geometry': {'radius_m': [5.5, 6.5], 'azimuth_deg': [0., 180.]}})
    @patch('pathlib.Path.read_text', return_value='')
    def test_b1_b2_share_protocol(self, *unused):
        assert_fair_b1_b2(self.b1,self.b2)
        other=copy.deepcopy(self.b2);other['loader']['num_workers']=0
        with self.assertRaises(ValueError):assert_fair_b1_b2(self.b1,other)
        other=copy.deepcopy(self.b2);other['train']['hold_views']=1
        with self.assertRaises(ValueError):assert_fair_b1_b2(self.b1,other)
