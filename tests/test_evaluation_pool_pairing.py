import unittest
import numpy as np
import torch
from pool_fixture import make_pool
from geophy_set3d.evaluation.protocols import evaluation_dataset
from geophy_set3d.data.collate import projection_pool_collate
from geophy_set3d.models import GeoLiftMeanRes3D
from geophy_set3d.training.utils import permute_views


class EvaluationPoolPairingTests(unittest.TestCase):
    def test_pool_b_projection_geometry_pair_and_permutation(self):
        temporary,root,rows,geometry=make_pool()
        self.addCleanup(temporary.cleanup)
        config=dict(dataset=dict(root=str(root)),normalization=dict(method='linear_p99_5',statistics_file='train_statistics.json'))
        a=evaluation_dataset(config,pool='train',views=4,selection='random')
        b=evaluation_dataset(config,pool='ood',views=4,selection='random')
        for dataset,pool in ((a,'train'),(b,'ood')):
            sample=dataset[0];ids=sample['input_station_indices'].numpy()
            disk=np.load(root/dataset.records[0][pool+'_projection_path'])
            np.testing.assert_array_equal(sample['input_projection_raw'][:,0].numpy(),disk[ids])
            np.testing.assert_allclose(sample['input_station_origin_m'].numpy(),geometry[pool]['station_origin_m'][ids],atol=1e-6)
            self.assertEqual(sample['geometry_pool'],pool)
            self.assertEqual(dataset.geometry_path.name,pool+'_geometry.npz')
        torch.set_num_threads(2);torch.manual_seed(81)
        model=GeoLiftMeanRes3D().eval();batch=projection_pool_collate([b[0]])
        with torch.no_grad():
            original=model(batch);other=model(permute_views(batch,torch.tensor([2,0,3,1])))
        torch.testing.assert_close(original,other,atol=2e-6,rtol=0)
