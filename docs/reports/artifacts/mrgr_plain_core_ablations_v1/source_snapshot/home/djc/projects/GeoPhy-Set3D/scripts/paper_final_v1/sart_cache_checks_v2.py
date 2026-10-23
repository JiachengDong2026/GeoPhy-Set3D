"""Regression for eviction of an already-requested station, without solver changes."""
from common import *
import ast
from unittest.mock import patch
import numpy as np
import sart


def main():
    archived=OUT/'sart/source_before_cache_fix_v2/sart.py';current=Path(sart.__file__)
    def definition(path,name):
        source=path.read_text();node=next(n for n in ast.parse(source).body if getattr(n,'name',None)==name)
        return ast.get_source_segment(source,node)
    for name in ('Block','solve','register','select','method'):assert definition(archived,name)==definition(current,name),name
    class Fake:
        def __init__(self,g):self.station=int(g['station'][0])
    cache={i:Fake(dict(station=np.array([i]))) for i in range(12)}
    with patch.object(sart,'Block',Fake):
        result=sart.cached_blocks(cache,[0,12,13,14],dict(station=np.arange(64)))
        assert [b.station for b in result]==[0,12,13,14] and len(cache)<=12 and 0 not in cache
        result=sart.cached_blocks(cache,[1,15,16,17,18,19],dict(station=np.arange(64)))
        assert [b.station for b in result]==[1,15,16,17,18,19] and len(cache)<=12
    preserved=verified(archived.parent/'preserved_predictions.json');verify_hashes(preserved['prediction_hashes'])
    for p in (OUT/'training').glob('*/protocol.json'):verify_hashes(verified(p)['sources'])
    freeze(OUT/'sart/cache_checks_v2.json',dict(at=now(),passed=True,eviction_of_current_station_reproduced_and_fixed=True,
        matrix_and_solver_source_identical=True,selection_and_configuration_unchanged=True,
        preserved_predictions_unchanged=preserved['count'],training_sources_unchanged=True,source_sha256=sha(__file__)))
    print('Cache recovery checks PASSED')


if __name__=='__main__':main()
