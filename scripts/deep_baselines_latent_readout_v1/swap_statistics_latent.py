"""Expand exact-donor swap statistics after independent validation of donor arrays."""
from latent_context import *

def main():
    verify_seal();assert read(ROOT/'independent_review.json')['passed']
    with (ROOT/'layer_statistics.csv').open() as f:base=list(csv.DictReader(f))
    lookup={(r['arm'],r['branch'],int(r['case']),r['layer']):r for r in base if r['condition']=='normal'}
    rows=[];n=0
    with Access('extract_swapstats'):
        ss=samples()
        for a in EXPECTED:
            ext=read(ROOT/f'extraction_{a}.json')
            for alias in ext['aliases']:
                if alias['kind']!='swap':continue
                b=alias['branch'];i=alias['case'];j=alias['donor'];recipient=old.batch([ss[i]],b);donor=old.batch([ss[j]],b)
                changed=dict(recipient)
                for field in ('input_projection','input_projection_raw'):changed[field]=donor[field].clone()
                assert old.value_hash(changed)==old.value_hash(donor)==alias['input_sha256']
                assert old.value_hash(old.nonmeasurement(changed))==old.value_hash(old.nonmeasurement(recipient))
                for layer in sorted({r['layer'] for r in base if r['arm']==a and r['branch']==b}):
                    source=lookup[a,b,j,layer];row=dict(source,case=i,donor=j,condition='swap',statistics_reused_from=f'{j}:normal',input_equal_to_donor=True);rows.append(row)
                    for k in ('shape','mean','std','finite_count','total_count','finite_fraction'):assert row[k]==source[k]
                n+=1
    assert n==224
    csvwrite(ROOT/'swap_layer_statistics.csv',rows)
    freeze(ROOT/'swap_statistics_audit.json',dict(at=now(),passed=True,directed_swaps=n,layer_records=len(rows),donor_stats_independently_verified=True,full_input_equal_to_donor=True,source_sha256=sha(__file__),csv_sha256=sha(ROOT/'swap_layer_statistics.csv'),neural_forward_calls=0,neural_updates=0))
if __name__=='__main__':main()
