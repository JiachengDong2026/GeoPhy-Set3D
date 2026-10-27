"""Check worker scheduling/RNG and checkpoint identity before boundary continuation."""
from feasible_context import *
from torch.utils.data import DataLoader


class Indices(torch.utils.data.Dataset):
    def __len__(self):return 166
    def __getitem__(self,index):return index


def main():
    setup();verify()
    assert read(ART/'prepared_input_checks.json')['passed']
    traces=[]
    for workers in [0,4]:
        setup();generator=torch.Generator().manual_seed(20261011)
        loader=DataLoader(Indices(),batch_size=2,shuffle=True,generator=generator,
            num_workers=workers,persistent_workers=False,
            **({'multiprocessing_context':'fork'} if workers else {}))
        records=[]
        for epoch in range(3):
            order=torch.cat(list(loader)).tolist()
            assert sorted(order)==list(range(166))
            records.append(dict(order=order,loader_rng=sha_tensor(generator.get_state()),torch_rng=sha_tensor(torch.get_rng_state())))
        traces.append(records)
    assert traces[0]==traces[1]
    write(ART/'workers_equivalence_checks.json',dict(passed=True,epochs=3,
        identical_case_order_and_rng=True,prepared_field_equivalence='prepared_input_checks.json',
        persistent_workers=False,scientific_updates=0,GT_used=False))
    print('PASS: 0/4 workers identical three-epoch case order and main/loader RNG; prepared fields already bitwise verified')


def sha_tensor(value):return hashlib.sha256(value.numpy().tobytes()).hexdigest()


if __name__=='__main__':main()
