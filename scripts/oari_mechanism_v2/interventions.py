"""Removable hooks; no shared model source or learned tensor is modified."""
from runtime import *
import math
import itertools
from torch.nn import functional as F


def permute_relation(relations, indices):
    rows = torch.arange(len(relations), device=relations.device)[:, None, None]
    return relations[rows, indices[:, :, None], indices[:, None, :]]


def controlled_maps(relations, mask, seed):
    """Eight distinct spatial fields; <=2 valid views are intrinsically ineligible.

    A field applies one common query/key permutation at each voxel. Different
    voxels can use different permutations. k=3 has at most five nonidentity
    local permutations, so eight fields do NOT imply eight independent local
    permutations. No sample-size multiplication is permitted.
    """
    n, v = mask.shape
    rng = np.random.default_rng(seed)
    identity = np.broadcast_to(np.arange(v, dtype=np.uint8), (n, v)).copy()
    codes = (mask * (1 << np.arange(v))).sum(1)
    all_maps = []
    eligible = np.zeros(n, dtype=bool)
    for rep in range(8):
        index = identity.copy()
        for code in np.unique(codes):
            ids = np.flatnonzero((int(code) & (1 << np.arange(v))) != 0)
            if len(ids) < 3:
                continue
            selected = np.flatnonzero(codes == code)
            block = relations[selected][:, ids][:, :, ids]
            off = ~np.eye(len(ids), dtype=bool)
            # A symmetric complete graph with constant off-diagonal channels
            # is invariant under every common index permutation.
            variable = np.ptp(block[:, off, :], axis=1).max(1) > 1e-7
            selected = selected[variable]
            if not len(selected):
                continue
            eligible[selected] = True
            pool = np.array([p for p in itertools.permutations(range(len(ids))) if p != tuple(range(len(ids)))], dtype=np.uint8)
            chosen = pool[rng.integers(len(pool), size=len(selected))]
            candidate = ids[chosen]
            # Reject graph automorphisms using only geometry, never outputs.
            remaining = np.arange(len(selected))
            for trial in range(40):
                local = selected[remaining]
                pp = candidate[remaining]
                before = relations[local][:, ids][:, :, ids]
                after = relations[local[:, None, None], pp[:, :, None], pp[:, None, :]]
                bad = np.max(np.abs(after - before), axis=(1, 2, 3)) <= 1e-7
                remaining = remaining[bad]
                if not len(remaining):
                    break
                candidate[remaining] = ids[pool[rng.integers(len(pool), size=len(remaining))]]
            if len(remaining):
                for row in remaining:
                    for perm in pool:
                        pp = ids[perm]
                        if np.max(np.abs(relations[selected[row]][pp][:, pp] - relations[selected[row]][ids][:, ids])) > 1e-7:
                            candidate[row] = pp
                            break
                    else:
                        raise AssertionError('Eligible geometry has no nontrivial common permutation')
            for j, original in enumerate(ids):
                index[selected, original] = candidate[:, j]
        all_maps.append(index)
    assert eligible.any(), 'This input cannot support the preregistered controlled diagnostic'
    assert len({hashlib.sha256(p[eligible].tobytes()).hexdigest() for p in all_maps}) == 8
    return np.stack(all_maps), eligible


class Intervention:
    def __init__(self, model, group, indices=None, legacy=None):
        self.model, self.group, self.indices, self.legacy = model, group, indices, legacy
        self.calls = 0
        self.linear_calls = 0
        self.audit = {}

    def before(self, module, args):
        self.calls += 1
        f, g, visibility, vm, relations, obs, *rest = args
        mask = (visibility & vm[:, :, None, None, None]).permute(0, 2, 3, 4, 1).reshape(-1, vm.shape[1])
        pair = mask[:, :, None] & mask[:, None, :]
        changed = relations
        source_invalid = torch.zeros_like(pair)
        if self.group == 'A2_P1':
            p = torch.as_tensor(self.legacy, device=relations.device)
            changed = relations[:, p][:, :, p]
            source_invalid = pair & ~(mask[:, p, None] & mask[:, None, p])
        elif self.group.startswith('A3_'):
            p = torch.as_tensor(self.indices, dtype=torch.long, device=relations.device)
            assert torch.equal(mask.gather(1, p), mask)
            assert torch.equal(p.sort(1).values, torch.arange(vm.shape[1], device=p.device).expand_as(p))
            changed = permute_relation(relations, p)
            assert torch.equal(changed.diagonal(dim1=1, dim2=2), relations.diagonal(dim1=1, dim2=2))
            assert torch.equal(changed, changed.transpose(1, 2))
            assert not (changed[~pair] != 0).any()
        changed_pair = ((changed - relations).abs().amax(-1) > 1e-7) & pair
        self.audit.update(valid_pair_count=int(pair.sum()), changed_valid_pairs=int(changed_pair.sum()),
                          changed_valid_pair_fraction=float(changed_pair.sum() / pair.sum().clamp_min(1)),
                          valid_pairs_receiving_invalid_source=int(source_invalid.sum()),
                          source_contamination_fraction=float(source_invalid.sum() / pair.sum().clamp_min(1)),
                          changed_diagonal_count=int(((changed - relations).abs().amax(-1).diagonal(dim1=1, dim2=2) > 1e-7).sum()),
                          symmetry_max_error=float((changed - changed.transpose(1, 2)).abs().max()),
                          visibility_unchanged=True, view_mask_unchanged=True)
        self.source_contaminated_query = source_invalid.any(-1)
        if changed is not relations:
            return (f, g, visibility, vm, changed, obs, *rest)

    def linear(self, module, args, output):
        self.linear_calls += 1
        columns = [0, 1, 2] if self.group == 'A1_P2' else CHANNELS.get(self.group, [])
        if columns:
            keep = [i for i in range(8) if i not in columns]
            x = args[0]
            replacement = F.linear(x[..., keep], module.weight[:, keep], module.bias)
            removed = F.linear(x[..., columns], module.weight[:, columns], None)
            torch.testing.assert_close(output, replacement + removed, atol=2e-6, rtol=2e-5)
            self.audit['first_affine_removed_columns'] = columns
            return replacement

    def token(self, module, args, output):
        if self.group in ('obs_token', 'obs_both'):
            # Zero the five descriptor input contributions, retain learned bias.
            return F.linear(torch.zeros_like(args[0]), module.weight, module.bias)

    def __enter__(self):
        self.hooks = [self.model.interaction.register_forward_pre_hook(self.before),
                      self.model.interaction.pairwise_bias[0].register_forward_hook(self.linear),
                      self.model.interaction.observability_embedding[0].register_forward_hook(self.token)]
        return self

    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()
        if args[0] is None:
            assert self.calls == self.linear_calls == 1


class Capture:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def before(self, module, args):
        self.args = args
        f, g, vis, vm, rel, obs, *_ = args
        self.mask = (vis & vm[:, :, None, None, None]).permute(0, 2, 3, 4, 1).reshape(-1, vm.shape[1])
        self.identity = {name: tensor_hash(t) for name, t in zip(('features', 'geometry', 'visibility', 'view_mask', 'observability'), (f, g, vis, vm, obs))}
        self.relations = rel
        active = torch.nonzero(self.mask.any(1)).flatten().cpu().numpy()
        self.sample_ids = np.unique(np.r_[np.linspace(0, len(self.mask) - 1, 3, dtype=int),
                                          active[np.linspace(0, len(active) - 1, min(8, len(active)), dtype=int)]])

    def token_input(self, module, args):
        self.sample_token_input = args[0][self.sample_ids].detach().cpu().numpy().copy()

    def qkv(self, module, args, output):
        n, v, _ = output.shape
        q, k, values = output.reshape(n, v, 3, 4, 8).permute(2, 0, 3, 1, 4).unbind(0)
        self.content = (q @ k.transpose(-1, -2)) / math.sqrt(8)
        self.qkv_hash = tensor_hash(output)
        self.sample_qkv = output[self.sample_ids].detach().cpu().numpy().copy()

    def after(self, module, args, output):
        self.calls += 1
        context, attention, bias = output
        v = self.mask.shape[1]
        self.context = context.permute(0, 3, 4, 5, 1, 2).reshape(-1, v, 32)
        self.attention = attention.permute(0, 4, 5, 6, 1, 2, 3).reshape(-1, 4, v, v)
        self.bias = bias

    def cpu(self):
        return {name: getattr(self, name).detach().cpu().numpy().copy() for name in ('mask', 'relations', 'content', 'context', 'attention', 'bias')}

    def __enter__(self):
        self.hooks = [self.model.interaction.register_forward_pre_hook(self.before),
                      self.model.interaction.token_projection.register_forward_pre_hook(self.token_input),
                      self.model.interaction.qkv.register_forward_hook(self.qkv),
                      self.model.interaction.register_forward_hook(self.after)]
        return self

    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()
        if args[0] is None:
            assert self.calls == 1
