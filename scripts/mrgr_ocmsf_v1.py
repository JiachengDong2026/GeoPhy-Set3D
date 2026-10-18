"""Independent OCMSF extension; frozen src modules remain unchanged."""
import weakref

import torch
from torch import nn
from torch.nn import functional as F

import probe_mrgr_multiscale_v1 as base

ARMS = ('A_original', 'B_multiscale', 'C_content', 'D_ocmsf')


def descriptor_volume(descriptor, batch_size, resolution=32):
    if descriptor.shape != (batch_size * resolution**3, 5):
        raise ValueError('Expected [B*32^3,5] observability descriptor')
    if not descriptor.is_floating_point() or not bool(torch.isfinite(descriptor).all()):
        raise ValueError('Descriptor must be finite floating point')
    if bool(((descriptor < -1e-6) | (descriptor > 1+1e-6)).any()):
        raise ValueError('Trace-normalized descriptor must lie in [0,1]')
    # OARI flattens tokens in B,X,Y,Z order. Never permute X/Y/Z.
    return descriptor.reshape(batch_size,resolution,resolution,resolution,5).permute(0,4,1,2,3).contiguous().detach()


class ScaleFusion(nn.Module):
    """Same 739 learned parameters for content-only and geometry-conditioned arms."""
    def __init__(self, conditioned):
        super().__init__()
        self.conditioned = conditioned
        self.content = nn.Conv3d(36,8,1)
        self.logits = nn.Conv3d(8,3,1)
        self.output = nn.Conv3d(12,32,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_diagnostics = None
        self.record_diagnostics = False

    def forward(self, scales, descriptor):
        if len(scales) != 3 or any(x.shape != scales[0].shape for x in scales):
            raise ValueError('Three aligned [B,12,32,32,32] scales required')
        x = scales[0]
        if x.ndim != 5 or x.shape[1:] != (12,32,32,32):
            raise ValueError('Unexpected aligned feature shape')
        if descriptor.shape != (x.shape[0],5,32,32,32):
            raise ValueError('Unexpected volume descriptor shape')
        if descriptor.dtype != x.dtype or descriptor.device != x.device:
            raise ValueError('Descriptor and features must share dtype/device')
        if any(not bool(torch.isfinite(t).all()) for t in (*scales,descriptor)):
            raise ValueError('Nonfinite fusion input')
        visible = descriptor[:,4:5] > 0
        clean = [torch.where(visible,s,0) for s in scales]
        hidden = self.content(torch.cat(clean,1))
        if self.conditioned:
            o = torch.where(visible,descriptor,0)
            # Fixed, parameter-free embedding: condition content processing without
            # introducing dead parameters or a different capacity in the control.
            embedded = torch.cat((o,o[:,0:1]-o[:,1:2],o[:,1:2]-o[:,2:3],1-o[:,4:5]),1)
            hidden = hidden + torch.where(visible,embedded,0)
        logits = self.logits(F.silu(hidden))
        logits = torch.where(visible,logits,0)
        alpha = logits.softmax(1)
        fused = sum(alpha[:,i:i+1]*s for i,s in enumerate(clean))
        delta = torch.where(visible,self.output(fused),0)
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError('Nonfinite OCMSF residual')
        if self.record_diagnostics:
            self.last_diagnostics = dict(alpha=alpha.detach(),delta=delta.detach(),visible=visible.detach())
        return delta


class OCMSFDecoder(base.ContextDecoder):
    def __init__(self, original, conditioned):
        # Do not recreate/reinitialize context: preserve exact original tensors.
        nn.Sequential.__init__(self)
        for name,module in original._modules.items():
            self.add_module(name,module)
        self.scale_fusion = ScaleFusion(conditioned)
        self._descriptor = None
        self.descriptor_calls = 0
        self.decoder_calls = 0

    def capture(self, descriptor):
        batch_size = descriptor.shape[0] // (32**3)
        self._descriptor = descriptor_volume(descriptor,batch_size)
        self.descriptor_calls += 1

    def forward(self, x):
        if self._descriptor is None:
            raise RuntimeError('OARI descriptor must be captured before shared decoder')
        if self._descriptor.shape[0] != x.shape[0]:
            raise ValueError('Descriptor batch mismatch')
        self.decoder_calls += 1
        for name,module in self._modules.items():
            if name in ('context','scale_fusion'):
                continue
            x = module(x)
            if name == '2':
                context = self.context
                # Same order and arithmetic as the frozen MultiscaleContext.
                x16 = context.local16(context.reduce(F.avg_pool3d(x,2)))
                x8 = context.context8(F.avg_pool3d(x16,2))
                y16 = context.fuse16(x16+F.interpolate(x8,size=x16.shape[-3:],mode='trilinear',align_corners=True))
                original = context.output(F.interpolate(y16,size=x.shape[-3:],mode='trilinear',align_corners=True))
                # Additional fine projection reuses existing context.reduce weights.
                # This does add computation; it is not claimed to be a preexisting
                # independent 32^3 branch in model B.
                aligned = (context.reduce(x),
                    F.interpolate(x16,size=x.shape[-3:],mode='trilinear',align_corners=True),
                    F.interpolate(x8,size=x.shape[-3:],mode='trilinear',align_corners=True))
                x = (x+original)+self.scale_fusion(aligned,self._descriptor)
        return x


def attach(model, conditioned):
    model.decoder = OCMSFDecoder(model.decoder,conditioned)
    decoder_ref = weakref.ref(model.decoder)
    def before(module,args):
        decoder = decoder_ref()
        decoder._descriptor = None
        decoder.descriptor_calls = decoder.decoder_calls = 0
    def capture(module,args,output):
        decoder_ref().capture(output)
    def after(module,args,output):
        decoder_ref()._descriptor = None
    # OARI computes descriptor once before C0; MRGR reuses it for C1. Always
    # clear at top-level exit, including exceptions, to prevent stale geometry.
    model._ocmsf_hooks = (model.register_forward_pre_hook(before),
        model.observability.register_forward_hook(capture),
        model.register_forward_hook(after,always_call=True))
    return model


def model_for(config, arm, device):
    if arm not in ARMS:
        raise ValueError(arm)
    model = base.model_for(config,'reference' if arm=='A_original' else 'multiscale','cpu')
    if arm in ('C_content','D_ocmsf'):
        state = torch.get_rng_state()
        torch.manual_seed(config['training']['seed']+2)
        attach(model,arm=='D_ocmsf')
        torch.set_rng_state(state)
    return model.to(device)
