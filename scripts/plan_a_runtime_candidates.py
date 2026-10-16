"""Isolated execution candidates; never imported by the production trainer.

Network parameters and state_dict keys stay unchanged. BF16 is confined to
encoder/decoder/prior calls; lifting, state updates and physical losses use FP32.
"""
from contextlib import nullcontext
import functools
import torch


def add_arguments(parser):
    parser.add_argument('--experimental-runtime', action='store_true')
    parser.add_argument('--network-layout', choices=('contiguous', 'channels_last_3d'), default='contiguous')
    parser.add_argument('--cudnn-autotune', action='store_true')
    parser.add_argument('--compile-prior', action='store_true')
    parser.add_argument('--network-precision', choices=('fp32', 'tf32', 'bf16'), default='fp32')


def settings(args):
    return dict(network_layout=args.network_layout, cudnn_autotune=args.cudnn_autotune,
                compile_prior=args.compile_prior, network_precision=args.network_precision)


def configure_backend(runtime):
    torch.backends.cudnn.benchmark = runtime['cudnn_autotune']
    # Only cuDNN convolution uses TF32. Physics matmul remains full FP32.
    torch.backends.cudnn.allow_tf32 = runtime['network_precision'] == 'tf32'
    torch.backends.cuda.matmul.allow_tf32 = False


def configure_model(model, runtime):
    layout = runtime['network_layout'] == 'channels_last_3d'
    bf16 = runtime['network_precision'] == 'bf16'
    if layout:
        torch.nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    modules = [(model.initializer.encoder, False), (model.initializer.decoder, True),
               (model.prior, True)]
    for module, volume in modules:
        original = module.forward
        if module is model.prior and runtime['compile_prior']:
            original = torch.compile(original, fullgraph=True, dynamic=False)

        @functools.wraps(original)
        def forward(x, _original=original, _volume=volume):
            if layout and _volume:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=torch.bfloat16) if bf16 else nullcontext():
                result = _original(x)
            # Never feed reduced-precision tensors into lifting or physics.
            return result.float() if bf16 else result

        module.forward = forward
    return model
