"""Independent plan-B implementation; frozen plan-A source is unchanged."""
from .model import FieldContext, SetConditionalField, sample_trilinear
from .physics import ContinuousGeometry, ContinuousRayIntegrator, mass_to_mole

__all__ = ['FieldContext', 'SetConditionalField', 'sample_trilinear',
           'ContinuousGeometry', 'ContinuousRayIntegrator', 'mass_to_mole']
