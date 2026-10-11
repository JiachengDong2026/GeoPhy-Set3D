#!/usr/bin/env python3
"""Audit saved Pool A measurements against C128/C64 differentiable projections."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from geophy_set3d.data.projection_audit import main

if __name__ == '__main__':
    main()
