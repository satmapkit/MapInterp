from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent
OCEANDB = (ROOT.parent / "OceanDB").resolve()

if not OCEANDB.exists():
    raise RuntimeError("Expected OceanDB repository next to MapInterp at ../OceanDB")

setup(
    install_requires=[
        f"OceanDB @ {OCEANDB.as_uri()}",
        "xarray~=2024.6.0",
        "matplotlib>=3.0",
    ],
)
