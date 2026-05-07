from pathlib import Path

from setuptools import find_packages, setup

readme = Path(__file__).parent / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="dual-channel-reliable-bus",
    version="0.1.0",
    description=(
        "Dual-channel reliability framework (SP-RISA + TTA) for breast "
        "ultrasound image classification."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hu, Lei et al.",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
