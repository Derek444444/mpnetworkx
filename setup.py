from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mpnetworkx-simple",
    version="1.0.0",
    description="Упрощенная оптимизация NetworkX через параллелизацию",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="mpnetworkx Team",
    packages=find_packages(),
    install_requires=["networkx>=2.6"],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
    ],
)
