from setuptools import find_packages, setup

with open("readme.md", "r") as f:
    long_description = f.read()

setup(
    name="pctree",
    version="0.0.11",
    description="Official Implementation of `Principal Component Trees` as referenced in the 2024 AAAI Paper `Principal Component Trees and their Persistent Homology`",
    long_description=long_description,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    author="Ben Kizaric",
    author_email="benkizaric@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/benkizaric/pctree",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy >= 1.24.0", "scikit-learn >= 1.2.0", "scipy >= 1.11.0"],
    python_requires=">= 3.10"
)