import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="elces",
    version="0.0.1",
    author="Louis Tamames, Paul Dufossé",
    author_email="paul.dufosse@inria.fr",
    description="Python implementation of the (1+1)-CMA-ES with active constraint handling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LsTam91/elitist_constrained_es",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    requires=["numpy", "matplotlib", "codetiming"],
    python_requires='>=3.6',
)