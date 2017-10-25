from setuptools import setup, find_packages


setup(
    name="bayesnet",
    version="0.0.1",
    description="Bayesian method library",
    author="ctgk",
    python_requires=">=3.6",
    install_requires=["numpy", "scipy"],
    packages=find_packages(exclude=["test", "test.*"]),
    test_suite="test"
)
