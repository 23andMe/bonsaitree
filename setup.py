from setuptools import setup, find_packages, Extension

NAME = "bonsaitree"
DESCRIPTION = "Python package for building pedigrees."

README = open("README.md", 'r').read()
LICENSE = open("LICENSE.txt", 'r').read()

setup(
    name=NAME,
    use_scm_version=True,
    author="23andMe Engineering",
    author_email="ejewett@23andme.com",
    description=DESCRIPTION,
    license=LICENSE,
    long_description=README,
    long_description_content_type="text/markdown",
    packages=[NAME],
    package_dir={NAME : NAME},
    package_data={NAME: ["models/*.json"]},
    include_package_data=True,
    url="https://github.com/23andme/" + NAME,
    zip_safe=True,
    install_requires=["Cython", "funcy", "numpy", "scipy", "six", "setuptools-scm", "wheel"],
    setup_requires=["Cython", "setuptools-scm", "wheel"],
    #ext_modules=[
    #    Extension("bonsaitree.copytools", sources=["bonsaitree/copytools.pyx"])
    #],
)
