from setuptools import setup, find_packages

setup(
    name="tpu",
    version="0.1.0",
    author="Sohaib Errabii",
    author_email="sohaib.errabii@inria.fr",
    description="TPU in amaranth",
    packages=find_packages(),    
    python_requires=">=3.10",
    install_requires=[
        "amaranth[builtin-yosys] @ git+https://github.com/amaranth-lang/amaranth.git",
        "amaranth_soc @ git+https://github.com/sohaiberrabii/amaranth-soc@patch-1",
    ],
    extras_require={
        "test": [ 
            "pytest",
            "numpy",
            "cocotb @ git+https://github.com/cocotb/cocotb.git",
            "torchvision",
        ],
    },

)
