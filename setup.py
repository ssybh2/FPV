from setuptools import find_packages, setup

setup(
    name="q250-uzh-isaaclab",
    version="0.3.0",
    description="Q250 dynamics workspace for UZH-style drone racing in Isaac Lab",
    packages=find_packages(),
    include_package_data=True,
    package_data={"q250_uzh.data": ["motor_lut.csv"]},
    python_requires=">=3.10",
)
