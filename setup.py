from pathlib import Path
import platform
from setuptools import find_packages, setup


# on_windows = platform.system() == "Windows"
def get_requirements(path: str):
    return [l.strip() for l in open(path)]

setup(
    name='PrivateLLM',
    version='0.1',
    long_description="A private LLM based MAS",
    author='Xavier',
    packages= find_packages(),
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Researchers",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=get_requirements("requirements.txt"),
)

# def get_version():
#     version_file = Path(
#         __file__).resolve().parent / "tensorrt_llm" / "version.py"
#     version = None
#     with open(version_file) as f:
#         for line in f:
#             if not line.startswith("__version__"):
#                 continue
#             version = line.split('"')[1]

#     if version is None:
#         raise RuntimeError(f"Could not set version from {version_file}")

#     return version
