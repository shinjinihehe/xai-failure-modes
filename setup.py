from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    # Direct VCS requirements are valid for pip but are not valid metadata in
    # setuptools' ``install_requires``. Keep them in requirements.txt for the
    # documented environment installation path.
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#") and not line.startswith("git+")
    ]

setup(
    name="ipta2026-xai",
    version="1.0.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="XAI Faithfulness Benchmarking for Medical Imaging (IPTA 2026)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shinjinihehe/xai-failure-modes",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
)
