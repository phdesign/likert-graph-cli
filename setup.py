from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="likert-graph-cli",  # Replace with your own username
    version="0.0.1",
    author="Paul Heasley",
    author_email="paul@phdesign.com.au",
    description="A likert survey result graph generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phdesign/likert-graph-cli",
    py_modules=["likert_graph_cli"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["click", "matplotlib", "numpy", "pandas", "colour",],
    extras_require={"dev": ["black==19.10b0", "flake8==3.8.3", "isort==4.3.21", "pytest==6.1.2", "coverage==5.3",]},
    entry_points="""
        [console_scripts]
        likert-graph-cli=likert_graph_cli:main
    """,
)
