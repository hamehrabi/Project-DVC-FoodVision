from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "Project-DVC-FoodVision"
AUTHOR_USER_NAME = "Hamed"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ["black",
                        "dvc==2.45.1",
                        "tqdm",
                        "tensorflow==2.8.0",
                        "joblib",
                        "Pillow",
                        "scipy"
                        ]


setup(
    name=SRC_REPO,
    version="0.0.3",
    author=AUTHOR_USER_NAME,
    description="A small package for DVC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/hamehrabi/Project-DVC-FoodVision",
    author_email="mehrabi.hamed@outlook.com",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.8",
    install_requires=LIST_OF_REQUIREMENTS
)
