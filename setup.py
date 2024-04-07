from setuptools import setup, find_packages

requirements = [
    "pythainlp"
]

setup(
    name="shannlp",
    version="0.1.0",
    description="Shan Natural Language Processing library",
    long_description_content_type="text/markdown",
    author="NoerNova",
    author_email="noernova666@gmail.com",
    url="https://github.com/NoerNova/ShanNLP",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    package_data={
        "shannlp": [
            "corpus/*",
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords=[
        "shannlp",
        "shan language",
        "tai",
        "NLP",
        "natural language processing",
        "text analytics",
        "text processing",
        "localization",
        "computational linguistics",
        "ShanNLP",
        "Shan NLP",
    ],
)
