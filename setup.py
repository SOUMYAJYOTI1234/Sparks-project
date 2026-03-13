from setuptools import find_packages, setup

setup(
    name="student_score_prediction",
    version="1.0.0",
    author="Soumyajyoti Chatterjee",
    author_email="soumyajyoti@example.com",
    description="End-to-end ML project to predict student scores based on study hours",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "flask",
        "dill",
    ],
)
