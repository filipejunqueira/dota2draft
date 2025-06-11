from setuptools import setup, find_packages

setup(
    name='dota2draft',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add dependencies from requirements.txt here
        # This makes the project's dependencies explicit
        'typer',
        'rich',
        'requests',
        'torch',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'PyYAML',
    ],
    entry_points={
        'console_scripts': [
            'dota2draft = dota2draft_cli:app',
        ],
    },
)
