# setup.py

from setuptools import setup, find_packages

setup(
    name='spam-detection',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'pandas',
        'numpy',
    ],
    # Optionally include data files
    include_package_data=True,
    package_data={
        # Include any files in the data directory
        '': ['data/*'],
        # Include pre-trained models
        'models': ['models/*/*']
    },
    entry_points={
        # Optionally create custom commands to run scripts
        'console_scripts': [
            'train_spam=spam_detector.trainer:main',
            # ...
        ],
    },
)
