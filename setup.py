"""
IGQK - Information-Geometric Quantum Compression

A theoretical framework for neural network compression combining:
- Information Geometry (Fisher metric on statistical manifolds)
- Quantum Mechanics (superposition and entanglement)
- Compression Theory (projection onto low-dimensional submanifolds)

This implementation realizes the mathematical theory described in:
"Entwicklung der IGQK-Theorie: Mathematische Details"

Requirements:
- Windows Visual Studio C++ (for building native extensions)
- Python 3.8+
- PyTorch 2.0+
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return __doc__

setup(
    name='igqk',
    version='0.1.0',
    author='IGQK Research Team',
    description='Information-Geometric Quantum Compression for Neural Networks',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/IGQK',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'jupyter>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'igqk-train=igqk.experiments.train:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
