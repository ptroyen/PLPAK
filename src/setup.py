from setuptools import setup, find_packages

setup(
    name='plpak',
    version='0.3.0',
    author='Sagar Pokharel',
    author_email='pokharel_sagar@tamu.edu',  # Primary email address
    description='Package for Low Temperature PLasma kinetics',
    long_description=open('README.md').read(),  # Ensure README.md is in src/
    long_description_content_type='text/markdown',  # or 'text/x-rst' if using reStructuredText
    packages=find_packages(where='.', include=['plpak']),
    package_dir={'': '.'},  # Directory containing the package code
    include_package_data=True,
    install_requires=[
        'cantera==2.6.0',  # Specific version of Cantera
        'sympy==1.11.1',   # Specific version of sympy
        'numpy==1.23.5',   # Specific version of numpy
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            # If you have command line scripts, add them here
            # 'script_name = plpak.module:function',
        ],
    }
)
