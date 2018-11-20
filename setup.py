from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['Click>=6.0', ]

setup(
    author="Jun Harashima",
    author_email='j.harashima@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Tool for segmenting ipadic-based analysis results into bunsetsu",
    entry_points={
        'console_scripts': [
            'bseg=bseg.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='bseg',
    name='bseg',
    packages=find_packages(include=['bseg']),
    test_suite='tests',
    url='https://github.com/jun-harashima/bseg',
    version='0.1.0',
    zip_safe=False,
)
