from setuptools import setup, find_packages


def main():
    console_scripts = [
        "train=handwriting_synthesis.train:main",
    ]
    setup(
        name='handwriting_synthesis',
        version='0.1',
        author='Vladislav Bataev',
        packages=find_packages("src"),
        package_dir={'': 'src'},
        description='Neural based handwriting synthesis implementation.',
        install_requires=[
            'torch==1.2.0',
            'numpy==1.14.5',
            'jupyter',
            'matplotlib',
        ],
        test_suite='nose.collector',
        tests_require=['nose', 'nose-cover3'],
        entry_points={
            'console_scripts': console_scripts,
        }
    )


if __name__ == "__main__":
    main()
