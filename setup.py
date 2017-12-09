from setuptools import setup

project = 'easyplot'
copyright = '2017, George Dadunashvili'
author = 'George Dadunashvili'
language = "en"

version = "0"
release = "1"
change = '7'
licence_type = "MIT"


package_name = project


v = f"{version}.{release}.{change}"

# release flag can be:
# None,
# 'a' for alpha version,
# 'b' for beta version
# and 'rc' for release candidate
release_flag = "a"
if release_flag is not None:
    v = f"{version}.{release}.{change}_{release_flag}"

description = 'collection of few tools and senceble defaults which make ' \
              'scientific plotting easier'

url = 'https://github.com/gdadunashvili/easyplot'


def readme():
    with open('README.rst') as f:
        return f.read()


def display_license():
    with open('LICENSE') as f:
        return f.read()


setup(name=package_name,
      version=v,
      description=description,
      url=url,
      author=author,
      license=licence_type,
      packages=['easyplot'],
      install_requires=[
          'numpy',
          'matplotlib'
      ],
      # test_suite='pytest',
      # tests_require=['hypothesis', 'pytest'],
      zip_safe=False)
