import os
import sys

from setuptools import setup
from distutils.sysconfig import get_python_lib

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write(
        """
        ==========================
        UNSUPPORTED PYTHON VERSION
        ==========================

        This version of ytbot requires Python {}.{} but you are attempting to install it on Python {}.{}.

        You may also be using a version of pip that doesn't understand the python_requires classifier. Make sure that you have pip >= 9.0 and setuptools >= 24.2, and try installing the package again. The following command allows you to do this.

        $ python -m pip install --upgrade pip setuptools
        $ python -m pip install -e .

        This will install the latest version of pip, setuptools, and ytbot. Older versions of ytbot may be available but are at end-of-life support.
        """.format(
            *(REQUIRED_PYTHON + CURRENT_PYTHON)
        )
    )
    sys.exit(1)

setup(
    name="cheapfake",
    version="0.0.1",
    packages=["cheapfake", "cheapfake.utils", "cheapfake.blazeface",],
    zip_safe=False,
)
