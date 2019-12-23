# AdaptivePIV

## pre-requisites
Before installing AdaptivePIV, you will require a Python 3.7 environment with numpy and cython installed
It is recommended to use a virtual environment for this:

    python -m venv env
Or

    conda create -n env python=3.7
Where in both cases `env` represents the name of the virtual environment. For more information see [Python virtual environments](https://docs.python.org/3/tutorial/venv.html) or [Conda virtual environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Numpy and cython are trivially installed using

    pip install numpy
    pip install cython
or
    
    conda install numpy
    conda install cython

## Installation
Checkout the repository using  

    git clone https://github.com/MattEdwards94/AdaptivePIV.git

With your virtual environment activated, there are 2 options to install AdaptivePIV:

### Installing a static version of the package
To install a static version, the distributable must first be built. 

    python setup.py sdist
This will build a `tar.gz` file under the sub-directory `dist\`
Installing the package and it's dependencies is then achieved via:

    pip install --upgrade dist\PIV-x.x.tar.gz
Where `x.x` is the version number being installed.

### Installing in developer mode
Alternatively, the package can be installed in developer mode. In doing so, changes to the source code are automatically reflected when running the code without having to rebuild (Note that you will likely have to restart the Python interpreter)

    pip install -e .

### Known installation issues
For some unknown reason, installing the 3rd party package `pillow` from the `conda` repository results in tiff files being unable to be opened. This is resolved by installing the package using `pip`. A similar issue is encountered when installing jupyter using conda - I suspect that there is a dependency issue somewhere along the chain. 


