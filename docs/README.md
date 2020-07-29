# Website readme

# AdaptivePIV
This repository is for the analysis of PIV images using adaptive PIV image analysis approaches

## pre-requisites
It is assumed that you have python set up on your machine already. If that is not the case, then look [here](https://www.python.org/about/gettingstarted/) to get started.
Before installing AdaptivePIV, you will require a Python 3.7 environment with numpy installed. During installation a number of other packages will be installed - see below.

I would strongly recommend to use a virtual environment for this:

    python -m venv env_name
    env_name\Scripts\activate.bat

Or

    conda create -n env_name python=3.7
    conda activate env_name

Where in both cases `env_name` represents the name of the virtual environment. For more information see [Python virtual environments](https://docs.python.org/3/tutorial/venv.html) or [Conda virtual environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Numpy is trivially installed using

    pip install numpy
or
    
    conda install numpy

**NOTE** While `numpy` is required to _install_ the package, a number of other packages are required to _run_ the code. These will be automatically installed when installing the PIV package. The installed packages are:

- scipy
- h5py
- pillow
- sklearn
- scikit-image
- matplotlib
- bottleneck

## Installation
Checkout the repository using  

    git clone https://github.com/MattEdwards94/AdaptivePIV.git

There are 2 options to install AdaptivePIV:

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

### Using Cython
The python package `Cython` is used in this repository to accelerate certain aspects of the cross-correlation analysis.
Files with the ending `.pyx` are `cythonized` into an equivalent C code which is built when the package is installed.   
This C code is shipped with the repository and therefore the `.pyx` should not need building again.   
If you would like to modify these Cython files then you must first install Cython:

    pip install cython  / conda install cython

Following this, the package will need rebuilding with an additional flag:

    python setup.py sdist --cyth True

### Known installation issues
For some unknown reason, installing the 3rd party package `pillow` from the `conda` repository results in tiff files being unable to be opened. This is resolved by installing the package using `pip`. A similar issue is encountered when installing jupyter using conda - I suspect that there is a dependency issue somewhere along the chain. 

## Usage
Included in this repository are two example PIV image pairs taken from an experiment of the flow over a backwards facing step.
The function `experimental_example(im_number=1, settings=None)` in the file `example.py` will analyse one of these pairs. Options for `im_number` are `1`(default) and `20`.   
If no settings are passed to this function, the following pre-configured defaults will be adopted:

    WidimSettings(init_WS=97,
                  final_WS=33,
                  WOR=0.5,
                  n_iter_main=3,
                  n_iter_ref=1,
                  vec_val='NMT',
                  interp='struc_cub')

There are two ways to see this function in action. Calling `python example.py` will run both the experimental example and the synthetic example (below) one after another. Alternatively, to just run one, launch a `python` session from within the `AdaptivePIV` directory and `import example`, followed by running the desired function `example.experimental_example()` or `example.synthetic_example()`.

If you would like to play about with the various settings, run `help(PIV.analysis.WidimSettings.__init__)` (with `PIV` imported) to see what each value can accept.

The `synthetic_example(settings=None)` shows how artificial images can be created and further analysed. By creating these artificial images, novel algorithms can be tested and validated against the known underlying displacement field.   
The displacement field used in this example is of a non-physical contra rotating vortex array.  
As above, this can either be seen by running the entire file `python example.py` or by importing `example` as a module and running `example.synthetic_example()`

## Contact details
If you have any issues with the above steps, or questions about the PIV analysis algorithm, please feel free to email me at [m.edwards@bristol.ac.uk](mailto:m.edwards@bristol.ac.uk).  


