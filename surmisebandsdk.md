# BAND SDK v0.2 Community Policy Compatibility for surmise


> This document summarizes the efforts of current and future BAND member packages to achieve compatibility with the BAND SDK community policies.  Additional details on the BAND SDK are available [here](https://raw.githubusercontent.com/bandframework/bandframework/main/resources/sdkpolicies/bandsdk.md) and should be considered when filling out this form. The most recent copy of this template exists [here](https://raw.githubusercontent.com/bandframework/bandframework/main/resources/sdkpolicies/template.md).
>
> This file should filled out and placed in the directory in the `bandframework` repository representing the software name appended by `bandsdk`.  For example, if you have a software `foo`, the compatibility file should be named `foobandsdk.md` and placed in the directory housing the software in the `bandframework` repository. No open source code can be included without this file.
>
> All code included in this repository will be open source.  If a piece of code does not contain a open-source LICENSE file as mentioned in the requirements below, then it will be automatically licensed as described in the LICENSE file in the root directory of the bandframework repository.
>
> Please provide information on your compatibility status for each mandatory policy and, if possible, also for recommended policies. If you are not compatible, state what is lacking and what are your plans on how to achieve compliance. For current BAND SDK packages: If you were not fully compatible at some point, please describe the steps you undertook to fulfill the policy. This information will be helpful for future BAND member packages.
>
> To suggest changes to these requirements or obtain more information, please contact [BAND](https://bandframework.github.io).
>
> Details on citing the current version of the BAND Framework can be found in the [README](https://github.com/bandframework/bandframework).


**Website:** https://github.com/bandframework/surmise

**Contact:** The surmise team, whose contact details are listed in [SUPPORT](SUPPORT.rst).

**Icon:** https://avatars.githubusercontent.com/u/77858356?s=200&v=4

**Description:** surmise is a Python package that is designed to provide a surrogate model interface for calibration, uncertainty quantification, and other tools.


### Mandatory Policies

**BAND SDK**

| # | Policy                 |Support| Notes                   |
|---|-----------------------|-------|-------------------------|
| 1. | Support BAND community GNU Autoconf, CMake, or other build options |Full| surmise is a Python package and provides a setup.py file for installation. This is compatible with Python's built-in installation feature (``python setup.py install``) and with the pip installer. GNU Autoconf or CMake are unsuitable for a Python package. |
| 2. | Have a README file in the top directory that states a specific set of testing procedures for a user to verify the software was installed and run correctly. | Full| README explains full test procedure.|
| 3. | Provide a documented, reliable way to contact the development team |Full| The surmise team can be contacted through the public [issues page on GitHub](https://github.com/bandframework/surmise/issues) or via an e-mail to [Özge Sürer](surero@miamioh.edu).|
| 4. | Come with an open-source license |Full| surmise uses the MIT license. [M4 details](#m4-details)|
| 5. | Provide a runtime API to return the current version number of the software |Full| The version can be returned within Python via: `surmise.__version__`.|
| 6. | Provide a BAND team-accessible repository |Full| https://github.com/bandframework/surmise |
| 7. | Must allow installing, building, and linking against an outside copy of all imported software that is externally developed and maintained |Full| surmise does not contain any other package's source code within. Note that Python packages are imported using the conventional `sys.path` system. Alternative instances of a package can be used, for example, by including them through an appropriate definition of the PYTHONPATH environment variable.|
| 8. |  Have no hardwired print or IO statements that cannot be turned off |Full| There are no mandatory print statements: any print statements for code feedback in a method can be suppressed via `verbose` argument. |

M4 details <a id="m4-details"></a>: This was chosen based on the MIT license being the default license for BAND.


### Recommended Policies

| #  | Policy                 |Support| Notes                   |
|---|------------------------|-------|-------------------------|
|**R1.**| Have a public repository. |Full| https://github.com/bandframework/surmise is publicly available. |
|**R2.**| Free all system resources acquired as soon as they are no longer needed. |Full| Python has built-in garbage collection that frees memory when it becomes unreferenced. |
|**R3.**| Provide a mechanism to export ordered list of library dependencies. |Full| The dependencies for surmise are given in `setup.py` and when pip install or pip setup.py egg_info are run, a file is created `surmise.egg-info/requires.txt` containing the list of required and optional dependencies. If installing through pip, these will automatically be installed if they do not exist. `pip install surmise` installs required dependencies. |
|**R4.**| Document versions of packages that it works with or depends upon, preferably in machine-readable form.  |Full| The detailed installation instructions come with a full list of tested external dependencies (available on github `README.rst`). |
|**R5.**| Have README, SUPPORT, LICENSE, and CHANGELOG files in top directory.  |Full| All files are included in the repository. |
|**R6.**| Have sufficient documentation to support use and further development. |Full| surmise provides documentation through a *Sphinx* framework. It is published on [readthedocs](https://surmise.readthedocs.io), which includes a user guide covering quick-start, installation, and many usage details. There are several tutorials and examples. The developer guide contains information on internal modules. |
|**R7.**| Be buildable using 64-bit pointers; 32-bit is optional |Full| There is no explicit use of pointers in surmise, as Python handles pointers internally and depends on the install of Python, which will generally be 64-bit on supported systems.|
|**R8.**| Do not assume a full MPI communicator; allow for user-provided MPI communicator |N/a| None. |
|**R9.**| Use a limited and well-defined name space (e.g., symbol, macro, library, include) |Full| surmise uses `surmise` namespace. Modules have `surmise` prefix, and in `surmise` folder.|
|**R10.**| Give best effort at portability to key architectures |Full| surmise is being regularly tested on Mac OS, Linux, and MS Windows. The current set of automatically tested, common architectures is viewable [here](https://github.com/bandframework/surmise/blob/main/.github/workflows/python-package.yml) |
|**R11.**| Install headers and libraries under `<prefix>/include` and `<prefix>/lib`, respectively |Full| The standard Python installation is used for Python dependencies. This installs external Python packages under `<install-prefix>/lib/python<X.Y>/site-packages/`.|
|**R12.**| All BAND compatibility changes should be sustainable |Full| The BAND-compatible package is in the standard release path. All the changes here should be sustainable.|
|**R13.**| Respect system resources and settings made by other previously called packages |Full| surmise does not modify system resources or settings.|
|**R14.**| Provide a comprehensive test suite for correctness of installation verification |Full| surmise contains a comprehensive set of unit tests that can be run, individually or all at once, via pytest with a high coverage. Running the provided ``.\run_tests.sh`` performs comprehensive testing. [R14 details](#r14-details)|

R14 details <a id="r14-details"></a>: See the `README.rst` file in the `tests` directory.
