# BAND SDK Community Policy Compatibility for SURmise

This document summarizes the efforts of current and future BAND member packages to achieve compatibility with the BAND SDK community policies. Additional details on the BAND SDK are available [here](https://github.com/bandframework/privateband/blob/team/Resources/bandsdk.md)
and should be considered when filling out this form.

*** A good example of how to complete this form can be found in the [ToBeCompleted](url://here).
*** The BAND SDK is inspired by [xSDK](http://xsdk.info); although the xSDK requirements differ, additional examples can be found in the [xSDK compatibility directory](https://github.com/xsdk-project/xsdk-policy-compatibility).

Please, provide information on your compability status for each mandatory policy, and if possible also for recommended policies.
If you are not compatible, state what is lacking and what are your plans on how to achieve compliance.

For current BAND SDK packages: If you were not fully compatible at some point, please describe the steps you undertook to fulfill the policy. This information will be helpful for future BAND member packages.

To suggest changes to these requirements or obtain more information, please contact the [Design & Oversight Committee](https://github.com/bandframework/privateband/blob/team/Resources/DesignandOversight.md)



**Website:** https://github.com/SURMISE/SURmise

### Mandatory Policies

**BAND SDK**

| # | Policy                 |Support| Notes                   |
|---|------------------------|-------|-------------------------|
| 1.  | Support BAND community GNU Autoconf, CMake, or other build options |Full| SURmise is a Python package and provides a setup.py file for installation. This is compatible with Python's built-in installation feature (``python setup.py install``) and with the pip installer. GNU Autoconf or CMake are unsuitable for a Python package.|
| 2.  | Provide a comprehensive test suite for correctness of installation verification |Full| SURmise contains a comprehensive set of unit tests which can be run individually, or all at once via pytest with a high coverage. ``.\run_tests.sh`` does comprehensive testing.|
| 3.  | Do not assume a full MPI communicator; allow for user-provided MPI communicator |N/a| SURmise provides no MPI functionality. |
| 4.  | Give best effort at portability to key architectures |Full| ? Mac OSX and MS Windows are fully supported.|
| 5.  | Provide a documented, reliable way to contact the development team |Full| The SURmise team can be contacted through the public [issues page on GitHub](https://github.com/SURMISE/SURmise/issues) or via an e-mail to [Ozge Surer](ozgesurer2019@u.northwestern.edu).|
| 6.  | Respect system resources and settings made by other previously called packages |Full| ? SURmise does not modify system resources or settings.|
| 7.  | Come with an open-source license |Full| Uses 2-clause BSD license.|
| 8.  | Provide a runtime API to return the current version number of the software |Full| The version can be returned within python via: `surmise.__version__`.|
| 9.  | Use a limited and well-defined name space (e.g., symbol, macro, library, include) |Full| ? Uses `surmise` namespace. Modules have `surmise` prefix, and in `surmise` folder.|
| 10. | Provide a BAND team-accessible repository |Full| https://github.com/ozgesurer/surmise is publicly available.|
| 11. | Have no hardwired print or IO statements that cannot be turned off |Full| None.|
| 12. | All building and linking against outside copy of external dependencies |Full| ? SURmise does not contain any other package's source code within. Note that Python packages are imported using the conventional `sys.path` system. Alternative instances of a package can be used by, for example, including in the PYTHONPATH environment variable.|
| 13. | Install headers and libraries under `<prefix>/include` and `<prefix>/lib`, respectively |Full| The standard Python installation is used for Python dependencies. This installs external Python packages under `<install-prefix>/lib/python<X.Y>/site-packages/`.|
| 14. | Be buildable using 64-bit pointers; 32-bit is optional |Full| ? There is no explicit use of pointers in SURmise, as Python handles pointers internally and depends on the install of Python, which will generally be 64-bit on supported systems.|
| 15. | All BAND compatibility changes should be sustainable |Full| ? The BAND-compatible package is in the standard release path. All the changes here should be sustainable.|

M1 details <a id="m1-details"></a>: optional: provide more details about approach to addressing topic M1.

M2 details <a id="m2-details"></a>: optional: See `README.rst` file in the `\tests` directory.

### Recommended Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**R1.** Have a public repository. |Full| https://github.com/ozgesurer/surmise |
|**R2.** Free all system resources acquired as soon as they are no longer needed. |Full| Python has built-in garbage collection that frees memory when it becomes unreferenced. |
|**R3.** Provide a mechanism to export ordered list of library dependencies. |Full| The dependencies for SURmise are given in `setup.py` and when pip install or pip setup.py egg_info are run, a file is created `surmise.egg-info/requires.txt` containing the list of required and optional dependencies. If installing through pip, these will automatically be installed if they do not exist. `pip install surmise` installs req. dependencies, while `pip install surmise[extras]` installs both required and optional dependencies. |
|**R4.** Document versions of packages that it works with or depends upon, preferably in machine-readable form.  |Full| The detailed installation instructions come with a full list of tested external dependencies (available on github `README.rst`) |
|**R5.** Have README, SUPPORT, LICENSE, and CHANGELOG files in top directory.  |Full| All files are included in the repository.|
|**R6.** Have sufficient documentation to support use and further development.  |Full| SURmise provides documentation through a *Sphinx* framework. It is published on [readthedocs](https://surmise.readthedocs.io), which includes a user guide covering quick-start, installation, and many usage details. There are several tutorials and examples. The developer guide contains information on workflow and internal modules. A [pdf](https://surmise.readthedocs.io/_/downloads/en/master/pdf/) version is also automatically generated. |
