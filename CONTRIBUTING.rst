Contributing to surmise
===========================

Contributions may be made via a GitHub pull request to

    https://github.com/surmising/surmise

surmise uses the Gitflow model. Contributors should branch from, and
make pull requests to, the develop branch. The master branch is used only
for releases. Pull requests may be made from a fork, for those without
repository write access.

Issues can be raised at

    https://github.com/surmising/surmise/issues

When a branch closes a related issue, the pull request message should include
the phrase "Closes #N," where N is the issue number. This will automatically
close out the issues when they are pulled into the default branch (currently
master).

Testing contributions
~~~~~~~

Code should pass flake8 tests, allowing for the exceptions given in the flake8_ 
file in the project directory.

As you develop your code, you may want to include tests specific to your code.
We encourage developers to provide their own tests in ``tests\`` directory. To run the
new tests, you can run the following::

  pytest tests/your-test.py


Documentation
~~~~~~~~~~~~~~~~~~~~

Clear and complete documentation is essential in order for users to be able to find and
 understand the code. 

Documentation for individual functions and classes – which includes at least a basic 
description, type and meaning of all parameters and returns values, and usage examples – 
is put in docstrings. Those docstrings can be read within the interpreter, and are 
compiled into a reference guide in html and pdf format.   If you want to contribute 
to the documentation of the architecture of surmise, you can write documentation
in reStructuredText format, and edit in ``\doc`` directory. If you run ``make html``
in the same directory, HTML pages can viewed.  As you develop your code, we recommend 
writing docstrings in your classes and methods.

On any code that is not self-documenting, provide clear comments when some important 
thing must be communicated to the user.  There is no general rul for the number of
needed comments, 

Some general guidence on commenting code can be found at::
  https://www.cs.utah.edu/~germain/PPS/Topics/commenting.html
and::
  http://ideas-productivity.org/wordpress/wp-content/uploads/2021/02/webinar049-softwaredocumentation.pdf

Developer's Certificate of Origin 1.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`surmise` is distributed under an MIT license (see LICENSE).  The
act of submitting a pull request (with or without an explicit
Signed-off-by tag) will be understood as an affirmation of the
following:

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.

.. _flake8: https://github.com/surmising/surmise/blob/develop/.flake8
