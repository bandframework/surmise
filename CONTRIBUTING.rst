Contributing to surmise
===========================

Contributions may be made via a GitHub pull request to

    https://github.com/bandframework/surmise

surmise uses the Gitflow model. Contributors should branch from, and
make pull requests to, the develop branch. The main branch is used only
for releases. Pull requests may be made from a fork, for those without
repository write access.

Issues can be raised at

    https://github.com/bandframework/surmise/issues

When a branch closes a related issue, the pull request message should include
the phrase "Closes #N," where N is the issue number. This will automatically
close out the issues when they are pulled into the default branch (currently
main).

Testing contributions
~~~~~~~~~~~~~~~~~~~~~

Code should pass flake8 tests, allowing for the exceptions given in the flake8_
file in the project directory. Some rules of flake8 can be seen at https://www.flake8rules.com.
To check if the code passes the flake8 tests, within the directory run::

  flake8

As you develop your code, we ask developers to include tests specific to their code, and
provide their own tests in the ``tests\`` directory. To run the new tests, you can run the following::

  pytest tests/your-test.py

When a pull request is done to include a new method, we ask developers to include their tests.

Documentation
~~~~~~~~~~~~~~~~~~~~

Clear and complete documentation is essential in order for users to be able to find and
understand the code.

Documentation for individual functions and classes – which includes at least a basic
description, type, and meaning of all parameters, and returns values and usage examples –
is put in docstrings. Those docstrings can be read within the interpreter, and are
compiled into a reference guide in html and pdf format.  If you want to contribute
to the documentation of the architecture of surmise, you can write documentation
in reStructuredText format, and edit in the ``\docs`` directory. If you run ``make html``
in the same directory, HTML pages can viewed.  As you develop your code, we recommend
writing docstrings in your classes and methods.

On any code that is not self-documenting, provide clear comments when some important
thing must be communicated to the user. There is no general rule for the number of
needed comments. Some examples of bad and good commenting habits are given below.

Lack of comments:

Bad Example::

  def mySqrt(x):

      r = x
      precision = 10 ** (-10)

      while abs(x - r * r) > precision:
          r = (r + x / r) / 2

      return r

Good Example::

  # square root of x with Newton-Raphson approximation
  def mySqrt(x):

      r = x
      precision = 10 ** (-10)

      while abs(x - r * r) > precision:
          r = (r + x / r) / 2

      return r

Do not use excessive comments.

Bad Example::

  # looping in range from 0 to 9 and printing the value to the console
  for x in range(10):
    print(x)

Good Example::

  # 0, 1 .. 9
  for x in range(10):
    print(x)

Some general guidance on commenting code can be found at:
  https://www.cs.utah.edu/~germain/PPS/Topics/commenting.html
and:
  http://ideas-productivity.org/wordpress/wp-content/uploads/2021/02/webinar049-softwaredocumentation.pdf

Developer's Certificate of Origin 1.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``surmise`` is distributed under an MIT license (see LICENSE_). The
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

.. _flake8: https://github.com/bandframework/surmise/blob/main/.flake8
.. _LICENSE: https://github.com/bandframework/surmise/blob/main/LICENSE
