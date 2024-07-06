

autopep8-lib:
	autopep8 --in-place --aggressive --aggressive churn_library.py

autopep8-test:
	autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py

pylint-lib:
	pylint churn_library.py

pylint-test:
	pylint churn_script_logging_and_tests.py

pylint-all:
	pylint *

test-log:
	ipython churn_script_logging_and_tests.py
