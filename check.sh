# check imports
isort --diff . || exit_code=$?
# code analysis
pylint . || exit_code=$?
# style guide enforcement
flake8 . || exit_code=$?
# end
if [[ "$exit_code" -ne "0" ]]; then echo "Previous command failed"; fi;
exit $exit_code