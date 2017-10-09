#!/bin/bash
set -e
. activate tencentGPU

PASSWD=sha1:'8c98103ae78c:a122ca8235f22615c06d6a3405b74e0070ae2e56'
jupyter notebook --no-browser --ip=* --NotebookApp.password="$PASSWD" --allow-root "$@"
