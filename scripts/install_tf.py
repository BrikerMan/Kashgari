import os
import sys

tf_args = str(sys.argv[1])
major_version, minor_version = tf_args.split('.')

command = (
    f"pip install 'tensorflow>={major_version}.{minor_version}.0,"
    f"<{major_version}.{int(minor_version)+1}.0'"
)
print(command)
os.system(command)
