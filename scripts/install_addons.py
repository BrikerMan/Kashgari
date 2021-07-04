import os
import sys

tf_version = str(sys.argv[1])

# TF 2.0, 2.1
if tf_version in ['2.0', '2.1']:
    addons_version = '0.9.1'
# TF 2.2
elif tf_version == '2.2':
    addons_version = '0.11.2'
# TF 2.3+
if tf_version in ['2.3', '2.4', '2.5']:
    addons_version = '0.13.0'

if addons_version:
    print(f'Should Install tensorflow-addons=={addons_version}')
    os.system(f"pip install 'tensorflow-addons=={addons_version}'")
