import os
from distutils.version import LooseVersion
from importlib.metadata import version

tf_version = LooseVersion(version('tensorflow'))

print(f'TF version: {tf_version}')

addons_version = ''

# TF 2.0, 2.1
if tf_version < LooseVersion('2.2.0'):
    addons_version = '0.9.1'
# TF 2.2
elif tf_version < LooseVersion('2.3.0'):
    addons_version = '0.11.2'
# TF 2.3+
elif tf_version < LooseVersion('2.6.0'):
    addons_version = '0.13.0'
else:
    print(f'New Version, {tf_version}.')

if addons_version:
    print(f'Should Install tensorflow-addons=={addons_version}')
    os.system(f"pip install tensorflow-addons=={addons_version}")
