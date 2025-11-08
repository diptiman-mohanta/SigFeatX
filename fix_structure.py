import os
import shutil

# Move files
files = [('aggregator.py', 'SigFeatX/aggregator.py'), ('utils.py', 'SigFeatX/utils.py')]
for src, dst in files:
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f'Moved {src} to {dst}')

# Move directories
dirs = [('decompose', 'SigFeatX/decompose'), ('features', 'SigFeatX/features')]
for src, dst in dirs:
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
        print(f'Moved {src} to {dst}')

print('Done!')
