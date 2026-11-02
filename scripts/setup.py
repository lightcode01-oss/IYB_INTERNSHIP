"""Create an isolated, reproducible project environment without global changes."""
import argparse,subprocess,sys,venv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check',action='store_true',help='Check setup prerequisites without installing')
args=parser.parse_args()
if sys.version_info[:2]!=(3,14): raise SystemExit('Use Python 3.14 for the captured environment')
assert (ROOT/'requirements-lock.txt').is_file()
if args.check:
    print('Python 3.14 and locked dependency file: PASS')
else:
    target=ROOT/'.venv'
    venv.EnvBuilder(with_pip=True).create(target)
    executable=target/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    subprocess.run([str(executable),'-m','pip','install','-r',str(ROOT/'requirements-lock.txt')],check=True)
    subprocess.run([str(executable),'-m','pip','check'],check=True)
    print('Environment ready:',target.name)
