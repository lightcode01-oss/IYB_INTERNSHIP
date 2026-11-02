"""Use a noninteractive plotting backend in headless app and notebook checks."""
import os
os.environ['MPLBACKEND'] = 'Agg'
