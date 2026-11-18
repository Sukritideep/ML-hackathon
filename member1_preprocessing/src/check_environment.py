
import sys
import importlib

print("Python version:", sys.version)
print("\nChecking installed libraries:\n")

libraries = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "shapely",
    "pyproj",
    "networkx",
    "folium"
]

for library in libraries:
    try:
        module = importlib.import_module(library)
        version = getattr(module, "__version__", "Version unavailable")
        print(f"{library}: INSTALLED ({version})")
    except ImportError:
        print(f"{library}: NOT INSTALLED")
