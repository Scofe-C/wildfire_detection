"""
End-to-end test for OBJ-2 fire spread simulation on Eaton Fire Jan 7 2025.

NOTE: This script has been retired. Fire spread simulation is now handled by
PythonFireSpreadSimulator (Rothermel) and PropagatorSpread via evaluate_obj2.py.

Run from model-pipeline root:
    cd <repo-root>/model-pipeline
    python evaluate_obj2.py --mode realtime
"""
import sys


def main():
    print(
        "This script has been retired. "
        "Use evaluate_obj2.py with --mode realtime for fire spread evaluation."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
