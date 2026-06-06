"""
tools/send_start.py — Writes start_command.json to trigger flight.
Use this instead of dashboard for testing.

Usage: python tools/send_start.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.makedirs("state", exist_ok=True)
cmd = {"command": "START", "sortie_id": "test_sortie", "timestamp": time.time()}
with open("state/start_command.json", "w") as f:
    json.dump(cmd, f, indent=2)
print("START command written to state/start_command.json")
