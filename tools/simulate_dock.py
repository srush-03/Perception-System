"""
tools/simulate_dock.py — Writes DOCKED state to mission_state.json.
Use this for testing post-flight pipeline without physical Arduino.

Usage:  python tools/simulate_dock.py
        python tools/simulate_dock.py --delay 5    (wait 5 seconds first)
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=int, default=0)
    args = parser.parse_args()

    if args.delay:
        print(f"Waiting {args.delay}s before sending dock signal...")
        time.sleep(args.delay)

    os.makedirs("state", exist_ok=True)
    state = {
        "state":          "DOCKED_FOR_CHARGING",
        "prev_state":     "LANDING_DETECTED",
        "sortie_id":      "test_sortie",
        "timestamp":      time.time(),
        "trigger_source": "simulate_dock.py",
    }
    with open("state/mission_state.json", "w") as f:
        json.dump(state, f, indent=2)
    print("DOCKED signal written to state/mission_state.json")


if __name__ == "__main__":
    main()
