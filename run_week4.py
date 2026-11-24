#!/usr/bin/env python3
"""
run_week4.py — Week 4 Standalone Web Application Launcher
Data Pipeline Failure Prediction System (ANNA Course 2307411T - Track 2: Sequential Models)

Group 12 (Equal Contributors):
 - Ayush Nagras     (PRN: 202301070001)
 - Siddhesh Bhivare (PRN: 202301070005)
 - Om Nikam         (PRN: 202301070006)
 - Vaibhav Tajane   (PRN: 202301070008)

Week 4 Deliverable (CO5 - Deployment & User Interface Application):
Launches the interactive production Streamlit Web Dashboard:
- Tab 1: Live Risk Monitor & Early Warning Alert
- Tab 2: Interactive What-If Scenario Simulator with live sliders
- Tab 3: Model Ablation & Figures (5 high-res charts)
- Tab 4: Airflow Data Engineering Pipeline Inspector
- Tab 5: System Architecture & Viva Defense Matrix
"""

import os
import sys
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("\n" + "#" * 80)
    print(" WEEK 4: STREAMLIT WEB DASHBOARD APPLICATION LAUNCHER (CO5)")
    print(" Course: 2307411T - Advanced Neural Network Architectures (ANNA) | Track 2")
    print(" Group 12: Ayush Nagras, Siddhesh Bhivare, Om Nikam, Vaibhav Tajane")
    print("#" * 80)
    
    # Verify required assets
    required_files = [
        os.path.join(PROJECT_ROOT, "app.py"),
        os.path.join(PROJECT_ROOT, "models", "best_attention_model.pth"),
        os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_telemetry.csv"),
        os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_logs.json")
    ]
    
    print("\n[STEP 1/2] Verifying Application Dependencies & Model Checkpoints...")
    for f in required_files:
        if not os.path.exists(f):
            print(f" ✗ Error: Missing required file: {f}")
            sys.exit(1)
        print(f" ✓ Verified: {os.path.relpath(f, PROJECT_ROOT)}")
        
    print("\n[STEP 2/2] Launching Interactive Streamlit Dashboard...")
    print(" • Local URL:   http://localhost:8501")
    print(" • Network URL: http://127.0.0.1:8501")
    print(" • Press Ctrl+C in terminal to stop server.\n")
    print("-" * 80)
    
    streamlit_bin = "/home/ayush/Desktop/projects/cx-backend/app/requirements/myenv/bin/streamlit"
    cmd = [
        streamlit_bin, "run",
        os.path.join(PROJECT_ROOT, "app.py"),
        "--server.port", "8501",
        "--server.headless", "true"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n[INFO] Dashboard server stopped cleanly.")
    except Exception as e:
        print(f"\n[ERROR] Failed to run dashboard: {e}")

if __name__ == "__main__":
    main()
