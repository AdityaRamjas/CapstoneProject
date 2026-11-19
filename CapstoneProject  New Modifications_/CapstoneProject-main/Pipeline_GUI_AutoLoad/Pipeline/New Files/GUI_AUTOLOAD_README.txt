Run python -u main.py from this folder on a Windows computer with NumPy, SciPy and OpenSim 4.6 installed at C:\OpenSim 4.6.
The pipeline now launches OpenSim64.exe and uses a Windows PowerShell keyboard automation helper to request opening P2_PointCloud_scaled.osim.
The helper uses Ctrl+O and the Windows clipboard; keep the desktop unlocked and do not type/click while OpenSim is starting.
This is best-effort GUI automation and has not been tested on your machine. If Ctrl+O does not open Open Model on your OpenSim build, open it manually via File > Open Model.
Copied .venv excluded because virtual environments are computer-specific.
