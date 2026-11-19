import os
import subprocess
from pathlib import Path

install_root = Path(os.environ.get("OPENSIM_HOME", r"C:\OpenSim 4.4"))
OPENSIM_CMD = next(
    (
        path
        for path in [
            install_root / "bin" / "opensim-cmd.exe",
            Path(r"C:\OpenSim 4.4\bin\opensim-cmd.exe"),
            Path(r"C:\OpenSim 4.6\bin\opensim-cmd.exe"),
        ]
        if path.is_file()
    ),
    None,
)

MODEL = next(
    (
        path
        for path in [
            install_root / "sdk" / "Python" / "opensim" / "tests" / "arm26.osim",
            Path(r"C:\OpenSim 4.4\sdk\Python\opensim\tests\arm26.osim"),
            Path(r"C:\OpenSim 4.6\sdk\Python\opensim\tests\arm26.osim"),
        ]
        if path.is_file()
    ),
    None,
)

if OPENSIM_CMD is None:
    raise FileNotFoundError(
        "OpenSim command-line tool not found. Set OPENSIM_HOME or update the path to the OpenSim bin folder."
    )

if MODEL is None:
    raise FileNotFoundError("OpenSim model not found.")

print(f"Launching OpenSim visualization with: {OPENSIM_CMD}")

subprocess.Popen(
    [
        str(OPENSIM_CMD),
        "viz",
        "model",
        str(MODEL)
    ],
    cwd=str(MODEL.parent)
)

print("OpenSim visualization launch requested!")