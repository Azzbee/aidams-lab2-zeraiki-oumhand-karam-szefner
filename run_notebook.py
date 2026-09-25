"""Execute the notebook with the Python interpreter running this script."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main() -> None:
    base = Path(__file__).resolve().parent
    notebook_path = base / "lab_2.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    with TemporaryDirectory(prefix="lab2-kernel-") as temporary:
        kernel_dir = Path(temporary) / "kernels" / "lab2"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Lab 2 Python",
                    "language": "python",
                }
            )
        )
        previous_path = os.environ.get("JUPYTER_PATH", "")
        os.environ["JUPYTER_PATH"] = temporary + (
            os.pathsep + previous_path if previous_path else ""
        )
        client = NotebookClient(
            notebook,
            timeout=1800,
            kernel_name="lab2",
            resources={"metadata": {"path": str(base)}},
        )
        try:
            client.execute()
        finally:
            nbformat.write(notebook, notebook_path)
            os.environ["JUPYTER_PATH"] = previous_path
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (base / "lab_2.html").write_text(html)
    print("Executed all cells and exported lab_2.html.")


if __name__ == "__main__":
    main()
