"""run_python — execute Python (numpy/scipy/sympy/matplotlib) in the sandbox; figures become artifacts."""
from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from app.services import sandbox
from app.tools.base import ToolContext, ToolResult, ToolSpec
from app.tools.bash import register_changes
from app.services.artifact_storage import IMAGE_EXT, resolve_in_workspace, snapshot, workspace_dir
from app.tools.common import register_file_artifact
from app.tools.registry import compact_json, registry

PRELUDE = """\
import os as _os
_os.makedirs('figures', exist_ok=True)
import matplotlib as _mpl
_mpl.use('Agg')
try:
    import matplotlib.pyplot as _plt
    _plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
except Exception:
    pass
"""

EPILOGUE = """
try:
    import matplotlib.pyplot as _plt
    _figs = _plt.get_fignums()
    for _i, _n in enumerate(_figs):
        _f = _plt.figure(_n)
        if not getattr(_f, '_stemos_saved', False):
            _f.savefig(f'figures/figure_{_STEMOS_RUN}_{_i + 1}.png')
except Exception as _e:
    print('[figure autosave failed]', _e)
"""


def check_output(path: Path, kind: str, slides: int | None) -> str:
    try:
        if kind == "presentation":
            from pptx import Presentation

            count = len(Presentation(str(path)).slides)
            if not count or (slides is not None and count != slides):
                return f"{path.name} contains {count} slides. Expected {slides or 'at least one'}."
        elif path.suffix.lower() == ".svg":
            if ElementTree.parse(path).getroot().tag != "{http://www.w3.org/2000/svg}svg":
                return f"{path.name} is not an SVG image."
        else:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
    except Exception as exc:
        return f"{path.name} is unreadable: {exc}"
    return ""


async def run_checked(ctx: ToolContext, args: dict[str, Any], code: str) -> tuple[sandbox.ExecResult, list[dict], str]:
    kind = args["output_type"]
    extensions = {".pptx"} if kind == "presentation" else IMAGE_EXT
    ws = workspace_dir(ctx.project_id).resolve()
    attempts = (ws / ".stemos" / "attempts").resolve()
    if not attempts.is_relative_to(ws):
        raise ValueError("The attempt directory escapes the project workspace.")
    attempts.mkdir(parents=True, exist_ok=True)
    expected = args.get("output_path")
    if expected:
        _, expected = resolve_in_workspace(ctx.project_id, expected)
    with tempfile.TemporaryDirectory(dir=attempts) as directory:
        stage = Path(directory)
        inputs = {}
        for name in args.get("inputs", []):
            source, rel = resolve_in_workspace(ctx.project_id, name)
            if source.is_file():
                target = stage / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(shutil.copyfile, source, target)
                inputs[rel] = target.stat().st_mtime_ns
        result = await sandbox.run_python(ctx.project_id, code, args.get("timeout"), cwd=stage)
        if result.exit_code or result.timed_out or ctx.cancel_event.is_set():
            return result, [], result.stderr[-500:] or "Execution did not finish."
        files = {
            path.relative_to(stage).as_posix(): path
            for path in sorted(stage.rglob("*"))
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(stage)
            and ".stemos" not in path.relative_to(stage).parts
            and inputs.get(path.relative_to(stage).as_posix()) != path.stat().st_mtime_ns
        }
        outputs = {
            rel: path for rel, path in files.items()
            if path.suffix.lower() in extensions and (not expected or rel == expected)
        }
        if not outputs:
            return result, [], f"Expected {expected or ('a PowerPoint deck' if kind == 'presentation' else 'an image file')}, but the program did not produce it."
        for path in outputs.values():
            problem = await asyncio.to_thread(check_output, path, kind, args.get("slide_count"))
            if problem:
                return result, [], problem
        if ctx.cancel_event.is_set():
            return result, [], "Execution was cancelled."
        destinations = [(path, *resolve_in_workspace(ctx.project_id, rel)) for rel, path in outputs.items()]
        if kind == "image":
            for rel, path in files.items():
                if rel not in inputs and rel not in outputs and path.suffix.lower() not in IMAGE_EXT and path.stat().st_size:
                    destinations.append((path, *resolve_in_workspace(ctx.project_id, rel)))
        for path, target, _ in destinations:
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
        arts = [await register_file_artifact(ctx, rel, meta={"origin": "execution"}) for _, _, rel in destinations]
        return result, arts, ""


async def run_python_tool(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    code = str(args["code"])
    if re.search(r"\b(subprocess|os\.system|socket|ctypes|pty)\b", code):
        # not a security boundary (the sandbox is), but keeps the model on-task
        return ToolResult.failure("run_python is for computation/plotting; process spawning and raw sockets are not allowed. Use the bash tool if a shell command is needed.")
    tag = ctx.run_id[:6]
    saves_explicitly = "savefig" in code
    full = PRELUDE + f"_STEMOS_RUN = '{tag}'\n" + code + ("\n" if saves_explicitly else EPILOGUE)
    if args.get("output_type") in ("image", "presentation"):
        res, arts, error = await run_checked(ctx, args, full)
    else:
        before = await asyncio.to_thread(snapshot, ctx.project_id)
        res = await sandbox.run_python(ctx.project_id, full, args.get("timeout"))
        error = res.stderr[-500:] or "Python execution failed." if res.exit_code or res.timed_out else ""
        arts = await register_changes(ctx, before) if not error and not ctx.cancel_event.is_set() else []
    images = [{"name": a["filename"], "url": a["url"], "path": a["path"]} for a in arts if a["type"] == "image"]
    payload = {**res.to_dict(), "images": images, "files": [a["path"] for a in arts if a["type"] != "image"]}
    if images:
        payload["note"] = "Images are attached to the answer automatically. Refer to them by file name; do not invent URLs."
    return ToolResult(content=compact_json(payload, 8000), data=payload, ok=not error, error=error, artifacts=arts)


registry.register(
    ToolSpec(
        name="run_python",
        description="Run Python 3 code in the sandboxed project workspace (numpy, scipy, sympy, pandas, matplotlib available). Matplotlib figures are saved and attached automatically.",
        parameters={
            "properties": {
                "code": {"type": "string", "description": "Complete Python program. Use print() for results. Save figures under figures/ (e.g. plt.savefig('figures/plot.png'))."},
                "timeout": {"type": "integer", "description": "Seconds (default 60)."},
                "output_type": {"type": "string", "enum": ["image", "presentation"], "description": "Artifact kind to validate before publishing."},
                "output_path": {"type": "string", "description": "Required output path, if specified."},
                "slide_count": {"type": "integer", "description": "Required number of slides for a presentation."},
                "inputs": {"type": "array", "items": {"type": "string"}, "description": "Existing workspace files to copy into a checked run."},
            },
            "required": ["code"],
        },
        handler=run_python_tool,
        category="compute",
        timeout=620,
        read_only=False,
        when_to_use="Plots/visualizations, numerical simulation, symbolic math (sympy), statistics, data processing, verifying derivations.",
        when_not_to_use="One-line arithmetic (use eval) or literature questions.",
        returns='JSON {"exit_code", "stdout", "stderr", "images":[{"name","url","path"}], "files":[...]}.',
        example_args={
            "code": "import numpy as np\nimport matplotlib.pyplot as plt\nx = np.linspace(0, 2*np.pi, 400)\nplt.plot(x, np.sin(x), label='sin x')\nplt.plot(x, np.cos(x), label='cos x')\nplt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.title('sin vs cos')\nplt.savefig('figures/sin_vs_cos.png')\nprint('max difference', np.max(np.abs(np.sin(x)-np.cos(x))))"
        },
        example_result='{"exit_code": 0, "stdout": "max difference 1.414\\n", "stderr": "", "images": [{"name": "sin_vs_cos.png", "url": "/api/artifacts/9f2c", "path": "figures/sin_vs_cos.png"}]}',
        interpretation="If exit_code != 0 read stderr, fix the code and retry once. Describe what the figure shows; it is already attached.",
    )
)
