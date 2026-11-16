"""run_python — execute Python (numpy/scipy/sympy/matplotlib) in the sandbox; figures become artifacts."""
from __future__ import annotations

import asyncio
import re
from typing import Any

from app.services import sandbox
from app.tools.base import ToolContext, ToolResult, ToolSpec
from app.tools.bash import register_changes
from app.services.artifact_storage import snapshot
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


async def run_python_tool(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    code = str(args["code"])
    if re.search(r"\b(subprocess|os\.system|socket|ctypes|pty)\b", code):
        # not a security boundary (the sandbox is), but keeps the model on-task
        return ToolResult.failure("run_python is for computation/plotting; process spawning and raw sockets are not allowed. Use the bash tool if a shell command is needed.")
    tag = ctx.run_id[:6]
    saves_explicitly = "savefig" in code
    full = PRELUDE + f"_STEMOS_RUN = '{tag}'\n" + code + ("\n" if saves_explicitly else EPILOGUE)
    before = await asyncio.to_thread(snapshot, ctx.project_id)
    res = await sandbox.run_python(ctx.project_id, full, args.get("timeout"))
    arts = await register_changes(ctx, before)
    images = [{"name": a["filename"], "url": a["url"], "path": a["path"]} for a in arts if a["type"] == "image"]
    payload = {**res.to_dict(), "images": images, "files": [a["path"] for a in arts if a["type"] != "image"]}
    if images:
        payload["note"] = "Images are attached to the answer automatically. Refer to them by file name; do not invent URLs."
    return ToolResult(content=compact_json(payload, 8000), data=payload, ok=res.exit_code == 0, error="" if res.exit_code == 0 else res.stderr[-500:], artifacts=arts)


registry.register(
    ToolSpec(
        name="run_python",
        description="Run Python 3 code in the sandboxed project workspace (numpy, scipy, sympy, pandas, matplotlib available). Matplotlib figures are saved and attached automatically.",
        parameters={
            "properties": {
                "code": {"type": "string", "description": "Complete Python program. Use print() for results. Save figures under figures/ (e.g. plt.savefig('figures/plot.png'))."},
                "timeout": {"type": "integer", "description": "Seconds (default 60)."},
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
