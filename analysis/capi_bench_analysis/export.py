"""Write every figure/table/summary to exports/ for pasting into the report."""

import shutil
import subprocess
import tempfile
from pathlib import Path

EXPORTS_DIR = Path(__file__).resolve().parents[1] / "exports"

_manifest: list[str] = []


def _target(subdir: str, filename: str) -> Path:
    d = EXPORTS_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def save_figure(fig, name: str):
    for ext, kwargs in (("svg", {}), ("png", {"dpi": 300})):
        path = _target("figures", f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        _manifest.append(str(path.relative_to(EXPORTS_DIR)))
    return fig


def save_mermaid(source: str, name: str):
    # Renders via mermaid-cli; needs mmdc on PATH, skips quietly otherwise.
    mmdc = shutil.which("mmdc")
    if mmdc is None:
        print(f"mmdc not found; skipped exporting {name}")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False) as f:
        f.write(source)
    for ext, extra in (("svg", []), ("png", ["-w", "1600"])):
        path = _target("figures", f"{name}.{ext}")
        subprocess.run([mmdc, "-i", f.name, "-o", str(path), "--quiet", *extra], check=True)
        _manifest.append(f"figures/{name}.{ext}")
    Path(f.name).unlink()


def save_table(df, name: str, float_format="%.2f"):
    df.to_csv(_target("tables", f"{name}.csv"), index=False, float_format=float_format)
    md = df.to_markdown(index=False, floatfmt=".2f")
    _target("tables", f"{name}.md").write_text(md + "\n")
    _manifest.extend([f"tables/{name}.csv", f"tables/{name}.md"])
    return df


def save_summary(text: str, name: str):
    _target("summaries", f"{name}.md").write_text(text.rstrip() + "\n")
    _manifest.append(f"summaries/{name}.md")
    return text


def write_manifest():
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Export manifest", ""] + [f"- {p}" for p in sorted(set(_manifest))]
    (EXPORTS_DIR / "manifest.md").write_text("\n".join(lines) + "\n")
    return len(set(_manifest))
