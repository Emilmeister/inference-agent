"""Convert SGLang JSX tables and vLLM definition-list docs into compact,
LLM-friendly markdown.

Reads:
  engine-docs/sglang/cli_params.md  (JSX-tables)
  engine-docs/vllm/cli_params.md    (definition lists)

Writes:
  engine-docs/sglang/cli_params.llm.md
  engine-docs/vllm/cli_params.llm.md

Output format (one row per CLI flag):
  ## Section name

  - `--flag` [type/choices, default=X] — description
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SGLANG_IN = ROOT / "engine-docs/sglang/cli_params.md"
VLLM_IN = ROOT / "engine-docs/vllm/cli_params.md"
SGLANG_OUT = ROOT / "engine-docs/sglang/cli_params.llm.md"
VLLM_OUT = ROOT / "engine-docs/vllm/cli_params.llm.md"
SGLANG_CURATED = ROOT / "engine-docs/sglang/cli_params.curated.md"
VLLM_CURATED = ROOT / "engine-docs/vllm/cli_params.curated.md"

# Sections kept in curated.md (relevance for perf-tuning of agentic / RAG inference,
# single-node, text-only models). Anything not listed here is stripped.
SGLANG_KEEP_SECTIONS = {
    "Model and tokenizer",
    "Quantization and data type",
    "Memory and scheduling",
    "Runtime options",
    "Data parallelism",
    "Kernel Backends (Attention, Sampling, Grammar, GEMM)",
    "Speculative decoding",
    "Multi-layer Eagle speculative decoding",
    "MoE",
    "Mamba Cache",
    "Hierarchical cache",
    "Optimization/debug options",
    "Dynamic batch tokenizer",
}

VLLM_KEEP_SECTIONS = {
    "ModelConfig",
    "LoadConfig",
    "AttentionConfig",
    "StructuredOutputsConfig",
    "ParallelConfig",
    "CacheConfig",
    "SchedulerConfig",
    "CompilationConfig",
    "KernelConfig",
    "VllmConfig",
}


# ── helpers ────────────────────────────────────────────────────────────────
_TAG_RE = re.compile(r"<[^>]+>", re.S)
_CODE_RE = re.compile(r"`([^`]+)`")
_WS_RE = re.compile(r"\s+")
_BACKSLASH_ESCAPE_RE = re.compile(r"\\([_*`\\\[\]])")


def _strip_html(text: str) -> str:
    """Remove HTML/JSX tags and decode common entities."""
    text = _TAG_RE.sub("", text)
    text = (text.replace("&lt;", "<").replace("&gt;", ">")
                .replace("&amp;", "&").replace("&quot;", '"')
                .replace("&#39;", "'").replace("&nbsp;", " "))
    text = _BACKSLASH_ESCAPE_RE.sub(r"\1", text)
    return _WS_RE.sub(" ", text).strip()


def _strip_code_marks(s: str) -> str:
    """Remove backticks and surrounding whitespace, keep inner content."""
    s = s.strip()
    if s.startswith("`") and s.endswith("`"):
        s = s[1:-1]
    return s.strip()


_FLAG_TOKEN_RE = re.compile(r"--?[A-Za-z][A-Za-z0-9_\-]*")


def _normalize_flag(s: str) -> str:
    """Extract flag tokens, joined by ' | '. Drops all noise (backticks,
    backslash-escapes, <code> tags, separator junk)."""
    s = s.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    s = _strip_html(s)
    s = s.replace("`", "")
    s = re.sub(r"\\(.)", r"\1", s)  # drop any backslash escape
    tokens = _FLAG_TOKEN_RE.findall(s)
    if not tokens:
        return _WS_RE.sub(" ", s).strip()
    # Dedup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return " | ".join(uniq)


# ── SGLang parser (JSX tables) ─────────────────────────────────────────────
def parse_sglang(text: str) -> list[tuple[str, list[dict]]]:
    """Return [(section_title, [param_dict, ...]), ...] from SGLang docs."""
    sections: list[tuple[str, list[dict]]] = []
    current_title: str | None = None
    current_rows: list[dict] = []

    # Split on H2 headers; keep text between them
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    # parts = [preamble, title1, body1, title2, body2, ...]
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""

        rows: list[dict] = []
        # Find all <tbody>...</tbody> blocks (skip <thead>)
        tbody_re = re.compile(r"<tbody[^>]*>(.*?)</tbody>", re.S)
        for tbody_match in tbody_re.finditer(body):
            tbody = tbody_match.group(1)
            tr_re = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
            for tr in tr_re.finditer(tbody):
                td_re = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
                cells = [m.group(1) for m in td_re.finditer(tr.group(1))]
                if len(cells) < 4:
                    continue
                flag = _normalize_flag(cells[0])
                desc = _strip_html(cells[1])
                default = _strip_code_marks(_strip_html(cells[2]))
                options = _strip_html(cells[3])

                # Type is often embedded in options as "Type: int" or as choice list
                rows.append({
                    "flag": flag,
                    "desc": desc,
                    "default": default,
                    "options": options,
                })
        if rows:
            sections.append((title, rows))

    return sections


def render_sglang(sections: list[tuple[str, list[dict]]]) -> str:
    out: list[str] = []
    out.append("# SGLang CLI parameters (LLM-friendly)\n")
    out.append("Format: `--flag` [type/choices, default=X] — description\n")
    for title, rows in sections:
        out.append(f"## {title}\n")
        for r in rows:
            meta_parts = []
            opts = r["options"]
            # Extract type: "Type: int" → "int"
            m = re.match(r"Type:\s*(\w+)", opts)
            if m:
                meta_parts.append(m.group(1))
            elif opts.startswith("bool flag"):
                meta_parts.append("bool")
            elif "," in opts and "<code>" not in opts and len(opts) < 200:
                # choices list
                meta_parts.append(opts)
            elif opts:
                meta_parts.append(opts[:100])

            default = r["default"]
            if default and default not in ("None", "{}", '""', "False"):
                meta_parts.append(f"default={default}")
            elif default == "False":
                meta_parts.append("default=False")

            meta = ", ".join(meta_parts)
            meta_str = f" [{meta}]" if meta else ""
            desc = r["desc"]
            out.append(f"- `{r['flag']}`{meta_str} — {desc}")
        out.append("")
    return "\n".join(out)


# ── vLLM parser (definition lists) ─────────────────────────────────────────
def parse_vllm(text: str) -> list[tuple[str, list[dict]]]:
    """Return [(section_title, [param_dict, ...]), ...] from vLLM docs."""
    sections: list[tuple[str, list[dict]]] = []
    current_title = "Misc"
    current_rows: list[dict] = []

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Section header (### Foo)
        m_section = re.match(r"^###\s+(.+)$", line)
        if m_section:
            if current_rows:
                sections.append((current_title, current_rows))
            current_title = m_section.group(1).strip()
            current_rows = []
            i += 1
            continue

        # Param header: #### `--flag`, `--flag2`
        m_param = re.match(r"^####\s+(.+)$", line)
        if m_param:
            flag_line = m_param.group(1).strip()
            # Extract `--flag` and aliases
            flags = _CODE_RE.findall(flag_line)
            flag = " | ".join(flags) if flags else _strip_html(flag_line)

            desc = ""
            choices = ""
            default = ""
            i += 1
            # Read continuation lines (`:   ...`) until next #### or ###
            while i < len(lines):
                nxt = lines[i]
                if re.match(r"^####\s", nxt) or re.match(r"^###\s", nxt) or re.match(r"^##\s", nxt):
                    break
                stripped = nxt.lstrip()
                if stripped.startswith(":"):
                    body = stripped[1:].strip()
                    if body.startswith("Possible choices:"):
                        choices = body[len("Possible choices:"):].strip()
                    elif body.startswith("Default:"):
                        default = body[len("Default:"):].strip().strip("`")
                    elif body:
                        desc = (desc + " " + body).strip() if desc else body
                elif stripped:
                    # Continuation of current `:` block
                    if desc:
                        desc = desc + " " + stripped
                i += 1

            current_rows.append({
                "flag": flag,
                "desc": desc,
                "default": default,
                "choices": choices,
            })
            continue

        i += 1

    if current_rows:
        sections.append((current_title, current_rows))
    return sections


def render_vllm(sections: list[tuple[str, list[dict]]]) -> str:
    out: list[str] = []
    out.append("# vLLM CLI parameters (LLM-friendly)\n")
    out.append("Format: `--flag` [choices, default=X] — description\n")
    for title, rows in sections:
        out.append(f"## {title}\n")
        for r in rows:
            meta_parts = []
            if r["choices"]:
                meta_parts.append(r["choices"])
            if r["default"] and r["default"] not in ('""', "{}"):
                meta_parts.append(f"default={r['default']}")
            meta_str = f" [{', '.join(meta_parts)}]" if meta_parts else ""
            desc = r["desc"] or "(no description)"
            out.append(f"- `{r['flag']}`{meta_str} — {desc}")
        out.append("")
    return "\n".join(out)


def render_curated(
    sections: list[tuple[str, list[dict]]],
    keep: set[str],
    engine_name: str,
    render_fn,
) -> str:
    """Filter sections to the keep-set and render via the engine-specific renderer."""
    filtered = [(t, rows) for t, rows in sections if t in keep]
    body = render_fn(filtered)
    # Replace the standard header with a curated-specific one
    lines = body.splitlines()
    # Skip first 3 lines (title + blank + format line) and replace
    rest = "\n".join(lines[3:]) if len(lines) > 3 else body
    header = (
        f"# {engine_name} CLI parameters — curated for perf tuning\n\n"
        "Subset of upstream docs relevant for single-node, text-only, "
        "agentic/RAG inference benchmarks. Frontend, multi-node, multimodal, "
        "LoRA, disaggregation, and offloading sections are omitted.\n\n"
        "Format: `--flag` [type/choices, default=X] — description\n"
    )
    return header + rest


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    sglang_text = SGLANG_IN.read_text(encoding="utf-8")
    vllm_text = VLLM_IN.read_text(encoding="utf-8")

    sglang_sections = parse_sglang(sglang_text)
    vllm_sections = parse_vllm(vllm_text)

    SGLANG_OUT.write_text(render_sglang(sglang_sections), encoding="utf-8")
    VLLM_OUT.write_text(render_vllm(vllm_sections), encoding="utf-8")

    SGLANG_CURATED.write_text(
        render_curated(sglang_sections, SGLANG_KEEP_SECTIONS, "SGLang", render_sglang),
        encoding="utf-8",
    )
    VLLM_CURATED.write_text(
        render_curated(vllm_sections, VLLM_KEEP_SECTIONS, "vLLM", render_vllm),
        encoding="utf-8",
    )

    sg_count = sum(len(rows) for _, rows in sglang_sections)
    vl_count = sum(len(rows) for _, rows in vllm_sections)
    sg_curated_count = sum(
        len(rows) for t, rows in sglang_sections if t in SGLANG_KEEP_SECTIONS
    )
    vl_curated_count = sum(
        len(rows) for t, rows in vllm_sections if t in VLLM_KEEP_SECTIONS
    )
    print(f"SGLang full:    {len(sglang_sections)} sections, {sg_count} params → {SGLANG_OUT.name}")
    print(f"SGLang curated: {len(SGLANG_KEEP_SECTIONS)} sections, {sg_curated_count} params → {SGLANG_CURATED.name}")
    print(f"vLLM full:      {len(vllm_sections)} sections, {vl_count} params → {VLLM_OUT.name}")
    print(f"vLLM curated:   {len(VLLM_KEEP_SECTIONS)} sections, {vl_curated_count} params → {VLLM_CURATED.name}")


if __name__ == "__main__":
    main()
