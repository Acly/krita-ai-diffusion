import argparse
import re
import json
from pathlib import Path

source_dir = Path(__file__).parent.parent / "ai_diffusion"
excluded_dirs = [".pytest_cache", "__pycache__", "icons", "websockets", "debugpy"]
expression = re.compile(r'_\(\s*"(.+?)"[\,|\s*\)]')


def extract_source_strings(filepath: Path):
    text = filepath.read_text(encoding="utf-8")
    return set(expression.findall(text))


def parse_source(dir: Path) -> set[str]:
    result: set[str] = set()
    for file in dir.iterdir():
        if file.is_dir() and file.name not in excluded_dirs:
            result |= parse_source(file)
        elif file.suffix == ".py":
            result |= extract_source_strings(file)
    return result


def write_language_file(strings: set[str], id: str, name: str, target_file: Path):
    defs = {"id": id, "name": name, "translations": {s: None for s in sorted(strings)}}
    with target_file.open("w", encoding="utf-8") as f:
        json.dump(defs, f, ensure_ascii=False, indent=2)


def update_template():
    template_file = source_dir / "language" / "new_language.json.template"
    strings = parse_source(source_dir)
    write_language_file(strings, "ex", "Example Language", template_file)


def update_all():
    strings = parse_source(source_dir)
    for lang_file in (source_dir / "language").iterdir():
        if lang_file.name != "en.json" and lang_file.suffix == ".json":
            existing = json.loads(lang_file.read_text(encoding="utf-8"))
            updated = {s: t for s, t in existing["translations"].items() if s in strings}
            updated.update({s: None for s in strings if s not in updated})
            existing["translations"] = updated
            with lang_file.open("w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    cmd_desc = "'template', 'update' or a language identifier (en, es, de, ...)"
    cmd = argparse.ArgumentParser()
    cmd.add_argument("command", type=str, help=cmd_desc)
    cmd.add_argument("--name", "-n", type=str, help="Language name for the UI")
    cmd.add_argument("--outdir", "-o", type=str, default=source_dir / "language")

    args = cmd.parse_args()
    if args.command == "template":
        update_template()
    elif args.command == "update":
        update_all()
    else:
        strings = parse_source(source_dir)
        outfile = Path(args.outdir) / f"{args.lang}.json"
        write_language_file(strings, args.lang, args.name, outfile)
