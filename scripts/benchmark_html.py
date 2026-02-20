#!/usr/bin/env python3
"""
Generate an HTML visualization of inpaint benchmark results.

Usage:
    python benchmark_html.py tests/benchmark/20260122-1124 [tests/benchmark/20260122-1200] [...]

This script creates a visual comparison of inpaint benchmark runs by generating
an HTML page with side-by-side comparisons of input images and benchmark results.
"""

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


def load_input_image(image_name: str, images_dir: Path) -> tuple[Image.Image, Image.Image] | None:
    """Load input image and mask for a given image name."""
    image_path = images_dir / f"{image_name}-image.webp"
    mask_path = images_dir / f"{image_name}-mask.webp"

    if not image_path.exists() or not mask_path.exists():
        return None

    try:
        image = Image.open(image_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L")
    except Exception as e:
        print(f"Warning: Failed to load {image_name}: {e}")
        return None
    else:
        return image, mask


def overlay_mask_on_image(
    image: Image.Image, mask: Image.Image, opacity: float = 0.6, bounds: list[int] | None = None
) -> Image.Image:
    """Overlay a mask on an image with semi-transparency and optionally draw bounds rectangle."""
    # Ensure mask is the same size as image
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)

    # Create overlay with red color
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))

    # Create red channel with mask opacity
    overlay_array = overlay.load()
    mask_array = mask.load()
    assert mask_array is not None and overlay_array is not None

    for y in range(image.size[1]):
        for x in range(image.size[0]):
            mask_val = mask_array[x, y]
            assert isinstance(mask_val, (int, float))
            alpha = int(mask_val * opacity)
            overlay_array[x, y] = (255, 0, 0, alpha)

    # Composite overlay on top of image
    result = Image.alpha_composite(image, overlay)

    # Draw bounds rectangle if provided
    if bounds is not None and len(bounds) == 4:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(result)
        x_offset, y_offset, width, height = bounds
        # Draw rectangle outline
        draw.rectangle(
            [(x_offset, y_offset), (x_offset + width, y_offset + height)],
            outline=(0, 255, 0, 255),  # Green outline
            width=3,
        )

    return result


def save_image_as_webp(image: Image.Image, output_dir: Path, filename: str) -> str:
    """Save an image as WebP format and return relative path from output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    # Convert to RGB if necessary (WebP doesn't support transparency well for lossy)
    if image.mode in ("RGBA", "LA", "P"):
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        rgb_image.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
        image = rgb_image
    image.save(output_path, "WEBP", quality=85)
    # Return relative path
    return output_path.name


def parse_benchmark_results(benchmark_dir: Path) -> dict:
    """Parse all benchmark result images in a directory using meta.json."""
    results = defaultdict(list)

    # Load meta.json
    meta_path = benchmark_dir / "meta.json"
    if not meta_path.exists():
        print(f"Warning: No meta.json found in {benchmark_dir}")
        return results

    try:
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
    except Exception as e:
        print(f"Error loading meta.json from {benchmark_dir}: {e}")
        return results

    # Iterate through PNG files and match them with meta.json entries
    for png_file in benchmark_dir.glob("*_local.png"):
        # Strip the _local.png suffix to get the key for meta.json
        key = png_file.stem.replace("_local", "")

        if key in meta_data:
            meta = meta_data[key]
            # Extract scenario (image name) from the meta data
            scenario = meta.get("scenario", "unknown")

            results[scenario].append({
                "path": png_file,
                "arch": meta.get("arch", "unknown"),
                "prompt": meta.get("user_prompt", "").strip().replace("\n\n", "\n"),
                "seed": meta.get("seed", 0),
                "filename": png_file.name,
                "meta": meta,  # Store full metadata for additional info
            })
        else:
            print(f"Warning: No metadata found for {png_file.name} (key: {key})")

    return results


def get_image_resolution(image_path: Path) -> tuple[int, int] | None:
    """Get image resolution."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None


def generate_html(
    benchmark_dirs: list[Path], output_path: Path | None = None, images_dir: Path | None = None
) -> str:
    """Generate HTML page for benchmark visualization."""
    if images_dir is None:
        images_dir = Path(__file__).parent.parent / "tests" / "images" / "inpaint"

    if output_path is None:
        output_path = Path(__file__).parent / "benchmark_results.html"

    # Create images subfolder next to output HTML
    output_images_dir = output_path.parent / "benchmark_images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Parse all benchmark results and organize by (benchmark, arch) columns
    all_columns = []
    column_map = {}  # Maps column_idx to column info

    for bench_dir in benchmark_dirs:
        results = parse_benchmark_results(bench_dir)

        # Group results by architecture
        results_by_arch = {}
        for image_name, image_results in results.items():
            for result in image_results:
                arch = result["arch"]
                if arch not in results_by_arch:
                    results_by_arch[arch] = {}
                if image_name not in results_by_arch[arch]:
                    results_by_arch[arch][image_name] = []
                results_by_arch[arch][image_name].append(result)

        # Create a column for each architecture
        for arch in sorted(results_by_arch.keys()):
            col_idx = len(all_columns)
            column_info = {
                "idx": col_idx,
                "dir": bench_dir,
                "bench_name": bench_dir.name,
                "arch": arch,
                "results": results_by_arch[arch],
            }
            all_columns.append(column_info)
            column_map[col_idx] = column_info

    # Collect all unique image names across all columns
    all_image_names = set()
    for column in all_columns:
        all_image_names.update(column["results"].keys())
    all_image_names = sorted(all_image_names)

    # Build HTML
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inpaint Benchmark Results</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
        }
        
        h1 {
            margin-bottom: 10px;
            color: #333;
        }
        
        .info {
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }
        
        thead th select {
            padding: 6px 8px;
            border: 1px solid #999;
            border-radius: 3px;
            font-size: 13px;
            background: white;
            cursor: pointer;
            width: 100%;
            margin-top: 4px;
        }
        
        .table-wrapper {
            margin-top: 20px;
        }
        
        .table-header {
            width: 100%;
            overflow: visible;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .table-body {
            width: 100%;
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 100%;
            table-layout: fixed;
        }
        
        thead th {
            background: #f8f9fa;
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            z-index: 100;
        }
        
        tbody tr {
            border: 1px solid #eee;
        }
        
        tbody tr:hover {
            background: #fafafa;
        }
        
        td {
            border: 1px solid #ddd;
            padding: 12px;
            vertical-align: top;
        }
        
        .image-cell {
            text-align: center;
        }
        
        .image-cell img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: inline-block;
        }
        
        .image-container {
            position: relative;
            display: inline-block;
            max-width: 100%;
        }
        
        .image-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(255, 255, 255, 0.75);
            color: black;
            padding: 12px;
            opacity: 0;
            transition: opacity 0.3s ease;
            font-size: 12px;
            line-height: 1.4;
            word-wrap: break-word;
            max-height: 40%;
            overflow-y: auto;
            pointer-events: none;
        }
        
        .image-container:hover .image-overlay {
            opacity: 1;
        }
        
        .metadata {
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 8px;
            font-size: 13px;
            color: #555;
        }
        
        .metadata-item {
            display: flex;
            justify-content: space-between;
            padding: 2px 0;
            gap: 10px;
        }
        
        .metadata-label {
            font-weight: 500;
            color: #333;
            min-width: 80px;
            flex-shrink: 0;
        }
        
        .metadata-value {
            color: #666;
            font-family: 'Courier New', monospace;
            word-break: break-word;
            white-space: pre-wrap;
            max-height: 100px;
            overflow-y: auto;
            text-align: right;
        }
        
        .hidden {
            display: none;
        }
        
        .column-1 { width: 25%; }
        .column-2 { width: 37.5%; }
        .column-3 { width: 37.5%; }
        
        @media (max-width: 1200px) {
            .column-1 { width: 30%; }
            .column-2 { width: 35%; }
            .column-3 { width: 35%; }
        }
        
        @media (max-width: 900px) {
            .table-wrapper {
                display: block;
                overflow-x: auto;
            }
            
            .column-1 { width: 35%; }
            .column-2 { width: 32.5%; }
            .column-3 { width: 32.5%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Inpaint Benchmark Results</h1>
""")

    # Add benchmark info
    bench_names = list(dict.fromkeys([col["bench_name"] for col in all_columns]))  # Unique, ordered
    html_parts.append(f'        <div class="info">Benchmarks: {", ".join(bench_names)}</div>\n')

    # Table headers
    html_parts.append('        <div class="table-wrapper">\n')
    html_parts.append('            <div class="table-header">\n')
    html_parts.append("                <table>\n")
    html_parts.append("                    <thead>\n")
    html_parts.append("                        <tr>\n")
    html_parts.append('                            <th class="column-1">Input Image</th>\n')

    if len(all_columns) <= 2:
        column_2_class = "column-2"
        column_3_class = "column-3"
        if len(all_columns) >= 1:
            col = all_columns[0]
            label = f"{col['bench_name']} ({col['arch']})"
            html_parts.append(
                f'                            <th class="{column_2_class}">{label}</th>\n'
            )
        if len(all_columns) >= 2:
            col = all_columns[1]
            label = f"{col['bench_name']} ({col['arch']})"
            html_parts.append(
                f'                            <th class="{column_3_class}">{label}</th>\n'
            )
    else:
        # Add selectors in headers for > 2 columns case
        html_parts.append('                            <th class="column-2" id="left-header">\n')
        html_parts.append("                                <div>Left Column:</div>\n")
        html_parts.append(
            '                                <select id="left-select" onchange="updateVisibility()">\n'
        )
        for col in all_columns:
            label = f"{col['bench_name']} ({col['arch']})"
            html_parts.append(
                f'                                    <option value="{col["idx"]}"{" selected" if col["idx"] == 0 else ""}>{label}</option>\n'
            )
        html_parts.append("                                </select>\n")
        html_parts.append("                            </th>\n")
        html_parts.append('                            <th class="column-3" id="right-header">\n')
        html_parts.append("                                <div>Right Column:</div>\n")
        html_parts.append(
            '                                <select id="right-select" onchange="updateVisibility()">\n'
        )
        html_parts.append('                                    <option value="">None</option>\n')
        for col in all_columns:
            label = f"{col['bench_name']} ({col['arch']})"
            html_parts.append(
                f'                                    <option value="{col["idx"]}"{" selected" if col["idx"] == 1 else ""}>{label}</option>\n'
            )
        html_parts.append("                                </select>\n")
        html_parts.append("                            </th>\n")

    html_parts.append("                        </tr>\n")
    html_parts.append("                    </thead>\n")
    html_parts.append("                </table>\n")
    html_parts.append("            </div>\n")
    html_parts.append('            <div class="table-body">\n')
    html_parts.append("                <table>\n")
    html_parts.append("                    <tbody>\n")

    # Table rows
    for image_name in all_image_names:
        # Load input image and mask
        input_data = load_input_image(image_name, images_dir)
        if input_data is None:
            print(f"Skipping {image_name}: input image not found")
            continue

        input_image, mask = input_data

        # Get bounds from first available result (same for all results of this image)
        bounds = None
        for col in all_columns:
            if image_name in col["results"]:
                first_result = col["results"][image_name][0]
                bounds = first_result.get("meta", {}).get("bounds")
                break

        masked_image = overlay_mask_on_image(input_image, mask, bounds=bounds)
        # Save input image as WebP
        input_filename = save_image_as_webp(
            masked_image, output_images_dir, f"input_{image_name}.webp"
        )
        input_img_path = f"./benchmark_images/{input_filename}"
        resolution = input_image.size

        # Find results for this image across all columns
        results_by_column = {}
        for col in all_columns:
            if image_name in col["results"]:
                results_by_column[col["idx"]] = col["results"][image_name]

        if not results_by_column:
            continue

        # Get all unique (prompt, seed) combinations from first available column
        first_col_results = results_by_column[next(iter(results_by_column.keys()))]
        sorted_results = sorted(
            first_col_results,
            key=lambda x: (x["prompt"], x["seed"]),
        )

        for result in sorted_results:
            prompt = result["prompt"]
            seed = result["seed"]
            meta = result.get("meta", {})

            html_parts.append("                    <tr>\n")

            # Input image cell with metadata
            html_parts.append('                        <td class="column-1">\n')
            html_parts.append('                            <div class="metadata">\n')
            html_parts.append(
                f'                                <div class="metadata-item"><span class="metadata-label">Image:</span><span class="metadata-value">{image_name}</span></div>\n'
            )
            html_parts.append(
                f'                                <div class="metadata-item"><span class="metadata-label">Resolution:</span><span class="metadata-value">{resolution[0]}x{resolution[1]}</span></div>\n'
            )
            if "mode" in meta:
                html_parts.append(
                    f'                                <div class="metadata-item"><span class="metadata-label">Mode:</span><span class="metadata-value">{meta["mode"]}</span></div>\n'
                )
            # Escape HTML and truncate long prompts for display
            prompt_display = html.escape(prompt) if prompt else "(no prompt)"
            html_parts.append(
                f'                                <div class="metadata-item"><span class="metadata-label">Prompt:</span><span class="metadata-value">{prompt_display}</span></div>\n'
            )
            html_parts.append(
                f'                                <div class="metadata-item"><span class="metadata-label">Seed:</span><span class="metadata-value">{seed}</span></div>\n'
            )
            html_parts.append("                            </div>\n")
            html_parts.append(
                f'                            <div class="image-cell"><img src="{input_img_path}" alt="Input: {image_name}"></div>\n'
            )
            html_parts.append("                        </td>\n")

            # Result columns
            for display_idx, col in enumerate(all_columns):
                col_class = "column-2" if display_idx == 0 else "column-3"

                if len(all_columns) > 2:
                    # Dynamic visibility based on selection
                    row_class = f"col-{col['idx']}"
                else:
                    row_class = ""

                html_parts.append(f'                        <td class="{col_class} {row_class}">\n')
                html_parts.append('                            <div class="image-cell">\n')

                if image_name in col["results"]:
                    # Find the matching result (by prompt and seed only, arch is column-specific)
                    matching_results = [
                        r
                        for r in col["results"][image_name]
                        if r["prompt"] == prompt and r["seed"] == seed
                    ]

                    if matching_results:
                        result_path = matching_results[0]["path"]
                        result_meta = matching_results[0].get("meta", {})
                        alt = f"{col['bench_name']} ({col['arch']})"
                        try:
                            with Image.open(result_path) as result_img:
                                # Save result image as WebP
                                result_filename = save_image_as_webp(
                                    result_img.convert("RGB"),
                                    output_images_dir,
                                    f"{Path(result_path).stem.replace('benchmark_inpaint', col['bench_name'])}.webp",
                                )
                                result_img_path = f"./benchmark_images/{result_filename}"
                                full_prompt = result_meta.get("full_prompt", "")
                                full_prompt = html.escape(full_prompt or "(no prompt)")
                                html_parts.append(
                                    '                                <div class="image-container">\n'
                                )
                                html_parts.append(
                                    f'                                    <img src="{result_img_path}" alt="{alt}">\n'
                                )
                                html_parts.append(
                                    f'                                    <div class="image-overlay">{full_prompt}</div>\n'
                                )
                                html_parts.append("                                </div>\n")
                        except Exception as e:
                            html_parts.append(
                                f"                                <p>Error loading image: {e}</p>\n"
                            )
                    else:
                        html_parts.append(
                            '                                <p style="color: #999;">No matching result</p>\n'
                        )
                else:
                    html_parts.append(
                        '                                <p style="color: #999;">Not available</p>\n'
                    )

                html_parts.append("                            </div>\n")
                html_parts.append("                        </td>\n")

            html_parts.append("                    </tr>\n")

    html_parts.append("                    </tbody>\n")
    html_parts.append("                </table>\n")
    html_parts.append("            </div>\n")
    html_parts.append("        </div>\n")

    # Add JavaScript for dynamic columns
    if len(all_columns) > 2:
        html_parts.append("""        <script>
            function updateVisibility() {
                const leftSelect = document.getElementById('left-select').value;
                const rightSelect = document.getElementById('right-select').value;
                
                // Update headers
                const headerMap = {};
""")
        for col in all_columns:
            label = f"{col['bench_name']} ({col['arch']})"
            html_parts.append(f"                headerMap['{col['idx']}'] = '{label}';\n")

        html_parts.append("""                // Update left header label
                const leftHeaderDiv = document.querySelector('#left-header div');
                if (leftHeaderDiv) {
                    leftHeaderDiv.textContent = 'Left Column: ' + (headerMap[leftSelect] || 'Select');
                }
                
                // Update right header label
                const rightHeaderDiv = document.querySelector('#right-header div');
                if (rightHeaderDiv) {
                    rightHeaderDiv.textContent = 'Right Column: ' + (rightSelect ? (headerMap[rightSelect] || 'Select') : 'None');
                }
                
                // Show/hide columns based on selection
                const rows = document.querySelectorAll('tbody tr');
                rows.forEach(row => {
                    const cells = row.querySelectorAll('td');
                    if (cells.length >= 3) {
                        // Hide all bench columns first
                        for (let i = 1; i < cells.length; i++) {
                            cells[i].style.display = 'none';
                        }
                        
                        // Show selected columns
                        const leftColIdx = parseInt(leftSelect) + 1;
                        const rightColIdx = rightSelect ? (parseInt(rightSelect) + 1) : null;
                        
                        if (leftColIdx < cells.length) {
                            cells[leftColIdx].style.display = '';
                        }
                        if (rightColIdx && rightColIdx < cells.length) {
                            cells[rightColIdx].style.display = '';
                        }
                    }
                });
            }
            
            // Initialize on page load
            window.addEventListener('DOMContentLoaded', updateVisibility);
        </script>
""")

    html_parts.append("""    </div>
</body>
</html>
""")

    html_content = "".join(html_parts)

    # Write to output file
    output_path.write_text(html_content)
    print(f"HTML report generated: {output_path}")

    return html_content


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML visualization of inpaint benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python benchmark_html.py tests/benchmark/20260122-1124
    python benchmark_html.py tests/benchmark/20260122-1124 tests/benchmark/20260122-1200
    python benchmark_html.py tests/benchmark/20260122-1124 tests/benchmark/20260122-1200 tests/benchmark/20260122-1300
        """,
    )

    parser.add_argument("benchmark_dirs", nargs="+", help="Path(s) to benchmark result folder(s)")

    parser.add_argument(
        "-o", "--output", help="Output HTML file path (default: scripts/benchmark_results.html)"
    )

    parser.add_argument(
        "-i", "--images-dir", help="Path to input images directory (default: tests/images/inpaint)"
    )

    args = parser.parse_args()

    # Convert to Path objects and validate
    benchmark_dirs = []
    for bench_dir_str in args.benchmark_dirs:
        bench_dir = Path(bench_dir_str)
        if not bench_dir.exists():
            print(f"Error: Directory not found: {bench_dir}")
            return 1
        if not bench_dir.is_dir():
            print(f"Error: Not a directory: {bench_dir}")
            return 1
        benchmark_dirs.append(bench_dir)

    if len(benchmark_dirs) < 1:
        print("Error: At least one benchmark directory is required")
        return 1

    output_path = Path(args.output) if args.output else None
    images_dir = (
        Path(args.images_dir)
        if args.images_dir
        else Path(__file__).parent.parent / "tests" / "images" / "inpaint"
    )

    try:
        generate_html(benchmark_dirs, output_path, images_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
