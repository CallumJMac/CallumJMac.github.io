# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown>=3.7", "beautifulsoup4>=4.12", "pillow>=11"]
# ///
"""Build Medium-importable copies of every blog post under public/medium/<slug>/.

Medium's importer only keeps plain HTML: paragraphs, headings, lists, code,
links and PNG/JPEG images. So each copy replaces SVG figures and tables with
PNG renders, turns audio players into links, and points its canonical link at
the original post. The copies are marked noindex so search engines keep
treating the original posts as the source.

    uv run scripts/medium_export.py
"""

from __future__ import annotations

import html
import re
from urllib.parse import quote
import subprocess
import tempfile
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from PIL import Image, ImageChops

SITE = "https://callumjmac.github.io"
ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / "src/data/blog"
PUBLIC = ROOT / "public"
OUTPUT = PUBLIC / "medium"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TABLE_CSS = """
body { margin: 0; background: #fff; font-family: Helvetica, Arial, sans-serif; color: #1d1d1f; }
table { border-collapse: collapse; font-size: 17px; margin: 24px; }
th, td { border: 1px solid #d6d6d9; padding: 8px 14px; text-align: left; }
th { background: #f3f4f6; font-weight: 700; }
td code, th code { font-family: Menlo, monospace; font-size: 15px; }
"""


def front_matter(text: str) -> tuple[dict[str, str], str]:
    _, header, body = text.split("---", 2)
    fields = {}
    for line in header.strip().splitlines():
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip('"')
    return fields, body


def screenshot(page: str, output: Path, width: int) -> None:
    """Render an HTML snippet in headless Chrome at 2x and crop to its content."""
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "page.html"
        source.write_text(page)
        raw = Path(directory) / "raw.png"
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--allow-file-access-from-files", "--blink-settings=preferredColorScheme=1",
             "--force-device-scale-factor=2", f"--window-size={width},2600",
             "--virtual-time-budget=3000", f"--screenshot={raw}", source.as_uri()],
            check=True, capture_output=True,
        )
        image = Image.open(raw).convert("RGB")
        background = Image.new("RGB", image.size, (255, 255, 255))
        box = ImageChops.difference(image, background).getbbox()
        if box is None:
            raise SystemExit(f"blank render for {output}")
        pad = 24
        box = (max(box[0] - pad, 0), max(box[1] - pad, 0),
               min(box[2] + pad, image.width), min(box[3] + pad, image.height))
        output.parent.mkdir(parents=True, exist_ok=True)
        image.crop(box).save(output, optimize=True)


def local_path(src: str) -> Path:
    return PUBLIC / src.lstrip("/")


def convert_image(src: str, out_dir: Path, slug: str, name: str, width: int) -> str:
    """Return a public URL for a Medium-compatible copy of the image at `src`."""
    path = local_path(src)
    if path.suffix.lower() == ".svg":
        target = out_dir / f"{name}.png"
        page = (f"<html><body style='margin:0;background:#fff'>"
                f"<img src='{path.as_uri()}' style='width:{width}px;display:block'></body></html>")
        screenshot(page, target, width + 40)
        return f"{SITE}/medium/{slug}/{target.name}"
    with path.open("rb") as handle:
        signature = handle.read(12)
    if signature[:4] == b"RIFF" and signature[8:12] == b"WEBP":
        target = out_dir / f"{name}.png"
        Image.open(path).convert("RGB").save(target, optimize=True)
        return f"{SITE}/medium/{slug}/{target.name}"
    return f"{SITE}{quote(src)}"


def build(post: Path) -> tuple[str, str, str]:
    fields, body = front_matter(post.read_text())
    slug, title = fields["slug"], fields["title"]
    canonical = f"{SITE}/posts/{slug}/"
    out_dir = OUTPUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = markdown.markdown(body, extensions=["fenced_code", "tables"])
    soup = BeautifulSoup(rendered, "html.parser")
    counter = {"figure": 0, "table": 0}

    for image in soup.find_all("img"):
        src = image.get("src", "")
        if src.startswith("http"):
            continue
        counter["figure"] += 1
        width = int(image.get("width") or 900)
        image["src"] = convert_image(src, out_dir, slug, f"figure-{counter['figure']}", width)
        for attribute in ("width", "height", "loading", "decoding", "style"):
            image.attrs.pop(attribute, None)

    for table in soup.find_all("table"):
        counter["table"] += 1
        target = out_dir / f"table-{counter['table']}.png"
        page = f"<html><head><style>{TABLE_CSS}</style></head><body>{table}</body></html>"
        screenshot(page, target, 1100)
        replacement = soup.new_tag("img", src=f"{SITE}/medium/{slug}/{target.name}",
                                   alt=re.sub(r"\s+", " ", table.get_text(" ")).strip()[:300])
        table.replace_with(replacement)

    for row in soup.select("div.listen-row"):
        label = row.find("p").get_text(" ", strip=True)
        audio = row.find("audio")
        paragraph = soup.new_tag("p")
        link = soup.new_tag("a", href=f"{SITE}{audio['src']}")
        link.string = label
        paragraph.append(link)
        row.replace_with(paragraph)
    for audio in soup.find_all("audio"):
        paragraph = soup.new_tag("p")
        link = soup.new_tag("a", href=f"{SITE}{audio['src']}")
        link.string = "Listen to this sample"
        paragraph.append(link)
        audio.replace_with(paragraph)

    for anchor in soup.find_all("a"):
        href = anchor.get("href", "")
        if href.startswith("/"):
            anchor["href"] = f"{SITE}{href}"
    for element in soup.find_all(True):
        element.attrs.pop("style", None)
        element.attrs.pop("class", None) if element.name != "code" else None

    footer = (f'<hr><p><em>Originally published at <a href="{canonical}">callumjmac.github.io</a>. '
              f"Figures, code and data are linked from the original post.</em></p>")
    page = f"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<meta name="robots" content="noindex">
<meta name="description" content="{html.escape(fields.get('description', ''))}">
<link rel="canonical" href="{canonical}">
</head>
<body>
<article>
<h1>{html.escape(title)}</h1>
<p><em>{html.escape(fields.get('description', ''))}</em></p>
{soup}
{footer}
</article>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page)
    return slug, title, f"{SITE}/medium/{slug}/"


def main() -> None:
    rows = [build(post) for post in sorted(POSTS.glob("*.md"))]
    for slug, title, url in rows:
        print(f"{url}  ->  canonical {SITE}/posts/{slug}/")


if __name__ == "__main__":
    main()
