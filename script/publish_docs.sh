#!/usr/bin/env bash
# Build the CrystOD documentation and copy it into the laboratory website
# repository, where it is served at https://mochizuki-tus.github.io/CrystOD/.
#
#   script/publish_docs.sh ~/mochizuki-tus.github.io
#
# The script writes only inside <site>/CrystOD/ and <site>/.nojekyll, and it
# never commits or pushes: review the result with `git status` in the website
# repository and push from there yourself.
set -euo pipefail

site=${1:-}
if [ -z "$site" ]; then
    echo "usage: $0 <path to a mochizuki-tus.github.io clone>" >&2
    exit 2
fi
if [ ! -d "$site/.git" ]; then
    echo "ERROR: '$site' is not a git clone." >&2
    exit 2
fi

repo=$(cd "$(dirname "$0")/.." && pwd)

# -W: a warning (a broken link, a missing file) stops the build instead of
# quietly publishing a damaged page.
# -d: keep the doctree pickles OUT of the HTML tree -- without it Sphinx puts
# them in <outdir>/.doctrees and they would be published (and committed) too
python -m sphinx -b html -W --keep-going -d "$repo/doc/_build/doctrees" \
    "$repo/doc" "$repo/doc/_build/html"

mkdir -p "$site/CrystOD"
# --delete-excluded: also clear the excluded files out of a site that was
# published before this exclusion existed (plain --delete keeps them)
rsync -a --delete --delete-excluded --exclude '.doctrees' --exclude '.buildinfo*' \
    "$repo/doc/_build/html/" "$site/CrystOD/"

# GitHub Pages runs Jekyll by default, and Jekyll drops every directory whose
# name begins with an underscore -- which is exactly Sphinx's _static/,
# _sources/ and _images/. Without this file the pages would load without their
# stylesheet and scripts.
touch "$site/.nojekyll"

# A sitemap of the documentation pages, plus one line in the site's robots.txt
# pointing at it: this is what tells a search engine that these pages exist
# (the hand-written sitemap.xml of the website lists the laboratory pages only).
SITE="$site" python - <<'PY'
import datetime
import os
import pathlib

site = pathlib.Path(os.environ["SITE"])
docs = site / "CrystOD"
base = "https://mochizuki-tus.github.io/CrystOD/"
today = datetime.date.today().isoformat()

urls = []
for path in sorted(docs.rglob("*.html")):
    rel = path.relative_to(docs).as_posix()
    if rel.startswith("_"):          # _static/embed/*.html are page fragments
        continue
    urls.append(base if rel == "index.html" else base + rel)

lines = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
for url in urls:
    lines += ["  <url>", f"    <loc>{url}</loc>",
              f"    <lastmod>{today}</lastmod>", "  </url>"]
lines.append("</urlset>")
(docs / "sitemap.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")

robots = site / "robots.txt"
entry = f"Sitemap: {base}sitemap.xml"
text = robots.read_text(encoding="utf-8") if robots.exists() else "User-agent: *\nAllow: /\n"
if entry not in text:
    robots.write_text(text.rstrip("\n") + "\n" + entry + "\n", encoding="utf-8")
    print(f"robots.txt: added '{entry}'")
print(f"sitemap: {len(urls)} pages -> CrystOD/sitemap.xml")
PY

echo
echo "Copied into $site/CrystOD"
echo "Review and push from the website repository:"
echo "  cd $site && git status"
echo "  git add CrystOD .nojekyll robots.txt && git commit -m 'docs: update CrystOD site' && git push"
