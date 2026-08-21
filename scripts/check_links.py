#!/usr/bin/env python
"""Verify every link in the rendered site resolves.

- Internal links (relative hrefs/srcs): the target file must exist in the site dir.
- External links (http/https): must answer < 400 (HEAD, falling back to GET;
  two retries with backoff — arXiv and publisher sites rate-limit).

Usage: python scripts/check_links.py _site [--skip-external]
Exit code 1 if any link is broken.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (DKR course link checker; +https://github.com/lukmanovr/dkr)"}
SKIP_SCHEMES = ("mailto:", "javascript:", "data:", "tel:")
# Hosts that block bots but are stable; verified manually at release time instead.
EXTERNAL_ALLOWLIST = ("doi.org", "dl.acm.org", "ieeexplore.ieee.org", "linkedin.com")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        for name, value in attrs:
            if value and name in ("href", "src", "data-src"):
                self.links.append(value)


def collect_links(site_dir: Path) -> tuple[set[tuple[Path, str]], set[str]]:
    internal: set[tuple[Path, str]] = set()
    external: set[str] = set()
    for html_file in site_dir.rglob("*.html"):
        parser = LinkParser()
        parser.feed(html_file.read_text(encoding="utf-8", errors="replace"))
        for link in parser.links:
            link = link.strip()
            if not link or link.startswith(SKIP_SCHEMES) or link.startswith("#"):
                continue
            if re.match(r"^https?://", link):
                external.add(link.split("#")[0])
            else:
                internal.add((html_file, urllib.parse.unquote(link.split("#")[0].split("?")[0])))
    return internal, external


SITE_PREFIX = "/dkr"  # GitHub Pages project subpath; absolute links carry it (e.g. on 404)


def check_internal(site_dir: Path, source: Path, target: str) -> str | None:
    if not target:
        return None
    if target.startswith(SITE_PREFIX + "/") or target == SITE_PREFIX:
        target = target[len(SITE_PREFIX):] or "/"
    base = site_dir if target.startswith("/") else source.parent
    resolved = (base / target.lstrip("/")).resolve()
    if resolved.is_dir():
        resolved = resolved / "index.html"
    if not resolved.exists():
        return f"{source.relative_to(site_dir)} -> {target} (missing: {resolved})"
    return None


def check_external(url: str) -> str | None:
    host = urllib.parse.urlparse(url).netloc.lower()
    if any(host.endswith(a) for a in EXTERNAL_ALLOWLIST):
        return None
    for attempt, method in enumerate(("HEAD", "GET", "GET")):
        try:
            req = urllib.request.Request(url, headers=UA, method=method)
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status < 400:
                    return None
        except Exception as exc:
            last = f"{type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(2 * (attempt + 1))
    return f"{url} ({last})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_dir", type=Path)
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()

    if not args.site_dir.exists():
        print(f"Site directory {args.site_dir} does not exist — run `quarto render` first.")
        return 1

    internal, external = collect_links(args.site_dir)
    print(f"Checking {len(internal)} internal and {len(external)} external links ...")

    broken = [msg for src, tgt in sorted(internal) if (msg := check_internal(args.site_dir, src, tgt))]

    if not args.skip_external:
        with ThreadPoolExecutor(max_workers=8) as pool:
            broken += [msg for msg in pool.map(check_external, sorted(external)) if msg]

    if broken:
        print(f"\n{len(broken)} broken link(s):")
        for msg in broken:
            print(f"  - {msg}")
        return 1
    print("All links ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
