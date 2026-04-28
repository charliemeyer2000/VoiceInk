#!/usr/bin/env python3
"""Prepend a new <item> to appcast.xml using release metadata.

Reads release metadata from CLI flags, computes DMG length from the file on
disk, and rewrites the appcast in place. Creates a fresh feed if none exists.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path


def xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def build_item(args, length: int, pub_date: str, bullets: str) -> str:
    return f"""        <item>
            <title>{xml_escape(args.tag)}</title>
            <pubDate>{pub_date}</pubDate>
            <sparkle:version>{xml_escape(args.sparkle_version)}</sparkle:version>
            <sparkle:shortVersionString>{xml_escape(args.short_version)}</sparkle:shortVersionString>
            <sparkle:minimumSystemVersion>{xml_escape(args.min_system)}</sparkle:minimumSystemVersion>
            <description><![CDATA[
                <h3>Fork build {xml_escape(args.tag)}</h3>
                <ul>
{bullets}
                </ul>
            ]]></description>
            <enclosure url="{xml_escape(args.url)}" length="{length}" type="application/octet-stream" sparkle:edSignature="{xml_escape(args.signature)}"/>
        </item>"""


def fresh_feed(item: str, channel_title: str) -> str:
    return f"""<?xml version="1.0" standalone="yes"?>
<rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle" version="2.0">
    <channel>
        <title>{xml_escape(channel_title)}</title>
{item}
    </channel>
</rss>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--appcast", required=True, type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--short-version", required=True)
    p.add_argument("--sparkle-version", required=True)
    p.add_argument("--dmg-path", required=True, type=Path)
    p.add_argument("--url", required=True)
    p.add_argument("--signature", required=True)
    p.add_argument("--notes-file", required=True, type=Path)
    p.add_argument("--min-system", default="14.4")
    p.add_argument("--channel-title", default="VoiceInk (charliemeyer2000 fork)")
    args = p.parse_args()

    length = args.dmg_path.stat().st_size
    pub_date = dt.datetime.now(dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    notes_lines = [n.strip() for n in args.notes_file.read_text().splitlines() if n.strip()]
    bullets = "\n".join(f"                    <li>{xml_escape(n)}</li>" for n in notes_lines) \
        or "                    <li>Internal build</li>"

    item = build_item(args, length, pub_date, bullets)

    if args.appcast.exists():
        xml = args.appcast.read_text()
    else:
        xml = ""

    if "<channel>" in xml:
        # Match through the closing </title> + ONE trailing newline only, so
        # the existing indentation of any following <item> is preserved.
        new_xml, n = re.subn(
            r"(<channel>\s*<title>[^<]*</title>\n)",
            lambda m: m.group(1) + item + "\n",
            xml, count=1,
        )
        if n == 0:
            print("ERROR: could not locate <channel><title>...</title> insertion point", file=sys.stderr)
            return 1
    else:
        new_xml = fresh_feed(item, args.channel_title)

    args.appcast.write_text(new_xml)
    print(f"Wrote {args.appcast} (length={length}, tag={args.tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
