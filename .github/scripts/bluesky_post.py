#!/usr/bin/env python3
"""Announce newly published posts on Bluesky.

Called by the `announce` job in .github/workflows/deploy.yml with the
paths of post files ADDED by the push. For each post it:

  1. parses the frontmatter (title, description, slug, draft, bluesky)
  2. skips drafts, and skips posts the account has already announced
     (checks the public author feed, so re-pushes never double-post)
  3. composes "New post: <title>\n\n<hook>\n\n<url>" within 300 chars,
     preferring the hand-written `bluesky:` frontmatter hook over the
     SEO description
  4. attaches a link facet (Bluesky does NOT auto-linkify URLs — plain
     text links are not clickable) and, when the live page is up, a
     link-preview card using the page's og:image
  5. posts via the AT Protocol using BLUESKY_IDENTIFIER and
     BLUESKY_APP_PASSWORD from the environment

Stdlib only — no pip installs on the runner. Use --dry-run to print
what would be posted without authenticating or posting.
"""

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone

SITE = "https://tinyweights.dev"
PDS = "https://bsky.social"
PUBLIC_API = "https://public.api.bsky.app"
MAX_LEN = 300
MAX_THUMB_BYTES = 950_000


def http(url, data=None, headers=None, timeout=30):
    # GitHub Pages' CDN 403s the default Python-urllib user agent
    headers = {"User-Agent": "tinyweights-announce/1.0", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers


def frontmatter(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m = re.match(r"(?s)^---\r?\n(.*?)\r?\n---", text)
    if not m:
        return {}
    fm = {}
    for key in ("title", "description", "slug", "bluesky"):
        km = re.search(r'(?m)^%s:\s*"(.*)"\s*$' % key, m.group(1))
        if km:
            fm[key] = km.group(1)
    fm["draft"] = bool(re.search(r"(?m)^draft:\s*true\s*$", m.group(1)))
    return fm


def already_posted(identifier, slug):
    url = (PUBLIC_API + "/xrpc/app.bsky.feed.getAuthorFeed?actor=%s&limit=50"
           % identifier)
    try:
        body, _ = http(url)
        feed = json.loads(body).get("feed", [])
    except Exception as e:
        print("  feed check failed (%s) — assuming not yet posted" % e)
        return False
    needle = "/posts/%s" % slug
    for item in feed:
        record = item.get("post", {}).get("record", {})
        if needle in record.get("text", ""):
            return True
        uri = record.get("embed", {}).get("external", {}).get("uri", "")
        if needle in uri:
            return True
    return False


def compose(fm, url):
    prefix = "New post: %s\n\n" % fm["title"]
    hook = fm.get("bluesky") or fm.get("description") or ""
    budget = MAX_LEN - len(prefix) - len("\n\n") - len(url)
    if len(hook) > budget:
        hook = hook[: budget - 1].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
    text = prefix + hook + "\n\n" + url
    facet = {
        "index": {
            "byteStart": len((prefix + hook + "\n\n").encode("utf-8")),
            "byteEnd": len(text.encode("utf-8")),
        },
        "features": [{"$type": "app.bsky.richtext.facet#link", "uri": url}],
    }
    return text, [facet]


def fetch_live_page(url, attempts=8, delay=15):
    """GitHub Pages needs a minute after the deploy job to serve the page."""
    for i in range(attempts):
        try:
            body, _ = http(url, timeout=20)
            return body.decode("utf-8", "replace")
        except Exception:
            if i < attempts - 1:
                time.sleep(delay)
    return None


def og_content(html, prop):
    m = re.search(
        r'<meta[^>]+property="?%s"?[^>]+content="([^"]*)"' % re.escape(prop),
        html) or re.search(
        r'<meta[^>]+property="?%s"?[^>]+content=([^\s"><]+)' % re.escape(prop),
        html)
    return m.group(1) if m else None


def build_card(fm, url, token):
    """Return an app.bsky.embed.external dict, or None if anything fails."""
    html = fetch_live_page(url)
    if not html:
        print("  page not live yet — posting without a link card")
        return None
    card = {
        "$type": "app.bsky.embed.external",
        "external": {
            "uri": url,
            "title": og_content(html, "og:title") or fm["title"],
            "description": og_content(html, "og:description")
                           or fm.get("description", ""),
        },
    }
    image_url = og_content(html, "og:image")
    if image_url and token:
        # the og image asset can lag the page HTML by a bit on the Pages
        # CDN right after a deploy, so retry before giving up on the thumb
        for attempt in range(4):
            try:
                img, headers = http(image_url, timeout=30)
                if len(img) <= MAX_THUMB_BYTES:
                    body, _ = http(
                        PDS + "/xrpc/com.atproto.repo.uploadBlob", data=img,
                        headers={
                            "Content-Type": headers.get("Content-Type", "image/png"),
                            "Authorization": "Bearer " + token,
                        })
                    card["external"]["thumb"] = json.loads(body)["blob"]
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(20)
                else:
                    print("  thumb upload failed (%s) — card without image" % e)
    return card


def create_session(identifier, password):
    body, _ = http(
        PDS + "/xrpc/com.atproto.server.createSession",
        data=json.dumps({"identifier": identifier,
                         "password": password}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    session = json.loads(body)
    return session["did"], session["accessJwt"]


def announce(path, identifier, password, dry_run):
    fm = frontmatter(path)
    if not fm.get("title") or not fm.get("slug"):
        print("  no title/slug frontmatter — skipping")
        return
    if fm["draft"]:
        print("  draft — skipping")
        return
    url = "%s/posts/%s/" % (SITE, fm["slug"])
    if already_posted(identifier, fm["slug"]):
        print("  already announced — skipping")
        return

    text, facets = compose(fm, url)
    print("  text (%d chars):\n%s" % (len(text), text))

    if dry_run:
        card = build_card(fm, url, token=None)
        print("  dry run — card: %s, not posting" % ("yes" if card else "no"))
        return

    did, token = create_session(identifier, password)
    record = {
        "$type": "app.bsky.feed.post",
        "text": text,
        "facets": facets,
        "langs": ["en"],
        "createdAt": datetime.now(timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
    }
    card = build_card(fm, url, token)
    if card:
        record["embed"] = card
    body, _ = http(
        PDS + "/xrpc/com.atproto.repo.createRecord",
        data=json.dumps({"repo": did, "collection": "app.bsky.feed.post",
                         "record": record}).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token})
    print("  posted: %s" % json.loads(body).get("uri", "?"))


def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv
    identifier = os.environ.get("BLUESKY_IDENTIFIER", "tinyweights.bsky.social")
    password = os.environ.get("BLUESKY_APP_PASSWORD", "")
    if not dry_run and not password:
        sys.exit("BLUESKY_APP_PASSWORD is not set")
    for path in args:
        print(path)
        try:
            announce(path, identifier, password, dry_run)
        except Exception as e:
            # one bad post shouldn't block announcing the others
            print("  FAILED: %s" % e)


if __name__ == "__main__":
    main()
