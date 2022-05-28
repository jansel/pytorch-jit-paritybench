import json
import logging
import os
import re
import time

import requests

log = logging.getLogger(__name__)


class CrawlGitHub(object):
    """
    Download projects from github with 100+ stars and the word "pytorch"
    """

    def __init__(self, download_dir, max_count=None, query=""):
        super(CrawlGitHub, self).__init__()
        self.download_dir = download_dir
        self.max_count = max_count # max number of projects to download
        self.usr_query = query

    def github_search(self):
        base = "https://api.github.com/search/repositories?per_page=100&sort=stars"
        query = "pytorch+language:Python+stars:>100+size:<100000"
        if self.usr_query != "":
            query = self.usr_query

        seen = set()
        # both orders gets us 20 pages (past 10 limit), need 12 for current query
        for order in ("desc", "asc"):
            page = 1
            while True:
                time.sleep(6)  # https://developer.github.com/v3/search/#rate-limit
                rs = requests.get(f"{base}&page={page}&order={order}&q={query}")
                rs.raise_for_status()
                result = rs.json()
                assert not result['incomplete_results']
                for project in result["items"]:
                    name = project["full_name"]
                    if self.max_count and len(seen) >= self.max_count:
                        return
                    if name not in seen:
                        seen.add(name)
                        yield project
                total_count = result['total_count']
                log.info(f"total_count={total_count} seen={len(seen)} page={page} {order}")
                page += 1
                if len(result["items"]) == 0 or len(seen) >= total_count or (self.max_count and len(seen) >= self.max_count):
                    return
                if page == 11:
                    break  # not allowed by API

    def download_project(self, project: dict):
        name = project["full_name"]
        url = project["html_url"]
        default_branch = project["default_branch"]
        output_filename = re.sub(r"[^a-zA-Z0-9]+", "_", name) + ".zip"
        output_path = os.path.join(self.download_dir, output_filename)
        if os.path.exists(output_path):
            return output_filename
        time.sleep(60)
        rs = requests.get(f"{url}/archive/{default_branch}.zip", stream=True)
        rs.raise_for_status()
        with open(output_path, "wb") as fd:
            for chunk in rs.iter_content(chunk_size=8192):
                fd.write(chunk)
        return output_filename

    def download(self):
        metadata_path = os.path.join(self.download_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            return

        os.path.exists(self.download_dir) or os.mkdir(self.download_dir)
        projects = list(self.github_search())
        metadata = dict()
        for i, project in enumerate(projects):
            log.info(f"Downloading {project['full_name']} ({i + 1} of {len(projects)})")
            metadata[self.download_project(project)] = project
        with open(metadata_path, "w") as fd:
            json.dump(metadata, fd)
