"""
%cd ../
%load_ext autoreload
%autoreload 2
"""

import os
import requests
import json
from pathlib import Path
from hashlib import md5
from typing import Dict
from loguru import logger
from tqdm import tqdm
from lxml import html
from html2text import HTML2Text
from util import config

class DataCrawler:
    def __init__(self, data_dir: str, url_prefix: str):
        self.url_prefix = url_prefix
        self.data_dir = data_dir
        Path(self.data_dir).mkdir(exist_ok=True)

    def __get_html(self, url: str, params: Dict[str, str] = None) -> str:
        response = requests.get(url, params)
        if response.status_code != 200:
            response.raise_for_status()
        return response.content

    def get_no_pages(self) -> int:
        resp = self.__get_html(self.url_prefix, {"page": "1"})
        xpath = "/html/body/div[7]/div[2]/div[1]/section/div[2]/nav/ul/li/a"
        tree = html.fromstring(resp)
        items = tree.xpath(xpath)
        page = items[-1].attrib.get("aria-label")
        return int(page)

    def get_list_articles(self, page: int) -> int:
        assert page > 0
        resp = self.__get_html(self.url_prefix, {"page": str(page)})
        xpath = "/html/body/div[7]/div[2]/div[1]/section/article[*]/a"
        tree = html.fromstring(resp)
        items = tree.xpath(xpath)
        articles = [x.attrib.get("href") for x in items]
        return articles

    def get_article_content(self, url: str) -> str:
        xpath = """//*[@id="news-content"]"""
        resp = self.__get_html(url)
        tree = html.fromstring(resp)
        item = tree.xpath(xpath)[0]
        return html.tostring(item).decode("utf-8")

    def __html_to_markdown(self, html: str) -> str:
        converter = HTML2Text()
        converter.ignore_links = False
        markdown_text = converter.handle(html)
        return markdown_text

    def crawl_article(self, url: str, extras: Dict[str, object]) -> str:
        content = self.get_article_content(url)
        markdown = self.__html_to_markdown(content)
        data = {
            "url": url,
            "md": markdown,
            "extras": extras
        }
        path = os.path.join(self.data_dir, f"{md5(url.encode()).hexdigest()}.json")
        json.dump(data, open(path, "w"), ensure_ascii=False)

def main():
    crawler = DataCrawler(config.DATA_ARTICLE_DIR, config.DATA_SRC_URL_PREFIX)
    no_pages = crawler.get_no_pages()
    cur_page = 1
    for page in tqdm(range(cur_page, no_pages + 1), desc="Crawling data..."):
        cur_page = page
        articles = crawler.get_list_articles(page)
        for url in articles:
            try:
                crawler.crawl_article(url, {"page": page})
            except Exception as e:
                logger.warning(e)

if __name__ == "__main__":
    main()