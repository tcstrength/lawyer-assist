# -*- coding: utf-8 -*-
"""
%cd ../
Author: tcstrength
Date: 2024-07-04
"""

import requests
import urllib
from util import config

    
def post_chat_stream(user_input: str):
    s = requests.Session()
    url = urllib.parse.urljoin(config.LAWYER_API_URL, "chat")
    with s.post(url, json={"user_input": user_input}, headers=None, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                yield line.decode("utf-8")