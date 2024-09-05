from pathlib import Path
from bs4 import BeautifulSoup
import json
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parents[2]
DATA_DIR = PROJECT_DIR / 'data' / 'raw'
CHUDE_JSON_PATH = DATA_DIR / 'ChuDe.json'
DEMUC_JSON_PATH = DATA_DIR / 'DeMuc.json'
ALL_JSON_PATH = DATA_DIR / 'All.json'
DEMUC_DIR = DATA_DIR / 'demuc'

def find_chude(id, chude_json):
    for chude in chude_json:
        if chude['Value'] == id:
            return chude['Text']
            
def parse_file(html_file, metadata:dict):
    with open(html_file) as f:
        soup = BeautifulSoup(f, 'html.parser')

    contents = []
    pdieu = soup.find_all('p', class_='pDieu')
    num_items = len(pdieu)
    if num_items > 0:
        for i in range(num_items - 1):
            item = {**metadata}
            item['id'] = pdieu[i].find('a').get('name')
            item['title'] = pdieu[i].text
            item['content'] = "\n".join(get_all_texts_between(pdieu[i], pdieu[i+1]))
            contents.append(item)
        last_item = {**metadata}
        last_item['id'] = pdieu[num_items-1].find('a').get('name')
        last_item['title'] = pdieu[num_items-1].text
        last_item['content'] = "\n".join(get_all_texts_between(pdieu[num_items-1]))
        contents.append(item)
    return contents

def get_all_texts_between(start_tag, end_tag=None):
    filtered_elements = []
    for sibling in start_tag.find_next_siblings():
        if end_tag != None and sibling == end_tag:
            break
        if sibling.name == "p":
            filtered_elements.append(sibling.text)

    return filtered_elements

def load_documents():
    with open(DEMUC_JSON_PATH) as f:
        demuc_json = json.load(f)
    with open(CHUDE_JSON_PATH) as f:
        chude_json = json.load(f)
    
    data = []
    for demuc in tqdm(demuc_json, desc="Load documents from raw data..."):
        
        meta_data = {}
        meta_data['de_muc'] = demuc['Text']
        meta_data['chu_de'] = find_chude(demuc['ChuDe'], chude_json)

        all_contents = parse_file(DEMUC_DIR / f"{demuc['Value']}.html", meta_data)
        data += all_contents
    return data