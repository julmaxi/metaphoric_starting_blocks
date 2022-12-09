"""
Most code taken from NAACL 2018 Figurative Workshop Shared Task on Metaphor Detection
See original note below:

Script to parse VUA XML corpus to get text fragment, sentence id, sentence text tuples.

:author: Ben Leong (cleong@ets.org)

"""

import configparser
import csv
import json
import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator


TRAINING_PARTION = [
    'a1e-fragment01',
    'a1f-fragment06',
    'a1f-fragment07',
    'a1f-fragment08',
    'a1f-fragment09',
    'a1f-fragment10',
    'a1f-fragment11',
    'a1f-fragment12',
    'a1g-fragment26',
    'a1g-fragment27',
    'a1h-fragment05',
    'a1h-fragment06',
    'a1j-fragment34',
    'a1k-fragment02',
    'a1l-fragment01',
    'a1m-fragment01',
    'a1n-fragment09',
    'a1n-fragment18',
    'a1p-fragment01',
    'a1p-fragment03',
    'a1x-fragment03',
    'a1x-fragment04',
    'a1x-fragment05',
    'a2d-fragment05',
    'a38-fragment01',
    'a39-fragment01',
    'a3c-fragment05',
    'a3e-fragment03',
    'a3k-fragment11',
    'a3p-fragment09',
    'a4d-fragment02',
    'a6u-fragment02',
    'a7s-fragment03',
    'a7y-fragment03',
    'a80-fragment15',
    'a8m-fragment02',
    'a8n-fragment19',
    'a8r-fragment02',
    'a8u-fragment14',
    'a98-fragment03',
    'a9j-fragment01',
    'ab9-fragment03',
    'ac2-fragment06',
    'acj-fragment01',
    'ahb-fragment51',
    'ahc-fragment60',
    'ahf-fragment24',
    'ahf-fragment63',
    'ahl-fragment02',
    'ajf-fragment07',
    'al0-fragment06',
    'al2-fragment23',
    'al5-fragment03',
    'alp-fragment01',
    'amm-fragment02',
    'as6-fragment01',
    'as6-fragment02',
    'b1g-fragment02',
    'bpa-fragment14',
    'c8t-fragment01',
    'cb5-fragment02',
    'ccw-fragment03',
    'cdb-fragment02',
    'cdb-fragment04',
    'clp-fragment01',
    'crs-fragment01',
    'ea7-fragment03',
    'ew1-fragment01',
    'fef-fragment03',
    'fet-fragment01',
    'fpb-fragment01',
    'g0l-fragment01',
    'kb7-fragment10',
    'kbc-fragment13',
    'kbd-fragment07',
    'kbh-fragment01',
    'kbh-fragment02',
    'kbh-fragment03',
    'kbh-fragment09',
    'kbh-fragment41',
    'kbj-fragment17',
    'kbp-fragment09',
    'kbw-fragment04',
    'kbw-fragment11',
    'kbw-fragment17',
    'kbw-fragment42',
    'kcc-fragment02',
    'kcf-fragment14',
    'kcu-fragment02',
    'kcv-fragment42']


TESTING_PARTION = [
    'a1j-fragment33',
    'a1u-fragment04',
    'a31-fragment03',
    'a36-fragment07',
    'a3e-fragment02',
    'a3m-fragment02',
    'a5e-fragment06',
    'a7t-fragment01',
    'a7w-fragment01',
    'aa3-fragment08',
    'ahc-fragment61',
    'ahd-fragment06',
    'ahe-fragment03',
    'al2-fragment16',
    'b17-fragment02',
    'bmw-fragment09',
    'ccw-fragment04',
    'clw-fragment01',
    'cty-fragment03',
    'ecv-fragment05',
    'faj-fragment17',
    'kb7-fragment31',
    'kb7-fragment45',
    'kb7-fragment48',
    'kbd-fragment21',
    'kbh-fragment04',
    'kbw-fragment09'
]


def read_config(configFilename):
    parser = configparser.ConfigParser()
    parser.read(configFilename)

    xml_file = parser['params']['xml_file']
    functions = set(parser['params']['functions'].split(','))
    types = set(parser['params']['types'].split(','))
    subtypes = set(parser['params']['subtypes'].split(','))
    function_override = bool(
        parser['params']['function_override'].lower() == 'true')

    return xml_file, functions, types, subtypes, function_override


def is_metaphor(seg, functions, types, subtypes, function_override):
    if seg is not None:
        if seg.get('function') in functions:
            if not function_override:
                return 1
            elif function_override and (seg.get('type') in types
                                    or seg.get('subtype') in subtypes):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0


def handle_anomaly(txt_id, sentence_id):
    if txt_id == 'as6-fragment01' and sentence_id == '26':
        return 'M_to'
    if txt_id == 'as6-fragment01' and sentence_id == '89':
        return 'M_sector'
    if txt_id == 'kb7-fragment48' and sentence_id == '13368':
        return 'like'


def extract_xml_tag_text(
    txt_id,
    sentence_id,
    namespace,
    t,
    functions,
    types,
    subtypes,
        function_override):

    final_token = None

    segs = t.findall('./' + namespace + 'seg')
    if len(segs) > 0:
        for seg in segs:
            if seg.text is None:
                return handle_anomaly(txt_id, sentence_id)

            flag = is_metaphor(seg, functions, types, subtypes, function_override)
            temp_token = seg.text.strip()
            temp_token = re.sub('[\[\]]', '', temp_token)  # replace non-word

            if flag == 1:
                temp_token = 'M_' + temp_token
                temp_token = re.sub(' +', ' M_', temp_token)

            prefix = t.text
            if prefix:
                temp_token = prefix.strip() + ' ' + temp_token

            suffix = seg.tail
            if suffix:
                temp_token = temp_token + ' ' + suffix.strip()

            temp_token = re.sub(' +', ' ', temp_token)
            if final_token:
                final_token += temp_token.strip()
            else:
                final_token = temp_token.strip()
    else:
        try:
            final_token = t.text.strip()
            # replace non-word
            final_token = re.sub('[\[\]]', '', final_token)
            final_token = re.sub(' +', ' ', final_token)
        except:
            pass

    if final_token and re.search(
            '-',
            final_token) and re.search(
            'M_',
            final_token):
        final_token = re.sub(' ', '', final_token)
        final_token = re.sub('M_', '', final_token)
        final_token = 'M_' + final_token

    if final_token and len(
            final_token.split()) > 1 and len(
            t.get('lemma').split()) == 1:
        final_token = re.sub(' ', '', final_token)

    # cleaning corrupted tokens due to annotator errors
    if final_token and re.search('^>[A-Za-z]+', final_token):
        final_token = re.sub('>', '', final_token)
    if final_token and re.search('^<[A-Za-z]+', final_token):
        final_token = re.sub('<', '', final_token)
    if final_token and re.search('^=[A-Za-z]+', final_token):
        final_token = re.sub('=', '', final_token)
    if final_token and re.search('^/[A-Za-z]+', final_token):
        final_token = re.sub('/', '', final_token)

    return final_token


def process_sentence(
        txt_id,
        sentence,
        tei_namespace,
        functions,
        types,
        subtypes,
        function_override):

    sentence_id = sentence.get('n')
    tokens_lst = []
    tokens = sentence.findall('*')
    for t in tokens:
        # special handling of cases with embedded words/puncts within a
        # <hi></hi> pair of tags
        if t.tag == tei_namespace + 'hi':
            subTokens = t.findall('*')
            for st in subTokens:
                token_text = extract_xml_tag_text(
                    txt_id, sentence_id,
                    tei_namespace, st, functions,
                    types, subtypes, function_override)

                if token_text is None or token_text == '':
                    continue
                tokens_lst.append(token_text.strip())
                continue  # done for this tag pair, continue

        token_text = extract_xml_tag_text(
            txt_id, sentence_id,
            tei_namespace,
            t,
            functions,
            types,
            subtypes,
            function_override)

        # skips empty, non-meaningful tokens
        if token_text is None or token_text == '':
            continue

        tokens_lst.append(token_text.strip())

    return sentence_id, ' '.join(tokens_lst)


def extract_xml(
    xml_file,
    functions,
    types,
    subtypes,
        function_override):

    tei_namespace = '{http://www.tei-c.org/ns/1.0}'
    xml_namespace = '{http://www.w3.org/XML/1998/namespace}'

    tree = ET.parse(xml_file)
    root = tree.getroot()
    texts = root.findall(
        './' +
        tei_namespace +
        'text/' +
        tei_namespace +
        'group/' +
        tei_namespace +
        'text')

    output = []

    for txt in texts:
        txt_id = txt.attrib[xml_namespace + 'id']
        #if txt_id not in TRAINING_PARTION:
        #    continue
        sents = txt.findall('.//' + tei_namespace + 's')
        for s in sents:
            sentence_id, sentence_txt = process_sentence(
                txt_id, s, tei_namespace, functions, types, subtypes, function_override)
            output.append({'txt_id': txt_id,
                            'sentence_id': sentence_id,
                            'sentence_txt': sentence_txt})

    return output


def main(temp_dir):
    xml_file = temp_dir / "2541" / "VUAMC.xml"
    functions = {"mrw"}
    types = {"impl"}
    subtypes = {"PP", "WIDLII"}
    function_override = False

    output = extract_xml(
        xml_file,
        functions,
        types,
        subtypes,
        function_override)

    return output


def download_zip_and_unpack(url, temp_dir, out_dir=None):
    if out_dir is not None:
        out_dir.mkdir()
    else:
        out_dir = temp_dir
    fname = os.path.basename(urllib.parse.urlparse(url).path)
    request = urllib.request.Request(url, data=None, headers={"Accept": "*/*"}, method="GET")
    with urllib.request.urlopen(request) as f:
        data = f.read()
        with open(temp_dir / fname, "wb") as f_o:
            f_o.write(data)
    zipfile.ZipFile(temp_dir / fname).extractall(out_dir)


from collections import namedtuple

LabelInstance = namedtuple("LabelInstance", "txt_id sentence_id token_id label")


def read_label_file(path):
    with open(path) as f:
        reader = csv.reader(f)
        for line in reader:
            txt_id, sentence_id, token_id = line[0].split("_")
            yield LabelInstance(txt_id=txt_id, sentence_id=(sentence_id), token_id=int(token_id), label=line[1])


class Tok2CharMapper:
    def __init__(self, tokens) -> None:
        self.tokens = list(tokens)
        self.token_starts = []
        cursor = 0

        for token in tokens:
            self.token_starts.append((cursor, cursor + len(token)))
            cursor += len(token) + 1
    
    def token_to_char_range(self, token_idx):
        return self.token_starts[token_idx]

    def get_raw_string(self):
        return " ".join(self.tokens)


def make_data_file_from_sentences_and_labels(sentence_database: dict[str, dict], labels: Iterator[LabelInstance]):
    instance_database = defaultdict(list)

    for label_instance in labels:
        sentence = sentence_database[label_instance.txt_id, label_instance.sentence_id]
        tokens = sentence["tokens"]
        mapper = sentence["mapper"]
        instance_database[label_instance.txt_id, label_instance.sentence_id].append(
            (
                (mapper.token_to_char_range(label_instance.token_id - 1), label_instance.label)
            )
        )
    
    out = []
    for key, instances in instance_database.items():
        sentence = sentence_database[key]

        out.append({
            "txt_id": sentence["txt_id"],
            "sentence_id": sentence["sentence_id"],
            "sentence": sentence["mapper"].get_raw_string(),
            "instances": [
                {"char_range": char_range, "label": int(label)} for char_range, label in instances
            ]
        })
    
    return out

def download_vua():
    temp_dir = Path(tempfile.mkdtemp())
    
    download_zip_and_unpack("https://web.archive.org/web/20151023150541/http://ota.ox.ac.uk/text/2541.zip", temp_dir)
    download_zip_and_unpack("https://github.com/EducationalTestingService/metaphor/releases/download/v1.0/naacl_flp_train_gold_labels.zip", temp_dir, temp_dir / "train_labels")
    download_zip_and_unpack("https://github.com/EducationalTestingService/metaphor/releases/download/v1.0/naacl_flp_test_gold_labels.zip", temp_dir, temp_dir / "test_labels")

    sentences = main(temp_dir)

    for sentence in sentences:
        tokens = sentence["sentence_txt"].split()
        tokens = [t if not t.startswith("M_") else t[2:] for t in tokens]
        sentence["tokens"] = tokens
        sentence["mapper"] = Tok2CharMapper(tokens)
    
    sentence_database = { (sentence["txt_id"], sentence["sentence_id"]): sentence for sentence in sentences }
    for task_name in "all_pos", "verb":
        train_labels = read_label_file(temp_dir / "train_labels" / f"{task_name}_tokens.csv")
        test_labels = read_label_file(temp_dir / "test_labels" / f"{task_name}_tokens.csv")

        train_out = make_data_file_from_sentences_and_labels(sentence_database, train_labels)
        test_out = make_data_file_from_sentences_and_labels(sentence_database, test_labels)

        with open(f"data/{task_name}_train", "w") as f:
            for elem in train_out:
                json.dump(elem, f)
                f.write("\n")

        with open(f"data/{task_name}_test", "w") as f:
            for elem in test_out:
                json.dump(elem, f)
                f.write("\n")
    
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    download_vua()
