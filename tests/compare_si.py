"""Compare an independently extracted Word SI and repository code on the same TIFF."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tifffile

from test_contact_angle_extraction import right_angle, ROOT
import contact_angle_extraction as repository

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
KEY = ['z', 'y', 'x']


def extract_source(path):
    with ZipFile(path) as doc:
        body = ET.fromstring(doc.read('word/document.xml')).find(W + 'body')
    paragraphs = []
    for paragraph in body.findall(W + 'p'):
        pieces = []
        for item in paragraph.iter():
            if item.tag == W + 't':
                pieces.append(item.text or '')
            elif item.tag == W + 'tab':
                pieces.append('\t')
            elif item.tag in (W + 'br', W + 'cr'):
                pieces.append('\n')
        paragraphs.append(''.join(pieces))
    first, = [i for i, p in enumerate(paragraphs) if p.startswith('"""Contact-angle extraction')]
    last, = [i for i, p in enumerate(paragraphs) if p == '    run_image(INPUT_PATH, OUTPUT_DIR)']
    return ('\n'.join(paragraphs[first:last + 1]) + '\n').encode('utf-8')


def load_results(folder):
    def read(name):
        return pd.read_csv(folder / (name + '.csv.gz'), float_precision='round_trip')
    features = read('candidate_features')
    states = read('criterion_states')
    if len(states) != len(features) or not features.candidate_id.is_unique:
        raise ValueError('Incomplete candidate/state pairing')
    # The SI exports states from the same unsorted frame. Verify the assignment
    # against its independent fail / unavailable lists before joining by location.
    for state, column in [('fail', 'rejection_reasons'), ('not_evaluated', 'not_evaluated_criteria')]:
        rendered = states.apply(lambda row: ';'.join(states.columns[row.eq(state)]), axis=1)
        np.testing.assert_array_equal(rendered, features[column].fillna(''))
    if states.shape[1] != 23:
        raise ValueError('Expected all 23 criterion columns')
    np.testing.assert_array_equal(states.eq('pass').all(axis=1), features.accepted_Q)
    tables = {'features': features, 'states': pd.concat([features[KEY], states], axis=1),
              'G': read('G'), 'Q': read('Q')}
    tables = {k: v.set_index(KEY, verify_integrity=True).sort_index() for k, v in tables.items()}
    for key, mask in [('G', 'geometry_available'), ('Q', 'accepted_Q')]:
        expected = tables['features'].loc[tables['features'][mask].astype(bool)]
        pd.testing.assert_frame_equal(tables[key], expected, check_exact=True, check_dtype=False)
    q_angles = tables['Q'].contact_angle_degrees
    np.testing.assert_array_equal(q_angles, tables['G'].loc[q_angles.index, 'contact_angle_degrees'])
    return tables, json.loads((folder / 'summary.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('si', type=Path)
    parser.add_argument('--tiff', type=Path, help='Use this labelled TIFF instead of the built-in 90-degree model')
    args = parser.parse_args()
    source = extract_source(args.si)
    code = (ROOT / 'contact_angle_extraction.py').read_bytes()
    compile(source, 'si_reference.py', 'exec')
    if ast.dump(ast.parse(code)) != ast.dump(ast.parse(source)) or code != source:
        raise ValueError('Repository and extracted SI source differ')
    with TemporaryDirectory() as temp:
        work = Path(temp)
        si_path = work / 'si_reference.py'
        si_path.write_bytes(source)
        spec = importlib.util.spec_from_file_location('si_reference', si_path)
        si = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = si
        spec.loader.exec_module(si)
        if asdict(si.Parameters()) != asdict(repository.Parameters()):
            raise ValueError('Different defaults')
        if args.tiff is None:
            image = work / 'orthogonal.tif'
            tifffile.imwrite(image, right_angle(), photometric='minisblack')
        else:
            image = args.tiff
        for name, implementation in [('SI', si), ('repository', repository)]:
            implementation.run_image(image, work / name)
        reference, summary = load_results(work / 'SI')
        actual, actual_summary = load_results(work / 'repository')
        for key in reference:
            pd.testing.assert_frame_equal(actual[key], reference[key], check_exact=True)
        if summary != actual_summary:
            raise ValueError('Summary mismatch')
        print(json.dumps({'sha256': hashlib.sha256(code).hexdigest(), 'source_bytes_and_ast': 'identical',
                          'raw': len(actual['features']), 'criteria': actual['states'].shape[1],
                          'G': len(actual['G']), 'Q': len(actual['Q']), 'outputs': 'exact'}, indent=2))


if __name__ == '__main__':
    main()
