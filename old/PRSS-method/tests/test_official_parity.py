import hashlib
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class TestOfficialParity(unittest.TestCase):
    def test_supervised_copy_matches_archived_upstream_commit(self):
        a=(ROOT/'official_tgn'/'train_supervised.py').read_bytes()
        b=(ROOT/'official_tgn'/'source'/'train_supervised.py').read_bytes()
        self.assertEqual(a,b)
        self.assertEqual(hashlib.sha256(a).hexdigest(), 'cc890c06d96cef5325632358db5fe1371c48bc21847082087aaeac5f6eda73ec')

    def test_upstream_commit_is_pinned(self):
        self.assertEqual((ROOT/'official_tgn'/'UPSTREAM_COMMIT').read_text().strip(),
                         'd55bbe678acabb9fc3879c408fd1f2e15919667c')

    def test_all_core_upstream_files_match_pinned_manifest(self):
        manifest=json.loads((ROOT/'official_tgn'/'UPSTREAM_CORE_SHA256.json').read_text())
        expected={
          'train_supervised.py':'cc890c06d96cef5325632358db5fe1371c48bc21847082087aaeac5f6eda73ec',
          'model/tgn.py':'c1a2b9124ad5573a6002da2a6bb2a14bf86cef4e5db3648bce539811e0156529',
          'modules/embedding_module.py':'c3f473989083b0b188f811ed88edfe5b0430a8a4e2e49747884c3c2a43f5c9d3',
          'modules/memory.py':'6bed733f666bb3c491b4a3b1ae2c012e1a3c4d0cc9b0e97848fefc5493a2e75d',
          'modules/memory_updater.py':'3e87a61db539b0ef8fdd6773f94bab84861fa9896da5f4c749e746dde39073e9',
          'utils/data_processing.py':'1c5ea765620e64b27238b5631253ae918ff125341cac5a7bbe40369b5f81439c',
          'utils/utils.py':'1498a1bfb3cb7ab44fa0d8af9aa21854c191bd6392daac7e9ff818cf2236ab23',
          'evaluation/evaluation.py':'8fa815e2ec42b7fadd7a3db05ada9c414ad2e4f48cb5bee3ec91fc13c705dd19',
        }
        self.assertEqual(manifest,expected)
        for rel, sha in expected.items():
            got=hashlib.sha256((ROOT/'official_tgn'/'source'/rel).read_bytes()).hexdigest()
            self.assertEqual(got,sha,rel)

if __name__=='__main__': unittest.main()
