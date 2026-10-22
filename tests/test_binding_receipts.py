"""Check published initialization receipts against current source; no model load."""
import ast
import csv
import hashlib
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class BindingReceiptChecks(unittest.TestCase):
    def test_cpu_initialized_state_equivalence_and_source_binding(self):
        with (ROOT/'data/PRETRAINED_INITIALIZATION_AUDIT.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),6)
        for row in rows:
            self.assertEqual(row['baseline_initialized_state_sha256'],row['bound_initialized_state_sha256'])
            self.assertEqual(row['forward_calls'],'0')
            for rel,column in [('orchid/models.py','current_factory_source_sha256'),('orchid/mamba_impl.py','current_composite_source_sha256')]:
                p=ROOT/row['stage']/rel
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row[column])

    def test_factories_use_verified_file_and_composite_forward_is_preserved(self):
        for stage in ('stage1','stage2'):
            tree=ast.parse((ROOT/stage/'orchid/models.py').read_text())
            factory=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='create_model')
            self.assertIn('verified_path',ast.dump(factory))
            self.assertIn('pretrained_state_dict',ast.dump(factory))
            tree=ast.parse((ROOT/stage/'orchid/mamba_impl.py').read_text())
            cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OrchidEffLite2_MSMambaDG')
            init=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
            self.assertIn('pretrained_cfg_overlay',ast.dump(init))


if __name__=='__main__':unittest.main(verbosity=2)
