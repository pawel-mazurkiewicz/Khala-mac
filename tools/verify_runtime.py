"""Phase-3 aggregate verification: runs the Unit A/B gates and the Phase-1 regression.
Run: KHALA_DEVICE=cpu .venv-mac/bin/python tools/verify_runtime.py"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    from tests import test_kv_cache as t_kv
    from tests import test_attention_mask as t_mask
    from tests import test_superres as t_sr

    print("== (a) backbone: cache equivalence + greedy gate + sampler ==")
    t_kv.test_cache_equivalence()
    t_kv.test_cache_equivalence_batch()
    t_kv.test_topk_sampler_runs()
    t_kv.test_sampler_greedy_gate()

    print("== (b) attention: non-causal padding-mask equivalence ==")
    t_mask.test_padding_mask_matches_prefix()

    print("== (c) super-res: forward trace + projection path ==")
    t_sr.test_superres_forward_trace()
    t_sr.test_superres_projection_runs()

    print("\nALL PHASE-3 GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
