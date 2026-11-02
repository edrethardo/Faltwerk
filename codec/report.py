"""summary table of codec.measure JSONs."""
import json, sys
for f in sys.argv[1:]:
    j = json.load(open(f))
    tb, mb = j['total_bytes'], j['mbit_s']
    print(f'== {f}: source {j["source"]} tracks {j["tracks"]}')
    print(f'   T={j["T"]} alive/frame {j["alive_mean"]:.0f} ks {j["ks"]} shared_knot {j["shared_knot"]} key_interval {j["key_interval"]}')
    print(f'   total bytes: rANS .falt {tb["rans"]} ({mb["rans"]} Mbit/s, {j["bytes_per_splat_frame_rans"]} B/splat/frame) | '
          f'zstd19 per stream {tb["zstd19_per_stream"]} ({mb["zstd19_per_stream"]}) | zstd19 concat {tb["zstd19_concat"]} ({mb["zstd19_concat"]})'
          + (f' | e6 layout zstd19 {j["e6_layout_zstd19_bytes"]} ({mb["e6_layout_zstd19"]})' if 'e6_layout_zstd19_bytes' in j else ''))
    print(f'   rANS vs best zstd: -{j["rans_vs_best_zstd_saving_pct"]} %   roundtrip all {j["roundtrip_all_ok"]}   '
          f'encode {j["secs_encode"]} s  decode {j["secs_decode"]} s  prep {j["secs_prep"]} s')
    print('   | stream | symbols | rANS B | zstd-19 B | saving |')
    for k, v in j['per_stream'].items():
        if v['nsym']:
            print(f'   | {k} | {v["nsym"]} | {v["rans"]} | {v["zstd19"]} | {v["saving_pct"]} % |')
    for r in j['sections']:
        nsym = sum(r['nsym'].values())
        print(f'   section {r["index"]} key={r["key"]} splats {r["splats"]} cont {r["n_cont"]} new {r["n_new"]}: {r["bytes_csgs"]} B '
              f'(as key {r.get("bytes_as_key")}, key roundtrip {r.get("key_roundtrip_ok")}), zstd concat {r["zstd19_concat"]}, '
              f'enc {r["secs_encode"]} s dec {r["secs_decode"]} s, {nsym} symbols -> {1e6 * r["secs_decode"] / max(nsym, 1):.2f} us/sym dec, '
              f'roundtrip {r["roundtrip_ok"]}')
    ex = {k: v for k, v in j['extra'].items() if k not in ('k_hist',)}
    print('   extra', ex, j['extra'].get('k_hist', ''))
