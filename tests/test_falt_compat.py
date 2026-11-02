"""Faltwerk format (.falt / magic FALT) next to the legacy .csgs / CSGS.

New sections carry FALT, the decoder reads both, unknown magic gives a clear error, directories may mix both suffixes,
`codec.csgs` and the old CLI flag still work, and a real legacy stream from data/ decodes to exactly the arrays the old
decoder dumped (web/bench/dump_ref.py refs; skipped when the data is not on this machine).
"""
import json
import os
import subprocess
import sys
import zlib

import numpy as np
import pytest

import codec.falt as falt
from codec.falt import (LEGACY_MAGIC, MAGIC, Decoder, check_magic, section_files, section_name, stream_dir, with_magic,
                        write_section)
from test_falt import encode_all, make_stream, params
from test_prep_s2 import synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.environ.get('FALT_REAL_DIR') or os.environ.get('CSGS_S1_REF_DIR') or os.path.join(
    'data/p14/s1')            # s0000.csgs, s0001.csgs + ref/ (dump_ref.py of the old decoder)


def stream():
    p = params(True, key_interval=2)
    secs = make_stream(p, 3, seed=5)
    return secs, encode_all(p, secs)


def test_encoder_writes_falt_and_legacy_decodes_the_same():
    secs, bufs = stream()
    assert MAGIC == b'FALT' and LEGACY_MAGIC == b'CSGS'
    assert all(b[:4] == b'FALT' for b in bufs)
    legacy = [with_magic(b, LEGACY_MAGIC) for b in bufs]
    for a, b in zip(bufs, legacy):                         # 4 magic bytes + the CRC32 (covers the magic) differ, nothing else
        assert len(a) == len(b) and a[4:-4] == b[4:-4] and a[:4] != b[:4]
    d1, d2, d3 = Decoder(), Decoder(), Decoder()
    mixed = [legacy[0], bufs[1], legacy[2]]                 # one stream, sections of both kinds
    for s, a, b, m in zip(secs, bufs, legacy, mixed):
        assert d1.decode(a) == s and d2.decode(b) == s and d3.decode(m) == s
    assert with_magic(legacy[0]) == bufs[0]


def test_unknown_magic_is_named():
    _, bufs = stream()
    bad = b'GLTF' + bufs[0][4:]
    for f in (check_magic, Decoder().decode, falt.stream_sizes):
        with pytest.raises(ValueError, match=r"not a Faltwerk section \(\.falt\): magic b'GLTF', expected b'FALT' or legacy b'CSGS'"):
            f(bad)
    with pytest.raises(ValueError, match='magic'):
        Decoder().decode(b'CS')
    broken = bytearray(bufs[0]); broken[-1] ^= 1            # known magic, damaged body: still the CRC error
    with pytest.raises(ValueError, match='CRC mismatch'):
        Decoder().decode(bytes(broken))


def test_section_files_and_names(tmp_path):
    assert section_name(7) == 's0007.falt' and section_name(7, '.csgs') == 's0007.csgs'
    (tmp_path / 's0000.csgs').write_bytes(b'x'); (tmp_path / 's0001.falt').write_bytes(b'y')
    (tmp_path / 'notes.txt').write_text('-')
    assert [os.path.basename(f) for f in section_files(str(tmp_path))] == ['s0000.csgs', 's0001.falt']
    (tmp_path / 's0001.csgs').write_bytes(b'z')
    with pytest.raises(ValueError, match='both .falt and .csgs'):
        section_files(str(tmp_path))
    with pytest.raises(FileExistsError, match='legacy'):
        write_section(str(tmp_path), 0, b'FALT')
    p = write_section(str(tmp_path / 'new'), 3, b'FALT...')
    assert p.endswith('s0003.falt') and open(p, 'rb').read() == b'FALT...'


def test_stream_dir_keeps_old_runs(tmp_path):
    assert stream_dir(str(tmp_path)).endswith('/falt')
    (tmp_path / 'csgs').mkdir()
    assert stream_dir(str(tmp_path)).endswith('/csgs')        # a run started before keeps its directory
    (tmp_path / 'falt').mkdir()
    assert stream_dir(str(tmp_path)).endswith('/falt')


def test_codec_csgs_is_an_alias():
    import codec.csgs as old
    from codec.csgs import Decoder as D2, MAGIC as M2
    assert old is falt and D2 is Decoder and M2 == b'FALT'
    import codec
    assert codec.csgs is falt


@pytest.mark.parametrize('extra,name', [([], 'falt'), (['--out-name', 'csgs'], 'csgs')])
def test_p44_encode_output_dirs(tmp_path, extra, name):
    """standard encoder: new runs write <out>/falt/ + falt.json; --out-name csgs keeps the pre-layout the old
    chains read (files inside are .falt either way); an <out> that already has csgs/ keeps it by default."""
    k = tmp_path / 'k'; k.mkdir()
    synth(k)
    out = tmp_path / 'enc'
    subprocess.run([sys.executable, os.path.join(ROOT, 'train/recipe/encode.py'), str(k), str(out), '--source', 'mesh',
                    '--footprint', '0.01', *extra], cwd=ROOT, check=True, capture_output=True)
    assert (out / f'{name}.json').is_file()
    files = section_files(str(out / name))
    assert files and all(f.endswith('.falt') and open(f, 'rb').read(4) == b'FALT' for f in files)
    assert json.load(open(out / 'measure.json'))['csgs_dir'].endswith('/' + name)


@pytest.mark.parametrize('flag', ['--write-falt', '--write-csgs'])
def test_measure_writes_falt_files(tmp_path, flag):
    synth(tmp_path)
    d = tmp_path / 'out'
    subprocess.run([sys.executable, '-m', 'codec.measure', '--source', 'curves', '--curves', str(tmp_path / 'curves.npz'),
                    '--tracks', str(tmp_path / 'tracks.npz'), '--quat', 'raw', '--key-interval', '4', flag, str(d),
                    '--out', str(tmp_path / 'm.json'), '--qpos', '0.001'], cwd=ROOT, check=True, capture_output=True)
    files = sorted(os.listdir(d))
    assert files and all(f.endswith('.falt') for f in files)
    assert all(open(d / f, 'rb').read(4) == b'FALT' for f in files)
    dec = Decoder()
    for f in section_files(str(d)):
        dec.decode(open(f, 'rb').read())


# ------------------------------------------------------------------ real legacy stream (data/, not in the repo)

def ref_arrays(sec, p):
    """the decoded arrays exactly as web/bench/dump_ref.py lists them (name -> int32 CRC32)."""
    arrays = [('ids', np.array(sec.ids, np.int64).astype('<u4').view('<i4')), ('birth', np.array(sec.birth)),
              ('death', np.array(sec.death))]
    for g, (tag, _) in enumerate(p.groups):
        arrays.append((f'k_{tag}', np.array(sec.k[g])))
        arrays.append((f'knots_{tag}', np.concatenate([np.array(r, np.int64) for r in sec.knots[g]])))
    arrays.append(('sn', np.array(sec.sn, np.int64).reshape(-1)))
    if p.sn_vq is not None:
        arrays.append(('sn_deg', np.array(sec.sn_deg, np.int64)))
        arrays.append(('sn_idx', np.array([x for row in sec.sn_idx for x in row], np.int64)))
        if sec.sn_cb is not None:
            arrays.append(('sn_cb', np.array([v for lvl in sec.sn_cb for st in lvl for e in st for v in e], np.int64)))
    return [dict(name=nm, len=int(a.size), crc32=zlib.crc32(np.ascontiguousarray(a.astype('<i4')).tobytes()) & 0xffffffff)
            for nm, a in arrays]


@pytest.mark.skipif(not os.path.isfile(os.path.join(REAL, 'ref', 's0000.ref.json')), reason='no real legacy stream')
def test_real_legacy_stream_decodes_as_before():
    files = section_files(REAL)
    assert [os.path.basename(f) for f in files][:2] == ['s0000.csgs', 's0001.csgs']
    old, new = Decoder(), Decoder()
    for f in files[:2]:
        buf = open(f, 'rb').read()
        assert buf[:4] == b'CSGS'
        ref = json.load(open(os.path.join(REAL, 'ref', os.path.basename(f)[:-5] + '.ref.json')))
        s_old = old.decode(buf)
        assert ref_arrays(s_old, old.params) == ref['arrays']          # same arrays as the pre-decoder dumped
        s_new = new.decode(with_magic(buf))                            # the same section relabelled FALT
        assert s_new == s_old
