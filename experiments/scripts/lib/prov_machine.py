#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prov_machine.py -- 測った計算機と Blackhole のボードを読む（scripts/provenance.sh の prov_machine・prov_smi_get が呼ぶ）

    tt-smi -s | python3 -B scripts/lib/prov_machine.py ident <デバイスファイル>
    tt-smi -s | python3 -B scripts/lib/prov_machine.py get <デバイスファイル> <欄>…

標準入力に tt-smi -s の JSON を取る（空でもよい）．
  ident  1 行 host=…,cpu=…,kmd=…,dev=…,board=…,board_id=…,asic_id=…,fw=…,cm_fw=…,dm_fw=…,eth_fw=…,gddr_fw=…,smi=…
         を出す．ボードの欄はデバイスファイルの sysfs（/sys/dev/char/<major>:<minor>/tt_*．tt-kmd が出す）を先に読み，
         無ければ tt-smi の JSON の同じ項目から取る．コンテナの中（/run/.containerenv がある）では host を unknown に
         する（コンテナのホスト名は起動ごとに変わる乱数で，計算機の識別にならない）．欄の並びは lib/constants.tsv の
         MACHINE_KEYS に記録だけの dev・smi を足したもの（verify/check_routes.py が突き合わせる）．値は , と空白を含まないよう
         [A-Za-z0-9._:+-] 以外を _ にし，読めなければ unknown にする
  get    tt-smi の JSON のうちこのデバイスの項目から，与えた欄（telemetry.asic_temperature など . 区切り）の値を
         1 行ずつ出す．このデバイスの項目が見つからなければ何も出さずに終了コード 1
tt-smi の device_info からこのデバイスの項目を選ぶのは，sysfs の ASIC ID（tt_asic_id）と smbus_telem の
ASIC_ID_HIGH・ASIC_ID_LOW，無ければボードの製造番号（tt_serial と board_info.board_id）の一致による．
デバイスファイルがあって sysfs が読めず（コンテナの中など）カードが 1 枚だけならその項目を使い，それ以外で
決まらなければ（デバイスファイルが無い・tt-kmd のデバイスでない（sysfs の subsystem が tenstorrent でない）ときも）
選ばない（別のカードや別のデバイスの値を記録しないように）．
（以前は provenance.sh の中に文字列として埋め込んでいた．構文の検査も単体の試しもできるようファイルに出した）
"""
import json
import os
import re
import socket
import sys


def clean(v):
    v = "" if v is None else str(v).strip()
    if v.upper() in ("", "N/A", "NA", "NONE", "NULL", "UNKNOWN"):
        return "unknown"
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9._:+-]", "_", v)).strip("_") or "unknown"


def rd(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, TypeError, UnicodeDecodeError):
        return ""


def sysdir(dev):
    """デバイスファイルの sysfs の場所．読めなければ None，tt-kmd のデバイスでなければ "" """
    try:
        st = os.stat(dev)
    except OSError:
        return None
    d = "/sys/dev/char/%d:%d" % (os.major(st.st_rdev), os.minor(st.st_rdev))
    if not os.path.isdir(d):
        return None
    return d if os.path.basename(os.path.realpath(d + "/subsystem")) == "tenstorrent" else ""


def get(d, dotted):
    for k in dotted.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return "" if d is None else str(d).strip()


def asic_of(d):
    t = d.get("smbus_telem") or {}
    try:
        return "%08x%08x" % (int(t.get("ASIC_ID_HIGH"), 16), int(t.get("ASIC_ID_LOW"), 16))
    except (TypeError, ValueError):
        return ""


def entry(js, dev, sd):
    try:
        top = json.loads(js[js.index("{"):])
        devs = [d for d in (top.get("device_info") or []) if isinstance(d, dict)]
    except (ValueError, AttributeError):
        return None, {}
    if not os.path.exists(dev) or sd == "":
        return None, top
    asic = rd(sd + "/tt_asic_id").lower() if sd else ""
    serial = rd(sd + "/tt_serial").lower() if sd else ""
    for key, want in ((asic_of, asic), (lambda d: get(d, "board_info.board_id").lower(), serial)):
        if want:
            hit = [d for d in devs if key(d) == want]
            if len(hit) == 1:
                return hit[0], top
    if not asic and not serial and len(devs) == 1:
        return devs[0], top
    return None, top


def ident(e, top, dev, sd):
    """ident の 1 行の欄の並び [(名前, 値)]"""
    e = e or {}
    S = lambda a: rd(sd + "/" + a) if sd else ""  # noqa: E731
    cpu = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("model name"):
                    cpu = ln.split(":", 1)[1]
                    break
    except OSError:
        pass
    incont = os.path.exists("/run/.containerenv") or os.path.exists("/.dockerenv")
    kmd = rd("/sys/module/tenstorrent/version") or re.sub(r"^TT-KMD\s*", "", get(top, "host_info.Driver"))
    out = [("host", "unknown" if incont else socket.gethostname()), ("cpu", cpu), ("kmd", kmd),
           ("dev", os.path.basename(dev)),
           ("board", S("tt_card_type") or get(e, "board_info.board_type")),
           ("board_id", S("tt_serial") or get(e, "board_info.board_id")),
           ("asic_id", S("tt_asic_id") or asic_of(e).upper()),
           ("fw", S("tt_fw_bundle_ver") or get(e, "firmwares.fw_bundle_version")),
           ("cm_fw", get(e, "firmwares.cm_fw")),
           ("dm_fw", S("tt_m3app_fw_ver") or get(e, "firmwares.dm_app_fw")),
           ("eth_fw", get(e, "firmwares.eth_fw")),
           ("gddr_fw", get(e, "firmwares.gddr_fw")),
           ("smi", get(top, "host_sw_vers.tt_smi"))]
    return out


def main(argv):
    if len(argv) < 3 or argv[1] not in ("ident", "get"):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    mode, dev = argv[1], argv[2]
    js = sys.stdin.read()
    sd = sysdir(dev)
    e, top = entry(js, dev, sd)
    if mode == "get":
        if e is None:
            return 1
        for k in argv[3:]:
            print(get(e, k))
        return 0
    print(",".join("%s=%s" % (k, clean(v)) for k, v in ident(e, top, dev, sd)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
