import inspect
import os
import random
import re
import subprocess
from copy import deepcopy
from typing import Any, Union
import psutil
import regex
from flatten_everything import flatten_everything
import keyboard as keyboard__
from generate_random_values_in_range import randomize_number
from sendevent_touch import SendEventTouch
from PrettyColorPrinter import add_printer
import pandas as pd
from a_pandas_ex_string_to_dtypes import pd_add_string_to_dtypes
import cv2
from a_cv_imwrite_imread_plus import add_imwrite_plus_imread_plus_to_cv2

add_imwrite_plus_imread_plus_to_cv2()
from a_cv2_imshow_thread import add_imshow_thread_to_cv2

add_imshow_thread_to_cv2()
from a_pandas_ex_plode_tool import qq_s_isnan
from shapely.geometry import Polygon

pd_add_string_to_dtypes()
add_printer(True)
from time import strftime

timest = lambda: strftime("%Y_%m_%d_%H_%M_%S")
from a_pandas_ex_xml2df import pd_add_read_xml_files
import numpy as np

pd_add_read_xml_files()


def copy_func(f):
    if callable(f):
        if inspect.ismethod(f) or inspect.isfunction(f):
            g = lambda *args, **kwargs: f(*args, **kwargs)
            t = list(filter(lambda prop: not ("__" in prop), dir(f)))
            i = 0
            while i < len(t):
                setattr(g, t[i], getattr(f, t[i]))
                i += 1
            return g
    dcoi = deepcopy([f])
    return dcoi[0]


class FlexiblePartial:
    def __init__(self, func: Any, this_args_first: bool = True, *args, **kwargs):

        self.this_args_first = this_args_first

        try:
            self.f = copy_func(func)
        except Exception:
            self.f = func
        try:
            self.args = copy_func(list(args))
        except Exception:
            self.args = args

        try:
            self.kwargs = copy_func(kwargs)
        except Exception:
            try:
                self.kwargs = kwargs.copy()
            except Exception:
                self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        newdic = {}
        newdic.update(self.kwargs)
        newdic.update(kwargs)
        if self.this_args_first:
            return self.f(*self.args, *args, **newdic)

        else:

            return self.f(*args, *self.args, **newdic)

    def __str__(self):
        return "()"

    def __repr__(self):
        return "()"


def cropimage(img, coords):
    return img[coords[1] : coords[3], coords[0] : coords[2]].copy()


def get_screenshot_adb(
    adb_executable, deviceserial,
):
    with subprocess.Popen(
        f"{adb_executable} -s {deviceserial} shell screencap -p", stdout=subprocess.PIPE
    ) as p:
        output = p.stdout.read()
    png_screenshot_data = output.replace(b"\r\n", b"\n")
    images = cv2.imdecode(
        np.frombuffer(png_screenshot_data, np.uint8), cv2.IMREAD_COLOR
    )
    images = cv2.imread_plus(images, channels_in_output=3)
    return images


def connect_to_adb(adb_path, deviceserial):
    _ = subprocess.run(f"{adb_path} start-server", capture_output=True, shell=False)
    _ = subprocess.run(
        f"{adb_path} connect {deviceserial}", capture_output=True, shell=False
    )


def get_label(va):
    try:
        stripped = va.strip("#")
        return pd.Series([stripped, int(stripped, base=16)])
    except Exception:
        return pd.Series([pd.NA, pd.NA])


def get_label_1(va):
    try:
        stripped = va.strip("#")
        return int(stripped, base=16)
    except Exception:
        return pd.NA


def execute_adb_command(
    cmd: str, subcommands: list, exit_keys: str = "ctrl+x", end_of_printline: str = ""
) -> list:
    if isinstance(subcommands, str):
        subcommands = [subcommands]
    elif isinstance(subcommands, tuple):
        subcommands = list(subcommands)
    popen = None

    def run_subprocess(cmd):
        nonlocal popen

        def kill_process():
            nonlocal popen
            try:
                print("Killing the process")
                p = psutil.Process(popen.pid)
                p.kill()
                try:
                    if exit_keys in keyboard__.__dict__["_hotkeys"]:
                        keyboard__.remove_hotkey(exit_keys)
                except Exception:
                    try:
                        keyboard__.unhook_all_hotkeys()
                    except Exception:
                        pass
            except Exception:
                try:
                    keyboard__.unhook_all_hotkeys()
                except Exception:
                    pass

        if exit_keys not in keyboard__.__dict__["_hotkeys"]:
            keyboard__.add_hotkey(exit_keys, kill_process)

        DEVNULL = open(os.devnull, "wb")
        try:
            popen = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True,
                stderr=DEVNULL,
                shell=False,
            )

            for subcommand in subcommands:
                if isinstance(subcommand, bytes):
                    subcommand = subcommand.rstrip(b"\n") + b"\n"

                    subcommand = subcommand.decode("utf-8", "replace")
                else:
                    subcommand = subcommand.rstrip("\n") + "\n"

                popen.stdin.write(subcommand)

            popen.stdin.close()

            for stdout_line in iter(popen.stdout.readline, ""):
                try:
                    yield stdout_line
                except Exception as Fehler:
                    continue
            popen.stdout.close()
            return_code = popen.wait()
        except Exception as Fehler:
            # print(Fehler)
            try:
                popen.stdout.close()
                return_code = popen.wait()
            except Exception as Fehler:
                yield ""

    proxyresults = []
    try:
        for proxyresult in run_subprocess(cmd):
            proxyresults.append(proxyresult)
            print(proxyresult, end=end_of_printline)
    except KeyboardInterrupt:
        try:
            p = psutil.Process(popen.pid)
            p.kill()
            popen = None
        except Exception as da:
            pass
            # print(da)

    try:
        if popen is not None:
            p = psutil.Process(popen.pid)
            p.kill()
    except Exception as da:
        pass

    try:
        if exit_keys in keyboard__.__dict__["_hotkeys"]:
            keyboard__.remove_hotkey(exit_keys)
    except Exception:
        try:
            keyboard__.unhook_all_hotkeys()
        except Exception:
            pass
    return proxyresults


def get_screenwidth(adb_path, deviceserial):
    try:
        screenwidth, screenheight = (
            subprocess.run(
                fr'{adb_path} -s {deviceserial} shell dumpsys window | grep cur= |tr -s " " | cut -d " " -f 4|cut -d "=" -f 2',
                shell=True,
                capture_output=True,
            )
            .stdout.decode("utf-8", "ignore")
            .strip()
            .split("x")
        )
        screenwidth, screenheight = int(screenwidth), int(screenheight)
        return screenwidth, screenheight
    except Exception:
        vax = subprocess.run(f'{adb_path} -s {deviceserial} shell dumpsys display', capture_output=True, shell=True)
        vax = vax.stdout.splitlines()
        vaxre = re.compile(rb'mBaseDisplayInfo.*width=\b(\d+)\b.*height=\b(\d+)\b', flags=re.I)
        scw, sch = 1280, 720
        for v in vax:
            fd = (vaxre.findall(v))
            if fd:
                try:
                    print(v)
                    scw, sch = int(fd[0][0]), int(fd[0][1])
                    break
                except Exception:
                    continue
        return scw, sch

def dumpsys_to_df(adb_path, deviceserial):
    activ = execute_adb_command(
        f"{adb_path} -s {deviceserial} shell", subcommands=["dumpsys activity top -c"]
    )
    activewinow = "\n".join(activ)
    activewinow2 = list(
        flatten_everything(
            [
                k.splitlines()
                for k in flatten_everything(
                    [
                        (
                            regex.findall(
                                r"View Hierarchy:.*[\r\n]\s{4}Looper",
                                activewinow,
                                flags=regex.DOTALL,
                            )
                        )
                    ]
                )
            ]
        )
    )

    if not any(activewinow2):
        activewinow2 = list(
            flatten_everything(
                [
                    k.splitlines()
                    for k in flatten_everything(
                        [
                            (
                                regex.findall(
                                    r"View Hierarchy:.*\{[^\}]+\}\s*[\r\n]+",
                                    activewinow,
                                    flags=regex.DOTALL,
                                )
                            )
                        ]
                    )
                ]
            )
        )
    activewinow3 = [x for x in activewinow2 if regex.search(r"\}\s*$", x) is not None]

    df = pd.DataFrame(activewinow3)
    df.columns = ["hiera"]
    spaces = df.hiera.str.extract(r"^(\s+)")[0]
    abslen = spaces.str.len()
    abslenwithoutmen = abslen - abslen.min()
    level = abslenwithoutmen // 2
    df = pd.concat([df, level], ignore_index=True, axis=1)
    df.columns = ["hiera", "level"]
    widgettype = df.hiera.str.extractall(r"^\s+([^{]+)").reset_index(drop=True)
    df = pd.concat([df, widgettype], axis=1).rename(columns={0: "widget_type"}).copy()
    return df, activewinow


def get_detailed_info_sep(df):
    detailedinformationtogether = df.hiera.str.extractall(
        r"^\s+[^{]+\{([^\}]+)"
    ).reset_index(drop=True)
    detailedinformationsep = (
        detailedinformationtogether[0].str.split(n=5, regex=False, expand=True).copy()
    )
    detailedinformationsep[1] = detailedinformationsep[1].apply(
        lambda x: (x + "..........")[:9]
    )
    detailedinformationsep[2] = detailedinformationsep[2].apply(
        lambda x: (x + "..........")[:8]
    )

    return detailedinformationsep


def get_widget_coords(detailedinformationsep):
    widgetcoords = (
        detailedinformationsep[3]
        .str.strip()
        .str.extractall(r"^(-?\d+),(-?\d+)-?(-?\d+),(-?\d+)")
        .reset_index(drop=True)
        .astype("string")
        .astype("Int64")
        .rename(columns={0: "x_start", 1: "y_start", 2: "x_end", 3: "y_end"})
        .copy()
    )
    return widgetcoords


def concat_df_widget_coords(df, widgetcoords):
    df = pd.concat([df, widgetcoords], axis=1).copy()
    df.columns = [
        "aa_complete_dump",
        "aa_depth",
        "aa_class_name",
        "aa_x_start",
        "aa_y_start",
        "aa_x_end",
        "aa_y_end",
    ]
    return df


def get_details(detailedinformationsep):
    details = (detailedinformationsep[1] + detailedinformationsep[2]).apply(
        lambda x: pd.Series(list(x))
    )
    details = details.replace(".", False)
    for col in details.columns[1:]:
        details.loc[(details[col].astype("string").str.len() == 1), col] = True
    flacol = [
        "visibility",
        "focusable",
        "enabled",
        "drawn",
        "scrollbars_horizontal",
        "scrollbars_vertical",
        "clickable",
        "long_clickable",
        "context_clickable",
        "pflag_is_root_namespace",
        "pflag_focused",
        "pflag_selected",
        "pflag_prepressed",
        "pflag_hovered",
        "pflag_activated",
        "pflag_invalidated",
        "pflag_dirty_mask",
    ]
    details.columns = flacol
    return details


def concat_df_details(df, details):
    df = pd.concat([df, details], axis=1).copy()
    return df


def calculate_center(df):
    df.loc[:, "aa_width"] = df["aa_x_end"] - df["aa_x_start"]
    df.loc[:, "aa_height"] = df["aa_y_end"] - df["aa_y_start"]
    df.loc[:, "aa_center_x"] = df["aa_x_start"] + (df["aa_width"] // 2)
    df.loc[:, "aa_center_y"] = df["aa_y_start"] + (df["aa_height"] // 2)
    df.loc[:, "aa_area"] = df["aa_width"] * df["aa_height"]
    return df


def hashcodes_ids_to_int(df, detailedinformationsep):
    int1 = detailedinformationsep[0].map(lambda x: get_label_1(x)).copy()
    hex1 = detailedinformationsep[0].copy()
    int1hex = (
        pd.concat([int1, hex1], axis=1, ignore_index=True)
        .rename(columns={0: "aa_hashcode_int", 1: "aa_hashcode_hex"})
        .copy()
    )
    df = pd.concat([df, int1hex], axis=1).copy()

    labeldf = (
        detailedinformationsep[4]
        .apply(get_label)
        .rename(columns={0: "aa_mID_hex", 1: "aa_mID_int"})
        .copy()
    )
    df = pd.concat([df, labeldf], axis=1).copy()
    return df


def fill_missing_ids_with_na(df, detailedinformationsep):
    idstuff = detailedinformationsep[5].fillna(pd.NA).copy()
    df = (
        pd.concat([df, idstuff], axis=1).rename(columns={5: "aa_id_information"}).copy()
    )
    return df


def remove_spaces(df):
    spaces = df["aa_complete_dump"].str.extract(r"^(\s+)")[0]
    abslen = spaces.str.len().min()
    df.loc[:, "aa_complete_dump"] = df["aa_complete_dump"].str.slice(abslen)
    pureid = (
        df["aa_id_information"]
        .fillna("")
        .str.split(":")
        .apply(lambda x: x[1] if len(x) == 2 else pd.NA)
        .copy()
    )
    df.loc[pureid.index, "pure_id"] = pureid.copy()
    for col_ in [x for x in df.columns if regex.search(r"^detail_[^0]\d*$", str(x))]:
        df.loc[~df[col_].isna(), col_] = True
        df.loc[df[col_].isna(), col_] = False
    df.columns = [
        "aa_" + regex.sub("^aa_", "", y) for y in df.columns.to_list()[:-1]
    ] + [df.columns.to_list()[-1]]
    return df


def reset_index_and_backup(df):
    df["old_index"] = df.index.__array__().copy()
    df = df.reset_index(drop=True)
    return df


def dump_uiautomator(adb_path, deviceserial):
    dumpstring2 = subprocess.run(
        fr"{adb_path} -s {deviceserial} shell uiautomator dump",
        shell=False,
        capture_output=True,
    )
    dumpstring3 = b""
    tempfile = os.path.join(os.getcwd(), "____window_dump.xml")
    if b"ERROR" not in dumpstring2.stderr:
        dumpstring3 = subprocess.run(
            f"{adb_path} -s {deviceserial} pull /sdcard/window_dump.xml {tempfile}",
            capture_output=True,
        )
        with open(tempfile, mode="r", encoding="utf-8") as f:
            dumpstring4 = f.read()

    dumpstring5 = regex.findall(r"<\?xml.*</hierarchy>", dumpstring4)[0]

    return dumpstring2, dumpstring3, dumpstring4, dumpstring5


def uiautomator_to_df(dumpstring):
    if isinstance(dumpstring, bytes):
        dumpstring = dumpstring.decode("utf-8", "replace")
    if not dumpstring.strip().startswith("<?xml"):
        dumpstring = "\n".join(dumpstring.splitlines()[1:])

    df2 = pd.Q_Xml2df(dumpstring).reset_index()
    df2 = df2.fillna(pd.NA).copy()
    df2["aa_all_keys_len"] = df2["aa_all_keys"].apply(lambda x: len(x))
    df = df2.copy()
    df3 = df.copy()
    df3 = df3.loc[~(df3["level_0"] == "rotation")]

    allpossiblecolumns = [
        "bounds",
        "checkable",
        "checked",
        "class",
        "clickable",
        "content-desc",
        "enabled",
        "focusable",
        "focused",
        "index",
        "long-clickable",
        "package",
        "password",
        "resource-id",
        "scrollable",
        "selected",
        "text",
    ]

    allgooddataf = []
    for col in df3.columns:
        if not "level_" in col:
            continue
        df4 = df3.loc[df3[col].isin(allpossiblecolumns)].copy()
        df4["keys_hierarchy"] = df4["aa_all_keys"].apply(lambda x: (x[:-1]))
        try:
            col_for_index = f"level_{df4.aa_all_keys_len.iloc[0] - 1}"
        except Exception:
            continue
        for name, group in df4.groupby("keys_hierarchy"):
            df5 = group[[col_for_index, "aa_value"]].copy()
            df5.columns = ["indi", "aa_value"]
            df5.index = df5["indi"].__array__().copy()
            df5 = df5.drop(columns="indi")
            df6 = df5.T.copy()
            df6 = df6.reset_index(drop=True)

            df6["keys_hierarchy"] = [name]
            allgooddataf.append(df6.copy())
    df = (
        pd.concat(allgooddataf, ignore_index=True)
        .reset_index(drop=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    coords_ = (
        df.bounds.str.extractall(r"(\d+)\W+(\d+)\W+(\d+)\W+(\d+)")
        .reset_index(drop=True)
        .rename(columns={0: "start_x", 1: "start_y", 2: "end_x", 3: "end_y"})
        .astype(np.uint16)
        .copy()
    )
    df = pd.concat([coords_, df], axis=1).copy()
    df.loc[:, "aa_width"] = df.end_x - df.start_x
    df.loc[:, "aa_height"] = df.end_y - df.start_y
    df["aa_center_x"] = df["start_x"] + (df["aa_width"] // 2)
    df["aa_center_y"] = df["start_y"] + (df["aa_height"] // 2)
    df["aa_area"] = df["aa_width"] * df["aa_height"]
    for col in df.columns:
        if col == "keys_hierarchy":
            continue
        try:
            df[col] = df[col].str.replace(r"^false$", "False", regex=True)

        except Exception as fe:
            pass
            # print(fe)

    for col in df.columns:
        if col == "keys_hierarchy":
            continue
        try:
            df[col] = df[col].str.replace(r"^true$", "True", regex=True)
        except Exception as fe:
            # print(fe)
            continue

    for col in df.columns:
        if col == "bounds":
            continue
        try:
            try:
                df.loc[:, col] = df.loc[:, col].ds_string_to_best_dtype()
            except Exception:
                pass

            df.loc[:, col] = df.loc[:, col].ds_reduce_memory_size(verbose=False)
        except Exception:
            pass
    df.bounds = list(zip(df.start_x, df.start_y, df.end_x, df.end_y))
    df = df.rename(
        columns={
            "start_x": "aa_start_x",
            "start_y": "aa_start_y",
            "end_x": "aa_end_x",
            "end_y": "aa_end_y",
        }
    )
    df = df.filter(list(sorted(df.columns))).copy()
    return df


def split_ui_columns_rename_cols(dfu):
    pureid = (
        dfu["resource-id"]
        .str.split(":")
        .apply(lambda x: x[1] if len(x) == 2 else pd.NA)
        .copy()
    )
    dfu.loc[pureid.index, "pure_id"] = pureid.copy()
    dfu.columns = [
        "bb_" + regex.sub("^aa_", "", y).replace("-", "_")
        for y in dfu.columns.to_list()[:-1]
    ] + [dfu.columns.to_list()[-1]]
    return dfu


def get_empty_dfu(df):
    allco = [
        "bb_area",
        "bb_center_x",
        "bb_center_y",
        "bb_x_end",
        "bb_y_end",
        "bb_height",
        "bb_x_start",
        "bb_y_start",
        "bb_width",
        "bb_bounds",
        "bb_checkable",
        "bb_checked",
        "bb_class",
        "bb_clickable",
        "bb_content_desc",
        "bb_enabled",
        "bb_focusable",
        "bb_focused",
        "bb_index",
        "bb_keys_hierarchy",
        "bb_long_clickable",
        "bb_package",
        "bb_password",
        "bb_resource_id",
        "bb_scrollable",
        "bb_selected",
        "bb_text",
        "pure_id",
    ]
    dfu = pd.DataFrame([[None] * len(allco)] * df.shape[0]).fillna(pd.NA)  # ?????
    dfu.columns = allco
    return dfu


def rename_dfu_cols(dfu):
    dfu = dfu.rename(
        columns={
            "bb_start_x": "bb_x_start",
            "bb_start_y": "bb_y_start",
            "bb_end_x": "bb_x_end",
            "bb_end_y": "bb_y_end",
        }
    )

    dfu["bb_screenshot"] = pd.NA
    return dfu


def screenshotcrop(img, x00, y00, x01, y01):
    try:
        cropa = cropimage(img=img, coords=(x00, y00, x01, y01))
        if qq_s_isnan(cropa, include_empty_iters=True):
            cropa = pd.NA
    except Exception:
        cropa = pd.NA
    return cropa


def take_screenshot(adb_path, deviceserial, channels_in_output=3):
    screens = get_screenshot_adb(adb_executable=adb_path, deviceserial=deviceserial,)
    screens = cv2.imread_plus(screens, channels_in_output=channels_in_output)
    return screens


def get_all_children(df, screenshot=pd.NA):
    if not qq_s_isnan(screenshot):
        screens = screenshot.copy()
    else:
        screens = pd.NA
    cropcoords = []
    group2 = df.copy()
    alldepths = group2.aa_depth.unique()
    alldepths = list(reversed(sorted(alldepths)))
    for ini, depth in enumerate(alldepths):
        subgroup = group2.loc[group2.aa_depth == depth]
        for key, item in subgroup.iterrows():
            oldrunvalue = depth
            goodstuff = []
            for ra in reversed(range(0, key)):
                if df.at[ra, "aa_depth"] < oldrunvalue:
                    goodstuff.append(df.loc[ra].to_frame().T)

                    oldrunvalue = df.at[ra, "aa_depth"]
            try:
                subdf = pd.concat(goodstuff).copy()
            except Exception as fe:
                continue

            singleitem = item.to_frame().T.copy()

            x00 = subdf.aa_x_start.sum() + item.aa_x_start
            y00 = subdf.aa_y_start.sum() + item.aa_y_start
            x01 = subdf.aa_x_start.sum() + item.aa_x_end
            y01 = subdf.aa_y_start.sum() + item.aa_y_end

            singleitem.loc[:, "aa_x_start_relative"] = singleitem.loc[:, "aa_x_start"]
            singleitem.loc[:, "aa_y_start_relative"] = singleitem.loc[:, "aa_y_start"]
            singleitem.loc[:, "aa_x_end_relative"] = singleitem.loc[:, "aa_x_end"]
            singleitem.loc[:, "aa_y_end_relative"] = singleitem.loc[:, "aa_y_end"]

            singleitem.loc[:, "aa_x_start"] = x00
            singleitem.loc[:, "aa_y_start"] = y00
            singleitem.loc[:, "aa_x_end"] = x01
            singleitem.loc[:, "aa_y_end"] = y01
            singleitem.loc[:, "aa_width"] = (
                singleitem["aa_x_end"] - singleitem["aa_x_start"]
            )
            singleitem.loc[:, "aa_height"] = (
                singleitem["aa_y_end"] - singleitem["aa_y_start"]
            )
            singleitem.loc[:, "aa_center_x"] = singleitem["aa_x_start"] + (
                singleitem["aa_width"] // 2
            )
            singleitem.loc[:, "aa_center_y"] = singleitem["aa_y_start"] + (
                singleitem["aa_height"] // 2
            )
            cropa = screenshotcrop(screens, x00, y00, x01, y01)
            singleitem.loc[:, "aa_is_child"] = True
            if qq_s_isnan(cropa, include_empty_iters=True):
                singleitem.loc[:, "aa_screenshot"] = pd.NA
                singleitem.loc[:, "aa_has_screenshot"] = False

            else:
                singleitem.loc[:, "aa_screenshot"] = [cropa]
                singleitem.loc[:, "aa_has_screenshot"] = True

            for ini_pa, pa_id in enumerate(subdf.old_index):
                singleitem.loc[:, f"parent_{str(ini_pa).zfill(3)}"] = pa_id
            cropcoords.append(singleitem.copy())
    df = pd.concat(cropcoords).reset_index(drop=True)
    return df


def shapely_poly_from4_coords(rect):
    try:
        X1, Y1, X2, Y2 = rect

        polygon = [(X1, Y1), (X2, Y1), (X2, Y2), (X1, Y2)]

        p = Polygon(polygon)
        if p.area > 0:
            return pd.Series([True, p])
        return pd.Series([False, p])
    except Exception:
        X1, Y1, X2, Y2 = 50000, 50000, 50000, 50000
        polygon = [(X1, Y1), (X2, Y1), (X2, Y2), (X1, Y2)]
        p = Polygon(polygon)
        return pd.Series([False, p])


def add_shapely_to_dataframes(df=None, dfu=None):
    d1, d2 = None, None
    if not qq_s_isnan(df):
        square1 = df.apply(
            lambda x: shapely_poly_from4_coords(
                (x.aa_x_start, x.aa_y_start, x.aa_x_end, x.aa_y_end)
            ),
            axis=1,
        )
        square1.columns = ["aa_valid_square", "aa_shapely"]
        d1 = pd.concat([df, square1], axis=1)

    if not qq_s_isnan(dfu):

        square2 = dfu["bb_bounds"].apply(lambda x: shapely_poly_from4_coords(x))
        square2.columns = ["bb_valid_square", "bb_shapely"]
        d2 = pd.concat([dfu, square2], axis=1)
    return d1, d2


def update_area(df=None, dfu=None):
    if not qq_s_isnan(dfu):
        dfu["bb_area"] = dfu["bb_shapely"].apply(lambda x: x.area)
    if not qq_s_isnan(df):

        df["aa_area"] = df["aa_shapely"].apply(lambda x: x.area)
    return df, dfu


def add_bounds_to_df(df):
    df["aa_bounds"] = df.apply(
        lambda x: tuple(
            (x["aa_x_start"], x["aa_y_start"], x["aa_x_end"], x["aa_y_end"])
        ),
        axis=1,
    )
    return df


def get_cropped_coords(max_x, max_y, df, pref="aa"):
    # pref = 'aa'
    df[f"{pref}_cropped_x_start"] = df[f"{pref}_x_start"]
    df[f"{pref}_cropped_y_start"] = df[f"{pref}_y_start"]
    df[f"{pref}_cropped_x_end"] = df[f"{pref}_x_end"]
    df[f"{pref}_cropped_y_end"] = df[f"{pref}_y_end"]

    df.loc[(df[f"{pref}_cropped_x_start"] <= 0), f"{pref}_cropped_x_start"] = 0
    df.loc[(df[f"{pref}_cropped_y_start"] <= 0), f"{pref}_cropped_y_start"] = 0
    df.loc[(df[f"{pref}_cropped_x_end"] <= 0), f"{pref}_cropped_x_end"] = 0
    df.loc[(df[f"{pref}_cropped_y_end"] <= 0), f"{pref}_cropped_y_end"] = 0

    df.loc[(df[f"{pref}_cropped_x_start"] >= max_x), f"{pref}_cropped_x_start"] = max_x
    df.loc[(df[f"{pref}_cropped_y_start"] >= max_y), f"{pref}_cropped_y_start"] = max_y
    df.loc[(df[f"{pref}_cropped_x_end"] >= max_x), f"{pref}_cropped_x_end"] = max_x
    df.loc[(df[f"{pref}_cropped_y_end"] >= max_y), f"{pref}_cropped_y_end"] = max_y
    df.loc[:, f"{pref}_width_cropped"] = (
        df[f"{pref}_cropped_x_end"] - df[f"{pref}_cropped_x_start"]
    )
    df.loc[:, f"{pref}_height_cropped"] = (
        df[f"{pref}_cropped_y_end"] - df[f"{pref}_cropped_y_start"]
    )
    df.loc[:, f"{pref}_center_x_cropped"] = df[f"{pref}_cropped_x_start"] + (
        df[f"{pref}_width_cropped"] // 2
    )
    df.loc[:, f"{pref}_center_y_cropped"] = df[f"{pref}_cropped_y_start"] + (
        df[f"{pref}_height_cropped"] // 2
    )
    return df


def get_activity_df(max_x, max_y, adb_path, deviceserial, screens=pd.NA):
    df, activewinow = dumpsys_to_df(adb_path, deviceserial)
    # print(df)
    detailedinformationsep = get_detailed_info_sep(df)
    # print(detailedinformationsep)
    widgetcoords = get_widget_coords(detailedinformationsep)
    # print(widgetcoords)
    df = concat_df_widget_coords(df, widgetcoords)
    # print(df)
    details = get_details(detailedinformationsep)
    # print(details)
    df = concat_df_details(df, details)
    # print(df)
    df = calculate_center(df)
    # print(df)
    df = hashcodes_ids_to_int(df, detailedinformationsep)
    # print(df)
    df = fill_missing_ids_with_na(df, detailedinformationsep)
    # print(df)
    df = remove_spaces(df)
    # print(df)
    df2 = reset_index_and_backup(df)
    # print(df2)
    df = get_all_children(df=df2, screenshot=screens)
    df, _ = add_shapely_to_dataframes(df=df)

    df, _ = update_area(df=df)
    df = add_bounds_to_df(df)

    df = df.rename(columns={"pure_id": "aa_pure_id", "old_index": "aa_old_index"})
    df = get_cropped_coords(max_x, max_y, df, pref="aa")
    return df


def get_view_df(
    adb_path, deviceserial, max_x, max_y, merge_it=False, df=None, screens=pd.NA
):
    dfu = pd.DataFrame()
    try:
        dumpstring2, dumpstring3, dumpstring4, dumpstring5 = dump_uiautomator(
            adb_path, deviceserial
        )
        # print(dumpstring2, dumpstring3, dumpstring4, dumpstring5)
        dfu = uiautomator_to_df(dumpstring=dumpstring5)
        # print(dfu)
        dfu = split_ui_columns_rename_cols(dfu)
    except Exception:
        if dfu.empty:
            if merge_it:
                dfu = get_empty_dfu(df)
    dfu = rename_dfu_cols(dfu)
    dfu = reset_index_and_backup(df=dfu)
    dfu["bb_screenshot"] = pd.NA
    if not qq_s_isnan(screens):
        try:
            dfu["bb_screenshot"] = dfu["bb_bounds"].apply(
                lambda x: screenshotcrop(screens, *x)
            )
        except Exception:
            pass
    _, dfu = add_shapely_to_dataframes(dfu=dfu)
    _, dfu = update_area(dfu=dfu)
    dfu = dfu.rename(columns={"pure_id": "bb_pure_id", "old_index": "bb_old_index"})
    dfu = get_cropped_coords(max_x, max_y, dfu, pref="bb")
    return dfu


def _get_show_parent_function(df, item):
    return FlexiblePartial(_execute_function_to_df_show_parents, True, df, item)


def _execute_function_to_df_show_parents(df, item):
    itemf = item.to_frame().T.dropna(axis=1)

    sortedcols = [
        x
        for x in list(reversed(sorted(itemf.columns.to_list())))
        if str(x).startswith("parent_")
    ]
    allparentscols = [
        df.loc[df.aa_old_index == int(itemf[x])]
        for x in sortedcols
        if str(x).startswith("parent")
    ]
    return pd.concat(allparentscols).sort_values(by="aa_depth", ascending=False)


def add_function_to_df_show_parents(df):
    df.loc[:, "ff_show_parents"] = df.apply(
        lambda item: _get_show_parent_function(df, item), axis=1
    )
    return df


def execute_function_df_dfu_show_screenshot(screenshot):
    if not qq_s_isnan(screenshot, include_empty_iters=True):
        cv2.imshow_thread(screenshot)


def get_function_df_dfu_show_screenshot(screenshot):
    return FlexiblePartial(execute_function_df_dfu_show_screenshot, True, screenshot)


def add_function_df_dfu_show_screenshot(
    df, column="aa_screenshot", new_column_name="ff_show_screenshot"
):
    df.loc[:, new_column_name] = df[column].apply(
        lambda item: get_function_df_dfu_show_screenshot(item)
    )
    return df


def execute_function_df_dfu_save_screenshot(screenshot):
    if not qq_s_isnan(screenshot, include_empty_iters=True):
        cv2.imshow_thread(screenshot)


def get_function_df_dfu_save_screenshot(screenshot, folder, filename):
    if qq_s_isnan(screenshot):
        return pd.NA
    return FlexiblePartial(
        cv2.imwrite_plus, True, os.path.join(folder, filename + ".png"), screenshot
    )


def add_function_df_dfu_save_screenshot(
    df, column="aa_screenshot", folder=None, new_column_name="ff_save_screenshot"
):
    if folder is None:
        folder = os.path.join(os.getcwd(), timest())
    df.loc[:, new_column_name] = df.apply(
        lambda item: get_function_df_dfu_save_screenshot(
            screenshot=item[column], folder=folder, filename=str(item.name).zfill(5)
        ),
        axis=1,
    )
    return df


def execute_df_tap_middle_offset(
    adb_path, deviceserial, x, columnx, columny, offset_x, offset_y
):
    execute_adb_command(
        f"{adb_path} -s {deviceserial} shell",
        [f"input tap {x[columnx]+offset_x} {x[columny]+offset_y}"],
    )


def execute_df_tap_middle(
    adb_path, deviceserial, x, columnx, columny, offset_x=0, offset_y=0
):
    try:
        return FlexiblePartial(
            execute_adb_command,
            True,
            f"{adb_path} -s {deviceserial} shell",
            [f"input tap {x[columnx]+offset_x} {x[columny]+offset_y}"],
        )
    except Exception as fas:
        print(fas)
    return pd.NA


def execute_df_tap_middle_variation(
    adb_path,
    deviceserial,
    x,
    columnx,
    columny,
    height,
    width,
    percent_x=10,
    percent_y=10,
    percentage_substract_vs_add=50,
):
    try:
        addtoy = randomize_number(
            value=x[columny],
            percent=percent_y,
            percentage_substract_vs_add=percentage_substract_vs_add,
            minimum_to_allow=x[columny],
            maximum_to_allow=x[columny] + x[height] // 4,
        )
        addtox = randomize_number(
            value=x[columnx],
            percent=percent_x,
            percentage_substract_vs_add=percentage_substract_vs_add,
            minimum_to_allow=x[columnx],
            maximum_to_allow=x[columnx] + x[width] // 4,
        )
        return FlexiblePartial(
            execute_adb_command,
            True,
            f"{adb_path} -s {deviceserial} shell",
            [f"input tap {addtox} {addtoy}"],
        )
    except Exception as fe:
        print(fe)
    return pd.NA


def get_function_for_offset_click(
    adb_path, deviceserial, x, columnx, columny,
):
    try:
        return FlexiblePartial(
            execute_df_tap_middle_offset,
            True,
            adb_path,
            deviceserial,
            x,
            columnx,
            columny,
        )
    except Exception as dsx:
        print(dsx)
    return pd.NA


def add_to_df_tap_middle_offset(
    adb_path, deviceserial, df, columnx, columny, new_column_name="ff_tap_center_offset"
):
    df.loc[:, new_column_name] = df.apply(
        lambda x: get_function_for_offset_click(
            adb_path, deviceserial, x, columnx, columny,
        ),
        axis=1,
    )
    return df


def add_to_df_tap_middle_exact(
    adb_path, deviceserial, df, columnx, columny, new_column_name="ff_tap_exact_center"
):
    df.loc[:, new_column_name] = df.apply(
        lambda x: execute_df_tap_middle(adb_path, deviceserial, x, columnx, columny),
        axis=1,
    )
    return df


def add_to_df_tap_middle_variation(
    adb_path,
    deviceserial,
    df,
    columnx,
    columny,
    height,
    width,
    new_column_name="ff_tap_center_variation",
    percent_x=10,
    percent_y=10,
    percentage_substract_vs_add=50,
):
    df.loc[:, new_column_name] = df.apply(
        lambda x: execute_df_tap_middle_variation(
            adb_path,
            deviceserial,
            x,
            columnx,
            columny,
            height,
            width,
            percent_x=percent_x,
            percent_y=percent_y,
            percentage_substract_vs_add=percentage_substract_vs_add,
        ),
        axis=1,
    )
    return df


def view_activity_dfs_merged(df, dfu):
    selection_for_merging_all = []
    nomatchingfound = []
    for key, item in dfu.iterrows():
        selection_for_merging = []
        step1 = df["aa_area"].apply(lambda x: x == item["bb_area"])
        if step1.empty:
            nomatchingfound.append(key)
            continue
        step2 = df["aa_shapely"].apply(lambda x: item["bb_shapely"].intersects(x))
        if step2.empty:
            nomatchingfound.append(key)
            continue
        step3 = df["aa_pure_id"].apply(lambda x: str(x) == str(item["bb_pure_id"]))
        step4 = df["aa_class_name"].apply(lambda x: str(x) == str(item["bb_class"]))
        step5 = df["aa_clickable"].apply(lambda x: str(x) == str(item["bb_clickable"]))
        step6 = df["aa_bounds"].apply(lambda x: str(x) == str(item["bb_bounds"]))

        selection_for_merging.append(key)
        selection_for_merging.append(step1.loc[step1].index.copy())
        selection_for_merging.append(step2.loc[step2].index.copy())
        selection_for_merging.append(step3.loc[step3].index.copy())
        selection_for_merging.append(step4.loc[step4].index.copy())
        selection_for_merging.append(step5.loc[step5].index.copy())
        selection_for_merging.append(step6.loc[step6].index.copy())
        selection_for_merging_all.append(selection_for_merging.copy())

    togea = []

    for _ in selection_for_merging_all:
        # print(_[0])
        ba = list(flatten_everything(_[1:]))
        # print(ba)
        soli = sorted(ba, key=lambda x: ba.count(x))
        bestres = soli[-1]
        r12 = dfu.loc[_[0]].to_frame().T.copy()
        r11 = df.loc[bestres].to_frame().T.copy()
        r12.index = [0]
        r11.index = [0]
        newtog = pd.concat([r12.copy(), r11.copy()], axis=1).copy()
        newtog["aa_matching"] = ba.count(bestres)

        togea.append(newtog.copy())
    dfmerged = pd.concat(togea).reset_index(drop=True).copy()
    dfmerged2 = pd.concat(
        [dfmerged, df.loc[~df.aa_old_index.isin(dfmerged.aa_old_index)]],
        ignore_index=True,
    ).copy()
    dfmerged3 = pd.concat(
        [dfmerged2, dfu.loc[~(dfu.bb_old_index.isin(dfmerged2.bb_old_index))]],
        ignore_index=True,
    ).copy()
    return dfmerged3


def execute_df_tap_middle_offset_long(
    adb_path, deviceserial, columnx, columny, delay, offset_x, offset_y,
):
    subcommands = [
        f"input touchscreen swipe {columnx + offset_x} {columny + offset_y} {columnx + offset_x} {columny + offset_y} {delay}"
    ]
    return execute_adb_command(f"{adb_path} -s {deviceserial} shell", subcommands,)


def execute_df_tap_middle_long(
    adb_path, deviceserial, x, columnx, columny, delay, offset_x=0, offset_y=0
):
    try:
        return FlexiblePartial(
            execute_adb_command,
            True,
            f"{adb_path} -s {deviceserial} shell",
            [
                f"input touchscreen swipe {x[columnx]+offset_x} {x[columny]+offset_y} {x[columnx]+offset_x} {x[columny]+offset_y} {delay}"
            ],
        )
    except Exception as fas:
        print(fas)
    return pd.NA


def execute_df_tap_middle_variation_long(
    adb_path,
    deviceserial,
    x,
    columnx,
    columny,
    height,
    width,
    delay,
    percent_x=10,
    percent_y=10,
    percentage_substract_vs_add=50,
):
    try:
        addtoy = randomize_number(
            value=x[columny],
            percent=percent_x,
            percentage_substract_vs_add=percentage_substract_vs_add,
            minimum_to_allow=x[columny],
            maximum_to_allow=x[columny] + x[height] // 4,
        )
        addtox = randomize_number(
            value=x[columnx],
            percent=percent_y,
            percentage_substract_vs_add=percentage_substract_vs_add,
            minimum_to_allow=x[columnx],
            maximum_to_allow=x[columnx] + x[width] // 4,
        )
        return FlexiblePartial(
            execute_adb_command,
            True,
            f"{adb_path} -s {deviceserial} shell",
            [f"input touchscreen swipe {addtox} {addtoy} {addtox} {addtoy} {delay}"],
        )
    except Exception as fe:
        print(fe)
    return pd.NA


def get_function_for_offset_click_long(adb_path, deviceserial, columnx, columny, delay):
    try:
        return FlexiblePartial(
            execute_df_tap_middle_offset_long,
            True,
            adb_path,
            deviceserial,
            columnx,
            columny,
            delay,
        )
    except Exception as dsx:
        print(dsx)
    return pd.NA


def add_to_df_tap_middle_offset_long(
    adb_path,
    deviceserial,
    df,
    columnx,
    columny,
    delay,
    new_column_name="ff_tap_center_offset",
):
    df.loc[:, new_column_name] = df.apply(
        lambda x: get_function_for_offset_click_long(
            adb_path, deviceserial, x[columnx], x[columny], int(random.uniform(*delay)),
        ),
        axis=1,
    )
    return df


def add_to_df_tap_middle_exact_long(
    adb_path,
    deviceserial,
    df,
    columnx,
    columny,
    delay,
    new_column_name="ff_tap_exact_center",
):
    df.loc[:, new_column_name] = df.apply(
        lambda x: execute_df_tap_middle_long(
            adb_path, deviceserial, x, columnx, columny, int(random.uniform(*delay))
        ),
        axis=1,
    )
    return df


def add_to_df_tap_middle_variation_long(
    adb_path,
    deviceserial,
    df,
    columnx,
    columny,
    height,
    width,
    delay,
    new_column_name="ff_tap_center_variation",
    percent_x=10,
    percent_y=10,
    percentage_substract_vs_add=50,
):
    df.loc[:, new_column_name] = df.apply(
        lambda x: execute_df_tap_middle_variation_long(
            adb_path,
            deviceserial,
            x,
            columnx,
            columny,
            height,
            width,
            int(random.uniform(*delay)),
            percent_x=percent_x,
            percent_y=percent_y,
            percentage_substract_vs_add=percentage_substract_vs_add,
        ),
        axis=1,
    )
    return df


def execute_upswipe(
    adb_path,
    deviceserial,
    startx,
    endx,
    starty,
    endy,
    width,
    height,
    delay=(1000, 2000),
    variation_startx=10,
    variation_endx=10,
    variation_starty=10,
    variation_endy=10,
):

    delay = random.randrange(delay[0], delay[1])
    wi = width
    hei = height
    addtox = randomize_number(
        value=startx,
        percent=variation_startx,
        percentage_substract_vs_add=100,
        minimum_to_allow=startx + wi // 200,
        maximum_to_allow=startx + wi // 150,
    )
    addtoy = randomize_number(
        value=starty,
        percent=variation_starty,
        percentage_substract_vs_add=100,
        minimum_to_allow=starty + hei // 10,
        maximum_to_allow=starty + hei // 5,
    )
    addtoxend = randomize_number(
        value=endx,
        percent=variation_endx,
        percentage_substract_vs_add=100,
        minimum_to_allow=endx - wi // 150,
        maximum_to_allow=endx - wi // 200,
    )

    addtoyend = randomize_number(
        value=endy,
        percent=variation_endy,
        percentage_substract_vs_add=100,
        minimum_to_allow=endy - hei // 10,
        maximum_to_allow=endy - hei // 5,
    )
    execute_adb_command(
        f"{adb_path} -s {deviceserial} shell",
        subcommands=[f"input swipe {addtox} {addtoy} {addtoxend} {addtoyend} {delay}"],
    )


def get_function_execute_upswipe(
    center_x_cropped_col,
    cropped_y_end_col,
    cropped_y_start_col,
    width_cropped_col,
    height_cropped_col,
    adb_path,
    deviceserial,
    variation_startx=10,
    variation_endx=10,
    variation_starty=10,
    variation_endy=10,
    delay=(1500, 2000),
):
    return FlexiblePartial(
        execute_upswipe,
        True,
        adb_path=adb_path,
        deviceserial=deviceserial,
        startx=center_x_cropped_col,
        endx=center_x_cropped_col,
        starty=cropped_y_start_col,
        endy=cropped_y_end_col,
        width=width_cropped_col,
        height=height_cropped_col,
        delay=delay,
        variation_startx=variation_startx,
        variation_endx=variation_endx,
        variation_starty=variation_starty,
        variation_endy=variation_endy,
    )


def add_to_df_upswipe(
    df,
    center_x_cropped_col,
    cropped_y_end_col,
    cropped_y_start_col,
    width_cropped_col,
    height_cropped_col,
    adb_path,
    deviceserial,
    variation_startx=10,
    variation_endx=10,
    variation_starty=10,
    variation_endy=10,
    new_column_name="ff_upswipe",
    delay=(1000, 2000),
):
    df.loc[:, new_column_name] = df.apply(
        lambda x: get_function_execute_upswipe(
            x[center_x_cropped_col],
            x[cropped_y_end_col],
            x[cropped_y_start_col],
            x[width_cropped_col],
            x[height_cropped_col],
            adb_path,
            deviceserial,
            variation_startx=variation_startx,
            variation_endx=variation_endx,
            variation_starty=variation_starty,
            variation_endy=variation_endy,
            delay=delay,
        ),
        axis=1,
    )
    return df


def execute_downswipe(
    adb_path,
    deviceserial,
    startx,
    endx,
    starty,
    endy,
    width,
    height,
    delay=(1000, 2000),
    variation_startx=10,
    variation_endx=10,
    variation_starty=10,
    variation_endy=10,
):

    delay = random.randrange(delay[0], delay[1])
    wi = width
    hei = height
    addtox = randomize_number(
        value=startx,
        percent=variation_startx,
        percentage_substract_vs_add=100,
        minimum_to_allow=startx + wi // 200,
        maximum_to_allow=startx + wi // 150,
    )
    addtoy = randomize_number(
        value=starty,
        percent=variation_starty,
        percentage_substract_vs_add=100,
        minimum_to_allow=starty + hei // 10,
        maximum_to_allow=starty + hei // 5,
    )
    addtoxend = randomize_number(
        value=endx,
        percent=variation_endx,
        percentage_substract_vs_add=100,
        minimum_to_allow=endx - wi // 150,
        maximum_to_allow=endx - wi // 200,
    )
    addtoyend = randomize_number(
        value=endy,
        percent=variation_endy,
        percentage_substract_vs_add=100,
        minimum_to_allow=endy - hei // 5,
        maximum_to_allow=endy - hei // 10,
    )

    execute_adb_command(
        f"{adb_path} -s {deviceserial} shell",
        subcommands=[f"input swipe {addtox} {addtoyend} {addtoxend} {addtoy} {delay}"],
    )


def get_function_execute_downswipe(
    center_x_cropped_col,
    cropped_y_end_col,
    cropped_y_start_col,
    width_cropped_col,
    height_cropped_col,
    adb_path,
    deviceserial,
    variation_startx=10,
    variation_endx=10,
    variation_starty=10,
    variation_endy=10,
    delay=(1000, 2000),
):
    return FlexiblePartial(
        execute_downswipe,
        True,
        adb_path=adb_path,
        deviceserial=deviceserial,
        startx=center_x_cropped_col,
        endx=center_x_cropped_col,
        starty=cropped_y_start_col,
        endy=cropped_y_end_col,
        width=width_cropped_col,
        height=height_cropped_col,
        delay=delay,
        variation_startx=variation_startx,
        variation_endx=variation_endx,
        variation_starty=variation_starty,
        variation_endy=variation_endy,
    )


def add_to_df_downswipe(
    df,
    center_x_cropped_col,
    cropped_y_end_col,
    cropped_y_start_col,
    width_cropped_col,
    height_cropped_col,
    adb_path,
    deviceserial,
    variation_startx=10,
    variation_endx=10,
    variation_starty=10,
    variation_endy=10,
    new_column_name="ff_downswipe",
    delay=(1000, 2000),
):
    df.loc[:, new_column_name] = df.apply(
        lambda x: get_function_execute_downswipe(
            x[center_x_cropped_col],
            x[cropped_y_end_col],
            x[cropped_y_start_col],
            x[width_cropped_col],
            x[height_cropped_col],
            adb_path,
            deviceserial,
            variation_startx=variation_startx,
            variation_endx=variation_endx,
            variation_starty=variation_starty,
            variation_endy=variation_endy,
            delay=delay,
        ),
        axis=1,
    )
    return df


def _sendevent_offset_long(
    sendtouch, x, y, struct_folder, duration, offset_x, offset_y
):
    sendtouch.longtouch(
        x + offset_x, y + offset_y, duration, struct_folder=struct_folder,
    )


def _sendevent_offset_long_bs(
    sendtouch, x, y, struct_folder, duration, offset_x, offset_y
):

    sendtouch.longtouch(
        x + offset_x, y + offset_y, duration, struct_folder=struct_folder
    )


def sendevent_offset_long(sendtouch, x, y, duration, structfolder):

    if structfolder == "struct":
        return FlexiblePartial(
            _sendevent_offset_long, True, sendtouch, x, y, structfolder, duration
        )
    return FlexiblePartial(
        _sendevent_offset_long_bs, True, sendtouch, x, y, structfolder, duration
    )


def _sendevent_offset(sendtouch, x, y, struct_folder, offset_x, offset_y):
    sendtouch.touch(
        x + offset_x, y + offset_y, struct_folder=struct_folder,
    )


def _sendevent_offset_bs(sendtouch, x, y, struct_folder, offset_x, offset_y):

    sendtouch.touch(x + offset_x, y + offset_y, struct_folder=struct_folder)


def sendevent_offset(sendtouch, x, y, structfolder):

    if structfolder == "struct":
        return FlexiblePartial(_sendevent_offset, True, sendtouch, x, y, structfolder)
    return FlexiblePartial(_sendevent_offset_bs, True, sendtouch, x, y, structfolder,)


def add_all_functions_to_df(
    sendtouch,
    adb_path,
    deviceserial,
    df,
    timesta,
    screenshotfolder=None,
    max_variation_percent_x=10,
    max_variation_percent_y=10,
    percentage_substract_vs_add=80,
    loung_touch_delay=(1000, 1500),
    swipe_variation_startx=10,
    swipe_variation_endx=10,
    swipe_variation_starty=10,
    swipe_variation_endy=10,
):
    if screenshotfolder is None:
        screenshotfolder = os.path.join(os.getcwd(), "__ANDROIDSCREENSHOTS")
    screenshotfolder = os.path.join(screenshotfolder, "0", timesta)
    if not os.path.exists(screenshotfolder):
        os.makedirs(screenshotfolder)
    df = add_function_to_df_show_parents(df)
    df = add_function_df_dfu_show_screenshot(
        df, column="aa_screenshot", new_column_name="ff_aa_show_screenshot"
    )
    df = add_function_df_dfu_save_screenshot(
        df,
        column="aa_screenshot",
        folder=screenshotfolder,
        new_column_name="ff_aa_save_screenshot",
    )
    df = add_to_df_tap_middle_offset(
        adb_path,
        deviceserial,
        df,
        columnx="aa_center_x_cropped",
        columny="aa_center_y_cropped",
        new_column_name="ff_aa_tap_center_offset",
    )

    df = add_to_df_tap_middle_exact(
        adb_path,
        deviceserial,
        df,
        columnx="aa_center_x_cropped",
        columny="aa_center_y_cropped",
        new_column_name="ff_aa_tap_exact_center",
    )

    df = add_to_df_tap_middle_variation(
        adb_path,
        deviceserial,
        df,
        columnx="aa_center_x_cropped",
        columny="aa_center_y_cropped",
        height="aa_height_cropped",
        width="aa_width_cropped",
        new_column_name="ff_aa_tap_center_variation",
        percent_x=max_variation_percent_x,
        percent_y=max_variation_percent_y,
        percentage_substract_vs_add=percentage_substract_vs_add,
    )
    df = add_to_df_tap_middle_offset_long(
        adb_path,
        deviceserial,
        df,
        columnx="aa_center_x_cropped",
        columny="aa_center_y_cropped",
        new_column_name="ff_aa_tap_center_offset_long",
        delay=loung_touch_delay,
    )

    df = add_to_df_tap_middle_exact_long(
        adb_path,
        deviceserial,
        df,
        columnx="aa_center_x_cropped",
        columny="aa_center_y_cropped",
        new_column_name="ff_aa_tap_exact_center_long",
        delay=loung_touch_delay,
    )

    df = add_to_df_tap_middle_variation_long(
        adb_path,
        deviceserial,
        df,
        columnx="aa_center_x_cropped",
        columny="aa_center_y_cropped",
        height="aa_height_cropped",
        width="aa_width_cropped",
        new_column_name="ff_aa_tap_center_variation_long",
        percent_x=max_variation_percent_x,
        percent_y=max_variation_percent_y,
        percentage_substract_vs_add=percentage_substract_vs_add,
        delay=loung_touch_delay,
    )
    df = add_to_df_upswipe(
        df=df,
        center_x_cropped_col="aa_center_x_cropped",
        cropped_y_end_col="aa_cropped_y_end",
        cropped_y_start_col="aa_cropped_y_start",
        width_cropped_col="aa_width_cropped",
        height_cropped_col="aa_height_cropped",
        adb_path=adb_path,
        deviceserial=deviceserial,
        variation_startx=swipe_variation_startx,
        variation_endx=swipe_variation_endx,
        variation_starty=swipe_variation_starty,
        variation_endy=swipe_variation_endy,
        new_column_name="ff_aa_upswipe",
        delay=loung_touch_delay,
    )

    df = add_to_df_downswipe(
        df=df,
        center_x_cropped_col="aa_center_x_cropped",
        cropped_y_end_col="aa_cropped_y_end",
        cropped_y_start_col="aa_cropped_y_start",
        width_cropped_col="aa_width_cropped",
        height_cropped_col="aa_height_cropped",
        adb_path=adb_path,
        deviceserial=deviceserial,
        variation_startx=swipe_variation_startx,
        variation_endx=swipe_variation_endx,
        variation_starty=swipe_variation_starty,
        variation_endy=swipe_variation_endy,
        new_column_name="ff_aa_downswipe",
        delay=loung_touch_delay,
    )

    # df= add_sendevent_touch(df, prefix="ee_aa", delay=loung_touch_delay,xcol='aa_center_x_cropped',ycol='aa_cropped_y_end')

    df.loc[:, "ee_aa_longtouch_offset"] = df.apply(
        lambda x: sendevent_offset_long(
            sendtouch,
            x.aa_center_x_cropped,
            x.aa_center_y_cropped,
            random.uniform(*loung_touch_delay) / 1000,
            structfolder="struct",
        ),
        axis=1,
    )

    df.loc[:, "ee_aa_longtouch_offset_bs"] = df.apply(
        lambda x: sendevent_offset_long(
            sendtouch,
            x.aa_center_x_cropped,
            x.aa_center_y_cropped,
            random.uniform(*loung_touch_delay) / 1000,
            structfolder="struct real",
        ),
        axis=1,
    )

    df.loc[:, "ee_aa_touch_offset"] = df.apply(
        lambda x: sendevent_offset(
            sendtouch,
            x.aa_center_x_cropped,
            x.aa_center_y_cropped,
            structfolder="struct",
        ),
        axis=1,
    )

    df.loc[:, "ee_aa_touch_offset_bs"] = df.apply(
        lambda x: sendevent_offset(
            sendtouch,
            x.aa_center_x_cropped,
            x.aa_center_y_cropped,
            structfolder="struct real",
        ),
        axis=1,
    )
    prefix = "ee_aa"
    df.loc[:, f"{prefix}_longtouch_bs"] = df.apply(
        lambda x: FlexiblePartial(
            sendtouch.longtouch,
            True,
            x.aa_center_x_cropped,
            x.aa_center_y_cropped,
            duration=random.uniform(*loung_touch_delay) / 1000,
            struct_folder="struct real",
        ),
        axis=1,
    )

    df.loc[:, f"{prefix}_touch_bs"] = df.apply(
        lambda x: FlexiblePartial(
            sendtouch.touch,
            True,
            x.aa_center_x_cropped,
            x.aa_center_y_cropped,
            struct_folder="struct real",
        ),
        axis=1,
    )

    df.loc[:, f"{prefix}_touch"] = df.apply(
        lambda x: FlexiblePartial(
            sendtouch.touch,
            True,
            x.aa_center_x_cropped,
            x.aa_center_y_cropped,
            struct_folder="struct",
        ),
        axis=1,
    )
    df.loc[:, f"{prefix}_longtouch"] = df.apply(
        lambda x: FlexiblePartial(
            sendtouch.longtouch,
            True,
            x.aa_center_x_cropped,
            x.aa_center_y_cropped,
            duration=random.uniform(*loung_touch_delay) / 1000,
            struct_folder="struct",
        ),
        axis=1,
    )
    return df


def add_all_functions_to_dfu(
    sendtouch,
    adb_path,
    deviceserial,
    dfu,
    tstamp,
    screenshotfolder=None,
    max_variation_percent_x=10,
    max_variation_percent_y=10,
    percentage_substract_vs_add=80,
    loung_touch_delay=(1000, 1500),
    swipe_variation_startx=10,
    swipe_variation_endx=10,
    swipe_variation_starty=10,
    swipe_variation_endy=10,
):
    if screenshotfolder is None:
        screenshotfolder = os.path.join(os.getcwd(), "__ANDROIDSCREENSHOTS")
    screenshotfolder = os.path.join(screenshotfolder, "1", tstamp)
    if not os.path.exists(screenshotfolder):
        os.makedirs(screenshotfolder)

    dfu = add_function_df_dfu_show_screenshot(
        dfu, column="bb_screenshot", new_column_name="ff_bb_show_screenshot"
    )

    dfu = add_function_df_dfu_save_screenshot(
        dfu,
        column="bb_screenshot",
        folder=screenshotfolder,
        new_column_name="ff_bb_save_screenshot",
    )

    dfu = add_to_df_tap_middle_offset(
        adb_path,
        deviceserial,
        dfu,
        columnx="bb_center_x",
        columny="bb_center_y",
        new_column_name="ff_bb_tap_center_offset",
    )
    dfu = add_to_df_tap_middle_exact(
        adb_path,
        deviceserial,
        dfu,
        columnx="bb_center_x",
        columny="bb_center_y",
        new_column_name="ff_bb_tap_exact_center",
    )
    dfu = add_to_df_tap_middle_variation(
        adb_path,
        deviceserial,
        dfu,
        columnx="bb_center_x",
        columny="bb_center_y",
        height="bb_height",
        width="bb_width",
        new_column_name="ff_bb_tap_center_variation",
        percent_x=max_variation_percent_x,
        percent_y=max_variation_percent_y,
        percentage_substract_vs_add=percentage_substract_vs_add,
    )

    dfu = add_to_df_tap_middle_offset_long(
        adb_path,
        deviceserial,
        dfu,
        columnx="bb_center_x",
        columny="bb_center_y",
        new_column_name="ff_bb_tap_center_offset_long",
        delay=loung_touch_delay,
    )
    dfu = add_to_df_tap_middle_exact_long(
        adb_path,
        deviceserial,
        dfu,
        columnx="bb_center_x",
        columny="bb_center_y",
        new_column_name="ff_bb_tap_exact_center_long",
        delay=loung_touch_delay,
    )
    dfu = add_to_df_tap_middle_variation_long(
        adb_path,
        deviceserial,
        dfu,
        columnx="bb_center_x",
        columny="bb_center_y",
        height="bb_height",
        width="bb_width",
        new_column_name="ff_bb_tap_center_variation_long",
        percent_x=max_variation_percent_x,
        percent_y=max_variation_percent_y,
        percentage_substract_vs_add=percentage_substract_vs_add,
        delay=loung_touch_delay,
    )
    dfu = add_to_df_upswipe(
        df=dfu,
        center_x_cropped_col="bb_center_x_cropped",
        cropped_y_end_col="bb_cropped_y_end",
        cropped_y_start_col="bb_cropped_y_start",
        width_cropped_col="bb_width_cropped",
        height_cropped_col="bb_height_cropped",
        adb_path=adb_path,
        deviceserial=deviceserial,
        variation_startx=swipe_variation_startx,
        variation_endx=swipe_variation_endx,
        variation_starty=swipe_variation_starty,
        variation_endy=swipe_variation_endy,
        new_column_name="ff_bb_upswipe",
        delay=loung_touch_delay,
    )

    dfu = add_to_df_downswipe(
        df=dfu,
        center_x_cropped_col="bb_center_x_cropped",
        cropped_y_end_col="bb_cropped_y_end",
        cropped_y_start_col="bb_cropped_y_start",
        width_cropped_col="bb_width_cropped",
        height_cropped_col="bb_height_cropped",
        adb_path=adb_path,
        deviceserial=deviceserial,
        variation_startx=swipe_variation_startx,
        variation_endx=swipe_variation_endx,
        variation_starty=swipe_variation_starty,
        variation_endy=swipe_variation_endy,
        new_column_name="ff_bb_downswipe",
        delay=loung_touch_delay,
    )

    dfu.loc[:, "ee_bb_longtouch_offset"] = dfu.apply(
        lambda x: sendevent_offset_long(
            sendtouch,
            x.bb_center_x_cropped,
            x.bb_center_y_cropped,
            random.uniform(*loung_touch_delay) / 1000,
            structfolder="struct",
        ),
        axis=1,
    )

    dfu.loc[:, "ee_bb_longtouch_offset_bs"] = dfu.apply(
        lambda x: sendevent_offset_long(
            sendtouch,
            x.bb_center_x_cropped,
            x.bb_center_y_cropped,
            random.uniform(*loung_touch_delay) / 1000,
            structfolder="struct real",
        ),
        axis=1,
    )

    dfu.loc[:, "ee_bb_touch_offset"] = dfu.apply(
        lambda x: sendevent_offset(
            sendtouch,
            x.bb_center_x_cropped,
            x.bb_center_y_cropped,
            structfolder="struct",
        ),
        axis=1,
    )

    dfu.loc[:, "ee_bb_touch_offset_bs"] = dfu.apply(
        lambda x: sendevent_offset(
            sendtouch,
            x.bb_center_x_cropped,
            x.bb_center_y_cropped,
            structfolder="struct real",
        ),
        axis=1,
    )
    prefix = "ee_bb"
    dfu.loc[:, f"{prefix}_longtouch_bs"] = dfu.apply(
        lambda x: FlexiblePartial(
            sendtouch.longtouch,
            True,
            x.bb_center_x_cropped,
            x.bb_center_y_cropped,
            duration=random.uniform(*loung_touch_delay) / 1000,
            struct_folder="struct real",
        ),
        axis=1,
    )

    dfu.loc[:, f"{prefix}_touch_bs"] = dfu.apply(
        lambda x: FlexiblePartial(
            sendtouch.touch,
            True,
            x.bb_center_x_cropped,
            x.bb_center_y_cropped,
            struct_folder="struct real",
        ),
        axis=1,
    )

    dfu.loc[:, f"{prefix}_touch"] = dfu.apply(
        lambda x: FlexiblePartial(
            sendtouch.touch,
            True,
            x.bb_center_x_cropped,
            x.bb_center_y_cropped,
            struct_folder="struct",
        ),
        axis=1,
    )
    dfu.loc[:, f"{prefix}_longtouch"] = dfu.apply(
        lambda x: FlexiblePartial(
            sendtouch.longtouch,
            True,
            x.bb_center_x_cropped,
            x.bb_center_y_cropped,
            duration=random.uniform(*loung_touch_delay) / 1000,
            struct_folder="struct",
        ),
        axis=1,
    )
    return dfu


class AndroDF:
    def __init__(
        self,
        adb_path: str,
        deviceserial: str,
        screenshotfolder: Union[str, None] = None,
        max_variation_percent_x: int = 10,
        max_variation_percent_y: int = 10,
        loung_touch_delay: tuple[int, int] = (1000, 1500),
        swipe_variation_startx: int = 10,
        swipe_variation_endx: int = 10,
        swipe_variation_starty: int = 10,
        swipe_variation_endy: int = 10,
        sdcard: str = "/storage/emulated/0/",
        tmp_folder_on_sd_card: str = "AUTOMAT",
        bluestacks_divider: int = 32767,
    ):
        self.screenshotfolder = screenshotfolder
        self.max_variation_percent_x = max_variation_percent_x
        self.max_variation_percent_y = max_variation_percent_y
        self.percentage_substract_vs_add = 100
        self.loung_touch_delay = loung_touch_delay
        self.swipe_variation_startx = swipe_variation_startx
        self.swipe_variation_endx = swipe_variation_endx
        self.swipe_variation_starty = swipe_variation_starty
        self.swipe_variation_endy = swipe_variation_endy
        self.adb_path = adb_path
        self.deviceserial = deviceserial
        connect_to_adb(adb_path=adb_path, deviceserial=deviceserial)
        self.max_x, self.max_y = get_screenwidth(
            adb_path=adb_path, deviceserial=deviceserial
        )
        self.df = pd.DataFrame()
        self.dfu = pd.DataFrame()

        self.df_merged = pd.DataFrame()
        self.screenshot = pd.NA
        self.sendtouch = SendEventTouch(
            adb_path=adb_path,
            deviceserial=deviceserial,
            sdcard=sdcard,  # it is probably better to pass the path, not the symlink
            tmp_folder_on_sd_card=tmp_folder_on_sd_card,  # if the folder doesn't exist, it will be created
            bluestacks_divider=bluestacks_divider,
            use_bluestacks_coordinates=True,
            # Recalculates the BlueStacks coordinates https://stackoverflow.com/a/73733261/15096247
        )
        self.sendtouch.connect_to_adb()
        self.timestamp = timest()

    def change_settings(
        self,
        screenshotfolder=None,
        max_variation_percent_x=None,
        max_variation_percent_y=None,
        loung_touch_delay=None,
        swipe_variation_startx=None,
        swipe_variation_endx=None,
        swipe_variation_starty=None,
        swipe_variation_endy=None,
    ):
        if screenshotfolder is not None:
            self.screenshotfolder = screenshotfolder
        if max_variation_percent_x is not None:
            self.max_variation_percent_x = max_variation_percent_x
        if max_variation_percent_y is not None:
            self.max_variation_percent_y = max_variation_percent_y
        if loung_touch_delay is not None:
            self.loung_touch_delay = loung_touch_delay
        if swipe_variation_startx is not None:
            self.swipe_variation_startx = swipe_variation_startx
        if swipe_variation_endx is not None:
            self.swipe_variation_endx = swipe_variation_endx
        if swipe_variation_starty is not None:
            self.swipe_variation_starty = swipe_variation_starty
        if swipe_variation_endy is not None:
            self.swipe_variation_endy = swipe_variation_endy
        return self

    def _get_timestamp(self):
        self.timestamp = timest()
        return self

    def get_screenshot(self):
        self.screenshot = take_screenshot(
            self.adb_path, self.deviceserial, channels_in_output=3
        )
        return self

    def get_df_from_activity(self, with_screenshot=True):
        self.df = pd.DataFrame()
        if with_screenshot:
            self.df = get_activity_df(
                self.max_x,
                self.max_y,
                self.adb_path,
                self.deviceserial,
                screens=self.screenshot,
            )
        else:
            self.df = get_activity_df(
                self.max_x, self.max_y, self.adb_path, self.deviceserial, screens=pd.NA
            )
        self.df = add_all_functions_to_df(
            self.sendtouch,
            self.adb_path,
            self.deviceserial,
            self.df,
            self.timestamp,
            screenshotfolder=self.screenshotfolder,
            max_variation_percent_x=self.max_variation_percent_x,
            max_variation_percent_y=self.max_variation_percent_y,
            percentage_substract_vs_add=self.percentage_substract_vs_add,
            loung_touch_delay=self.loung_touch_delay,
            swipe_variation_startx=self.swipe_variation_startx,
            swipe_variation_endx=self.swipe_variation_endx,
            swipe_variation_starty=self.swipe_variation_starty,
            swipe_variation_endy=self.swipe_variation_endy,
        )
        self.df = self.df.filter(list(sorted(self.df.columns)))
        for col in self.df:
            if str(col).startswith("parent_"):
                try:
                    self.df[col] = self.df[col].astype("Int64")
                except Exception:
                    pass
        return self

    def get_df_from_view(self, with_screenshot=True):
        self.dfu = pd.DataFrame()
        try:
            if with_screenshot:
                self.dfu = get_view_df(
                    self.adb_path,
                    self.deviceserial,
                    self.max_x,
                    self.max_y,
                    merge_it=False,
                    df=None,
                    screens=self.screenshot,
                )
            else:
                self.dfu = get_view_df(
                    self.adb_path,
                    self.deviceserial,
                    self.max_x,
                    self.max_y,
                    merge_it=False,
                    df=None,
                    screens=pd.NA,
                )

            self.dfu = add_all_functions_to_dfu(
                self.sendtouch,
                self.adb_path,
                self.deviceserial,
                self.dfu,
                self.timestamp,
                screenshotfolder=self.screenshotfolder,
                max_variation_percent_x=self.max_variation_percent_x,
                max_variation_percent_y=self.max_variation_percent_y,
                percentage_substract_vs_add=self.percentage_substract_vs_add,
                loung_touch_delay=self.loung_touch_delay,
                swipe_variation_startx=self.swipe_variation_startx,
                swipe_variation_endx=self.swipe_variation_endx,
                swipe_variation_starty=self.swipe_variation_starty,
                swipe_variation_endy=self.swipe_variation_endy,
            )
            self.dfu = self.dfu.filter(list(sorted(self.dfu.columns)))

        except Exception as fe:
            print(fe)
            pass
        return self

    def get_all_results(self):
        return self.df.copy(), self.dfu.copy(), self.df_merged.copy()

    def get_dfs_from_view_and_activity(self, with_screenshot=True):
        self.dfu = pd.DataFrame()
        self.df = pd.DataFrame()
        self.df_merged = pd.DataFrame()

        self.get_df_from_activity(with_screenshot=with_screenshot)
        self.dfu = pd.DataFrame()
        if with_screenshot:
            self.dfu = get_view_df(
                self.adb_path,
                self.deviceserial,
                self.max_x,
                self.max_y,
                merge_it=True,
                df=self.df,
                screens=self.screenshot,
            )
        else:
            self.dfu = get_view_df(
                self.adb_path,
                self.deviceserial,
                self.max_x,
                self.max_y,
                merge_it=True,
                df=self.df,
                screens=pd.NA,
            )

        self.dfu = add_all_functions_to_dfu(
            self.sendtouch,
            self.adb_path,
            self.deviceserial,
            self.dfu,
            self.timestamp,
            screenshotfolder=self.screenshotfolder,
            max_variation_percent_x=self.max_variation_percent_x,
            max_variation_percent_y=self.max_variation_percent_y,
            percentage_substract_vs_add=self.percentage_substract_vs_add,
            loung_touch_delay=self.loung_touch_delay,
            swipe_variation_startx=self.swipe_variation_startx,
            swipe_variation_endx=self.swipe_variation_endx,
            swipe_variation_starty=self.swipe_variation_starty,
            swipe_variation_endy=self.swipe_variation_endy,
        )
        self.df_merged = view_activity_dfs_merged(self.df, self.dfu)
        self.df_merged = self.df_merged.filter(list(sorted(self.df_merged.columns)))
        return self
