import re
from .array import pad_array
from pandas import read_csv
from numpy import array

# good reference:
# https://docs.microsoft.com/en-us/windows/deployment/usmt/usmt-recognized-environment-variables

VARIABLE_MAP = {
    r"%systemdrive%": r"[drive]", 
    r"%systemroot%": r"[drive]\windows",
    r"%windir%": r"[drive]\windows", 
    r"%allusersprofile%": r"[drive]\programdata",
    r"%programdata%": r"[drive]\programdata",
    r"%programfiles%": r"[drive]\program files",
    r"%programfiles(x86)%": r"[drive]\program files (x86)",
    r"%programw6432%": r"[drive]\program files",
    r"%commonprogramfiles%": r"[drive]\program files\common files",
    r"%commonprogramfiles(x86)%": r"[drive]\program files (x86)\common files",
    r"%commonprogramw6432%": r"[drive]\program files\common files",
    r"%commonfiles%": r"[drive]\program files\common files",
    r"%profiles%": r"[drive]\users",
    r"%public%": r"[drive]\users\public",
    r"%userprofile%": r"[drive]\users\[user]"
}
# more user variables
VARIABLE_MAP.update({
    r"%homepath%": VARIABLE_MAP[r"%userprofile%"],
    r"%downloads%": VARIABLE_MAP[r"%userprofile%"] + r"\downloads",
    r"%desktop%": VARIABLE_MAP[r"%userprofile%"] + r"\desktop",
    r"%favorites%": VARIABLE_MAP[r"%userprofile%"] + r"\favorites",
    r"%documents%": VARIABLE_MAP[r"%userprofile%"] + r"\documents",
    r"%mydocuments%": VARIABLE_MAP[r"%userprofile%"] + r"\documents", # obsolete
    r"%personal%": VARIABLE_MAP[r"%userprofile%"] + r"\documents", # obsolete
    r"%localsettings%": VARIABLE_MAP[r"%userprofile%"] + r"\documents", # obsolete
    r"%mypictures%": VARIABLE_MAP[r"%userprofile%"] + r"\documents\my pictures",
    r"%mymusic%": VARIABLE_MAP[r"%userprofile%"] + r"\documents\my music",
    r"%myvideos%": VARIABLE_MAP[r"%userprofile%"] + r"\documents\my videos",
    r"%localappdata%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local",
    r"%appdata%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\roaming",
    r"%usertemp%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\temp",
    r"%temp%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\temp",
    r"%tmp%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\temp",
    r"%cache%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\microsoft\windows\temporary internet files"
})    

def load_txt(filename, padding_length):
    try:
        txtdata = read_csv(filename, header=None, on_bad_lines='skip') # 0.5s
    except TypeError:
        txtdata = read_csv(filename, header=None, error_bad_lines=False) # 0.5s

    txtdata.columns = ["x"]
    txtdata = txtdata.x.apply(normalize_path) # 2s
    txtdata = txtdata.str.encode("utf-8", "ignore").apply(lambda x: array(list(x), dtype=int)) # 2s
    txtdata = txtdata.apply(pad_array, args=(padding_length,)) # 2s
    return txtdata


def normalize_path(path, verbose=False):
    """Function that takes a path string and returns a normalized version,
    but substituting: (1) drive letters with [drive] (2) network hosts with [net] 
                      (3) arbitrary, non-default usernames with [user] 
                      (4) environment variables with fullpath equivalent

    Args:
        path (str): String containing file-path, e.g. "C:\\users\\ieuser\\desktop\\my.exe"
        verbose (bool, optional): Whether to print file-path normalization steps. Defaults to False.

    Returns:
        str: Normalized path, e.g. "[drive]\\users\\[user]\\desktop\\my.exe"
    """
    # X. Avast TI proprietary: some paths have "*raw:", "*amsiprocess:", "script started by " auxiliary strings...
    path = path.lower().replace("*raw:","").replace("*amsiprocess:","").replace("a script started by ","").strip()
    if verbose:
        print(path)

    # 1a. Normalize drive
    # "c:\\"" or "c:\"" will be [drive]\
    path = re.sub(r"\w:\\{1,2}", r"[drive]\\", path)
    if verbose:
        print(path)

    # 2. Normalize network paths, i.e. need to change "\\host\" to "[net]\"
    # before that take care of "\;lanmanredirector" or "\;webdavredirector" paths
    path = re.sub(r"[\\]{1,2};((lanmanredirector|webdavredirector)\\;){0,1}\w\:[a-z0-9]{16}", "\\\\", path)
    # [\w\d\.\-]+ is DNS pattern, comes from RFC 1035 and:
    # https://docs.microsoft.com/en-us/troubleshoot/windows-server/identity/naming-conventions-for-computer-domain-site-ou#dns-host-names
    path = re.sub(r"\\\\[\w\d\.\-]+\\", r"[net]\\", path)
    if verbose:
        print(path)

    # 1b. Normalize drive (2) - you can refer to path as "dir \user\ieuser\desktop" in windows
    # if starts with \, and not \\ (captures not-\\ as \1 group and preserves), then add [drive]
    path = re.sub(r"^\\([^\\])", r"[drive]\\\1", path)
    if verbose:
        print(path)
    
    # 1c. Normalize "\\?\Volume{614d36cf-0000-0000-0000-10f915000000}\" format
    path = re.sub(r"\\[\.\?]\\volume\{[a-z0-9\-]{36}\}", "[drive]", path)
    
    # 3. normalize non-default users
    default_users = ["administrator", "public", "default"]
    if "users\\" in path:
        # default user path, want to preserve them
        if not any([True if "users\\"+x+"\\" in path else False for x in default_users]):
            path = re.sub(r"users\\[^\\]+\\", r"users\\[user]\\", path)
    if verbose:
        print(path)

    # 4. Normalize environment variables with actual paths
    for k,v in VARIABLE_MAP.items():
        path = path.replace(k, v)
    if verbose:
        print(path)

    return path
