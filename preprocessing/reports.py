from pandas import json_normalize
import json

import sys
sys.path.append("..")
from utils.functions import flatten

def report_to_apiseq(reportfile_fullpath):
    """Functiona parses JSON reports provided by emulaton/emulate_samples.py
    which generates files in a format of /path/to/report/<hash>.json

    Args:
        reportfile_fullpath (str): Fullpath of a reportfile.

    Returns:
        dict: Dictionary containing sha256, api sequence, and api sequence length
    """

    filehash = reportfile_fullpath.strip(".json").split("/")[-1]
    try:
        report = json.load(open(reportfile_fullpath))
    except json.decoder.JSONDecodeError as ex:
        print(reportfile_fullpath)
        print(ex)
        import pdb;pdb.set_trace()
    report_fulldf = json_normalize(report)


    # for now only hash and API sequence
    apiseq = list(flatten(report_fulldf["apis"].apply(lambda x: [y["api_name"].lower() for y in x]).values))
    data = {"sha256": filehash, "api.seq": apiseq, "api.seq.len": len(apiseq)}

    return data
