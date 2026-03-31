#!/usr/bin/env python
"""Model selection between different multinest runs."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import os.path
import re
import sys
from typing import Dict, List

import numpy as np

from isochrones import StarModel


def main() -> None:
    parser = argparse.ArgumentParser(description='Model select between different multinest runs.')

    parser.add_argument('folders', nargs='*', default=['.'])
    parser.add_argument('--models', default='dartmouth')

    args = parser.parse_args()

    for folder in args.folders:
        chainsdir = folder + '/chains'
        if not os.path.exists(chainsdir):
            logging.error('No chains directory in {}'.format(folder))
            sys.exit()
            
        starmodels = glob.glob('{}/{}_starmodel_*.h5'.format(folder,
                                                             args.models))
        names: List[str] = []
        evidence: Dict[str, float] = {}
        evidences: List[float] = []
        for sm in starmodels:
            m = re.search('{}_starmodel_(\w+)\.h5'.format(args.models), sm)
            if m:
                name = m.group(1)
                names.append(name)
                model = StarModel.load_hdf(sm)
                evidence[name] = model.evidence
                evidences.append(model.evidence)

        evidences_arr = np.array(evidences)
        evidences_arr = evidences_arr - evidences_arr.max()

        print(names, evidences_arr)


if __name__ == '__main__':
    main()
