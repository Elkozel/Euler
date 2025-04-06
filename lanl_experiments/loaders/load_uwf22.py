from copy import deepcopy
from datetime import datetime
import os 
import pickle 
from elasticsearch import Elasticsearch
from joblib import Parallel, delayed

import pandas as pd
import torch 
from torch_geometric.data import Data 
from tqdm import tqdm

from .elastic_helpers import ElasticDataFetcher, generate_node_map, make_fielddata 

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized

# We want to go from Dec 17, 2021 08:00 (1639724400)
# Till Feb 19, 2022 21:00 (1645300800), which is a difference of 5576400 seconds
# Dates are in standard UNIX format:
DATE_OFFSET = 1639724400
DATE_START = DATE_OFFSET
DATE_END = DATE_START + 5576400

DATE_OF_EVIL_UWF22 = 150885
FILE_DELTA = 10000


TIMES = {
    20      : 254429, # First 20 anoms
    100     : 823578, # First 100 anoms
    500     : 1212490, # First 500 anoms
    'all'  : 5576400  # Full
}

def empty_uwf22(nodemap):
    return make_data_obj([],None,None, nodemap)

def make_data_obj(eis, ys, ew_fn, nodemap, ews=None, **kwargs):
    cl_cnt = len(nodemap)
    x = torch.eye(cl_cnt+1)
    
    # Build time-partitioned edge lists
    eis_t = []
    masks = []

    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)

        # This is training data if no ys present
        if isinstance(ys, None.__class__):
            masks.append(edge_tv_split(ei)[0])

    # Balance the edge weights if they exist
    if not isinstance(ews, None.__class__):
        cnt = deepcopy(ews)
        ews = ew_fn(ews)
    else:
        cnt = None

    # Finally, return Data object
    return TData(
        eis_t, x, ys, masks, ews=ews, cnt=cnt, node_map=nodemap
    )

def load_uwf22_dist(workers, start=0, end=DATE_END-DATE_START, delta=10000, is_test=False, ew_fn=std_edge_w):
    if start == None or end == None:
        return empty_uwf22()

    num_slices = ((end - start) // delta)
    remainder = (end-start) % delta
    num_slices = num_slices + 1 if remainder else num_slices
    workers = 1 # min(num_slices, workers)

    # Can't distribute the job if not enough workers
    if workers <= 1:
        return load_partial_uwf22(start, end, delta, is_test, ew_fn)

    per_worker = [num_slices // workers] * workers
    remainder = num_slices % workers

    # Give everyone a balanced number of tasks 
    # put remainders on last machines as last task 
    # is probably smaller than a full delta
    if remainder:
        for i in range(workers, workers-remainder, -1):
            per_worker[i-1] += 1

    kwargs = []
    prev = start 
    for i in range(workers):
        end_t = prev + delta*per_worker[i]
        kwargs.append({
            'start': prev, 
            'end': min(end_t-1, end),
            'delta': delta,
            'is_test': is_test,
            'ew_fn': ew_fn
        })
        prev = end_t
    
    # Now start the jobs in parallel 
    datas = Parallel(n_jobs=workers, prefer='processes')(
        delayed(load_partial_uwf22_job)(i, kwargs[i]) for i in range(workers)
    )

    # Helper method to concatonate one field from all of the datas
    data_reduce = lambda x : sum([getattr(datas[i], x) for i in range(workers)], [])

    # Just join all the lists from all the data objects
    print("Joining Data objects")
    x = datas[0].xs
    eis = data_reduce('eis')
    masks = data_reduce('masks')
    ews = data_reduce('ews')
    node_map = datas[0].node_map

    if is_test:
        ys = data_reduce('ys')
        cnt = data_reduce('cnt')
    else:
        ys = None
        cnt = None

    # After everything is combined, wrap it in a fancy new object, and you're
    # on your way to coolsville flats
    print("Done")
    return TData(
        eis, x, ys, masks, ews=ews, node_map=node_map, cnt=cnt
    )

def process_pandas_data(data, is_test=False):
    """
    Process the pandas DataFrame to extract edges, edge weights, and labels.
    """
    edges = []
    ews = []
    edges_t = {}
    ys = []

    # Helper function to add edges
    def add_edge(et, is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = (max(is_anom, val[0]), val[1] + 1)
        else:
            edges_t[et] = (is_anom, 1)

    for _, row in data.iterrows():
        src = int(row["src_ip_zeek"])
        dst = int(row["dest_ip_zeek"])
        label = int(row["label_tactic"] if is_test else 0)
        et = (src, dst)

        # Skip self-loops
        if src == dst:
            continue

        add_edge(et, is_anom=label)

    # Convert edges_t to lists of edges, weights, and labels
    if len(edges_t):
        ei = list(zip(*edges_t.keys()))
        edges.append(ei)

        y, ew = list(zip(*edges_t.values()))
        ews.append(torch.tensor(ew))

        if is_test:
            ys.append(torch.tensor(y))

    ys = ys if is_test else None
    return edges, ews, ys

def load_partial_uwf22(start, end, delta, is_test, ew_fn):
    UWF22_INDEX = "uwf-zeekdata22"
    es = Elasticsearch("https://localhost:9200", api_key="MGZGTDhaVUJHWEpfZm5CYVB1bXo6dXBxVk5ucF9Rc3F6dWh5RjVRVDQzUQ==", verify_certs=False, ssl_show_warn=False)

    make_fielddata(es, UWF22_INDEX, "src_ip_zeek")
    make_fielddata(es, UWF22_INDEX, "dest_ip_zeek")

    nodemap = generate_node_map(es, UWF22_INDEX, "src_ip_zeek")
    dest_nodemap = generate_node_map(es, UWF22_INDEX, "dest_ip_zeek")

    for node in dest_nodemap:
        if node not in nodemap:
            nodemap.append(node)

    # Query Elasticsearch for the data
    from_date = datetime.fromtimestamp(start + DATE_OFFSET)
    to_date = datetime.fromtimestamp(end + DATE_OFFSET)

    query = {
        "range": {
            "datetime": {
                "gte": from_date,
                "lt": to_date
            }
        }
    }
    data_fetcher = ElasticDataFetcher(es, UWF22_INDEX, query)
    data = pd.DataFrame(tqdm(data_fetcher, desc=f"Loading {from_date} to {to_date}"))
    if data.empty:
        print(f"No data found for the time range {from_date} to {to_date}.")
    else:
        # Drop the columns we don't need
        data = data[["src_ip_zeek", "dest_ip_zeek", "label_tactic", "duration"]]
        

        # TODO: Figure out why some IPs are not in the nodemap
        data["src_ip_zeek"] = data["src_ip_zeek"].map(lambda x : nodemap.index(x) if x in nodemap else 1)
        data["dest_ip_zeek"] = data["dest_ip_zeek"].map(lambda x : nodemap.index(x) if x in nodemap else 1)
        label_map = {
            "none": 0,
            "Reconnaissance": 0,
            "Discovery": 1,
            "Credential Access": 1,
            "Privilege Escalation": 1,
            "Exfiltration": 1,
            "Lateral Movement": 1,
            "Resource Development": 1,
            "Defense Evasion": 1,
            "Initial Access": 1,
            "Persistence": 1        
        }
        data["label_tactic"] = data["label_tactic"].map(label_map)

    # Process the pandas DataFrame to extract edges, weights, and labels
    edges, ews, ys = process_pandas_data(data, is_test=is_test)

    # Call make_data_obj with the processed data
    return make_data_obj(
        edges, ys, ew_fn,
        ews=ews, nodemap=nodemap
    )

# wrapper bc its annoying to send kwargs with Parallel
def load_partial_uwf22_job(pid, args):
    data = load_partial_uwf22(**args)
    return data