import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def process_instance(el):
    """
        Process each 'process instance' element from the .mxml file
        and returns as dict
    """
    resp = []
    for entry in el[1:]:
        r = {
            "TraceId": el.get("id")
        }
        for item in entry:
            if item.tag == 'Data':
                r[item.tag] = item[-1].text
            else:
                r[item.tag] = item.text
        resp.append(r)
    return resp

def read_mxml(file):
    """
        Read MXML file into a Pandas DataFrame
    """
    root = ET.parse(file).getroot()
    process = root[-1]
    
    resp = []
    for p_instance in process:
        for r in process_instance(p_instance):
            resp.append(r)
    
    return pd.DataFrame.from_dict(resp)

def get_completed_only(mxml_df):
    """
        Filter only completed activities 
    """
    complete = mxml_df[mxml_df.EventType == "complete"].rename({
        "Timestamp": "Timestamp_Complete"
    }, axis=1).set_index(["TraceId", "WorkflowModelElement"])
    
    return complete.drop(["Originator", "EventType"], errors='ignore', axis=1).reset_index()

def cumulative_counting(traces):
    """
        Cumulative counting in column
    """
    t_ant = None
    cnt = 0
    
    resp = []
    for t in traces:
        if t != t_ant:
            cnt += 1
            t_ant = t
        resp.append(cnt)
        
    return(pd.Series(resp) - 1)

def all_prep(file, aliases=None, replace_whitespaces="_"):
    """
        Runs all basic prep and return preped DataFrame
    """
    df = read_mxml(file)
    df = get_completed_only(df)
    
    df["WorkflowModelElement"] = df.WorkflowModelElement.apply(lambda x: x.replace(' ', replace_whitespaces))
    
    if aliases is not None:
        df["Activity"] = df.WorkflowModelElement.replace(aliases)
    else:
        df["Activity"] = df.WorkflowModelElement
        
    df["Trace_order"] = cumulative_counting(df["TraceId"])
    
    return df
