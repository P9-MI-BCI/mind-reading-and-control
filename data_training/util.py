import concurrent.futures
import os
from utility.logger import get_logger

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def multiproc_classifier(model, target, channels, preds_data, feats, labels):

    max_proc = os.cpu_count()-2
    chunks_list = chunks(channels, int(max_proc))
    get_logger().info(f"Chunking channel power set into {max_proc} chunks")
    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        get_logger().info(f"Creating {max_proc} processes")
        finished_procs = [executor.submit(target, model=model, channels=chunk, pred_data=preds_data, feats=feats, labels=labels) for chunk in chunks_list]
        for f in concurrent.futures.as_completed(finished_procs):
            results.append(f.result())
    get_logger().info("Done with multiprocessing, joining parent")

    return results
