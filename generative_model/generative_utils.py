# utils for generative model

# create a single-layer metrics dict for output
# since the pickler can't handle multilevel dicts
def cleanup_metrics(metrics):
#    for k in metrics['SSRT']:
#        metrics['SSRT_' + k] = metrics['SSRT'][k]
    del metrics['SSRT']
    return(metrics)


