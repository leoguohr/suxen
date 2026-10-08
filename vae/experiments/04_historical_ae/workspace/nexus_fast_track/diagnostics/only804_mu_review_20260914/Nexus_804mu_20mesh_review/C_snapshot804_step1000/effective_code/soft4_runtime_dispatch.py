def full_objective(rows,data,scales,base=None):
    if base is not None:return c.objective(rows,data,scales,base)
    c.weights=full_weights;sums_globals['soft4_sums']=full_sums
    try:return c.objective(rows,data,scales)
    finally:c.weights=old_weights;sums_globals['soft4_sums']=old_sums
