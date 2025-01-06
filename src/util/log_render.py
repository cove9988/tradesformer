import os

def render_to_file(**kwargs):
    log_header = kwargs.get("log_header", False)
    log_filename = kwargs.get("log_filename", "./data/log/log_")
    # printout = kwargs.get("printout", False)
    # balance = kwargs.get("balance")
    # balance_initial = kwargs.get("balance_initial")
    positions = kwargs.get("positions", [])

    step = kwargs.get("step",0)

    tr_lines = ""
    _header = ''

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    if log_header:
        _header = f'{"tranactions_id":10},{"currency_pair":8},{"direction":8},{"open_step":16},{"open_price":16},{"maxDD":10},{"close_step":16},{"close_price":16},{"pips":8},{"close_reason":6},{"duration":10}\n'

    for _tr in positions:
        if _tr["close_step"] + 1 == step:
            tr_lines += f'{_tr["tranactions_id"]:10},{_tr["currency_pair"]:8},{_tr["direction"]:8},{_tr["open_step"]:16},{_tr["open_price"]:11.5f},{_tr["maxDD"]:8.2f},{_tr["close_step"]:16},{_tr["close_price"]:11.5f},{_tr["pips"]:6.2f},{_tr["close_reason"]:6},{_tr["duration"]:10}\n'


    if not os.path.exists(log_filename):
        with open(log_filename, 'a+') as _f:
            _f.write(_header + tr_lines)
    else:
        with open(log_filename, 'a+') as _f:
            _f.write(tr_lines)
3