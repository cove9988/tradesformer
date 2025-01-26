
import os
"""
Renders transaction data to a specified log file and optionally prints it to the console.

Parameters:
    log_header (bool): If True, includes a header row in the log output.
    log_filename (str): File path where the log is written or appended.
    printout (bool): If True, prints transaction information to the console.
    balance (float): Current balance value.
    balance_initial (float): Initial balance value.
    transaction_close_this_step (list): List of dictionaries with details of closed transactions.
    done_information (str): Additional information appended to the log upon completion.

Returns:
    None
"""
def render_to_file(**kwargs):
    log_header = kwargs.get("log_header",False)
    log_filename=kwargs.get("log_filename","")
    printout=kwargs.get("printout",False)
    balance=kwargs.get("balance")
    balance_initial=kwargs.get("balance_initial")
    transaction_close_this_step=kwargs.get("transaction_close_this_step",[])
    done_information=kwargs.get("done_information","")
    profit = balance - balance_initial
    tr_lines = ""
    tr_lines_comma = ""
    _header = ''
    _header_comma = ''
    if log_header:
        _header = f'{"Ticket":>8} {"Type":>4} {"ActionStep":16} \
                    {"ActionPrice":>12} {"CloseStep":8} {"ClosePrice":>12} \
                    {"pips":>6} {"SL":>6} {"PT":>6} {"DeltaStep":8}\n'


        _header_comma = f'{"Ticket,Type,ActionTime,ActionStep,ActionPrice,CloseTime,ClosePrice,pips,SL,PT,CloseStep,DeltaStep"}\n'
    if transaction_close_this_step:
        for _tr in transaction_close_this_step:
            if _tr["CloseStep"] >=0:
                tr_lines += f'{_tr["Ticket"]:>8} {_tr["Type"]:>4} {_tr["ActionStep"]:16} \
                    {_tr["ActionPrice"]:6.5f} {_tr["CloseStep"]:8} {_tr["ClosePrice"]:6.5f} \
                    {_tr["pips"]:4.0f} {_tr["SL"]:4.0f} {_tr["PT"]:4.0f} {_tr["DeltaStep"]:8}\n'

                tr_lines_comma += f'{_tr["Ticket"]},{_tr["Type"]},{_tr["ActionTime"]},{_tr["ActionStep"]}, \
                    {_tr["ActionPrice"]},{_tr["CloseTime"]},{_tr["ClosePrice"]}, \
                    {_tr["pips"]},{_tr["SL"]},{_tr["PT"]},{_tr["CloseStep"]},{_tr["DeltaStep"]}\n'

    log = _header_comma + tr_lines_comma
    # log = f"Step: {current_step}   Balance: {balance}, Profit: {profit} \
    #     MDD: {max_draw_down_pct}\n{tr_lines_comma}\n"
    if done_information:
        log += done_information
    if log:
        # os.makedirs(log_filename, exist_ok=True)
        dir_path = os.path.dirname(log_filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(log_filename, 'a+') as _f:
            _f.write(log)
            _f.close()

    tr_lines = _header + tr_lines
    if printout and tr_lines:
        print(tr_lines)
        if done_information:
            print(done_information)