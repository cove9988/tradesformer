import os
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
        _header = f'{"Ticket":>8}{"Type":4}{"ActionTime":>16}{"ActionStep":>16} \
                            {"ActionPrice":12}{"CloseTime":>16}{"ClosePrice":12} \
                            {"pips":4}{"SL":4}{"PT":4}{"CloseStep":8}{"DeltaStep":8}\n'

        _header_comma = f'{"Ticket,Type,ActionTime,ActionStep,ActionPrice,CloseTime,ClosePrice,pips,SL,PT,CloseStep,DeltaStep"}\n'
    if transaction_close_this_step:
        for _tr in transaction_close_this_step:
            if _tr["CloseStep"] >=0:
                tr_lines += f'{_tr["Ticket"]:>8} {_tr["Type"]:>4} {_tr["ActionTime"]:16} {_tr["ActionStep"]:16} \
                    {_tr["ActionPrice"]:6.5f} {_tr["CloseTime"]:16} {_tr["ClosePrice"]:6.5f} \
                    {_tr["pips"]:4.0f} {_tr["SL"]:4.0f} {_tr["PT"]:4.0f} {_tr["CloseStep"]:8} {_tr["DeltaStep"]:8}\n'

                tr_lines_comma += f'{_tr["Ticket"]:>8},{_tr["Type"]:>4},{_tr["ActionTime"]:>16},{_tr["ActionStep"]:16}, \
                    {_tr["ActionPrice"]:6.5f},{_tr["CloseTime"]:16},{_tr["ClosePrice"]:6.5f}, \
                    {_tr["pips"]:4.0f},{_tr["SL"]:4.0f},{_tr["PT"]:4.0f},{_tr["CloseStep"]:8},{_tr["DeltaStep"]:8}\n'

    log = _header_comma + tr_lines_comma
    # log = f"Step: {current_step}   Balance: {balance}, Profit: {profit} \
    #     MDD: {max_draw_down_pct}\n{tr_lines_comma}\n"
    if done_information:
        log += done_information
    if log:
        # os.makedirs(log_filename, exist_ok=True)
        with open(log_filename, 'a+') as _f:
            _f.write(log)
            _f.close()

    tr_lines += _header
    if printout and tr_lines:
        print(tr_lines)
        if done_information:
            print(done_information)