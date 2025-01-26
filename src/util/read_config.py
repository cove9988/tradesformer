import json

class EnvConfig():
    """environment configuration from json file
       tgym requires you configure your own parameters in json file.
        Args:
            config_file path/file.json

    """
    def __init__(self,config_file):
        self.config = {}
        with open(config_file) as j: 
            self.config = json.load(j)

    def env_parameters(self,item=''):   
        """environment variables 
        """ 
        if item:
            return self.config["env"][item]
        else:
            return self.config["env"]
        
    def symbol(self, asset="GBPUSD", item='') :
        """get trading pair (symbol) information

        Args:
            asset (str, optional): symbol in config. Defaults to "GBPUSD".
            item (str, optional): name of item, if '' return dict, else return item value. Defaults to ''.

        Returns:
            [type]: [description]
        """
        if item:
            return self.config["symbol"][asset][item]
        else:
            return self.config["symbol"][asset]
        
    def data_processing_parameters(self, item=''):
        """Get data processing config"""
        if item:
            return self.config["data_processing"][item]
        return self.config["data_processing"]
            
    def trading_hour(self,place="New York"):
        """forex trading hour from different markets

        Args:
            place (str, optional): [Sydney,Tokyo,London] Defaults to "New York".

        Returns:
            [dict]: from time, to time
        """
        if place:
            return self.config["trading_hour"][place]
        else:
            return self.config["trading_hour"]
    
if __name__ == '__main__':    
    env_config_file ='/home/paulg/github/tradesformer/src/configure.json'
    cf = EnvConfig(env_config_file)
    print(f'{cf.env_parameters()}')
    print(cf.env_parameters("observation_list"))
    print(f'symbol_col: {cf.env_parameters()["symbol_col"]}')
    print(cf.symbol("GBPUSD")["point"])
    print(f'trading hour new york: {cf.trading_hour("New York")}')