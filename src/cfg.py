from typing import List


class Config:

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class MathConfig:

    def __init__(self, **kwargs) -> None:
        self.dataset_name = ""
        self.model_name = ""

        self.num_epochs = 500
        self.batch_size = 8
        self.lr = 2e-5
        self.lr_alpha = 1.0
        self.weight_decay = 1e-2
        
        self.max_step_size = 35

        self.use_data_aug = False
        self.debug = False
        self.device = "cuda:0"
        self.save_result = False
        
        self.quant_size = 35
        self.const_quant_size = None
        self.ext_tokens = None
        self.expr_size = 4
        self.beam_size = 4
        
        self.__dict__.update(kwargs)

"""
class KBQAConfig:

    def __init__(self, **kwargs) -> None:
        self.dataset_name = ""
        self.model_name = ""

        self.batch_size = 8
        self.num_epochs = 50
        self.lr = 2e-5
        self.weight_decay = 1e-2

        self.debug = False
        self.device = "cuda:0"
        
        self.ext_tokens: List[str] = None
        
        self.variable_size = 20
        self.expr_size     = 5
        self.relation_size = 100
        self.type_size     = 100
        self.bucket_size   = 100
        self.ranker_bucket_size = 100

        self.__dict__.update(kwargs)
"""