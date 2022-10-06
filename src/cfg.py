class Config:

    def __init__(self, **kwargs) -> None:

        self.model_name = "bert-base-chinese"
        self.model_hidden_size = 768

        self.batch_size = 8
        self.scheduler_step_size = 10
        self.num_epochs = 100  # 80

        self.max_nums_size = 35

        self.device = "cuda:0"

        kwargs = {k: v for k, v in kwargs.items() if k in self.__dict__}
        self.__dict__.update(kwargs)


class MathConfig:

    def __init__(self, **kwargs) -> None:
        self.dataset_name = ""
        # self.model_name = "bert-base-chinese"
        self.model_name = "hfl/chinese-roberta-wwm-ext"

        self.batch_size = 8
        self.scheduler_step_size = 10
        self.num_epochs = 80
        
        self.max_nums_size = 35

        self.debug = False
        self.device = "cuda:0"

        kwargs = {k: v for k, v in kwargs.items() if k in self.__dict__}
        self.__dict__.update(kwargs)


class KBQAConfig:

    def __init__(self, **kwargs) -> None:
        self.dataset_name = ""
        self.model_name = "facebook/bart-base"

        self.batch_size = 8
        self.scheduler_step_size = 10
        self.num_epochs = 80
        
        self.debug = False
        self.device = "cuda:0"

        kwargs = {k: v for k, v in kwargs.items() if k in self.__dict__}
        self.__dict__.update(kwargs)
