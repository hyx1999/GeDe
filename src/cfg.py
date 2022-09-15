class Config:

    def __init__(self, **kwargs) -> None:

        self.model_name = "bert-base-chinese"
        self.model_hidden_size = 768

        self.batch_size = 16
        self.scheduler_step_size = 10
        self.num_epochs = 100  # 80

        self.max_nums_size = 35

        self.device = "cuda:0"

        kwargs = {k: v for k, v in kwargs.items() if k in self.__dict__}
        self.__dict__.update(kwargs)


class RecConfig:

    def __init__(self, **kwargs) -> None:
        self.model_name = "bert-base-chinese"

        self.batch_size = 16
        self.scheduler_step_size = 10
        self.num_epochs = 100
        
        self.max_nums_size = 35
        self.max_const_nums_size = 10
        self.max_step_size = 35
        self.use_bracket = False

        self.device = "cuda:0"

        kwargs = {k: v for k, v in kwargs.items() if k in self.__dict__}
        self.__dict__.update(kwargs)
