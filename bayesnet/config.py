class Config:
    __is_training = False

    @property
    def is_training(self):
        return self.__is_training

    @is_training.setter
    def is_training(self, flg):
        assert(isinstance(flg, bool))
        self.__is_training = flg
