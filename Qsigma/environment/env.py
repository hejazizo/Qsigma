from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self, config):
        '''
        Generic class for all environment.
        '''
        self.world = None
        self.config = config
        self.reset()

    @abstractmethod
    def act(self):
        '''
        The main method, that does one step of actions for the agents
        in the world. This is empty in the base class.
        '''
        pass

    @abstractmethod
    def reset(self):
        '''
        resets the environment world.
        '''
        pass
