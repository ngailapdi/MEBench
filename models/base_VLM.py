from abc import ABC, abstractmethod

class BaseVLModel:

    @abstractmethod
    def get_prompt(self):
        pass

    @abstractmethod
    def build_input_ids(self):
        pass
    
    @abstractmethod
    def generate_content(self):
        pass

    @abstractmethod
    def get_model_kwargs(self):
        pass

    @abstractmethod
    def get_model_input(self):
        pass
    
    @abstractmethod
    def process_model_output(self):
        pass
