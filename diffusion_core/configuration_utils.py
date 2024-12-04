import functools
import inspect


class MethodStorage(dict):    
    @staticmethod
    def register(method_name):
        def decorator(method):
            method._method_name = method_name
            method._is_decorated = True
            return method
        return decorator
    
    def register_methods(self):
        self.registered_methods = {}
        for method_name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            print(method_name)
            if getattr(method, '_is_decorated', False):
                self.registered_methods[method._method_name] = method
    
    def __getitem__(self, method_name):
        return self.registered_methods[method_name]
