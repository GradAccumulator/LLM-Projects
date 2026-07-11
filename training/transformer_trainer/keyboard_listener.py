from pynput import keyboard

class KeyboardListener:
    def __init__(self, key_map:dict, trainer):
        self.key_map = key_map
        self.trainer = trainer
        self.listener = keyboard.Listener(self._on_press)
    
    def _on_press(self, key):
        char = getattr(key, "char", None)
        command_type = self.key_map.get(char)
        if command_type is not None:
            self.trainer.submit_command(command_type())
        
    
    def start(self):
        return self.listener.start()
    
    def stop(self):
        return self.listener.stop()
    