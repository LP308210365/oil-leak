from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.modules import inspector


inspector.create_inspector(Window, Button)  #启动调试模式