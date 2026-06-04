import pygame
import numpy as np

class PygameController:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Robot Control (Focus Here!)")
        self.font = pygame.font.SysFont(None, 32)

        # 初始化手柄
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"检测到手柄: {self.joystick.get_name()}")
        else:
            print("未检测到手柄，继续使用键盘控制")
            
        # 初始化控制内部状态（从主程序中移过来的初始值）
        self.paused = False
        self.control_enabled = False
        self.target_length = 0.1
        self.target_v = 0.0
        self.target_w = 0.0

    def update(self, current_v, target_length_left, target_length_right, roll, viewer):
        """
        更新事件、读取键盘/手柄，并刷新 UI。
        传入主程序中的实时运行数据用于界面显示。
        """
        # --- 处理 Pygame 事件 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                viewer.close()
                pygame.quit()
                exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RETURN:
                    self.control_enabled = not self.control_enabled
                    print(f"当前控制状态: {'开启 (ON)' if self.control_enabled else '关闭 (OFF)'}")
                elif event.key in [pygame.K_EQUALS, pygame.K_PLUS]:
                    self.target_length = float(np.clip(self.target_length + 0.01, 0.10, 0.14))
                    print(f"目标腿长增加，当前腿长为: {self.target_length:.3f}")
                elif event.key == pygame.K_MINUS:
                    self.target_length = float(np.clip(self.target_length - 0.01, 0.10, 0.14))
                    print(f"目标腿长减少，当前腿长为: {self.target_length:.3f}")
            
            # 处理手柄按键按下事件
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 4:  # LB
                    self.target_length = float(np.clip(self.target_length - 0.005, 0.10, 0.14))
                    print(f"手柄按键 4 被按下，目标腿长减少，当前腿长为: {self.target_length:.3f}")
                elif event.button == 5: # RB
                    self.target_length = float(np.clip(self.target_length + 0.005, 0.10, 0.14))
                    print(f"手柄按键 5 被按下，目标腿长增加，当前腿长为: {self.target_length:.3f}")

        # --- 读取键盘状态 ---
        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_UP]:
            self.target_v = 0.6
        elif keys_pressed[pygame.K_DOWN]:
            self.target_v = -0.6
        else:
            self.target_v = 0.0
            
        if keys_pressed[pygame.K_LEFT]:
            self.target_w = 2.0  
        elif keys_pressed[pygame.K_RIGHT]:
            self.target_w = -2.0
        else:
            self.target_w = 0.0
            
        # --- 读取手柄轴数据 ---
        if self.joystick is not None:
            deadzone = 0.1
            
            # 轴 1：左摇杆上下
            axis_1 = self.joystick.get_axis(1)
            if abs(axis_1) > deadzone:
                self.target_v = -axis_1 * 0.6
                
            # 轴 3：右摇杆左右
            axis_3 = self.joystick.get_axis(3)
            if abs(axis_3) > deadzone:
                self.target_w = -axis_3 * 2.0

        # --- 刷新控制面板 UI ---
        self.screen.fill((30, 30, 30))
        text1 = self.font.render(f"Ground_detection (ENTER) : {'ON' if self.control_enabled else 'OFF'}", True, (255, 255, 255))
        text2 = self.font.render(f"Target V (UP/DOWN): {self.target_v:.2f} | Cur: {current_v:.2f}", True, (255, 255, 255))
        text3 = self.font.render(f"Leg L: {target_length_left:.3f} | R: {target_length_right:.3f} | Roll: {roll:.2f}", True, (255, 255, 255))
        text4 = self.font.render(f"Paused (SPACE)    : {self.paused}", True, (255, 255, 255))
        
        self.screen.blit(text1, (20, 30))
        self.screen.blit(text2, (20, 80))
        self.screen.blit(text3, (20, 130))
        self.screen.blit(text4, (20, 180))
        pygame.display.flip()