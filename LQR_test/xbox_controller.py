import pygame
import sys

# 初始化 pygame
pygame.init()

# 设置手柄
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"检测到手柄: {joystick.get_name()}")
else:
    print("未检测到手柄")
    pygame.quit()
    sys.exit()

# 设置窗口
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Xbox 手柄测试")

# 字体设置
font = pygame.font.Font(None, 36)

# 主循环
running = True
while running:
    screen.fill((30, 30, 30))  # 设置背景颜色

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 获取轴的数量
    num_axes = joystick.get_numaxes()
    
    # 打印手柄所有轴的值，并显示在屏幕左侧
    for i in range(num_axes):
        axis_value = joystick.get_axis(i)
        axis_text = font.render(f"轴 {i}: {axis_value:.2f}", True, (255, 255, 255))
        screen.blit(axis_text, (20, 20 + i * 40))

    # 获取按键的数量
    num_buttons = joystick.get_numbuttons()

    # 打印手柄所有按键的值，分列显示在屏幕右边
    for i in range(num_buttons):
        button_value = joystick.get_button(i)
        # 按钮按下时显示绿色，否则显示白色
        color = (0, 255, 0) if button_value else (200, 200, 200)
        btn_text = font.render(f"按键 {i}: {button_value}", True, color)
        
        # 将按键分成多列显示，每列最多放 9 个
        col = i // 9
        row = i % 9
        screen.blit(btn_text, (200 + col * 150, 20 + row * 40))

    # 刷新屏幕
    pygame.display.flip()

    # 设置延迟，防止刷新过快
    pygame.time.wait(100)

pygame.quit()

