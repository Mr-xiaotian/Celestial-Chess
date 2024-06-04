from PIL import Image, ImageDraw, ImageFont

# 创建一个大小为 64x64 的图像，背景为透明
img = Image.new('RGBA', (48, 48), color=(255, 255, 255, 0))

# 初始化绘图工具
d = ImageDraw.Draw(img)

# 设置字体样式和大小（使用PIL内置的字体）
try:
    # 尝试使用更小尺寸的字体
    font = ImageFont.truetype("arial.ttf", 48)  # 调整字体大小以适应图标大小
except IOError:
    # 备用选项：简单的内置字体
    font = ImageFont.load_default()

# 绘制两个'C'，使它们紧密贴合
d.text((12, 0), "C", font=font, fill="lightcoral")
d.text((0, 0), "C", font=font, fill="#6495ED")


# 保存图像
img.save('Custom_CC_Favicon.png')

# 显示图像
img.show()
