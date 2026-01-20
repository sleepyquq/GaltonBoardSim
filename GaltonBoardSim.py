import pygame
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.stats import norm

# 设置 matplotlib 为非交互模式
matplotlib.use("Agg")

# --- 常量定义 ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
PANEL_WIDTH = 400

# 布局关键常量
FLOOR_OFFSET = 30          # 地板距离窗口底部的距离
MAX_BIN_HEIGHT = 220       # 收集框允许的最大视觉高度 (包含堆积区)
BIN_PEG_GAP = 40           # 钉子最底层与收集框顶部的安全间距

BG_COLOR = (30, 30, 35)
PANEL_COLOR = (50, 50, 60)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (0, 150, 255)
BALL_COLOR = (255, 100, 100)
PEG_COLOR = (200, 200, 200)
BIN_WALL_COLOR = (80, 80, 90)

FPS = 60

# --- UI 组件 (滑条) ---
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial, label, int_only=True):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial
        self.label = label
        self.int_only = int_only
        self.dragging = False
        self.handle_rect = pygame.Rect(x, y - 5, 10, h + 10)
        self.update_handle()

    def update_handle(self):
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        self.handle_rect.centerx = self.rect.x + ratio * self.rect.width

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        
        if self.dragging and event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION]:
            mouse_x = pygame.mouse.get_pos()[0]
            ratio = (mouse_x - self.rect.x) / self.rect.width
            ratio = max(0, min(1, ratio))
            self.val = self.min_val + ratio * (self.max_val - self.min_val)
            if self.int_only:
                self.val = int(round(self.val))
            self.update_handle()
            return True
        return False

    def draw(self, surface, font):
        label_surf = font.render(f"{self.label}: {self.val}", True, TEXT_COLOR)
        surface.blit(label_surf, (self.rect.x, self.rect.y - 25))
        pygame.draw.rect(surface, (100, 100, 100), self.rect, border_radius=5)
        fill_rect = list(self.rect)
        fill_rect[2] = self.handle_rect.centerx - self.rect.x
        pygame.draw.rect(surface, ACCENT_COLOR, fill_rect, border_radius=5)
        pygame.draw.rect(surface, (255, 255, 255), self.handle_rect, border_radius=5)

# --- 核心实体类：小球 ---

class Ball:
    def __init__(self, x, y, radius, level_height, speed_mult, bin_width, start_row_y, peg_spacing, rows):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = 0
        self.vy = 0
        self.target_row = 0
        self.state = "falling"
        self.level_height = level_height
        self.speed_mult = speed_mult
        
        self.gravity = 0.5 * speed_mult
        self.bin_width = bin_width
        self.start_row_y = start_row_y
        self.peg_spacing = peg_spacing
        self.total_rows = rows
        
        self.vx = 0

    def update(self, bins, floor_y, bin_start_x):
        if self.state == "stacked":
            return True

        # 1. 物理运动
        self.vy += self.gravity
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98

        # --- 2. 左右边界约束 ---
        min_x = PANEL_WIDTH + self.radius + 2
        max_x = SCREEN_WIDTH - self.radius - 2
        
        if self.x < min_x:
            self.x = min_x
            self.vx *= -0.5 
        elif self.x > max_x:
            self.x = max_x
            self.vx *= -0.5

        # --- 3. 钉子碰撞逻辑 ---
        if self.target_row < self.total_rows:
            current_row_y = self.start_row_y + self.target_row * self.level_height
            if self.y >= current_row_y - self.radius:
                direction = 1 if random.random() < 0.5 else -1
                force = (1.5 + random.random()) * self.speed_mult
                self.vx += direction * force
                self.y = current_row_y - self.radius
                self.vy *= 0.4 
                self.target_row += 1
        
        # --- 4. 底部收集判定 (吸附逻辑) ---
        if self.y > floor_y - 150: 
            relative_x = self.x - bin_start_x
            phys_bin_idx = int(relative_x // self.bin_width)
            phys_bin_idx = max(0, min(phys_bin_idx, len(bins) - 1))
            
            if self.y >= floor_y - self.radius:
                # 强制吸附到槽中心
                bin_center_x = bin_start_x + phys_bin_idx * self.bin_width + self.bin_width / 2
                jitter = random.uniform(-self.radius/2, self.radius/2)
                self.x = bin_center_x + jitter
                
                self.y = floor_y - self.radius
                self.state = "stacked"
                bins[phys_bin_idx] += 1
                return True 

        return False

# --- 主程序类 ---

class GaltonBoardSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Galton Board Sim - Explore Gaussian Distribution")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.title_font = pygame.font.SysFont("Arial", 20, bold=True)

        self.board_rect = pygame.Rect(PANEL_WIDTH, 0, SCREEN_WIDTH - PANEL_WIDTH, SCREEN_HEIGHT)

        # 控件
        self.slider_count = Slider(50, 80, 300, 20, 20, 3000, 1000, "Total Balls")
        self.slider_levels = Slider(50, 150, 300, 20, 6, 20, 12, "Rows (Levels)")
        self.slider_speed = Slider(50, 220, 300, 20, 0, 0.7, 0.35, "Gravity/Speed", int_only=False)  #下限 上限 初始值
        self.sliders = [self.slider_count, self.slider_levels, self.slider_speed]

        # 数据状态
        self.balls = []
        self.pegs = [] 
        self.bins = [] 
        self.bin_lines = []
        self.spawning_active = True
        self.balls_spawned = 0
        
        # 布局缓存变量
        self.floor_y = SCREEN_HEIGHT - FLOOR_OFFSET
        self.bin_start_x = 0
        self.bin_width = 0
        self.start_row_y = 0
        self.level_height = 0
        self.peg_spacing = 0
        self.ball_radius = 0
        
        self.graph_surf = None
        self.needs_graph_update = True
        
        self.reset_board()

    def reset_board(self):
        self.balls = []
        self.balls_spawned = 0
        self.spawning_active = True
        
        rows = self.slider_levels.val
        self.bins = [0] * (rows + 1)
        
        # --- 严格的布局计算 ---
        
        # 1. 确定底部和顶部边界
        margin_top = 80
        # 底部留给 collecting bins 的绝对空间 = Max Bin Height + Gap
        # 这样钉子绝对不会画在这个区域内
        reserved_bottom_height = MAX_BIN_HEIGHT + BIN_PEG_GAP + FLOOR_OFFSET
        
        # 2. 计算钉子板可用的垂直空间
        available_h = SCREEN_HEIGHT - margin_top - reserved_bottom_height
        
        # 3. 计算每一层的高度
        # 注意：这里 rows 表示间隔数，实际钉子排数是 rows? 不，通常是 rows 层障碍
        # 我们让最底下一层钉子的 Y 坐标 = start_y + (rows-1) * level_height
        # 加上最后一层小球掉落的空间
        self.level_height = available_h / rows
        
        # 4. 计算水平间距
        board_w = self.board_rect.width
        # 限制水平间距，防止层数少时钉子太稀疏，也防止层数多时太密
        # 同时受限于高度 (通常是一个等边或等腰三角形结构)
        self.peg_spacing = min(board_w / (rows + 4), self.level_height * 1.5)
        
        # 5. 确定球的大小 (自适应)
        self.ball_radius = max(3, min(8, int(self.peg_spacing / 3.5)))
        
        center_x = self.board_rect.x + board_w // 2
        self.start_row_y = margin_top

        # 6. 生成钉子坐标
        self.pegs = []
        for r in range(rows):
            row_y = self.start_row_y + r * self.level_height
            row_width = r * self.peg_spacing
            row_start_x = center_x - row_width / 2
            
            # 我们这里用简化的三角形排列：第r行有 r+1 个钉子
            for c in range(r + 1):
                px = row_start_x + c * self.peg_spacing
                self.pegs.append((px, row_y))

        # 7. 计算收集槽位置
        num_bins = rows + 1
        self.bin_width = self.peg_spacing
        total_bins_width = num_bins * self.bin_width
        self.bin_start_x = center_x - total_bins_width / 2
        
        self.bin_lines = []
        for i in range(num_bins + 1):
            x = self.bin_start_x + i * self.bin_width
            self.bin_lines.append(x)
        
        self.update_graph()

    def update_graph(self):
        fig = plt.figure(figsize=(4, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('#1E1E23')
        ax.set_facecolor('#32323C')
        
        rows = self.slider_levels.val
        x = np.arange(rows + 1)
        counts = np.array(self.bins)
        total_landed = sum(counts)
        
        var_real = 0
        var_theo = 0
        
        if total_landed > 0:
            density = counts / total_landed
            ax.bar(x, density, color='#0096FF', alpha=0.8, width=0.8, label='Simulation')
            mean_real = np.average(x, weights=counts)
            var_real = np.average((x - mean_real)**2, weights=counts)
        else:
            ax.bar(x, counts, color='#0096FF', alpha=0.8)

        n = rows
        p = 0.5
        mu = n * p
        var_theo = n * p * (1 - p)
        sigma = math.sqrt(var_theo)
        x_smooth = np.linspace(0, rows, 100)
        
        if total_landed > 0:
            y_smooth = norm.pdf(x_smooth, mu, sigma)
            ax.plot(x_smooth, y_smooth, 'r-', linewidth=2.5, label='Theory')
        
        # --- 拟合度判断逻辑 ---
        # 如果真实方差比理论方差大很多，说明球竖直/水平惯性过大，不符合i.i.d的假设
        fit_color = 'white'
        status_text = "Status: Gathering Data..."
        
        if total_landed > 50:
            diff_ratio = abs(var_real - var_theo) / var_theo
            if diff_ratio < 0.15:
                status_text = "Status: Perfect Fit!"
                fit_color = '#00FF00' # Green
            elif diff_ratio < 0.5:
                status_text = "Status: Deviating..."
                fit_color = '#FFFF00' # Yellow
            else:
                status_text = "Status: High Inertia!"
                fit_color = '#FF5555' # Red

        ax.set_title("Distribution Analysis", color='white', fontsize=10)
        ax.tick_params(axis='x', colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.spines['left'].set_visible(True); ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_visible(True); ax.spines['bottom'].set_color('white')
        
        info_text = (f"Landed: {total_landed}\n"
                     f"Var(Real): {var_real:.2f}\n"
                     f"Var(Theo): {var_theo:.2f}")
        
        # 基础数据框
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', color='white', fontsize=9,
                bbox=dict(boxstyle="round", facecolor='#404040', edgecolor='none', alpha=0.7))
        
        # 状态提示框 (右侧)
        ax.text(0.98, 0.98, status_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', 
                color=fit_color, fontsize=9, fontweight='bold')

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        self.graph_surf = pygame.image.fromstring(raw_data, size, "RGB")
        plt.close(fig)
        self.needs_graph_update = False

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            
            slider_changed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                for slider in self.sliders:
                    if slider.handle_event(event):
                        slider_changed = True
            
            if slider_changed:
                self.reset_board()

            # --- 生成与更新 ---
            max_balls = self.slider_count.val
            if self.spawning_active and self.balls_spawned < max_balls:
                spawn_rate = int(1 + self.slider_speed.val * 3)
                for _ in range(spawn_rate):
                    if self.balls_spawned < max_balls:
                        start_x = self.pegs[0][0] + random.uniform(-1, 1)
                        ball = Ball(start_x, 30, self.ball_radius, self.level_height, self.slider_speed.val,
                                   self.bin_width, self.start_row_y, self.peg_spacing, self.slider_levels.val)
                        self.balls.append(ball)
                        self.balls_spawned += 1
            elif self.balls_spawned >= max_balls and len(self.balls) == 0:
                self.spawning_active = False

            active_balls = []
            landed_this_frame = False
            
            for ball in self.balls:
                just_landed = ball.update(self.bins, self.floor_y, self.bin_start_x)
                if just_landed:
                    landed_this_frame = True
                if ball.state == "falling":
                    active_balls.append(ball)
            
            self.balls = active_balls

            if landed_this_frame:
                self.needs_graph_update = True

            # --- 绘制 ---
            self.screen.fill(BG_COLOR)
            
            # 左侧面板
            pygame.draw.rect(self.screen, PANEL_COLOR, (0, 0, PANEL_WIDTH, SCREEN_HEIGHT))
            pygame.draw.line(self.screen, (20, 20, 25), (PANEL_WIDTH, 0), (PANEL_WIDTH, SCREEN_HEIGHT), 3)
            title_surf = self.title_font.render("Galton Board Controls", True, ACCENT_COLOR)
            self.screen.blit(title_surf, (20, 20))
            for slider in self.sliders: slider.draw(self.screen, self.font)
            pygame.draw.line(self.screen, (80, 80, 80), (20, 300), (380, 300), 1)
            chart_title = self.title_font.render("Real-time Statistics", True, ACCENT_COLOR)
            self.screen.blit(chart_title, (20, 320))
            if self.needs_graph_update: self.update_graph()
            if self.screen and self.graph_surf: self.screen.blit(self.graph_surf, (0, 360))

            # --- 右侧绘制 (重叠修复核心) ---
            
            # 1. 绘制容器壁 (使用 MAX_BIN_HEIGHT 限制高度)
            for x in self.bin_lines:
                pygame.draw.line(self.screen, BIN_WALL_COLOR, (x, self.floor_y), (x, self.floor_y - MAX_BIN_HEIGHT), 2)
            pygame.draw.line(self.screen, BIN_WALL_COLOR, (self.bin_lines[0], self.floor_y), (self.bin_lines[-1], self.floor_y), 2)

            # 2. 绘制钉子
            for px, py in self.pegs:
                pygame.draw.circle(self.screen, PEG_COLOR, (int(px), int(py)), 3)

            # 3. 堆积球 (动态压缩，且受 MAX_BIN_HEIGHT 限制)
            max_count = max(self.bins) if self.bins else 0
            physical_stack_height = max_count * (self.ball_radius * 2)
            
            compression_scale = 1.0
            # 如果堆积高度超过了最大视觉高度，进行压缩
            if physical_stack_height > MAX_BIN_HEIGHT:
                compression_scale = MAX_BIN_HEIGHT / physical_stack_height

            for i, count in enumerate(self.bins):
                center_x = self.bin_start_x + i * self.bin_width + self.bin_width / 2
                for c in range(count):
                    y_offset = c * (self.ball_radius * 2) * compression_scale
                    cy = self.floor_y - self.ball_radius - y_offset
                    pygame.draw.circle(self.screen, BALL_COLOR, (int(center_x), int(cy)), self.ball_radius - 1)

            # 4. 下落的球
            for ball in self.balls:
                pygame.draw.circle(self.screen, BALL_COLOR, (int(ball.x), int(ball.y)), ball.radius)

            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    app = GaltonBoardSim()
    app.run()