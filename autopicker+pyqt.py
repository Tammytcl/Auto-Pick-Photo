import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QComboBox, QSpinBox, 
                             QGroupBox, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

class ImageComparisonTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像对比工具 (支持Prompt信息)")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量
        self.original_dir = ""
        self.our_dir = ""
        self.other_dir = ""
        self.current_index = 0
        self.image_files = []
        self.distance_method = "L2"
        self.image_distances = []  # 用于存储计算的距离结果
        
        # 创建主界面
        self.init_ui()
        
    def init_ui(self):
        # 主窗口布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()
        
        # 文件夹选择
        self.btn_select_original = QPushButton("选择Original文件夹")
        self.btn_select_original.clicked.connect(lambda: self.select_folder("original"))
        self.btn_select_our = QPushButton("选择Our文件夹")
        self.btn_select_our.clicked.connect(lambda: self.select_folder("our"))
        self.btn_select_other = QPushButton("选择Other文件夹")
        self.btn_select_other.clicked.connect(lambda: self.select_folder("other"))
        
        # 对比方法选择
        self.method_combo = QComboBox()
        self.method_combo.addItems(["L1距离", "L2距离", "PSNR", "SSIM"])
        self.method_combo.setCurrentText("L2距离")
        self.method_combo.currentTextChanged.connect(self.update_distance_method)
        
        # 图片导航
        self.spin_index = QSpinBox()
        self.spin_index.setMinimum(1)
        self.spin_index.valueChanged.connect(self.go_to_image)
        
        self.btn_prev = QPushButton("上一张")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("下一张")
        self.btn_next.clicked.connect(self.next_image)
        
        # 添加到控制面板
        control_layout.addWidget(self.btn_select_original)
        control_layout.addWidget(self.btn_select_our)
        control_layout.addWidget(self.btn_select_other)
        control_layout.addWidget(QLabel("对比方法:"))
        control_layout.addWidget(self.method_combo)
        control_layout.addWidget(QLabel("当前图片:"))
        control_layout.addWidget(self.spin_index)
        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.btn_next)
        control_group.setLayout(control_layout)
        
        # 筛选功能面板
        filter_group = QGroupBox("筛选功能")
        filter_layout = QHBoxLayout()
        
        self.filter_method_combo = QComboBox()
        self.filter_method_combo.addItems(["L1距离", "L2距离", "PSNR", "SSIM"])
        self.filter_method_combo.setCurrentText("L2距离")
        
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setMinimum(1)
        self.top_n_spin.setMaximum(500)
        self.top_n_spin.setValue(10)
        
        self.btn_find_best = QPushButton("筛选Our最接近Original且Other最远离Original的图像")
        self.btn_find_best.clicked.connect(self.find_best_difference)
        
        filter_layout.addWidget(QLabel("筛选方法:"))
        filter_layout.addWidget(self.filter_method_combo)
        filter_layout.addWidget(QLabel("显示数量:"))
        filter_layout.addWidget(self.top_n_spin)
        filter_layout.addWidget(self.btn_find_best)
        filter_group.setLayout(filter_layout)
        
        # 图像显示区域
        image_group = QGroupBox("图像对比")
        image_layout = QHBoxLayout()
        
        # Original 图像
        self.original_group = QGroupBox("Original")
        original_inner = QVBoxLayout()
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_prompt = QLabel("Prompt信息将显示在这里")
        self.original_prompt.setWordWrap(True)
        original_inner.addWidget(self.original_label)
        original_inner.addWidget(self.original_prompt)
        self.original_group.setLayout(original_inner)
        
        # Our 图像
        self.our_group = QGroupBox("Our")
        our_inner = QVBoxLayout()
        self.our_label = QLabel()
        self.our_label.setAlignment(Qt.AlignCenter)
        self.our_label.setMinimumSize(400, 400)
        self.our_distance = QLabel("距离: -")
        our_inner.addWidget(self.our_label)
        our_inner.addWidget(self.our_distance)
        self.our_group.setLayout(our_inner)
        
        # Other 图像
        self.other_group = QGroupBox("Other")
        other_inner = QVBoxLayout()
        self.other_label = QLabel()
        self.other_label.setAlignment(Qt.AlignCenter)
        self.other_label.setMinimumSize(400, 400)
        self.other_distance = QLabel("距离: -")
        other_inner.addWidget(self.other_label)
        other_inner.addWidget(self.other_distance)
        self.other_group.setLayout(other_inner)
        
        # 添加到图像区域
        image_layout.addWidget(self.original_group)
        image_layout.addWidget(self.our_group)
        image_layout.addWidget(self.other_group)
        image_group.setLayout(image_layout)
        
        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(["文件名", "Our距离", "Other距离", "差值(Other-Our)", "Prompt"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.result_table.cellClicked.connect(self.table_item_clicked)
        
        # 添加到主布局
        main_layout.addWidget(control_group)
        main_layout.addWidget(filter_group)
        main_layout.addWidget(image_group)
        main_layout.addWidget(self.result_table)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 状态栏
        self.statusBar().showMessage("准备好选择文件夹")
        
    def select_folder(self, folder_type):
        folder = QFileDialog.getExistingDirectory(self, f"选择{folder_type}文件夹")
        if folder:
            if folder_type == "original":
                self.original_dir = folder
                self.btn_select_original.setText(f"Original: {os.path.basename(folder)}")
            elif folder_type == "our":
                self.our_dir = folder
                self.btn_select_our.setText(f"Our: {os.path.basename(folder)}")
            elif folder_type == "other":
                self.other_dir = folder
                self.btn_select_other.setText(f"Other: {os.path.basename(folder)}")
            
            self.check_folders_ready()
    
    def check_folders_ready(self):
        if self.original_dir and self.our_dir and self.other_dir:
            self.load_image_list()
            self.statusBar().showMessage("文件夹已加载，可以开始对比")
    
    def load_image_list(self):
        # 获取三个文件夹共有的图像文件
        original_files = set(f for f in os.listdir(self.original_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        our_files = set(f for f in os.listdir(self.our_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        other_files = set(f for f in os.listdir(self.other_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        
        common_files = original_files & our_files & other_files
        self.image_files = sorted(common_files)
        
        if self.image_files:
            self.spin_index.setMaximum(len(self.image_files))
            self.current_index = 0
            self.spin_index.setValue(1)
            self.show_current_image()
        else:
            self.statusBar().showMessage("没有找到共有的图像文件")
    
    def update_distance_method(self, method):
        mapping = {
            "L1距离": "L1",
            "L2距离": "L2",
            "PSNR": "PSNR",
            "SSIM": "SSIM"
        }
        self.distance_method = mapping.get(method, "L2")
        if self.image_files:
            self.show_current_image()
    
    def show_current_image(self):
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        filename = self.image_files[self.current_index]
        
        # 加载并显示Original图像
        original_path = os.path.join(self.original_dir, filename)
        self.show_image(original_path, self.original_label)
        self.original_prompt.setText(self.extract_prompt(original_path))
        
        # 加载并显示Our图像
        our_path = os.path.join(self.our_dir, filename)
        self.show_image(our_path, self.our_label)
        our_distance = self.calculate_distance(original_path, our_path)
        self.our_distance.setText(f"距离 ({self.method_combo.currentText()}): {our_distance:.4f}")
        
        # 加载并显示Other图像
        other_path = os.path.join(self.other_dir, filename)
        self.show_image(other_path, self.other_label)
        other_distance = self.calculate_distance(original_path, other_path)
        self.other_distance.setText(f"距离 ({self.method_combo.currentText()}): {other_distance:.4f}")
        
        # 更新状态栏
        self.statusBar().showMessage(f"正在显示: {filename} ({self.current_index + 1}/{len(self.image_files)})")
    
    def show_image(self, path, label):
        image = cv2.imread(path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))
    
    def calculate_distance(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return float('nan')
        
        # 调整尺寸一致
        if img1.shape != img2.shape:
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        
        if self.distance_method == "L1":
            return np.mean(np.abs(img1.astype(float) - img2.astype(float)))
        elif self.distance_method == "L2":
            return np.sqrt(np.mean((img1.astype(float) - img2.astype(float)) ** 2))
        elif self.distance_method == "PSNR":
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            if mse == 0:
                return float('inf')
            return 10 * np.log10(255.0 ** 2 / mse)
        elif self.distance_method == "SSIM":
            # 简化的SSIM计算
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1_sq = np.var(img1)
            sigma2_sq = np.var(img2)
            sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
            
            ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
            return 1 - ssim  # 转换为距离
        
        return float('nan')
    
    def extract_prompt(self, image_path):
        try:
            img = Image.open(image_path)
            if img.format == 'PNG':
                metadata = img.info
                prompt = metadata.get("prompt", "无Prompt信息")
            else:
                exif_data = img._getexif()
                if exif_data:
                    prompt = {TAGS.get(tag): value for tag, value in exif_data.items()}.get("ImageDescription", "无Prompt信息")
                else:
                    prompt = "无Prompt信息"
            return prompt
        except Exception as e:
            return f"无法提取元数据: {e}"
    
    def go_to_image(self, index):
        if 1 <= index <= len(self.image_files):
            self.current_index = index - 1
            self.show_current_image()
    
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.spin_index.setValue(self.current_index + 1)
    
    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.spin_index.setValue(self.current_index + 1)
    
    def find_best_difference(self):
        """
        查找同时满足条件的图像：
        1. Our方法最接近Original
        2. Other方法最远离Original
        3. 两者差距最大
        """
        if not self.image_files:
            QMessageBox.warning(self, "警告", "请先加载图像文件")
            return
        
        # 获取筛选方法
        method_mapping = {
            "L1距离": "L1",
            "L2距离": "L2",
            "PSNR": "PSNR",
            "SSIM": "SSIM"
        }
        filter_method = method_mapping.get(self.filter_method_combo.currentText(), "L2")
        
        # 保存当前方法
        current_method = self.distance_method
        # 临时切换到筛选方法
        self.distance_method = filter_method
        
        # 计算所有图像的距离
        self.image_distances = []
        self.statusBar().showMessage("正在计算距离...")
        
        for filename in self.image_files:
            original_path = os.path.join(self.original_dir, filename)
            our_path = os.path.join(self.our_dir, filename)
            other_path = os.path.join(self.other_dir, filename)
            
            our_distance = self.calculate_distance(original_path, our_path)
            other_distance = self.calculate_distance(original_path, other_path)
            
            # 计算差距
            diff = other_distance - our_distance
            prompt = self.extract_prompt(original_path)
            
            self.image_distances.append((filename, our_distance, other_distance, diff, prompt))
        
        # 恢复当前方法
        self.distance_method = current_method
        
        # 根据指标类型确定排序
        # 对于PSNR，值越大表示越相似；对于其他方法，值越小表示越相似
        if filter_method == "PSNR":
            # 我们希望：Our的PSNR高(更好)，Other的PSNR低(更差)，差值大
            # 创建复合排序键: (our_quality, other_badness, difference)
            # 对于PSNR: our_quality = our_distance (越大越好)
            # 对于PSNR: other_badness = -other_distance (越小越差)
            self.image_distances.sort(key=lambda x: (x[1], -x[2], x[3]), reverse=True)
        else:
            # 对于其他距离度量(L1, L2, SSIM as distance)，值越小表示越相似
            # 我们希望：Our的距离小(更好)，Other的距离大(更差)，差值大
            # 对于L1/L2/SSIM距离: our_quality = -our_distance (越小越好)
            # 对于L1/L2/SSIM距离: other_badness = other_distance (越大越差)
            self.image_distances.sort(key=lambda x: (-x[1], x[2], x[3]), reverse=True)
        
        # 显示结果
        top_n = min(self.top_n_spin.value(), len(self.image_distances))
        filtered_results = self.image_distances[:top_n]
        
        # 更新表格
        self.result_table.setRowCount(top_n)
        for row, (filename, our_distance, other_distance, diff, prompt) in enumerate(filtered_results):
            self.result_table.setItem(row, 0, QTableWidgetItem(filename))
            
            our_item = QTableWidgetItem(f"{our_distance:.4f}")
            self.result_table.setItem(row, 1, our_item)
            
            other_item = QTableWidgetItem(f"{other_distance:.4f}")
            self.result_table.setItem(row, 2, other_item)
            
            diff_item = QTableWidgetItem(f"{diff:.4f}")
            self.result_table.setItem(row, 3, diff_item)
            
            prompt_item = QTableWidgetItem(prompt)
            self.result_table.setItem(row, 4, prompt_item)
        
        message = "筛选完成: Our方法最接近Original且Other方法最远离Original的图像"
        self.statusBar().showMessage(message)
    
    def table_item_clicked(self, row, column):
        """当表格项被点击时，显示对应的图像"""
        if row < len(self.image_distances):
            filename = self.image_distances[row][0]
            # 找到对应文件在image_files中的索引
            try:
                new_index = self.image_files.index(filename)
                self.current_index = new_index
                self.spin_index.setValue(new_index + 1)
            except ValueError:
                pass  # 如果找不到文件，不做任何操作

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageComparisonTool()
    window.show()
    sys.exit(app.exec_())
