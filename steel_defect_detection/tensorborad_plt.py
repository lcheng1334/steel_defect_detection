import pandas as pd
import matplotlib.pyplot as plt

def csv_plot(csv_path, x_label, y_label, title, legend, png_name):
    # 读取 CSV 文件
    csv_file = csv_path # 替换为实际 CSV 文件路径
    data = pd.read_csv(csv_file)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(data['Step'], data['Value'], label=legend)

    # 设置图形标题和标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 显示网格和图例
    plt.grid(True)
    plt.legend()
    plt.savefig(png_name)
    # 显示图形
    plt.show()

def smooth_plot(csv_path, x_label, y_label, title,legend, png_name):
    # 读取原始的 CSV 文件
    csv_file = csv_path  # 替换为实际 CSV 文件路径
    data = pd.read_csv(csv_file)

    # 进行平滑处理，使用 pandas 的 rolling mean
    window_size = 2  # 平滑窗口大小，可以根据需要调整
    data['Value'] = data['Value'].rolling(window=window_size).mean()

    # 将平滑后的数据保存为新的 CSV 文件
    # data.to_csv('smoothed_training_curve.csv', index=False)

    # 可视化原始数据和平滑后的数据
    plt.figure(figsize=(10, 6))
    # plt.plot(data['Step'], data['Value'], label='Original', alpha=0.6)
    plt.plot(data['Step'], data['Value'], label=legend)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(png_name)
    plt.show()


def csv_plot_multiple(csv_paths, x_label, y_label, title, legends, png_name):
    """
    绘制多个 CSV 文件数据到同一个图上。

    Parameters:
        csv_paths (list): CSV 文件路径的列表。
        x_label (str): X 轴标签。
        y_label (str): Y 轴标签。
        title (str): 图表标题。
        legends (list): 每条曲线的图例名称列表。
        png_name (str): 保存图像的文件名。
    """
    plt.figure(figsize=(10, 6))

    # 遍历每个 CSV 文件
    for csv_path, legend in zip(csv_paths, legends):
        data = pd.read_csv(csv_path)
        plt.plot(data["Step"], data["Value"], label=legend)

    # 设置图形标题和标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 显示网格和图例
    plt.grid(True)
    plt.legend()
    plt.savefig(png_name)
    plt.show()


def main():
    csv_files = ["./train_log_pic/train/run-logs_log_pl-tag-train_loss_dice.csv",
                 "./train_log_pic/train/run-logs_medium_precision_version_0-tag-train_loss.csv", 
                 "./train_log_pic/train/run-logs_medium_precision_version_0-tag-train_top-1_acc.csv",
                 "./train_log_pic/train/run-logs_medium_precision_version_0-tag-train_top-3_acc.csv"]  # 替换为实际 CSV 文件路径
    legends = ["train_loss", "train_loss", "top1", "top3"]  # 替换为对应的图例
    csv_plot_multiple(
    csv_files, 
    x_label="Step", 
    y_label="Value", 
    title="Training Curves", 
    legends=legends, 
    png_name="training_curves.png"
)
    pass
if __name__ == '__main__':
    main()
