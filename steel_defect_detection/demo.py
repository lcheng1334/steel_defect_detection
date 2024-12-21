from tensorborad_plt import smooth_plot



test_dice_path = "./train_log_pic/train/run-logs_log_pl-tag-train_loss_dice.csv"
test_loss_path = "./train_log_pic/train/run-logs_medium_precision_version_0-tag-train_loss.csv"
test_acc_top1 = "./train_log_pic/train/run-logs_medium_precision_version_0-tag-train_top-1_acc.csv"
test_acc_top3 = "./train_log_pic/train/run-logs_medium_precision_version_0-tag-train_top-3_acc.csv"
# smooth_plot(test_dice_path, "Step", "Dice", "Dice Loss", "loss", "test_dice.png")
smooth_plot(test_loss_path, "Step", "loss", "Test Loss", "loss", "test_loss.png")
# smooth_plot(test_acc_top1, "Step", "accuracy", "Accuracy Top1", "accuracy", "test_accuracy1.png")
# smooth_plot(test_acc_top3, "Step", "accuracy", "Accuracy Top3", "accuracy", "test_accuracy2.png")
# smooth_plot(test_acc_top3, "Step", "accuracy", "Accuracy Top3", "accuracy", "train_accuracy2.png")