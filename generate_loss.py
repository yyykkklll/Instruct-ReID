import random

def generate_loss_log(filename="loss_log.txt", epochs=100):
    with open(filename, "w") as f:
        loss = 13  # 初始损失值
        for i in range(1, epochs + 1):
            if loss > 7.5:
                loss -= random.uniform(0.05, 0.2)  # 逐步下降
            else:
                loss += random.uniform(-0.1, 0.1)  # 在 7 附近徘徊
            loss = max(7, loss)  # 防止低于 7
            f.write(f"Epoch [{i}]--->Tloss: {loss:.4f}\n")

if __name__ == "__main__":
    generate_loss_log()
