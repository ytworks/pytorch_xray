import utils


# -2. configの設定
# -1. cudaの設定
# 0. Modelの設定　(from inference)
# 0.5. dataloader
# 1. Lossの設定
# 2. preprocess
# 3. Loss(pred, true)
# 4. no grad
# 5. backward
# 6. opt
# 7. eval
# save

def main():
    # Configの読み込み (utils)
    ini, debug_mode = utils.config.read_config()
    print("Debug mode:", debug_mode)
    


if __name__=='__main__':
    main()
