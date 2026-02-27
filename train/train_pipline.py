import torch
from train_data import start_train_data, save_train_data, load_train_data
from train_model import ModelTrainer, get_model_info_dict, save_model_info_dict


flag = torch.cuda.is_available()
print(flag)

if flag:
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3,3).cuda())

train_num = 10
mcts_iter = 1000
train_data = start_train_data(train_num, mcts_iter, execution_mode='serial')
train_data_path = save_train_data(train_data, train_num, mcts_iter)

# train_data_path = r"data\train_data\2025-11-20\MCTS1.0k_Games10.0k_Length133927(18-34).json"
# train_data = load_train_data(train_data_path)

trainer = ModelTrainer()
model_path, model = trainer.train_model(train_data, 10)

model_info_dict = get_model_info_dict(model_path, train_data_path, model)
save_model_info_dict(model_info_dict, "DeepLearningAI")

