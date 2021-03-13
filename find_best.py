import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default=None, required=True, type=str, help="folder path to find")
args= parser.parse_args()
model_path = args.model_dir
hyperparam_set = 5 #num of hyper-parameters sets to print out
evalset_list = []
for lr in os.listdir(model_path):
    lr_path = os.path.join(model_path,lr)
    print(lr_path)
    for coef in os.listdir(lr_path):
        coef_path = os.path.join(lr_path, coef)
        print(coef_path)
        eval_file = os.path.join(coef_path,"eval_dev_results.txt")
        print(eval_file)
        with open(eval_file, 'r') as f:
            while True:
                line = f.readline()
                splitted_line = line.strip().split(" ")
                if len(splitted_line) == 3:
                    if splitted_line[0] == "mean_intent_slot":
                        l1 = splitted_line[2]
                    if splitted_line[0] == "intent_acc":
                        l2 = splitted_line[2]
                    if splitted_line[0] == "slot_f1":
                        l3 = splitted_line[2]
                    if splitted_line[0] == "semantic_frame_acc":
                        l4 = splitted_line[2]
                if not line:
                    break
            eval_set = (coef_path,l1,l2,l3,l4,)
            evalset_list.append(eval_set)
sorted_evalset_by_mean= sorted(evalset_list, key=lambda tup: tup[1], reverse = True)
print("Model mean_intent_slot  intent_acc  slot_F1  sentence")
for x in sorted_evalset_by_mean[:5]:
    print(x)
