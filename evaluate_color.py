import os
import argparse
import json

def arg_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output_dir", type=str, default='output_backup/output', help="Path to the output folder where the results will be saved.")
    parser.add_argument("--output_dir", type=str, default='output_intervlm', help="Path to the output folder where the results will be saved.")
    parser.add_argument("--eval_dir", type=str, default='evaluation_res_intervlm', help="Path to the evaluation folder where the results will be saved.")
    parser.add_argument("--template", type=int ,default=4, help="Number of the template to use.")
    return parser.parse_args()

def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data 

def find_color_acc(output_dir, template, gt_word=None):
    video_acc = {}
    for filename in os.listdir(os.path.join(output_dir, str(template))):
        input_folder = os.path.join(output_dir, str(template), filename)
        object_acc = []
        for subfile in os.listdir(input_folder):
            if subfile.endswith('.json'):
                data = load_json(os.path.join(input_folder, subfile))
                gt_color = gt_word if gt_word != None  else data["gt_color"]
                frames = data["frames"]
                pred = []
                for fr, sentence in frames.items():
                    pred.append(1 if gt_color in sentence else 0)
                object_acc.append(sum(pred)/len(pred))
                # print(sum(pred)/len(pred), gt_color)
        video_acc[filename] = sum(object_acc)/len(object_acc)
    return sum(video_acc.values())/len(video_acc)

def template_color(output_dir, template, gt_word=None):
    video_accuracy = find_color_acc(output_dir, template, gt_word)
    print(f"Template {template} accuracy: {video_accuracy}")
    return video_accuracy

def model_acc(output_dir):
    # acc_t1 = template_color(output_dir, 1)
    # acc_t2 = template_color(output_dir, 2)
    # acc_t3 = template_color(output_dir, 3)
    # total_acc = (acc_t1 + acc_t2 + acc_t3)/3
    total_acc={}

    color_confusion = template_color(output_dir, 4, gt_word="Yes")
    return total_acc, color_confusion

if __name__ == "__main__":
    args = arg_parser()
    os.makedirs(args.eval_dir, exist_ok=True)
    models_2 = []
    models_5 = []


    for models in os.listdir(args.output_dir):
        for sec in os.listdir(os.path.join(args.output_dir, models)):
            acc, color_confusion = model_acc(os.path.join(args.output_dir,models,str(sec)))
            print(f"Model {models} with duration {sec}secs has accuracy: {acc}")
            if sec == "2.0":
                models_2.append((models, sec, acc, color_confusion))
            else:
                models_5.append((models, sec, acc, color_confusion))
            # break

    with open(os.path.join(args.eval_dir, "evaluation.txt"), "w") as f:
        f.write("| Model Name | Duration | Accuracy | Color Confusion|\n")
        print("| Model Name | Duration | Accuracy | Color Confusion|")
        for model in models_2:
            f.write(f"| {model[0]} | {model[1]} | {model[2]} | {model[3]} |\n")
            print(f"| {model[0]} | {model[1]} | {model[2]} | {model[3]} |")
        print("------")
        for model in models_5:
            f.write(f"| {model[0]} | {model[1]} | {model[2]} | {model[3]} |\n")
            print(f"| {model[0]} | {model[1]} | {model[2]} | {model[3]} |")
    print("Evaluation saved in evaluation.txt")



