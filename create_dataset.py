import json
import os
import random

with open("result_splited.json", "r", encoding="utf-8") as f:
    data = json.load(f)

context_window = 10
window = 20
chat_history = 20

all_question_ids = []
for i, message in enumerate(data):
    if message["from"] == "Соня" and '?' in message["text"]:
        all_question_ids.append(i)

if os.path.exists("dataset.json"):
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
else:
    dataset = []

random.seed(42)
random.shuffle(all_question_ids)

# Remove questions that are already in the dataset
question_ids = []
for i, id in enumerate(all_question_ids):
    question = data[id]['text']
    if question in [d['question'] for d in dataset]:
        continue
    question_ids.append(id)
for i, id in enumerate(question_ids):
    question = data[id]['text']
    print(f'Вопрос {i+1}/{len(question_ids)}:\n')
    print(f'Cоня: {question}\n')

    counter = 0
    answers = []
    j = 0
    while counter <= window and id + j < len(data):
        if data[id + j]['from'] == 'Леша':
            answers.append(data[id + j]['text'])
            counter += 1
        j += 1
    print('\n'.join([f'{j+1}. {answer}' for j, answer in enumerate(answers)]) + '\n')
    inp = input('Выберите ответ: ').strip().split()

    if len(inp) == 1:
        if inp[0] == '-1':
            with open("dataset.json", "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            exit()
        elif inp[0] == '0':
            answers = ""
        elif str(inp[0]) == '?':
            print(f'\nПредыдущие {chat_history} сообщений диалога:')
            print('\n'.join([f'{data[k]['from']}: {data[k]['text']}' for k in range(max(0, id - chat_history), id)]) + '\n')
            print(f'Cоня: {question}\n')
            print('\n'.join([f'{j+1}. {answer}' for j, answer in enumerate(answers)]) + '\n')
            inp = input('Выберите ответ: ').strip().split()

            if len(inp) == 1:
                if inp[0] == '-1':
                    with open("dataset.json", "w", encoding="utf-8") as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)
                    exit()
                elif inp[0] == '0':
                    answers = ""

    ids = list(map(int, inp))


    new_data = {
        'question': question,
        'answers': '\n'.join([answers[id - 1] for id in ids]),
        'context': '\n'.join([f'{data[k]['from']}: {data[k]['text']}' for k in range(max(0, id - context_window), id)]),
    }
    dataset.append(new_data)

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print('-'*50 + '\n')
    # break

