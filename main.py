from transformers import AutoTokenizer, AutoConfig
from fine_tuning_and_training import DMModel, id2marker
import torch

# Load model and tokenizer
model_name = "./RussianDMrecognizer"
tokenizer = AutoTokenizer.from_pretrained("./RussianDMrecognizer")
config = AutoConfig.from_pretrained("./RussianDMrecognizer")

model = DMModel(config=config, num_marker_labels=len(id2marker))
model.load_state_dict(torch.load("./RussianDMrecognizer/pytorch_model.bin"))

# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")

# Function to test any sentence
def classify_marker(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        marker_pred_id = torch.argmax(outputs.marker_logits, dim=-1).item()
        binary_pred = torch.argmax(outputs.binary_logits, dim=-1).item()

    marker_word = id2marker.get(marker_pred_id, "UNKNOWN")
    is_discourse_marker = binary_pred == 1

    return marker_word, is_discourse_marker
# Try it
test_sents = [
    "Видимо дверь была плохо закрыта — вот и хлопнула.",
    "Напротив, некоторые участники высказались резко против.",
    "Он, вероятно, просто не понял инструкции.",
    "По-моему, это было отличное выступление.",
    "Итак, мы пришли к главному вопросу.",
    "Он выбрал вариант по-моему вкусу.",
    "Словом, решение было принято единогласно.",
    "Пожалуй, стоит начать подготовку заранее.",
    "Она работает в Итак — новой технологической компании.",
    "Напротив дома стоял странный фургон.",
    "Пожалуй номер восемь был самым неожиданным.",
    "Словом «сила» можно описать их стремление.",
    "Это мнение, по-моему, не отражает всей картины.",
    "Конечно, я помогу тебе с проектом.",
    "Он конечно плотник, но больше похож на скульптора.",
    "Во-первых, нужно изучить материалы, а уже потом решать задачи.",
    "Конечно было темно, но не страшно.",
    "Впрочем, никто и не ожидал другого исхода.",
    "Он работает в компании «Впрочем» — смешное совпадение.",
    "Видимо, встреча перенеслась на завтра."

]

for sent in test_sents:
    marker, is_marker = classify_marker(sent)
    print(f"'{sent}' → Marker: {marker} | {'✔️ DM' if is_marker else '✖️ not DM'}")


