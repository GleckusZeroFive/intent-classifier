"""
Dataset generator for intent classifier.

Expands 279 seed examples into ~3000+ training examples using:
1. Original seeds (high quality)
2. Template-based generation (medium quality, high volume)
3. Augmentation: typos, lowercase/uppercase, punctuation variations

Output: dataset.csv with columns [text, label]
"""

import csv
import random
import re
from pathlib import Path

from seeds import get_all_seeds

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# TEMPLATE-BASED GENERATION
# ═══════════════════════════════════════════════════════════════

# Pre-declined topic phrases: (nominative concept, ready-to-use prepositional phrases)
# Each tuple: (topic_name, "про X", "X-related question")
# This avoids broken Russian grammar from naive template substitution
_TOPICS_PHRASES = [
    ("возврат товара", "про возврат товара", "условия возврата товара"),
    ("доставка", "про доставку", "сроки доставки"),
    ("гарантия", "про гарантию", "условия гарантии"),
    ("оплата", "про оплату", "порядок оплаты"),
    ("штрафы", "про штрафы", "размер штрафов"),
    ("скидки", "про скидки", "условия получения скидки"),
    ("бонусы", "про бонусы", "начисление бонусов"),
    ("тарифы", "про тарифы", "стоимость тарифов"),
    ("подписка", "про подписку", "условия подписки"),
    ("страхование", "про страхование", "условия страхования"),
    ("кредитование", "про кредитование", "условия кредитования"),
    ("рассрочка", "про рассрочку", "условия рассрочки"),
    ("неустойка", "про неустойку", "размер неустойки"),
    ("компенсация", "про компенсацию", "порядок получения компенсации"),
    ("отпуск", "про отпуск", "продолжительность отпуска"),
    ("больничный", "про больничный", "оформление больничного"),
    ("увольнение", "про увольнение", "процедура увольнения"),
    ("приём на работу", "про приём на работу", "порядок приёма на работу"),
    ("командировка", "про командировки", "оформление командировки"),
    ("зарплата", "про зарплату", "расчёт зарплаты"),
    ("премия", "про премии", "условия начисления премии"),
    ("налоговый вычет", "про налоговые вычеты", "размер налогового вычета"),
    ("аренда", "про аренду", "условия аренды"),
    ("ремонт", "про ремонт", "порядок ремонта"),
    ("обслуживание", "про обслуживание", "стоимость обслуживания"),
    ("рекламация", "про рекламацию", "порядок подачи рекламации"),
    ("персональные данные", "про персональные данные", "обработка персональных данных"),
    ("расторжение", "про расторжение", "условия расторжения договора"),
    ("индексация", "про индексацию", "порядок индексации"),
    ("перерасчёт", "про перерасчёт", "порядок перерасчёта"),
]

_DOC_NAMES = [
    "договор", "контракт", "соглашение", "политика", "регламент",
    "инструкция", "приказ", "положение", "устав", "памятка",
    "руководство", "справочник", "каталог", "прайс-лист",
    "техническое задание", "спецификация", "акт", "протокол",
]

_FILE_NAMES = [
    "contract.pdf", "policy.docx", "readme.md", "report.pdf",
    "manual.pdf", "agreement.docx", "terms.pdf", "faq.txt",
    "rules.pdf", "guide.docx", "spec.pdf", "invoice.pdf",
    "договор.pdf", "политика.docx", "инструкция.pdf",
    "регламент.docx", "отчёт.pdf", "памятка.txt",
]

_BOT_QUESTIONS = [
    "что ты умеешь", "как ты работаешь", "кто тебя создал",
    "какие у тебя функции", "что ты можешь", "зачем ты нужен",
    "как тебя зовут", "ты умный", "ты всё знаешь",
    "ты понимаешь русский", "ты живой", "ты настоящий",
    "сколько тебе лет", "где ты живёшь", "ты мальчик или девочка",
]

_EMOTIONS = [
    "классно", "здорово", "отлично", "прекрасно", "замечательно",
    "ужасно", "плохо", "грустно", "обидно", "досадно",
    "странно", "удивительно", "невероятно", "восхитительно",
    "нормально", "неплохо", "так себе", "пойдёт",
    "вот это да", "ну ничего себе", "офигеть", "ого",
]


def _generate_rag_templates() -> list[str]:
    """Generate RAG examples from templates."""
    examples = []

    # Using pre-declined phrases for correct Russian grammar
    for name, pro_phrase, question_phrase in _TOPICS_PHRASES:
        examples.extend([
            f"расскажи {pro_phrase}",
            f"что ты знаешь {pro_phrase}?",
            f"есть информация {pro_phrase}?",
            f"мне нужна информация {pro_phrase}",
            f"подскажи {pro_phrase}",
            f"какие {question_phrase}?",
            f"объясни {question_phrase}",
        ])

    # Document reference patterns
    for doc in _DOC_NAMES:
        examples.extend([
            f"что написано в {doc}е?",
            f"найди в {doc}е",
            f"покажи {doc}",
            f"согласно {doc}у",
            f"в {doc}е есть что-то про сроки?",
        ])

    # File reference patterns
    for fn in _FILE_NAMES:
        examples.extend([
            f"посмотри в {fn}",
            f"что в файле {fn}?",
            f"открой {fn}",
        ])

    # Naturally phrased questions (no templates — just more examples)
    natural_rag = [
        "какая минимальная сумма заказа?",
        "есть ли бесплатная доставка?",
        "можно ли оплатить картой?",
        "принимаете ли вы наличные?",
        "какой график работы офиса?",
        "как связаться с поддержкой?",
        "есть ли техподдержка по выходным?",
        "какие способы связи доступны?",
        "можно ли получить счёт на оплату?",
        "как получить акт сверки?",
        "нужна ли предоплата?",
        "есть ли пробный период?",
        "сколько длится пробный период?",
        "какие ограничения у бесплатного тарифа?",
        "чем отличается про от бесплатного?",
        "можно ли перейти на другой тариф?",
        "как отменить подписку?",
        "есть ли возврат денег при отмене?",
        "за сколько дней нужно предупредить?",
        "какой минимальный срок договора?",
        "можно ли продлить договор?",
        "что происходит после окончания договора?",
        "есть ли автопродление?",
        "как отключить автопродление?",
        "какие данные вы собираете?",
        "как удалить мои данные?",
        "где хранятся данные?",
        "есть ли шифрование?",
        "соответствуете ли вы 152-ФЗ?",
        "какая ответственность за утечку данных?",
        "нужно ли согласие на обработку данных?",
        "можно ли отозвать согласие?",
        "кто имеет доступ к моим документам?",
        "какие права у сотрудника при увольнении?",
        "положена ли компенсация за неиспользованный отпуск?",
        "сколько дней отпуска в году?",
        "как рассчитывается отпускные?",
        "какой испытательный срок?",
        "можно ли уволиться на испытательном сроке?",
        "какие выплаты при сокращении?",
    ]
    examples.extend(natural_rag)

    return examples


def _generate_chat_templates() -> list[str]:
    """Generate chat examples from templates."""
    examples = []

    # Greeting + name patterns
    greetings = ["привет", "здравствуйте", "добрый день", "хай", "салют"]
    suffixes = ["", "!", "!!", " :)", ", бот", ", друг", ", помощник"]
    for g in greetings:
        for s in suffixes:
            examples.append(f"{g}{s}")

    # Bot questions with variations
    for q in _BOT_QUESTIONS:
        examples.extend([f"{q}?", f"а {q}?", f"скажи, {q}?"])

    # Emotion reactions
    for e in _EMOTIONS:
        examples.extend([e, f"{e}!", f"ну {e}"])

    # Meta questions about the bot/service
    meta_templates = [
        "как загрузить {что}",
        "можно ли загрузить {что}",
        "поддерживается ли формат {что}",
        "как удалить {что}",
        "как обновить {что}",
    ]
    meta_objects = [
        "документ", "файл", "pdf", "картинку", "фото",
        "таблицу", "архив", "несколько файлов",
    ]
    for tmpl in meta_templates:
        for obj in meta_objects:
            examples.append(tmpl.format(что=obj) + "?")

    # General chat (not about documents)
    general = [
        "расскажи анекдот про программистов",
        "какой сегодня праздник?",
        "посоветуй книгу",
        "что думаешь о нейросетях?",
        "тебе нравится твоя работа?",
        "ты лучше chatgpt?",
        "сколько будет 15 умножить на 7?",
        "переведи hello на русский",
        "какая столица Франции?",
        "кто написал Войну и мир?",
        "во сколько закат сегодня?",
        "помоги составить план",
        "напиши стихотворение",
        "давай поиграем в игру",
        "мне нужен совет",
        "что приготовить на ужин?",
        "какой курс доллара?",
        "когда ближайшие выходные?",
        "расскажи интересный факт",
        "ты знаешь какие-нибудь фокусы?",
        # More general chat
        "что ты думаешь о жизни?",
        "какой твой любимый цвет?",
        "ты умеешь шутить?",
        "расскажи о себе",
        "откуда ты?",
        "на каком языке ты говоришь?",
        "ты понимаешь сарказм?",
        "можешь петь?",
        "у тебя есть чувства?",
        "ты устаёшь?",
        "тебе бывает скучно?",
        "ты запоминаешь наши разговоры?",
        "сколько людей с тобой разговаривают?",
        "ты работаешь 24/7?",
        "у тебя есть выходные?",
        "кто тебя придумал?",
        "ты опенсорсный?",
        "на чём ты написан?",
        "python или javascript?",
        "ты на нейросети работаешь?",
        "какая у тебя версия?",
        "тебя можно обмануть?",
        "ты врёшь иногда?",
        "можно тебе доверять?",
        "ты безопасный?",
        "мои данные в безопасности?",
        "с наступающим!",
        "с праздником!",
        "с днём рождения!",
        "с новым годом!",
        "поздравляю",
        "молодец",
        "красавчик",
        "умница",
        "гений",
        "тупой бот",
        "ты глупый",
        "бесполезный",
        "не помогаешь",
        "ерунда какая-то",
        "фигня",
        "не верю",
        "враньё",
        "ты ошибаешься",
        "это неправильно",
        "чушь",
        # Filler / acknowledgements
        "ну ладно",
        "допустим",
        "может быть",
        "возможно",
        "наверное",
        "не знаю",
        "мне всё равно",
        "как хочешь",
        "без разницы",
        "пофиг",
        "забей",
        "проехали",
        "ладно проехали",
        "не важно",
        "забудь",
        "отмена",
        "стоп",
        "хватит",
        "достаточно",
        "всё",
        "всё понятно",
        "разобрался",
        "спасибо разобрался",
        "всё спасибо",
        "больше не нужно",
        "на этом всё",
        "я понял спасибо",
    ]
    examples.extend(general)

    return examples


def _generate_followup_templates() -> list[str]:
    """Generate followup examples from templates."""
    examples = []

    # "Tell me more about X" patterns — using pre-declined phrases
    for name, pro_phrase, question_phrase in random.sample(
        _TOPICS_PHRASES, min(15, len(_TOPICS_PHRASES))
    ):
        examples.extend([
            f"расскажи подробнее {pro_phrase}",
            f"а что насчёт {question_phrase}?",
            f"а {name} тоже?",
            f"а если говорить {pro_phrase}?",
            f"а {pro_phrase} подробнее можно?",
        ])

    # Generic clarification patterns
    clarifications = [
        "а можно конкретнее?",
        "то есть как это?",
        "я не понял последний пункт",
        "а первый пункт поясни",
        "а второе условие?",
        "это обязательно?",
        "а без этого можно?",
        "а если я не согласен?",
        "а сроки какие?",
        "а стоимость?",
        "а гарантии?",
        "а если нарушить?",
        "и какие последствия?",
        "а штраф большой?",
        "а можно без штрафа?",
        "а на практике как?",
        "а в реальности?",
        "а были прецеденты?",
        "а часто такое бывает?",
        "а что обычно делают?",
        "а другие как решают?",
        "это только для России?",
        "а в других странах?",
        "а онлайн можно?",
        "а по телефону?",
        "а лично надо приходить?",
        "а какие документы взять?",
        "а сколько ждать?",
        "а куда обращаться?",
        "а к кому?",
        "подожди, я запутался",
        "стой, давай сначала",
        "а предыдущий пункт?",
        "вернись к началу",
        "повтори последнее",
        "не расслышал, ещё раз",
        "перефразируй пожалуйста",
        "попроще можно?",
        "слишком сложно, упрости",
        "а для чайников?",
        # More clarifications and continuations
        "а что это значит на практике?",
        "и как это применить?",
        "а есть нюансы?",
        "а подводные камни?",
        "а минусы какие?",
        "а плюсы?",
        "а риски есть?",
        "а если не получится?",
        "а план Б?",
        "а что будет если не сделать?",
        "а обязательно всё это?",
        "а можно частично?",
        "а поэтапно можно?",
        "а с чего начать?",
        "а что в первую очередь?",
        "а что самое важное?",
        "а что можно пропустить?",
        "а это точно законно?",
        "а если поймают?",
        "а кто контролирует?",
        "а кто проверяет?",
        "а были случаи?",
        "а как у других компаний?",
        "а по опыту как лучше?",
        "а ты бы что посоветовал?",
        "а ты сам как думаешь?",
        "давай разберём первый пункт",
        "вернёмся ко второму вопросу",
        "а третий пункт не понял",
        "последнее предложение поясни",
        "начало ответа не понял",
        "а середину можно подробнее?",
        "а вот этот момент раскрой",
        "стоп, а вот тут подробнее",
        "погоди, это важный момент",
        "а вот это интересно, расскажи больше",
        "хм, а почему именно так?",
        "а нельзя ли по-другому?",
        "а если сделать наоборот?",
        "а обратный процесс?",
        "а отменить можно?",
        "а вернуть как было?",
        "а исправить ошибку?",
        "а если я уже сделал по-другому?",
        "а задним числом можно?",
    ]
    examples.extend(clarifications)

    return examples


# ═══════════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def _augment_text(text: str) -> list[str]:
    """Create variations of a text: case, punctuation, typos."""
    variations = []

    # Lowercase (if not already)
    if text != text.lower():
        variations.append(text.lower())

    # Remove punctuation
    no_punct = text.rstrip("?!.,;:")
    if no_punct != text:
        variations.append(no_punct)

    # Add question mark (if not present and looks like a question)
    if not text.endswith("?") and any(
        text.lower().startswith(w) for w in
        ["как", "что", "где", "когда", "сколько", "какой", "какая",
         "какие", "каков", "зачем", "почему", "кто", "чем", "откуда"]
    ):
        variations.append(text.rstrip(".,!;:") + "?")

    # Add "а " prefix (conversational style)
    if not text.lower().startswith("а ") and len(text.split()) >= 2:
        if random.random() < 0.5:
            variations.append("а " + text[0].lower() + text[1:])

    # Add "подскажи, " or "скажи, " prefix
    if random.random() < 0.3 and len(text.split()) >= 3:
        prefix = random.choice(["подскажи, ", "скажи, ", "слушай, ",
                                "а подскажи, ", "напомни, "])
        variations.append(prefix + text[0].lower() + text[1:])

    # Exclamation / emphasis
    if random.random() < 0.2 and not text.endswith("!"):
        variations.append(text.rstrip("?.,:;") + "!")

    # "пожалуйста" suffix
    if random.random() < 0.15 and len(text.split()) >= 2:
        variations.append(text.rstrip("?.!,:;") + " пожалуйста")

    return variations


# ═══════════════════════════════════════════════════════════════
# DATASET ASSEMBLY
# ═══════════════════════════════════════════════════════════════

def generate_dataset() -> list[tuple[str, str]]:
    """Generate the full dataset as list of (text, label) tuples."""
    seeds = get_all_seeds()
    dataset: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(text: str, label: str) -> None:
        """Add example if not duplicate."""
        key = text.lower().strip()
        if key and key not in seen:
            seen.add(key)
            dataset.append((text.strip(), label))

    # 1. Add all seeds
    for label, examples in seeds.items():
        for text in examples:
            _add(text, label)

    # 2. Add template-generated examples
    for text in _generate_rag_templates():
        _add(text, "rag")
    for text in _generate_chat_templates():
        _add(text, "chat")
    for text in _generate_followup_templates():
        _add(text, "followup")

    # 3. Augment existing examples
    current = list(dataset)  # snapshot
    for text, label in current:
        for variation in _augment_text(text):
            _add(variation, label)

    # 4. Shuffle
    random.shuffle(dataset)

    return dataset


def save_dataset(dataset: list[tuple[str, str]], path: str = "dataset.csv") -> None:
    """Save dataset to CSV."""
    filepath = Path(path)
    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(dataset)
    print(f"Saved {len(dataset)} examples to {filepath}")


def print_stats(dataset: list[tuple[str, str]]) -> None:
    """Print dataset statistics."""
    from collections import Counter
    counts = Counter(label for _, label in dataset)

    print("\n=== Dataset Statistics ===")
    print(f"Total examples: {len(dataset)}")
    print()
    for label in ["rag", "chat", "followup"]:
        count = counts[label]
        pct = count / len(dataset) * 100
        bar = "#" * int(pct)
        print(f"  {label:10s} {count:5d}  ({pct:5.1f}%)  {bar}")

    print("\n=== Sample examples ===")
    for label in ["rag", "chat", "followup"]:
        examples = [t for t, l in dataset if l == label]
        print(f"\n  [{label}]")
        for ex in random.sample(examples, min(5, len(examples))):
            print(f"    {ex}")


if __name__ == "__main__":
    dataset = generate_dataset()
    print_stats(dataset)
    save_dataset(dataset)
