def template_0(SRC_LANG, TARG_LANG, BGD_INFO=None):
    if BGD_INFO is not None:
        bg_insert = f"As you translate, you can use the following background information:\n\n{BGD_INFO}\n\n"
    else:
        bg_insert = ""
    core_instuction = f"You are a conference interpreter. {bg_insert}Taking into account the original {SRC_LANG} text, complete its translation into {TARG_LANG}. "
    extra_instruction = "\nDo not add any notes or comments to the translation."
    return core_instuction + extra_instruction


def template_1(SRC_LANG, TARG_LANG, BGD_INFO=None):
    if BGD_INFO is not None:
        bg_insert = f"As you translate, you can use the following background information:\n\n{BGD_INFO}\n\n"
    else:
        bg_insert = ""
    core_instuction = f"You are a conference interpreter. {bg_insert}Taking into account the original {SRC_LANG} text, complete its translation into {TARG_LANG}. "
    extra_instruction = "Prioritize not word-for-word correspondence, but the overall meaning and style.\nDo not add any notes or comments to the translation."
    return core_instuction + extra_instruction


def template_2(SRC_LANG, TARG_LANG, BGD_INFO=None):
    if BGD_INFO is not None:
        bg_insert = f"As you translate, you can use the following background information:\n\n{BGD_INFO}\n\n"
    else:
        bg_insert = ""
    core_instuction = f"You are a conference interpreter. {bg_insert}Taking into account the original {SRC_LANG} text, complete its translation into {TARG_LANG}. "
    extra_instruction = """Be very terse, refraining from word-for-word translation, but aiming for extremly concise interpretation (that makes implicit what the listener can easily infer). The relative order of the subject, predicate and object do not need to follow the order in which they appear in the source.

Examples:

Source: The most important question these days is how can we speed up the solutions to the climate crisis?
Verbose: Самый важный вопрос наших дней - это как нам ускорить решение климатического кризиса.
Concise translation: Самое важное это понять как решить проблемы с климатом.

Source: I'm convinced we are going to solve the climate crisis, we've got this, but the question remains, will we solve it in time?
Verbose: Я уверен, что мы решим проблему с климатом, и у нас для этого всё есть, но отстаётся вопрос, решим ли мы её вовремя?
Concise: Уверен, у нас получится, вот только вовремя ли?

Source: I'll give you the the shortest definition of the problem: if I was going to give a one-slide slide show, it would be this slide.
Verbose: Я дам вам самое короткое определение проблемы: если бы я делал перезентацию на один слайд, то это был бы вот такой слайд.
Concise: Если кратко, буквально одим слайдом, то вот в чём проблема.

Это тропосфера, не буду напоминать, почему она синяя: здесь кислород преломляет синий свет.
Do not add any notes or comments to the translation."""

    return core_instuction + extra_instruction


def template_3(SRC_LANG, TARG_LANG, BGD_INFO=None):
    if BGD_INFO is not None:
        bg_insert = f"As you translate, you can use the following background information:\n\n{BGD_INFO}\n\n"
    else:
        bg_insert = ""
    core_instuction = f"You are a conference interpreter. {bg_insert}Taking into account the original {SRC_LANG} text, complete its translation into {TARG_LANG}. "
    extra_instruction = """As you make your translation, avoid word-for-word corrensondence, but convey the core message tersely, specifically
(1) express the original thought using as few words as possible
(2) prefer shorter synonyms to loger ones
(3) omit non-essential conjunctions, adverbs and adjectives

**Example or using fewer words**
Source: Мы настроены не просто на повышение цен в связи с тем, что денег много, а на достижение конкретного результата, который люди почувствуют.
Translation: We don't just want to raise prices because we have extra cash, but to reach tangible results for the people.
Explanation: "настроены на повышение цен" actually means "want to raise prices"; "в связи с тем, что денег много" means "because we have extra cash"; "на достижение конкретного результата, который люди почувствуют" essentially means "for the people".

**Example of compression by omitting non-essential words and phrases**
Source: При этом обрабатывающие отрасли производства растут чуть быстрее.
Translation: The manufacturing industries are growing at a slightly higher pace.
Explanation: The conjunction "При этом" is omitted for brevity, while important information is conveyed.

**Example of syntactic compression**
Source: И вторая задача – это цифровая экономика, развитие робототехники, и так далее...
Translation: And the second thing we need is digital economy, robotics etc.
Explanation: Here, "развитие робототехники" is translated "robotics", because essentially the need is not for the process of developing robotics, but the end result of having (developed) robotics.

Do not add any notes, comments, explanations or revisions to the translation."""

    return core_instuction + extra_instruction


def template_4(SRC_LANG, TARG_LANG, BGD_INFO=None):
    if BGD_INFO is not None:
        bg_insert = f"As you translate, you can use the following background information:\n\n{BGD_INFO}\n\n"
    else:
        bg_insert = ""
    core_instuction = f"You are a conference interpreter. {bg_insert}Taking into account the original {SRC_LANG} text, complete its translation into {TARG_LANG}. "
    extra_instruction = f"Prioritize not word-for-word correspondence, but the overall meaning and style. Do not add any notes or comments to the translation. Be sure to spell out all the acronyms and only use {TARG_LANG}."
    return core_instuction + extra_instruction


PROMPT_TEMPLATES = {
    0: template_0,  # plain
    1: template_1,  # emphasis on meaning
    2: template_2,  # concise
    3: template_3,  # compression strategy
    4: template_4,  # compression strategy
}
