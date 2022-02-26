from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from word2number import w2n
import re

def findIndex(arr, warr):
    index = []
    for i in range(len(arr)):
        for j in range(len(warr)):
            if arr[i]==warr[j]:
                index.append(i)
    index.sort()
    return index

signs = {
    '': '',
    '=': '',
    '!=': 'not ',
    '>': 'more than ',
    '<': 'less than ',
    '>=': 'more or equal than ',
    '<=': 'less or equal than ',
    'LIKE': 'like ',
    'like': 'like '
}

rules = {                                               # setul de reguli: daca e cuvantul x, se pune intrabrea respectiva
    'WHERE': '{} is {}what?',
    'AND': '{} is {}what?',
    'OR': '{} is {}what?',
    'HAVING': '{} is {}what?',
    'Count(*)': '{} is {}what?',
    'LIKE': '{} is {}like what?',
    'BETWEEN': '{} is between {}what?',
    'where': '{} is {}what?',
    'and': '{} is {}what?',
    'or': '{} is {}what?',
    'having': '{} is {}what?',
    'count(*)': '{} is {}what?',
    'like': '{} is {}like what?',
    'between': '{} is between {}what?'
}

def question(word: str, subject: str, quantity = ''):
    try:
        return rules[word].format(subject, signs[quantity])
    except:
        return ' '

def answer(question: str, context: str):
    model_name = "deepset/roberta-base-squad2"
    
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return res["answer"]


def robertaQnA(quest, cont):
    context = cont
    init = quest
    sql = ['SELECT', 'FROM', 'HAVING', 'JOIN', 'JOIN LEFT', 'JOIN RIGHT', 'ON', 'AND', 'OR', 'WHERE',  'INTERSECT', 'Count(*)', 'BETWEEN', \
           'select', 'from', 'having', 'join', 'join left', 'join right', 'on', 'and', 'or', 'where', 'intersect', 'count(*)', 'between']
    exclude = [',',':',';','?']
    query=init                                              # copiaza textul de query
    for x in exclude:                                       # modifica toate chestiile de nu ne trebuie cu spatiu
        query = query.replace(x, ' ')
    query=query.split()                                     # se sparge in list
    sqlIndex = findIndex(query, sql)                        # se cauta pozitiile cuvintelor cheie
    terminalIndex = findIndex(query, ["'terminal'"])      # se cauta pozitiile cuvantului terminal
    pairIndex = []
    for x in terminalIndex:                                 # se fac perechi de forma: ultimul cuvant cheie dinaintea lui terminal, terminal
        last = 0
        for y in sqlIndex:
            if y > x:
                break
            last = y
        pairIndex.append((last, x))
    replace = []
    for pair in pairIndex:                                  # pentru fiecare pereche gasita, se face intrebarea, se raspunde si se ia raspunsul
        q = query[pair[0]]
        s = ' '.join(query[pair[0]+1:pair[1]])
        quant = query[pair[1]-1]
        if quant == q:
            quant = ''
        ans = answer(question(q, s, quant), context)

        copy = re.split(' and | or | to ', ans)
        for i in range(len(copy)):
            try:
                copy[i] = w2n.word_to_num(copy[i])
            except:
                copy[i] = ''.join(re.findall(r'\d+', copy[i]))
        try:
            copy = list(filter(('').__ne__, copy))
        except:
            pass
        if len(copy) > 0:
            for j in range(len(copy)):
                copy[j] = str(copy[j])
            replace+=copy
        else:
            copy = re.split(' and | or ', ans)
            replace += copy

    for x in replace:                                       # pentru fiecare raspuns, se modifica cuvintele terminal pe rand
        init = init.replace("'terminal'","'{}'".format(x) if x.isnumeric()==False else x, 1)
    return init