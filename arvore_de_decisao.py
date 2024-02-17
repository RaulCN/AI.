import math

## Retorna a nova árvore de decisão com base nos exemplos fornecidos
#@param data: Cópia da lista de dados.
#@param attributes: Lista de todos os atributos 
#@param target_attr: O atributo alvo que será avaliado
#@param fitness_func: A função alvo

## Retorna os diferentes valores que um atributo específico pode ter em um conjunto de dados.
def get_values(data, attribute):
    """
    Retorna os diferentes valores que um atributo específico pode ter em um conjunto de dados.

    Args:
    - data: Lista de dicionários, cada dicionário representa um exemplo com seus atributos e valores.
    - attribute: O atributo para o qual os valores serão retornados.

    Returns:
    - Uma lista dos diferentes valores que o atributo específico pode ter no conjunto de dados.
    """
    values = set()
    for record in data:
        values.add(record[attribute])
    return list(values)

## Retorna uma lista de exemplos que têm o valor específico para o atributo escolhido.
def get_examples(data, attribute, value):
    """
    Retorna uma lista de exemplos que têm o valor específico para o atributo escolhido.

    Args:
    - data: Lista de dicionários, cada dicionário representa um exemplo com seus atributos e valores.
    - attribute: O atributo para o qual os exemplos serão filtrados.
    - value: O valor do atributo para o qual os exemplos serão filtrados.

    Returns:
    - Uma lista de exemplos que têm o valor específico para o atributo escolhido.
    """
    examples = []
    for record in data:
        if record[attribute] == value:
            examples.append(record)
    return examples

## Escolhe o próximo melhor atributo para classificar os dados.
def choose_attribute(data, attributes, target_attr, fitness_func):
    """
    Escolhe o próximo melhor atributo para classificar os dados.

    Args:
    - data: Lista de dicionários, cada dicionário representa um exemplo com seus atributos e valores.
    - attributes: Lista de todos os atributos disponíveis nos exemplos.
    - target_attr: O atributo alvo que será avaliado para tomar decisões.
    - fitness_func: A função usada para determinar a melhor divisão dos dados.

    Returns:
    - O próximo melhor atributo para classificar os dados.
    """
    # Por simplicidade, retornaremos o primeiro atributo da lista de atributos
    return attributes[0]

## Calcula o valor mais frequente do atributo alvo em um conjunto de dados.
def majority_value(data, target_attr):
    """
    Calcula o valor mais frequente do atributo alvo em um conjunto de dados.

    Args:
    - data: Lista de dicionários, cada dicionário representa um exemplo com seus atributos e valores.
    - target_attr: O atributo alvo para o qual o valor mais frequente será calculado.

    Returns:
    - O valor mais frequente do atributo alvo.
    """
    val_freq = {}
    
    # Calcula a frequência de cada valor no atributo alvo
    for record in data:
        if record[target_attr] in val_freq:
            val_freq[record[target_attr]] += 1
        else:
            val_freq[record[target_attr]] = 1
    
    # Encontra o valor mais frequente
    majority_value = max(val_freq, key=val_freq.get)
    
    return majority_value

def create_decision_tree(data, attributes, target_attr, fitness_func):
    """
    Constrói uma árvore de decisão recursivamente com base nos exemplos fornecidos.

    Args:
    - data: Lista de dicionários, cada dicionário representa um exemplo com seus atributos e valores.
    - attributes: Lista de todos os atributos disponíveis nos exemplos.
    - target_attr: O atributo alvo que será avaliado para tomar decisões.
    - fitness_func: A função usada para determinar a melhor divisão dos dados.

    Returns:
    - A árvore de decisão construída.
    """

    data = data[:]  # Cria uma cópia dos dados para não modificar a lista original
    vals = [record[target_attr] for record in data]
    default = majority_value(data, target_attr)  # Obtém o valor mais frequente do atributo alvo
    
    print("Valores do atributo alvo:", vals)  # Print dos valores do atributo alvo para diagnóstico
    
    # Se o conjunto de dados estiver vazio ou a lista de atributos estiver vazia, retorne o valor padrão
    if not data or (len(attributes) - 1) <= 0:
        print("Conjunto de dados vazio ou lista de atributos vazia. Retornando valor padrão:", default)
        return default
    # Se todos os registros no conjunto de dados tiverem a mesma classificação, retorne a classificação
    elif vals.count(vals[0]) == len(vals):
        print("Todos os registros têm a mesma classificação. Retornando classificação:", vals[0])
        return vals[0]
    
    else:
        # Escolha o próximo melhor atributo para classificar nossos dados
        best = choose_attribute(data, attributes, target_attr, fitness_func)
        print("Melhor atributo escolhido:", best)  # Print do melhor atributo para diagnóstico
        
        # Crie uma nova árvore de decisão/nó com o melhor atributo
        tree = {best:{}}
        
        # Crie uma nova árvore de decisão/sub-nó para cada um dos valores no melhor campo de atributo
        for val in get_values(data, best):
            print("Valor atual:", val)  # Print do valor atual do atributo para diagnóstico
            
            # Crie uma subárvore para o valor atual sob o campo "melhor"
            subtree = create_decision_tree(get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func)
        
            # Adicione a nova subárvore ao objeto de dicionário vazio em nossa nova árvore/nó que acabamos de criar
            tree[best][val] = subtree
        
    return tree

## Calcula a entropia do conjunto de dados fornecido para o atributo alvo.
#@param data: a lista de dados
#@param target_attr: o atributo alvo
def entropy(data, target_attr):
    """
    Calcula a entropia do conjunto de dados para o atributo alvo.

    Args:
    - data: Lista de dicionários, cada dicionário representa um exemplo com seus atributos e valores.
    - target_attr: O atributo alvo para o qual a entropia será calculada.

    Returns:
    - A entropia do conjunto de dados para o atributo alvo.
    """
    val_freq = {}
    data_entropy = 0.0
    
    # Calcula a frequência de cada um dos valores no atributo alvo
    for record in data:
        if record[target_attr] in val_freq:
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]] = 1.0
    
    # Calcula a entropia dos dados para o atributo alvo
    for freq in val_freq.values():
            data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)
    
    return data_entropy

## Calcula o ganho de informação (redução na entropia) que resultaria ao dividir os dados no atributo escolhido (attr).
def gain(data, attr, target_attr):
    """
    Calcula o ganho de informação que resultaria ao dividir os dados no atributo escolhido.

    Args:
    - data: Lista de dicionários, cada dicionário representa um exemplo com seus atributos e valores.
    - attr: O atributo pelo qual os dados serão divididos.
    - target_attr: O atributo alvo para o qual a entropia será calculada.

    Returns:
    - O ganho de informação resultante ao dividir os dados no atributo escolhido.
    """
    val_freq = {}
    subset_entropy = 0.0
    
    # Calcula a frequência de cada um dos valores no atributo escolhido
    for record in data:
        if record[attr] in val_freq:
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0
    
    # Calcula a soma da entropia para cada subconjunto de registros ponderados pela sua probabilidade de ocorrência no conjunto de treinamento
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)
 
    # Subtrai a entropia do atributo escolhido da entropia do conjunto de dados inteiro em relação ao atributo alvo (e retorna)
    return entropy(data, target_attr) - subset_entropy

## Função para imprimir a árvore de decisão
def print_tree(tree, indent=''):
    """
    Imprime a árvore de decisão de forma legível.

    Args:
    - tree: A árvore de decisão representada como um dicionário.
    - indent: A string usada para indentar a árvore para torná-la mais legível.
    """
    for key, value in tree.items():
        if isinstance(value, dict):
            print(indent + str(key))
            print_tree(value, indent + '  ')
        else:
            print(indent + str(key) + ': ' + str(value))

## Exemplo de uso
def main():
    # Exemplo de dados
    data = [
        {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Windy': 'False', 'Play': 'No'},
        {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Windy': 'True', 'Play': 'No'},
        {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Windy': 'False', 'Play': 'Yes'},
        {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Windy': 'False', 'Play': 'Yes'},
        {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Windy': 'False', 'Play': 'Yes'},
        {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Windy': 'True', 'Play': 'No'},
        {'Outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Windy': 'True', 'Play': 'Yes'},
        {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High', 'Windy': 'False', 'Play': 'No'},
        {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Windy': 'False', 'Play': 'Yes'},
        {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Windy': 'False', 'Play': 'Yes'},
        {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Windy': 'True', 'Play': 'Yes'},
        {'Outlook': 'Overcast', 'Temperature': 'Mild', 'Humidity': 'High', 'Windy': 'True', 'Play': 'Yes'},
        {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'Normal', 'Windy': 'False', 'Play': 'Yes'},
        {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Windy': 'True', 'Play': 'No'}
    ]

    attributes = ['Outlook', 'Temperature', 'Humidity', 'Windy']

    target_attr = 'Play'

    # Chamada da função para construir a árvore de decisão
    tree = create_decision_tree(data, attributes, target_attr, gain)

    # Imprime a árvore de decisão
    print("Árvore de Decisão:")
    print_tree(tree)

# Chamada da função principal
if __name__ == "__main__":
    main()
