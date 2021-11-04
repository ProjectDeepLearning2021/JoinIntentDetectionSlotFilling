import torch
from torch.autograd import Variable
from collections import Counter
import pickle
import random
import os


# Se verifica si es que el soporte de CUDA está disponible (para utilizar la GPU
# en el cómputo de tensores)
USE_CUDA = torch.cuda.is_available()

# Se define función para convertir la sequencia de texto a índices en formato de
# tensor con soporte de CUDA
def prepare_sequence(seq, to_ix):
    # Crea una lista a partir de una secuencia. Para cada palabra de la secuencia
    # asigna un índice de acuerdo al diccionario "to_ix", ya sea que pertenece o
    # no (token desconocido <UNK>)
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    # Convierte la lista en tensor con soporte de CUDA (si está disponible)
    tensor = Variable(torch.LongTensor(idxs)).cuda() if USE_CUDA else Variable(torch.LongTensor(idxs))
    return tensor

# Se define función para obtener los elementos dentro de las listas de una tupla
flatten = lambda l: [item for sublist in l for item in sublist]

# Función para el pre procesamiento
def preprocessing(file_path,length):
    """
    atis-2.train.w-intent.iob
    """

    # Nos ubicamos en la ruta donde se guardará la data pre porcesada
    processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/")
    print("processed_data_path : %s" % processed_path)

    # Verificamos si existe el archivo processed_train_data.pkl
    if os.path.exists(os.path.join(processed_path,"processed_train_data.pkl")):
        train_data, word2index, tag2index, intent2index = pickle.load(open(os.path.join(processed_path,"processed_train_data.pkl"),"rb"))
        return train_data, word2index, tag2index, intent2index

    # Si no existe la ruta para el pre procesamiento, se procede a crearla
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    # Se carga la data de entrenamiento
    try:
        train = open(file_path,"r").readlines()
        print("Successfully load data. # of set : %d " % len(train))
    except:
        print("No such file!")
        return None,None,None,None

    # Se realiza la lectura de la data de entrenamiento, de acuerdo a algunos separadores
    # Además, se crea el vocabulario, los slot y los intentos
    try:
        # Se eliminan los saltos de línea al final de la oración
        train = [t[:-1] for t in train]
        # Se obtiene la secuencia de entrada (oraciones en inglés), la secuencia de
        # salida (secuencia de slots) y la intención relacionada a la secuencia
        train = [[t.split("\t")[0].split(" "),t.split("\t")[1].split(" ")[:-1],t.split("\t")[1].split(" ")[-1]] for t in train]
        # Se retiran los tokens de inicio y fin en la secuencia de entrada, el token de
        # inicio en la secuencia de salida (ya que se encuentra desfasada una posición
        # respecto a la entrada)
        train = [[t[0][1:-1],t[1][1:],t[2]] for t in train]

        # Se obtiene las tuplas de las secuencias de entrada, salida y las intenciones
        # por separado
        seq_in,seq_out, intent = list(zip(*train))

        # Se conforma el vocabulario de las palabras, las etiquetas de los slots y las
        # etiquetas de las intenciones
        vocab = set(flatten(seq_in))
        slot_tag = set(flatten(seq_out))
        intent_tag = set(intent)
        print("# of vocab : {vocab}, # of slot_tag : {slot_tag}, # of intent_tag : {intent_tag}".format(vocab=len(vocab),slot_tag=len(slot_tag),intent_tag=len(intent_tag)))
    except:
        print("Please, check data format! It should be 'raw sentence \t BIO tag sequence intent'. The following is a sample.")
        print("BOS i want to fly from baltimore to dallas round trip EOS\tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight")
        return None,None,None,None

    # Se definen listas para almacenar las secuencias de entrada y salida luego del
    # preprocesamiento para añadir tokens de fin de secuencia y para hacer padding
    sin=[]
    sout=[]

    # Se añade nuevamente el token de fin de secuencia (<EOS>) y se hace padding
    # hasta completar el tamaño máximo de secuencia
    for i in range(len(seq_in)):
        # Secuencia de entrada
        temp = seq_in[i]
        if len(temp)<length:
            # Añade el token de fin de secuencia
            temp.append('<EOS>')
            while len(temp)<length:
                # Completa con token de padding hasta completar tamaño máximo de secuencia
                temp.append('<PAD>')
        else:
            # Trunca la secuencia en el tamaño máximo
            temp = temp[:length]
            # Reemplaza el último elemento por el token de fin de secuencia
            temp[-1]='<EOS>'
        sin.append(temp)

        # Secuencia de salida
        temp = seq_out[i]
        if len(temp)<length:
            while len(temp)<length:
                # Completa con token de padding hasta completar tamaño máximo de secuencia
                temp.append('<PAD>')
        else:
            # Trunca la secuencia en el tamaño máximo
            temp = temp[:length]
            # Reemplaza el último elemento por el token de fin de secuencia
            temp[-1]='<EOS>'
        sout.append(temp)

    # Se define un diccionario para mapear las palabras a índices
    word2index = {'<PAD>': 0, '<UNK>':1,'<SOS>':2,'<EOS>':3}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token]=len(word2index)

    # Se define un diccionario para mapear las etiquetas de slots a índices
    tag2index = {'<PAD>' : 0}
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)

    # Se define un diccionario para mapear las etiquetas de intenciones a índices
    intent2index={}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)

    # Se reconstruye una lista general con las listas de secuencias de entrada,
    # salida y las intenciones
    train = list(zip(sin,sout,intent))

    # Se define una lista vacía para almacenar los tensores con soporte de CUDA
    # correspondientes a las secuencias de entrada, salida y las intenciones
    train_data=[]

    for tr in train:
        # Se usa la función "prepare_sequence" para obtener el tensor de entrada con
        # soporte de CUDA
        temp = prepare_sequence(tr[0],word2index)
        temp = temp.view(1,-1)
        # Se usa la función "prepare_sequence" para obtener el tensor de salida con
        # soporte de CUDA
        temp2 = prepare_sequence(tr[1],tag2index)
        temp2 = temp2.view(1,-1)
        # Se usa la función "prepare_sequence" para obtener el tensor de intención
        # con soporte de CUDA
        temp3 = Variable(torch.LongTensor([intent2index[tr[2]]])).cuda() if USE_CUDA else Variable(torch.LongTensor([intent2index[tr[2]]]))
        # Almacena los tensores en una lista
        train_data.append((temp,temp2,temp3))
    
    pickle.dump((train_data,word2index,tag2index,intent2index),open(os.path.join(processed_path,"processed_train_data.pkl"),"wb"))
    pickle
    print("Preprocessing complete!")
              
    return train_data, word2index, tag2index, intent2index

# Se define una función para producir batches aleatorios del conjunto de
# entrenamiento
def getBatch(batch_size,train_data):
    random.shuffle(train_data)
    sindex=0
    eindex=batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp
        
        yield batch

# Se define una función para cargar el diccionario
def load_dictionary(dic_path):
    
    processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/")

    if os.path.exists(os.path.join(processed_path,"processed_train_data.pkl")):
        _, word2index, tag2index, intent2index = pickle.load(open(os.path.join(processed_path,"processed_train_data.pkl"),"rb"))
        return word2index, tag2index, intent2index
    else:
        print("Please, preprocess data first")
        return None,None,None