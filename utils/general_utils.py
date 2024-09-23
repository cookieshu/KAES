import numpy as np


# def get_score_vector_positions():
#     '''
#     返回一个字典，其中包含评分类型和对应的索引位置。字典的键是评分类型的名称，值是对应的索引位置。
#     'score': '1', 'content': '1', 'prompt_adherence': '1', 'language': '1', 'narrativity': '1',
#     'score': 0: 总评分位于索引位置 0。
#     'content': 1: 内容评分位于索引位置 1。
#     'organization': 2: 组织评分位于索引位置 2。
#     'word_choice': 3: 词汇选择评分位于索引位置 3。
#     'sentence_fluency': 4: 句子流畅度评分位于索引位置 4。
#     'conventions': 5: 书面语规范评分位于索引位置 5。
#     'prompt_adherence': 6: 贴合提示评分位于索引位置 6。
#     'language': 7: 语言评分位于索引位置 7。
#     'narrativity': 8: 叙事性评分位于索引位置 8
#     '''
#     return {
#         'score': 0,
#         'content': 1,
#         'organization': 2,
#         'word_choice': 3,
#         'sentence_fluency': 4,
#         'conventions': 5,
#         'prompt_adherence': 6,
#         'language': 7,
#         'narrativity': 8,
#         # 'style': 9,
#         # 'voice': 10
#     }

def get_score_vector_positions():# todo 替换了
    '''
    返回一个字典，其中包含评分类型和对应的索引位置。字典的键是评分类型的名称，值是对应的索引位置。
    'content': 0: 内容评分位于索引位置 0。
    'conventions': 1: 书面语规范评分位于索引位置 1。
    '''
    return {
        'content': 0,
        'conventions': 1,
    }


def get_keystroke_score_vector_positions():
    '''
    返回一个字典，其中包含评分类型和对应的索引位置。字典的键是评分类型的名称，值是对应的索引位置。
    示例：'score': '1', 'content': '1', 'prompt_adherence': '1', 'language': '1', 'narrativity': '1',

    'content': 0: 内容评分位于索引位置 0。
    'conventions': 1: 书面语规范评分位于索引位置 1。

    '''
    return {
        'content': 0,
        'conventions': 1,
    }


# def get_min_max_scores():
#     '''
#     这个函数返回了每个提示ID及其各评分属性的最小值和最大值，用于归一化评分。
#     返回一个字典，其中键是提示ID（1到8），值是另一个字典，这个字典包含了评分属性及其对应的最小值和最大值。
#     '''
#     return {
#         1: {'score': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
#             'sentence_fluency': (1, 6), 'conventions': (1, 6)},
#         2: {'score': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
#             'sentence_fluency': (1, 6), 'conventions': (1, 6)},
#         3: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
#         4: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
#         5: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
#         6: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
#         7: {'score': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6)},
#         8: {'score': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12),
#             'sentence_fluency': (2, 12), 'conventions': (2, 12)}}
def get_min_max_scores():# todo 替换
    '''
    这个函数返回了每个提示ID及其各评分属性的最小值和最大值，用于归一化评分。
    返回一个字典，其中键是提示ID（1到8），值是另一个字典，这个字典包含了评分属性及其对应的最小值和最大值。
    '''
    return {
        1: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
        2: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
        3: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
        4: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
        5: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
        6: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
    }


def get_keystroke_min_max_scores():
    '''
    这个函数返回了每个提示ID及其各评分属性的最小值和最大值，用于归一化评分。
    返回一个字典，其中键是提示ID（1到8），值是另一个字典，这个字典包含了评分属性及其对应的最小值和最大值。
    '''
    return {
        1: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
        2: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
        3: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
        4: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
        5: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
        6: {'content': (1.0, 5.0), 'conventions': (1.0, 5.0)},
    }


def get_scaled_down_scores(scores, prompts):
    '''
    这个函数通过最小-最大归一化方法将评分缩放到 [0, 1] 范围内，然后将归一化后的评分向量返回。这个过程有助于将不同范围的评分标准化，使其在后续的处理或建模过程中具有可比性。
    scores: 一个包含评分向量的列表，每个评分向量包含多个评分属性。
    prompts: 一个包含提示（prompt）ID的列表，对应每个评分向量
    '''
    # 获取评分属性及其在评分向量中的位置，存储在字典 score_positions 中。
    score_positions = get_score_vector_positions()
    # 获取每个提示（prompt）及其各评分属性的最小值和最大值，存储在字典 min_max_scores 中
    min_max_scores = get_min_max_scores()
    # 将评分向量和提示ID配对，生成一个可迭代对象 score_prompts
    score_prompts = zip(scores, prompts)
    scaled_score_list = []
    # 遍历每个评分向量及其对应的提示ID，并初始化一个长度与评分属性数相同的列表 rescaled_score_vector，初始值为 -1。
    for score_vector, prompt in score_prompts:
        rescaled_score_vector = [-1] * len(score_positions)
        # 遍历评分向量中的每个评分值及其索引 ind。如果评分值不为 -1，则进行归一化处理
        for ind, att_val in enumerate(score_vector):
            if att_val != -1:
                # 获取当前评分属性的名称 attribute_name
                attribute_name = list(score_positions.keys())[list(score_positions.values()).index(ind)]
                # 获取该属性在当前提示下的最小值 min_val 和最大值 max_val
                min_val = min_max_scores[prompt][attribute_name][0]
                max_val = min_max_scores[prompt][attribute_name][1]
                # 计算归一化后的评分 scaled_score
                scaled_score = (att_val - min_val) / (max_val - min_val)
                # 将归一化后的评分存储在 rescaled_score_vector 中对应的位置
                rescaled_score_vector[ind] = scaled_score
        scaled_score_list.append(rescaled_score_vector)
    assert len(scaled_score_list) == len(scores)
    for scores in scaled_score_list:
        assert min(scores) >= -1
        assert max(scores) <= 1
    return scaled_score_list


def get_single_scaled_down_score(scores, prompts, attribute_name):
    min_max_scores = get_min_max_scores()
    score_prompts = zip(scores, prompts)
    scaled_score_list = []
    for score_vector, prompt in score_prompts:
        for ind, att_val in enumerate(score_vector):
            min_val = min_max_scores[prompt][attribute_name][0]
            max_val = min_max_scores[prompt][attribute_name][1]
            scaled_score = (att_val - min_val) / (max_val - min_val)
        scaled_score_list.append([scaled_score])
    assert len(scaled_score_list) == len(scores)
    return scaled_score_list


def rescale_tointscore(scaled_scores, set_ids):
    '''
    rescale scaled scores range[0,1] to original integer scores based on  their set_ids
    :param scaled_scores: list of scaled scores range [0,1] of essays
    :param set_ids: list of corresponding set IDs of essays, integer from 1 to 8
    '''
    if isinstance(set_ids, int):
        prompt_id = set_ids
        set_ids = np.ones(scaled_scores.shape[0], ) * prompt_id
    assert scaled_scores.shape[0] == len(set_ids)
    int_scores = np.zeros((scaled_scores.shape[0], 1))
    for k, i in enumerate(set_ids):
        assert i in range(1, 9)
        # TODO
        if i == 1:
            minscore = 2
            maxscore = 12
        elif i == 2:
            minscore = 1
            maxscore = 6
        elif i in [3, 4]:
            minscore = 0
            maxscore = 3
        elif i in [5, 6]:
            minscore = 0
            maxscore = 4
        elif i == 7:
            minscore = 0
            maxscore = 30
        elif i == 8:
            minscore = 0
            maxscore = 60
        else:
            print("Set ID error")

        int_scores[k] = scaled_scores[k] * (maxscore - minscore) + minscore
    return np.around(int_scores).astype(int)


def rescale_single_attribute(scores, set_ids, attribute_name):
    min_max_scores = get_min_max_scores()
    score_id_combined = list(zip(scores, set_ids))
    rescaled_scores = []
    for score, set_id in score_id_combined:
        min_score = min_max_scores[set_id][attribute_name][0]
        max_score = min_max_scores[set_id][attribute_name][1]
        rescaled_score = score * (max_score - min_score) + min_score
        rescaled_scores.append(np.around(rescaled_score).astype(int))
    return np.array(rescaled_scores)


def separate_attributes_for_scoring(scores, set_ids):
    score_vector_positions = get_score_vector_positions()
    min_max_scores = get_min_max_scores()
    individual_att_scores_dict = {att: [] for att in score_vector_positions.keys()}
    score_set_comb = list(zip(scores, set_ids))
    for att_scores, set_id in score_set_comb:
        for relevant_attribute in min_max_scores[set_id].keys():
            att_position = score_vector_positions[relevant_attribute]
            individual_att_scores_dict[relevant_attribute].append(att_scores[att_position])
    return individual_att_scores_dict


def separate_and_rescale_attributes_for_scoring(scores, set_ids):
    score_vector_positions = get_score_vector_positions()
    min_max_scores = get_min_max_scores()
    individual_att_scores_dict = {}
    score_set_comb = list(zip(scores, set_ids))
    for att_scores, set_id in score_set_comb:
        for relevant_attribute in min_max_scores[set_id].keys():
            min_score = min_max_scores[set_id][relevant_attribute][0]
            max_score = min_max_scores[set_id][relevant_attribute][1]
            att_position = score_vector_positions[relevant_attribute]
            att_score = att_scores[att_position]
            rescaled_score = att_score * (max_score - min_score) + min_score
            try:
                individual_att_scores_dict[relevant_attribute].append(np.around(rescaled_score).astype(int))
            except KeyError:
                individual_att_scores_dict[relevant_attribute] = [np.around(rescaled_score).astype(int)]
    return individual_att_scores_dict


def pad_flat_text_sequences(index_sequences, max_essay_len):
    X = np.empty([len(index_sequences), max_essay_len], dtype=np.int32)

    for i, essay in enumerate(index_sequences):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)
        for j in range(num):
            word_id = sequence_ids[j]
            X[i, j] = word_id
        length = len(sequence_ids)
        X[i, length:] = 0
    return X


def pad_hierarchical_text_sequences(index_sequences, max_sentnum, max_sentlen):
    '''
    这段代码定义了一个函数，用于填充层次文本序列，使其达到指定的最大句子数和最大句子长度。
    这个函数通过将层次文本序列进行填充，使其在句子和文本级别达到指定的最大长度和数量。填充后的序列可以用于深度学习模型的训练和预测，确保输入数据的形状一致。
    index_sequences: 包含索引序列的列表，每个索引序列代表一个文本。
    max_sentnum: 所有文本中包含的最大句子数。
    max_sentlen: 每个句子中包含的最大单词数。
    '''
    # 创建一个空的 NumPy 数组 X，其形状为 (文本数, 最大句子数, 最大句子长度)，数据类型为整数。
    X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)
    # 遍历所有文本
    for i in range(len(index_sequences)):
        # 获取当前文本的索引序列，并计算该序列中包含的句子数量
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)
        # 遍历当前文本中的每个句子，并获取每个句子中的单词索引序列以及该句子的长度
        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            # 将当前句子中的单词索引填充到 X 数组的对应位置
            for k in range(length):
                wid = word_ids[k]
                X[i, j, k] = wid
            # 如果当前句子长度小于 max_sentlen，则在其后面用零填充，保证所有句子的长度相同
            X[i, j, length:] = 0

        # 如果当前文本包含的句子数量小于 max_sentnum，则在其后面用零填充，保证所有文本的句子数量相同。
        X[i, num:, :] = 0
    return X


def get_attribute_masks(score_matrix):
    mask_value = -1
    mask = np.cast['int32'](np.not_equal(score_matrix, mask_value))
    return mask


def load_word_embedding_dict(embedding_path):
    print("Loading GloVe ...")
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim, True


def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, caseless):
    '''
    这个函数构建了一个词嵌入矩阵，每一行代表词汇表中的一个词汇。如果词汇存在于预训练词嵌入字典中，则使用对应的嵌入向量；
    否则，生成一个随机初始化的嵌入向量。函数还会输出未出现在预训练词嵌入字典中的词汇数量及其比例。
    word_alphabet: 一个词汇表，包含所有词汇及其对应的索引。
    embedd_dict: 预训练的词嵌入字典。
    embedd_dim: 词嵌入向量的维度。
    caseless: 一个布尔值，表示是否忽略大小写。
    '''
    # 计算随机初始化词嵌入向量的范围 scale。这个范围基于嵌入向量的维度，用于生成均匀分布的随机数。
    scale = np.sqrt(3.0 / embedd_dim)
    # 初始化一个空的嵌入矩阵 embedd_table，其行数为词汇表的大小，列数为嵌入向量的维度。
    embedd_table = np.empty([len(word_alphabet), embedd_dim])
    # 将嵌入矩阵的第一行全部置零，通常用于填充或表示未定义的词汇。
    embedd_table[0, :] = np.zeros([1, embedd_dim])
    # 初始化一个计数器 oov_num，用于记录未出现在预训练词嵌入字典中的词汇数量
    oov_num = 0
    # 遍历词汇表中的每一个词汇。
    for word in word_alphabet:
        # 如果 caseless 为 True，将词汇转换为小写，否则保持原样。
        ww = word.lower() if caseless else word
        # 检查词汇（转换后的小写形式）是否在预训练词嵌入字典中。如果在，则使用对应的嵌入向量；否则，为该词汇生成一个范围在 [-scale, scale] 的均匀分布随机向量，并将 oov_num 计数器加一。
        if ww in embedd_dict:
            embedd = embedd_dict[ww]
        else:
            embedd = np.random.uniform(-scale, scale, [1, embedd_dim])
            oov_num += 1
        # 将词汇的嵌入向量存入嵌入矩阵 embedd_table 的对应行。word_alphabet[word] 返回该词汇在词汇表中的索引。
        embedd_table[word_alphabet[word], :] = embedd
    # 计算未出现在预训练词嵌入字典中的词汇比例 oov_ratio。
    oov_ratio = float(oov_num) / (len(word_alphabet) - 1)
    # 打印未出现在预训练词嵌入字典中的词汇数量和比例。
    print("OOV number =%s, OOV ratio = %f" % (oov_num, oov_ratio))
    return embedd_table
