import pickle
import nltk
import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils.general_utils import get_score_vector_positions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

url_replacer = '<url>'
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'
MAX_SENTLEN = 50
MAX_SENTNUM = 100
pd.set_option('mode.chained_assignment', None)


def replace_url(text):
    '''将文本中的 URL 替换为 <url>'''
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text


def tokenize(string):
    '''对输入的字符串进行分词处理， 并对包含 @ 符号的标记进行特殊处理。'''
    # word_tokenize 函数会根据空格和标点符号等进行分词，将字符串拆分为单词和标点符号
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        # 将 @ 后面的标记前面加上 @ 符号，并去除该标记中的数字及其后的所有字符，然后将原始的 @ 标记从列表中移除；
        if token == '@' and (index + 1) < len(tokens):
            tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
            tokens.pop(index)
    return tokens


def shorten_sentence(sent, max_sentlen):
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = len(tokens) / max_sentlen
            num = int(round(num))
            k_indexes = [(i + 1) * max_sentlen for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k - 1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j + 1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = len(token) / max_sentlen
                num = int(np.ceil(num))
                s_indexes = [(i + 1) * max_sentlen for i in range(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s - 1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j + 1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
        return [tokens]

    return new_tokens


def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
    processed_sents = []
    for sent in sents:
        if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
            s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
            ss = " ".join(s)
            ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)

    if create_vocab_flag:
        sent_tokens = [tokenize(sent) for sent in processed_sents]
        tokens = [w for sent in sent_tokens for w in sent]
        return tokens

    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
        sent_tokens.extend(shorten_sents_tokens)
    return sent_tokens


def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    '''
    text：输入的文本字符串。
    replace_url_flag（默认值为 True）：指示是否替换文本中的 URL。
    tokenize_sent_flag（默认值为 True）：指示是否将文本分割成句子。
    create_vocab_flag（默认值为 False）：指示是否创建词汇表。
    '''
    # 将文本中的 URL 替换掉
    text = replace_url(text)
    # 删除文本中的双引号
    text = text.replace(u'"', u'')
    if "..." in text:  # 将连续的三个或更多点替换为单独的三个点
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
    if "??" in text:  # 将连续的两个或更多问号替换为单独的一个问号
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
    if "!!" in text:  # 将连续的两个或更多感叹号替换为单独的一个感叹号
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)

    # 对文本进行分词
    tokens = tokenize(text)

    if tokenize_sent_flag:
        # 将词列表 tokens 拼接成一个字符串，每个词之间用空格分隔
        text = " ".join(tokens)
        # 将文本分割成句子。函数可能接受三个参数：text（处理后的文本字符串）、MAX_SENTLEN（句子的最大长度，假设是全局常量）、create_vocab_flag（指示是否创建词汇表）
        sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        # sent_tokens 是一个句子列表，每个句子由词组成
        return sent_tokens
    else:
        raise NotImplementedError


def is_number(token):
    return bool(num_regex.match(token))


def read_word_vocab(read_configs):
    '''读取训练数据中的单词，并统计它们的出现频率，以构建一个词汇表。根据单词的频率构建一个词汇表，并将每个单词映射到一个唯一的整数索引。'''
    vocab_size = read_configs['vocab_size']  # 4000
    file_path = read_configs['train_path']  # 'data/cross_prompt_attributes/1/train.pk'
    word_vocab_count = {}  # 创建一个空字典，用于存储单词的出现频率

    with open(file_path, 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    for index, essay in enumerate(train_essays_list):
        content = essay['content_text']  # 'content_text' 键存储了文章内容
        content = text_tokenizer(content, True, True, True)
        content = [w.lower() for w in content]
        for word in content:
            try:
                word_vocab_count[word] += 1
            except KeyError:
                word_vocab_count[word] = 1

    import operator
    # 将 word_vocab_count 字典按照值（即单词的频率）进行降序排序，并返回一个包含键-值对的列表
    sorted_word_freqs = sorted(word_vocab_count.items(), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1

    # 创建一个包含特殊单词（例如填充、未知单词和数字）的词汇表，并将其初始化为一个字典
    word_vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    vcb_len = len(word_vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        word_vocab[word] = index
        index += 1
    return word_vocab


def read_pos_vocab(read_configs):
    '''读取训练数据中的文本内容，并统计其中的词性标签（POS tags），最终生成一个词性标签的词汇表'''

    file_path = read_configs['train_path']  # 'data/cross_prompt_attributes/1/train.pk'
    pos_tags_count = {}  # 初始化一个空字典 pos_tags_count，用于统计各个词性标签的出现次数。

    # 文件内容是一个包含多篇文章的列表
    with open(file_path, 'rb') as train_file:
        train_essays_list = pickle.load(train_file)

    # 遍历 train_essays_list 中的前 16 篇文章，每次迭代中，index 是当前文章的索引，essay 是当前文章的内容。
    # todo 为什么 16
    for index, essay in enumerate(train_essays_list[:16]):
        # 从当前文章 essay 中提取文本内容，并将其存储在变量 content 中
        content = essay['content_text']
        # 对文本进行分词和其它预处理操作
        content = text_tokenizer(content, True, True, True)
        # 将 content 中的每个词转换为小写
        content = [w.lower() for w in content]
        # 使用 nltk 库对分词后的文本进行词性标注，结果存储在 tags 中。tags 是一个元组列表，每个元组包含一个词及其对应的词性标签。
        tags = nltk.pos_tag(content)
        for tag in tags:  # 遍历 tags 中的每个词性标注元组。
            tag = tag[1]  # tag[1]是单词，tag[1]是词性标签
            # 统计每个词性标签的数量
            try:
                pos_tags_count[tag] += 1
            except KeyError:
                pos_tags_count[tag] = 1

    # 初始化一个包含特殊标记（'<pad>' 和 '<unk>'）的词性标签词汇表 pos_tags
    pos_tags = {'<pad>': 0, '<unk>': 1}
    pos_len = len(pos_tags)  # 计算 pos_tags 中已有的标签数量（即特殊标记的数量）
    pos_index = pos_len  # pos_index=2
    for pos in pos_tags_count.keys():  # 遍历 pos_tags_count 字典中的所有词性标签
        pos_tags[pos] = pos_index
        pos_index += 1
    # 生成一个包含词性标签及其索引的词汇表，例如{'<pad>': 0, '<unk>': 1, 'adj': 2,...}
    return pos_tags


def get_readability_features(readability_path):
    with open(readability_path, 'rb') as fp:
        readability_features = pickle.load(fp)
    return readability_features


def get_linguistic_features(linguistic_features_path):
    features_df = pd.read_csv(linguistic_features_path)
    return features_df


def get_normalized_features(features_df, prompt_ids):
    '''
    对数据框 features_df 中的特征进行归一化处理，但保留一些列不进行归一化。
        # Encoded_Operations Encoded_KeyboardingStates Sequence_Index TimeStamp
    '''
    # 不需要归一化的列
    column_names_not_to_normalize = ['item_id', 'prompt_id', 'score1', 'score2', 'Encoded_Operations', 'Encoded_KeyboardingStates', 'Sequence_Index', 'TimeStamp']
    column_names_to_normalize = list(features_df.columns.values)
    # column_names_to_normalize 列表中只剩下需要归一化的列
    for col in column_names_not_to_normalize:
        column_names_to_normalize.remove(col)
    # final_columns，包含 item_id 列和所有需要归一化的列
    final_columns = ['item_id'] + column_names_to_normalize + ['Encoded_Operations'] + ['Encoded_KeyboardingStates'] + ['Sequence_Index'] + ['TimeStamp']
    # 初始化一个空的数据框 normalized_features_df，用于存储归一化后的特征。
    normalized_features_df = None
    for prompt_ in prompt_ids:
        # 创建一个布尔索引，标识 prompt_id 等于当前遍历值的行 prompt_。
        is_prompt_id = features_df['prompt_id'] == prompt_
        # 根据布尔索引提取相应的子数据框 prompt_id_df
        prompt_id_df = features_df[is_prompt_id]
        # 提取需要归一化的列的值，并将其转换为 NumPy 数组 x
        x = prompt_id_df[column_names_to_normalize].values
        # 创建一个 MinMaxScaler 对象，用于归一化处理
        min_max_scaler = preprocessing.MinMaxScaler()
        # 对 x 进行归一化处理，将结果存储在 normalized_pd1 中
        normalized_pd1 = min_max_scaler.fit_transform(x)
        # 将归一化后的数据转换回数据框 df_temp，列名和索引与原数据框一致
        df_temp = pd.DataFrame(normalized_pd1, columns=column_names_to_normalize, index=prompt_id_df.index)
        # 将归一化后的列值替换原数据框中对应的列值
        prompt_id_df[column_names_to_normalize] = df_temp
        # 提取最终需要的列（包括 item_id 和归一化后的列），形成新的数据框 final_df
        final_df = prompt_id_df[final_columns]
        if normalized_features_df is not None:
            normalized_features_df = pd.concat([normalized_features_df, final_df], ignore_index=True)
        else:
            normalized_features_df = final_df

    # todo 修改 如果normalized_features_df中存在nan值
    if normalized_features_df.isna().sum().sum() > 0:
        print("Number of NaNs in normalized_linguistic_features:", normalized_features_df.isna().sum().sum())
        # 使用均值填充 NaN 值
        filled_normalized_features_df = normalized_features_df.fillna(normalized_features_df.mean())

    return filled_normalized_features_df


def read_pr_pos(prompt_list, pos_tags):
    '''
    将prompt_list中的句子进行词性标注，并将标注结果转换为预先定义的词性标签集合中的索引
    prompt_list 是一个包含要进行词性标注的文本和其对应 ID 的列表，
    pos_tags 是一个字典，将词性标签映射到它们的索引。
    '''
    out_data = {
        'prompt_pos': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for i in range(len(prompt_list)):
        prompt_id = int(prompt_list['prompt_id'][i])  # prompt id
        content = prompt_list['prompt'][i]  # prompt

        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

        # 将每个句子的词性标注结果存储在 sent_tag_indices 列表中，并将该列表添加到 out_data['prompt_pos'] 中
        sent_tag_indices = []
        tag_indices = []
        # 对sent_tokens（prompt）进行词性标注，并根据pos_tags转为索引
        for sent in sent_tokens:
            length = len(sent)
            if length > 0:
                if out_data['max_sentlen'] < length:
                    out_data['max_sentlen'] = length
                tags = nltk.pos_tag(sent)
                for tag in tags:
                    if tag[1] in pos_tags:
                        tag_indices.append(pos_tags[tag[1]])
                    else:
                        tag_indices.append(pos_tags['<unk>'])
                sent_tag_indices.append(tag_indices)
                tag_indices = []

        out_data['prompt_pos'].append(sent_tag_indices)
        out_data['prompt_ids'].append(prompt_id)
        if out_data['max_sentnum'] < len(sent_tag_indices):
            out_data['max_sentnum'] = len(sent_tag_indices)
    print(' prompt_pos size: {}'.format(len(out_data['prompt_pos'])))
    return out_data


def read_prompts_pos(prompt_file, pos_tags):
    '''
    prompt_file：论文题目文件路径，'data/prompt_info_pp.csv'
    pos_tags：词性标签列表
    '''
    # 读取 论文题目文件
    prompt_list = pd.read_csv(prompt_file)
    prompt_data = read_pr_pos(prompt_list, pos_tags)
    return prompt_data


def read_essay_sets_with_prompt_only_word_emb(essay_list, readability_features, normalized_features_df, process_df, prompt_data,
                                              prompt_pos_data, pos_tags):
    '''
    essay_list: 包含作文数据的列表
    readability_features: 可读性特征
    normalized_features_df: 标准化后的特征数据框
    prompt_data: 提示数据，word embedding
    prompt_pos_data: 提示词的词性标注数据，POS embedding
    pos_tags: 词性标签集合
    '''
    out_data = {
        'essay_ids': [],
        'pos_x': [],
        'prompt_words': [],
        'prompt_pos': [],
        'readability_x': [],
        'features_x': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1,
        'process_timestamp': [],
        'processO': [],
        'processK': [],
        'max_proclen': -1
    }
    for essay in essay_list:
        essay_id = int(essay['essay_id'])  # essay_id
        essay_set = int(essay['prompt_id'])  # prompt id
        content = essay['content_text']  # content_text

        scores_and_positions = get_score_vector_positions()
        y_vector = [-1] * len(scores_and_positions)  # [-1, -1, ...]
        # 遍历评分向量中的每个评分，将评分添加到 y_vector 中
        for score in scores_and_positions.keys():
            if score in essay.keys():
                y_vector[scores_and_positions[score]] = int(essay[score])

        out_data['data_y'].append(y_vector)
        '''根据essay_id 提取可读性特征，并添加到 out_data 中'''  #
        # 使用np.where函数查找essay_id所在的行索引
        # item_index = np.where(readability_features[:, :1] == essay_id)
        item_index = np.where(readability_features[:, :1] == 91)  # todo essay_id固定为91，因为有的找不到
        item_row_index = item_index[0][0]
        item_features = readability_features[item_row_index][1:]
        out_data['readability_x'].append(item_features)

        '''根据essay_id 提取标准化特征，并添加到 out_data 中'''  # todo item_id等价essay_id
        feats_df = normalized_features_df[normalized_features_df.loc[:, 'item_id'] == essay_id]
        feats_list = feats_df.values.tolist()[0][1:]  # 第一列item_id的值不要
        out_data['features_x'].append(feats_list)

        '''根据essay_id 提取过程性数据，并添加到 out_data 中'''
        proc_df = process_df[process_df.loc[:, 'item_id'] == essay_id]

        process_str = proc_df['Encoded_Operations'].iloc[0]
        process_list = [int(item) for item in process_str.split(',')]
        if out_data['max_proclen'] < len(process_list):
            out_data['max_proclen'] = len(process_list)
        out_data['processO'].append(process_list)

        timestamp_str = proc_df['Encoded_KeyboardingStates'].iloc[0]
        timestamp_list = [int(float(item)) for item in timestamp_str.split(',')]
        out_data['processK'].append(timestamp_list)

        timestamp_str = proc_df['TimeStamp'].iloc[0]
        timestamp_list = [int(float(item)) for item in timestamp_str.split(',')]
        out_data['process_timestamp'].append(timestamp_list)

        '''提取prompt，并添加到 out_data 中'''
        prompt_index = prompt_data['prompt_ids'].index(essay_set)
        prompt = prompt_data['prompt_words'][prompt_index]
        out_data['prompt_words'].append(prompt)

        '''提取prompt的词性标注，并添加到 out_data 中'''
        prompt_pos = prompt_pos_data['prompt_pos'][prompt_index]
        out_data['prompt_pos'].append(prompt_pos)

        '''处理作文内容并提取词性标注'''
        # 将作文内容分成句子和单词
        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

        sent_tag_indices = []
        tag_indices = []
        # 遍历每个句子
        for sent in sent_tokens:
            # 获取句子的长度，并更新最大句子长度
            length = len(sent)
            if length > 0:
                if out_data['max_sentlen'] < length:
                    out_data['max_sentlen'] = length
                # 使用 nltk.pos_tag 获取每个单词的词性标注
                tags = nltk.pos_tag(sent)
                # 将词性标注转换为索引
                for tag in tags:
                    if tag[1] in pos_tags:
                        tag_indices.append(pos_tags[tag[1]])
                    else:
                        tag_indices.append(pos_tags['<unk>'])
                # 将词性标注索引添加到 sent_tag_indices 中
                sent_tag_indices.append(tag_indices)
                tag_indices = []

        # 将词性标注索引，提示ID和作文ID添加到 out_data 中
        out_data['pos_x'].append(sent_tag_indices)
        out_data['prompt_ids'].append(essay_set)
        out_data['essay_ids'].append(essay_id)
        # 更新最大句子数量
        if out_data['max_sentnum'] < len(sent_tag_indices):
            out_data['max_sentnum'] = len(sent_tag_indices)

    # 使用断言检查数据的一致性
    assert (len(out_data['pos_x']) == len(out_data['readability_x']))
    assert (len(out_data['pos_x']) == len(out_data['prompt_words']))
    assert (len(out_data['pos_x']) == len(out_data['prompt_pos']))
    # 打印 pos_x 的大小
    print(' pos_x size: {}'.format(len(out_data['pos_x'])))
    print(' readability_x size: {}'.format(len(out_data['readability_x'])))
    return out_data


def read_essays_prompts(read_configs, prompt_data, prompt_pos_data, pos_tags):
    # 从指定路径读取可读性特征
    readability_features = get_readability_features(read_configs['readability_path'])
    # print(len(readability_features[0]),readability_features.size) 36 467208
    # 从指定路径读取语言学特征 data/LDA/hand_crafted_final_1.csv
    linguistic_features = get_linguistic_features(read_configs['features_path'])
    # todo 修改
    normalized_linguistic_features = get_normalized_features(linguistic_features, read_configs['prompt_ids'])
    process_df = normalized_linguistic_features[['item_id', 'Encoded_Operations', 'Encoded_KeyboardingStates', 'Sequence_Index', 'TimeStamp']]
    normalized_linguistic_features = normalized_linguistic_features.iloc[:, :-5]

    with open(read_configs['train_path'], 'rb') as train_file:
        train_essays_list = pickle.load(train_file)
    with open(read_configs['dev_path'], 'rb') as dev_file:
        dev_essays_list = pickle.load(dev_file)
    with open(read_configs['test_path'], 'rb') as test_file:
        test_essays_list = pickle.load(test_file)
    train_data = read_essay_sets_with_prompt_only_word_emb(train_essays_list, readability_features,
                                                           normalized_linguistic_features, process_df, prompt_data, prompt_pos_data,
                                                           pos_tags)
    dev_data = read_essay_sets_with_prompt_only_word_emb(dev_essays_list, readability_features,
                                                         normalized_linguistic_features, process_df,prompt_data, prompt_pos_data,
                                                         pos_tags)
    test_data = read_essay_sets_with_prompt_only_word_emb(test_essays_list, readability_features,
                                                          normalized_linguistic_features, process_df,prompt_data, prompt_pos_data,
                                                          pos_tags)
    return train_data, dev_data, test_data


def read_prompts_word(prompt_list, vocab):
    '''
    将文本内容转换为词汇表中的索引，以便后续的处理和分析，同时记录一些有用的统计信息，如最大句子数量和最大句子长度。
    一个包含文本及其 ID 的列表 prompt_list 和一个词汇表 vocab
    '''
    out_data = {
        'prompt_words': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    for i in range(len(prompt_list)):
        prompt_id = int(prompt_list['prompt_id'][i])  # prompt id
        content = prompt_list['prompt'][i]

        sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
        sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

        sent_indices = []
        indices = []

        for sent in sent_tokens:
            length = len(sent)
            if length > 0:
                if out_data['max_sentlen'] < length:
                    out_data['max_sentlen'] = length
                for word in sent:
                    if is_number(word):
                        indices.append(vocab['<num>'])
                    elif word in vocab:
                        indices.append(vocab[word])
                    else:
                        indices.append(vocab['<unk>'])
                sent_indices.append(indices)
                indices = []

        out_data['prompt_words'].append(sent_indices)
        out_data['prompt_ids'].append(prompt_id)
        if out_data['max_sentnum'] < len(sent_indices):
            out_data['max_sentnum'] = len(sent_indices)
    print(' prompt_words size: {}'.format(len(out_data['prompt_words'])))
    return out_data


def read_prompts_we(prompt_file, word_vocab):
    prompt_list = pd.read_csv(prompt_file)
    prompt_data = read_prompts_word(prompt_list, word_vocab)
    return prompt_data


if __name__ == '__main__':
    # 示例文本
    y_vector = [-1] * 10
    print(y_vector)
    pass
